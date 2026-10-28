//! Resume an interrupted eval from its trace stream (port of `eval/resume.py`).
//!
//! The stream records what landed, so it is what a resume continues from. Successful episodes
//! rejoin the epoch; failed and cut-off ones are owed again. Each attempt's saved resolved config
//! must match the current one (except `resume`) unless `--resume.skip-checks` is set.

use std::fs;
use std::io::{BufRead, BufReader, Read};
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use prl_monitors::file::chunks::{chunk_numbers, chunk_path};
use prl_monitors::file::{file_monitor_dir, trace_stream};
use serde::Serialize;
use serde_json::{Map, Value};

/// The resolved config an attempt stamps into its file monitor directory.
pub const CONFIG_NAME: &str = "eval.json";

/// Write the resolved config beside the attempt's episodes, as `orjson.dumps(config, OPT_INDENT_2)`.
pub fn stamp_config(run_dir: &Path, config: &Value) -> Result<()> {
    let directory = file_monitor_dir(run_dir);
    fs::create_dir_all(&directory)?;
    fs::write(directory.join(CONFIG_NAME), to_orjson_indent2(config)?)?;
    Ok(())
}

/// `orjson.dumps(value, option=OPT_INDENT_2)`: two-space pretty JSON with orjson float formatting.
pub fn to_orjson_indent2(value: &Value) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    let mut ser = serde_json::Serializer::with_formatter(&mut out, OrjsonPretty::default());
    value.serialize(&mut ser)?;
    Ok(out)
}

#[derive(Default)]
struct OrjsonPretty<'a> {
    inner: serde_json::ser::PrettyFormatter<'a>,
}

macro_rules! delegate {
    ($($name:ident($($arg:ident: $ty:ty),*);)*) => {
        $(fn $name<W: ?Sized + std::io::Write>(&mut self, writer: &mut W $(, $arg: $ty)*) -> std::io::Result<()> {
            self.inner.$name(writer $(, $arg)*)
        })*
    };
}

impl serde_json::ser::Formatter for OrjsonPretty<'_> {
    delegate! {
        begin_array();
        end_array();
        begin_array_value(first: bool);
        end_array_value();
        begin_object();
        end_object();
        begin_object_key(first: bool);
        begin_object_value();
        end_object_value();
    }

    fn write_f64<W: ?Sized + std::io::Write>(&mut self, writer: &mut W, value: f64) -> std::io::Result<()> {
        let mut buf = Vec::new();
        prl_wire::json::write_f64(&mut buf, value);
        writer.write_all(&buf)
    }
}

/// Every record of a trace stream (one JSON document per line), in chunk order. A torn line
/// (the interrupted process died mid-append) ends the stream.
pub fn read_records(stream: &Path) -> Result<Vec<(Vec<u8>, Value)>> {
    let mut records = Vec::new();
    for number in chunk_numbers(stream)?.into_keys() {
        let plain = chunk_path(stream, number);
        let reader: Box<dyn Read> = match fs::File::open(&plain) {
            Ok(file) => Box::new(file),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                let sealed = PathBuf::from(format!("{}.zst", plain.display()));
                let file = fs::File::open(&sealed).with_context(|| format!("open {}", sealed.display()))?;
                Box::new(zstd::stream::read::Decoder::new(file)?)
            }
            Err(e) => return Err(e.into()),
        };
        let mut reader = BufReader::new(reader);
        let mut line = Vec::new();
        loop {
            line.clear();
            if reader.read_until(b'\n', &mut line)? == 0 {
                break;
            }
            match serde_json::from_slice::<Value>(&line) {
                Ok(record) => records.push((line.clone(), record)),
                Err(_) => return Ok(records),
            }
        }
    }
    Ok(records)
}

/// File monitor directories of the run's earlier attempts (`monitors/file.attempt_N`), oldest first.
pub fn archives(run_dir: &Path) -> Result<Vec<PathBuf>> {
    let monitors = file_monitor_dir(run_dir).parent().expect("file monitor dir has a parent").to_path_buf();
    let mut found: Vec<(u64, PathBuf)> = Vec::new();
    if monitors.is_dir() {
        for entry in fs::read_dir(&monitors)? {
            let path = entry?.path();
            let name = path.file_name().unwrap_or_default().to_string_lossy().into_owned();
            if let Some(n) = name.strip_prefix("file.attempt_") {
                let number = n.parse().with_context(|| format!("archive name {name}"))?;
                found.push((number, path));
            }
        }
    }
    found.sort();
    Ok(found.into_iter().map(|(_, path)| path).collect())
}

/// Python `==` over JSON values: numbers compare numerically (`1 == 1.0`), objects ignore key order.
pub fn py_eq(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Number(x), Value::Number(y)) => match (x.as_i64(), y.as_i64()) {
            (Some(x), Some(y)) => x == y,
            _ => match (x.as_u64(), y.as_u64()) {
                (Some(x), Some(y)) => x == y,
                _ => x.as_f64() == y.as_f64(),
            },
        },
        (Value::Bool(x), Value::Number(y)) | (Value::Number(y), Value::Bool(x)) => y.as_f64() == Some(f64::from(u8::from(*x))),
        (Value::Array(x), Value::Array(y)) => x.len() == y.len() && x.iter().zip(y).all(|(x, y)| py_eq(x, y)),
        (Value::Object(x), Value::Object(y)) => {
            x.len() == y.len() && x.iter().all(|(k, v)| y.get(k).is_some_and(|w| py_eq(v, w)))
        }
        _ => a == b,
    }
}

/// Top-level keys whose values differ between two config dumps (or exist on one side only), sorted.
pub fn changed_keys(expected: &Map<String, Value>, previous: &Map<String, Value>) -> Vec<String> {
    let mut keys: Vec<String> = expected
        .keys()
        .chain(previous.keys())
        .filter(|key| match (expected.get(*key), previous.get(*key)) {
            (Some(a), Some(b)) => !py_eq(a, b),
            _ => true,
        })
        .cloned()
        .collect();
    keys.sort();
    keys.dedup();
    keys
}

fn without_resume(config: &Value) -> Result<Map<String, Value>> {
    let Value::Object(map) = config else { bail!("a resolved eval config must be a JSON object") };
    let mut map = map.clone();
    map.shift_remove("resume");
    Ok(map)
}

/// Successful episodes from every attempt (the rollout planner deduplicates them). The current
/// file monitor directory joins the archives so the resumed attempt writes a fresh stream; nothing
/// is deleted. `config` is the current `model_dump(mode="json")`; `parse` validates one record line
/// into an episode and reports its `ok`.
pub fn take_landed<E>(
    run_dir: &Path,
    config: &Value,
    skip_checks: bool,
    mut parse: impl FnMut(&[u8]) -> Result<(E, bool)>,
) -> Result<Vec<E>> {
    let current = file_monitor_dir(run_dir);
    let stream_rel = trace_stream(run_dir)
        .strip_prefix(&current)
        .expect("the trace stream lives in the file monitor dir")
        .to_path_buf();
    let mut directories = archives(run_dir)?;
    if current.is_dir() || directories.is_empty() {
        directories.push(current.clone());
    }
    let expected = without_resume(config)?;
    if skip_checks {
        tracing::warn!("Skipping resume compatibility checks; saved episodes may use different experiment settings.");
    }
    let mut landed = Vec::new();
    for directory in &directories {
        if !skip_checks {
            let saved_path = directory.join(CONFIG_NAME);
            if !saved_path.is_file() {
                bail!(
                    "--resume: no saved config at {}. Set --resume.skip-checks to reuse episodes without a saved config.",
                    saved_path.display()
                );
            }
            let previous: Value = serde_json::from_slice(&fs::read(&saved_path)?)
                .with_context(|| format!("read {}", saved_path.display()))?;
            let changed = changed_keys(&expected, &without_resume(&previous)?);
            if !changed.is_empty() {
                bail!(
                    "--resume: config differs from {} in [{}]. Use the saved config, start a fresh run, or set \
                     --resume.skip-checks if the changes are intentional.",
                    saved_path.display(),
                    changed.join(", ")
                );
            }
        }
        let stream = directory.join(&stream_rel);
        if stream.is_dir() {
            for (line, record) in read_records(&stream)? {
                let (episode, ok) = parse(&line)?;
                if ok && record.get("traces").is_some() {
                    landed.push(episode);
                }
            }
        }
    }
    if current.is_dir() {
        let number = archives(run_dir)?.len() + 1;
        fs::rename(&current, current.with_file_name(format!("file.attempt_{number}")))?;
    }
    Ok(landed)
}

/// [`take_landed`] over wire episodes (`vf.WireEpisode.model_validate(record)`).
pub fn take_landed_episodes(run_dir: &Path, config: &Value, skip_checks: bool) -> Result<Vec<prl_wire::Episode>> {
    take_landed(run_dir, config, skip_checks, |line| {
        let episode = prl_wire::Episode::from_record_json(line)
            .map_err(|e| anyhow::anyhow!("invalid saved episode at {}: {}", e.path, e.message))?;
        let ok = episode.ok;
        Ok((episode, ok))
    })
}
