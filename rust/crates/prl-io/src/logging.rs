//! Console logging identical to prime-rl's loguru setup (`utils/logger.py`).
//!
//! Every crate logs with `tracing::{debug, info, warn, error}!`; SUCCESS (loguru level 25) is
//! `prl_io::success!(...)`, an INFO event carrying `success = true`.

use std::fmt::Write as _;
use std::fs::File;
use std::io::Write as _;
use std::path::Path;
use std::sync::Mutex;

use tracing::field::{Field, Visit};
use tracing::{Event, Level, Subscriber};
use tracing_subscriber::filter::filter_fn;
use tracing_subscriber::layer::{Context, Layer, SubscriberExt};
use tracing_subscriber::registry::Registry;
use tracing_subscriber::util::SubscriberInitExt;
use prl_wire::json::to_python_json;
use prl_wire::Value;

/// Log at loguru's SUCCESS level.
#[macro_export]
macro_rules! success {
    ($($arg:tt)+) => { ::tracing::info!(success = true, $($arg)+) };
}

const NO_BOLD: &str = "\x1b[22m";
const RESET: &str = "\x1b[0m";
const DIM: &str = "\x1b[2m";

/// Loguru levels, ordered by severity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum LogLevel {
    Trace,
    Debug,
    Info,
    Success,
    Warning,
    Error,
}

impl LogLevel {
    fn parse(level: &str) -> anyhow::Result<Self> {
        Ok(match level.to_ascii_uppercase().as_str() {
            "TRACE" => Self::Trace,
            "DEBUG" => Self::Debug,
            "INFO" => Self::Info,
            "SUCCESS" => Self::Success,
            "WARNING" | "WARN" => Self::Warning,
            "ERROR" | "CRITICAL" => Self::Error,
            other => anyhow::bail!("Level '{other}' does not exist"),
        })
    }

    fn name(self) -> &'static str {
        match self {
            Self::Trace => "TRACE",
            Self::Debug => "DEBUG",
            Self::Info => "INFO",
            Self::Success => "SUCCESS",
            Self::Warning => "WARNING",
            Self::Error => "ERROR",
        }
    }

    /// Loguru's default color markup for the level.
    fn color(self) -> &'static str {
        match self {
            Self::Trace => "\x1b[36m\x1b[1m",
            Self::Debug => "\x1b[34m\x1b[1m",
            Self::Info => "\x1b[1m",
            Self::Success => "\x1b[32m\x1b[1m",
            Self::Warning => "\x1b[33m\x1b[1m",
            Self::Error => "\x1b[31m\x1b[1m",
        }
    }
}

struct Record<'a> {
    level: LogLevel,
    message: &'a str,
    time: chrono::DateTime<chrono::Local>,
    file: &'a str,
    line: u32,
}

/// The colorized console line.
fn format_console(record: &Record, with_location: bool) -> String {
    let color = record.level.color();
    let mut line = format!(
        "{DIM}{}{RESET} {color}{:>7}{RESET} {color}{NO_BOLD}{}{RESET}{RESET}",
        record.time.format("%H:%M:%S"),
        record.level.name(),
        record.message,
    );
    if with_location {
        let _ = write!(line, "{color}{NO_BOLD} [{}::{}]{RESET}{RESET}", record.file, record.line);
    }
    line
}

/// The flat JSON line (Loki/Grafana) as Python's `json.dumps` renders it.
fn format_json(record: &Record) -> String {
    let entry: prl_wire::Map = [
        ("timestamp", Value::Str(record.time.format("%Y-%m-%dT%H:%M:%S%.6f%:z").to_string())),
        ("level", Value::Str(record.level.name().into())),
        ("message", Value::Str(record.message.into())),
        ("module", Value::Str(record.file.strip_suffix(".rs").unwrap_or(record.file).into())),
        ("function", Value::Str(String::new())),
        ("line", Value::Int(record.line.into())),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_owned(), v))
    .collect();
    to_python_json(&Value::Map(entry), false)
}

/// `format_time` of `utils/logger.py`.
pub fn format_time(seconds: f64) -> String {
    if seconds < 1.0 {
        return format!("{:.0}ms", seconds * 1000.0);
    }
    if seconds < 60.0 {
        return format!("{seconds:.1}s");
    }
    if seconds < 3600.0 {
        let m = (seconds / 60.0).floor();
        let s = seconds - m * 60.0;
        return format!("{}m {}s", m as i64, s as i64);
    }
    if seconds < 86400.0 {
        let h = (seconds / 3600.0).floor();
        let m = ((seconds - h * 3600.0) / 60.0).floor();
        return format!("{}h {}m", h as i64, m as i64);
    }
    let d = (seconds / 86400.0).floor();
    let h = ((seconds - d * 86400.0) / 3600.0).floor();
    format!("{}d {}h", d as i64, h as i64)
}

#[derive(Default)]
struct MessageVisitor {
    message: String,
    success: bool,
    console: bool,
    fields: String,
}

impl Visit for MessageVisitor {
    fn record_bool(&mut self, field: &Field, value: bool) {
        if field.name() == "success" {
            self.success = value;
        } else if field.name() == "console" {
            self.console = value;
        } else {
            self.record_debug(field, &value);
        }
    }

    fn record_str(&mut self, field: &Field, value: &str) {
        if field.name() == "message" {
            self.message.push_str(value);
        } else {
            let _ = write!(self.fields, " {}={}", field.name(), value);
        }
    }

    fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
        if field.name() == "message" {
            let _ = write!(self.message, "{value:?}");
        } else {
            let _ = write!(self.fields, " {}={:?}", field.name(), value);
        }
    }
}

struct LoguruLayer {
    json: bool,
    /// Loguru's debug format appends ` [file::line]`.
    with_location: bool,
    level: LogLevel,
    console_level: LogLevel,
    file: Option<Mutex<File>>,
}

fn level_of(event_level: &Level, success: bool) -> LogLevel {
    match *event_level {
        Level::TRACE => LogLevel::Trace,
        Level::DEBUG => LogLevel::Debug,
        Level::INFO if success => LogLevel::Success,
        Level::INFO => LogLevel::Info,
        Level::WARN => LogLevel::Warning,
        Level::ERROR => LogLevel::Error,
    }
}

impl<S: Subscriber> Layer<S> for LoguruLayer {
    fn on_event(&self, event: &Event<'_>, _ctx: Context<'_, S>) {
        let mut visitor = MessageVisitor::default();
        event.record(&mut visitor);
        let level = level_of(event.metadata().level(), visitor.success);
        if level < self.level {
            return;
        }
        let message = visitor.message + &visitor.fields;
        let path = event.metadata().file().unwrap_or("");
        let record = Record {
            level,
            message: &message,
            time: chrono::Local::now(),
            file: path.rsplit('/').next().unwrap_or(path),
            line: event.metadata().line().unwrap_or(0),
        };
        // Records carrying `console = true` (e.g. a banner) pass the console threshold.
        if level >= self.console_level || visitor.console {
            let line = if self.json { format_json(&record) } else { format_console(&record, self.with_location) };
            let mut stdout = std::io::stdout().lock();
            let _ = writeln!(stdout, "{line}");
            let _ = stdout.flush();
        }
        if let Some(file) = &self.file {
            let mut file = file.lock().expect("log file lock");
            let _ = writeln!(file, "{} {:>7} {}", record.time.format("%H:%M:%S"), level.name(), record.message);
        }
    }
}

/// Install the process logger: colorized loguru-format console lines (or JSON lines) on stdout and,
/// with `log_file`, a plain-text file sink (appended). `console_level` raises the console
/// threshold above `level`; the file keeps everything. Only events from prime-rl crates (`prl_*`
/// targets) pass `level`; third-party crates log at WARNING and above.
pub fn setup_logger(level: &str, json_logging: bool, log_file: Option<&Path>, console_level: Option<&str>) -> anyhow::Result<()> {
    let level = LogLevel::parse(level)?;
    let console_level = console_level.map(LogLevel::parse).transpose()?.unwrap_or(level);
    let file = match log_file {
        Some(path) => Some(Mutex::new(File::options().create(true).append(true).open(path)?)),
        None => None,
    };
    let layer = LoguruLayer { json: json_logging, with_location: level == LogLevel::Debug, level, console_level, file };
    let filter = filter_fn(|meta| meta.target().starts_with("prl_") || *meta.level() <= Level::WARN);
    Registry::default().with(layer.with_filter(filter)).try_init()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    fn record(level: LogLevel, message: &str) -> Record<'_> {
        let time = chrono::Local.with_ymd_and_hms(2026, 10, 28, 8, 22, 44).unwrap();
        Record { level, message, time, file: "io_log.py", line: 6 }
    }

    #[test]
    fn lines_match_loguru() {
        // Captured from Python's `setup_logger("info")` / `setup_logger("debug")`.
        assert_eq!(
            format_console(&record(LogLevel::Success, "ok message"), false),
            "\x1b[2m08:22:44\x1b[0m \x1b[32m\x1b[1mSUCCESS\x1b[0m \x1b[32m\x1b[1m\x1b[22mok message\x1b[0m\x1b[0m"
        );
        assert_eq!(
            format_console(&record(LogLevel::Debug, "dbg message"), true),
            "\x1b[2m08:22:44\x1b[0m \x1b[34m\x1b[1m  DEBUG\x1b[0m \x1b[34m\x1b[1m\x1b[22mdbg message\x1b[0m\x1b[0m\x1b[34m\x1b[1m\x1b[22m [io_log.py::6]\x1b[0m\x1b[0m"
        );
        let json = format_json(&record(LogLevel::Info, "é"));
        assert!(json.ends_with(r#""level": "INFO", "message": "\u00e9", "module": "io_log.py", "function": "", "line": 6}"#), "{json}");
    }

    #[test]
    fn format_time_matches_python() {
        let cases = [(0.0004, "0ms"), (0.9996, "1000ms"), (59.96, "60.0s"), (61.5, "1m 1s"), (7322.0, "2h 2m"), (200000.0, "2d 7h")];
        for (seconds, expected) in cases {
            assert_eq!(format_time(seconds), expected, "{seconds}");
        }
    }
}
