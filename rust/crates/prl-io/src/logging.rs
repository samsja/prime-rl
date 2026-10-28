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
pub enum LogLevel {
    Trace,
    Debug,
    Info,
    Success,
    Warning,
    Error,
}

impl LogLevel {
    pub fn parse(level: &str) -> anyhow::Result<Self> {
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

    pub fn name(self) -> &'static str {
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

/// One log record, independent of tracing.
#[derive(Debug, Clone)]
pub struct Record<'a> {
    pub level: LogLevel,
    pub message: &'a str,
    pub time: chrono::DateTime<chrono::Local>,
    pub file: &'a str,
    pub line: u32,
    pub module: &'a str,
}

/// The colorized console line (without trailing newline).
pub fn format_console(record: &Record, tag: Option<&str>, with_location: bool) -> String {
    let color = record.level.color();
    let tag_prefix = tag.map(|t| format!("[{t}] ")).unwrap_or_default();
    let mut line = format!(
        "{DIM}{}{RESET} {color}{:>7}{RESET} {color}{NO_BOLD}{tag_prefix}{}{RESET}{RESET}",
        record.time.format("%H:%M:%S"),
        record.level.name(),
        record.message,
    );
    if with_location {
        let _ = write!(line, "{color}{NO_BOLD} [{}::{}]{RESET}{RESET}", record.file, record.line);
    }
    line
}

/// The plain file-sink line (without trailing newline).
pub fn format_plain(record: &Record, tag: Option<&str>) -> String {
    let tag_prefix = tag.map(|t| format!("[{t}] ")).unwrap_or_default();
    format!("{} {:>7} {tag_prefix}{}", record.time.format("%H:%M:%S"), record.level.name(), record.message)
}

/// The flat JSON line (Loki/Grafana), as Python's `json.dumps` renders it.
pub fn format_json(record: &Record, tag: Option<&str>) -> String {
    let timestamp = record.time.format("%Y-%m-%dT%H:%M:%S%.6f%:z").to_string();
    let mut out = format!(
        "{{\"timestamp\": {}, \"level\": {}, \"message\": {}, \"module\": {}, \"function\": \"\", \"line\": {}",
        py_json_str(&timestamp),
        py_json_str(record.level.name()),
        py_json_str(record.message),
        py_json_str(record.module),
        record.line,
    );
    if let Some(tag) = tag {
        let _ = write!(out, ", \"tag\": {}", py_json_str(tag));
    }
    out.push('}');
    out
}

/// A JSON string literal with `ensure_ascii=True` escaping.
fn py_json_str(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 || (c as u32) > 0x7e => {
                let mut buf = [0u16; 2];
                for unit in c.encode_utf16(&mut buf) {
                    let _ = write!(out, "\\u{unit:04x}");
                }
            }
            c => out.push(c),
        }
    }
    out.push('"');
    out
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
    fields: String,
}

impl Visit for MessageVisitor {
    fn record_bool(&mut self, field: &Field, value: bool) {
        if field.name() == "success" {
            self.success = value;
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Sink {
    Console { with_location: bool },
    Json,
}

struct LoguruLayer {
    sink: Sink,
    level: LogLevel,
    tag: Option<String>,
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
        let file = path.rsplit('/').next().unwrap_or(path);
        let record = Record {
            level,
            message: &message,
            time: chrono::Local::now(),
            file,
            line: event.metadata().line().unwrap_or(0),
            module: file.strip_suffix(".rs").unwrap_or(file),
        };
        let line = match self.sink {
            Sink::Console { with_location } => format_console(&record, self.tag.as_deref(), with_location),
            Sink::Json => format_json(&record, self.tag.as_deref()),
        };
        let mut stdout = std::io::stdout().lock();
        let _ = writeln!(stdout, "{line}");
        let _ = stdout.flush();
        if let Some(file) = &self.file {
            let mut file = file.lock().expect("log file lock");
            let _ = writeln!(file, "{}", format_plain(&record, self.tag.as_deref()));
        }
    }
}

/// Install the process logger: colorized loguru-format console lines (or JSON lines) on stdout and,
/// with `log_file`, a plain-text file sink. Only events from prime-rl crates (`prl_*` targets) pass
/// `level`; third-party crates log at WARNING and above.
pub fn setup_logger(level: &str, json_logging: bool, log_file: Option<&Path>, tag: Option<&str>) -> anyhow::Result<()> {
    let level = LogLevel::parse(level)?;
    let sink = if json_logging { Sink::Json } else { Sink::Console { with_location: level == LogLevel::Debug } };
    let file = match log_file {
        Some(path) => Some(Mutex::new(File::options().create(true).append(true).open(path)?)),
        None => None,
    };
    let layer = LoguruLayer { sink, level, tag: tag.map(str::to_owned), file };
    let filter = filter_fn(|meta| meta.target().starts_with("prl_") || *meta.level() <= Level::WARN);
    Registry::default().with(layer.with_filter(filter)).try_init()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    fn record(level: LogLevel, message: &str) -> Record<'_> {
        Record {
            level,
            message,
            time: chrono::Local.with_ymd_and_hms(2026, 10, 28, 8, 22, 44).unwrap(),
            file: "io_log.py",
            line: 6,
            module: "io_log",
        }
    }

    #[test]
    fn console_lines_match_loguru() {
        // Captured from `setup_logger("info")` / `setup_logger("debug")` in Python.
        assert_eq!(
            format_console(&record(LogLevel::Info, "info message"), None, false),
            "\x1b[2m08:22:44\x1b[0m \x1b[1m   INFO\x1b[0m \x1b[1m\x1b[22minfo message\x1b[0m\x1b[0m"
        );
        assert_eq!(
            format_console(&record(LogLevel::Success, "ok message"), None, false),
            "\x1b[2m08:22:44\x1b[0m \x1b[32m\x1b[1mSUCCESS\x1b[0m \x1b[32m\x1b[1m\x1b[22mok message\x1b[0m\x1b[0m"
        );
        assert_eq!(
            format_console(&record(LogLevel::Warning, "warn message"), None, false),
            "\x1b[2m08:22:44\x1b[0m \x1b[33m\x1b[1mWARNING\x1b[0m \x1b[33m\x1b[1m\x1b[22mwarn message\x1b[0m\x1b[0m"
        );
        assert_eq!(
            format_console(&record(LogLevel::Error, "err message"), None, false),
            "\x1b[2m08:22:44\x1b[0m \x1b[31m\x1b[1m  ERROR\x1b[0m \x1b[31m\x1b[1m\x1b[22merr message\x1b[0m\x1b[0m"
        );
        assert_eq!(
            format_console(&record(LogLevel::Debug, "dbg message"), None, true),
            "\x1b[2m08:22:44\x1b[0m \x1b[34m\x1b[1m  DEBUG\x1b[0m \x1b[34m\x1b[1m\x1b[22mdbg message\x1b[0m\x1b[0m\x1b[34m\x1b[1m\x1b[22m [io_log.py::6]\x1b[0m\x1b[0m"
        );
    }

    #[test]
    fn json_line_uses_python_separators() {
        let line = format_json(&record(LogLevel::Info, "json, message: \"x\""), None);
        assert!(line.starts_with("{\"timestamp\": \"2026-10-28T08:22:44.000000"), "{line}");
        assert!(line.contains("\"level\": \"INFO\", \"message\": \"json, message: \\\"x\\\"\""), "{line}");
        assert!(line.ends_with("\"module\": \"io_log\", \"function\": \"\", \"line\": 6}"), "{line}");
        assert_eq!(py_json_str("é😀\u{7f}"), "\"\\u00e9\\ud83d\\ude00\\u007f\"");
    }

    #[test]
    fn format_time_matches_python() {
        // Expected values from prime_rl.utils.logger.format_time.
        for (seconds, expected) in [
            (0.0, "0ms"),
            (0.0004, "0ms"),
            (0.5, "500ms"),
            (0.9996, "1000ms"),
            (1.0, "1.0s"),
            (59.96, "60.0s"),
            (61.5, "1m 1s"),
            (3599.9, "59m 59s"),
            (3600.0, "1h 0m"),
            (7322.0, "2h 2m"),
            (86400.0, "1d 0h"),
            (200000.0, "2d 7h"),
        ] {
            assert_eq!(format_time(seconds), expected, "{seconds}");
        }
    }
}
