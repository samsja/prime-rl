//! A JSON writer byte-compatible with `orjson.dumps` (default options, plus
//! `OPT_SERIALIZE_NUMPY` for float32 arrays held as `Value::F32`).
//!
//! orjson's float format: the shortest round-trip digits, in decimal notation when the
//! decimal exponent is in `-5..16` (`100.0`, `0.00001`) and scientific otherwise (`1e-6`,
//! `1.5e+16`); numpy float32 uses `-6..13`. NaN and infinities are `null`. Strings escape only `"`, `\` and control
//! characters.

use crate::value::Value;

pub fn to_vec(value: &Value) -> Vec<u8> {
    let mut out = Vec::with_capacity(128);
    write(&mut out, value);
    out
}

pub fn write(out: &mut Vec<u8>, value: &Value) {
    match value {
        Value::Nil => out.extend_from_slice(b"null"),
        Value::Bool(true) => out.extend_from_slice(b"true"),
        Value::Bool(false) => out.extend_from_slice(b"false"),
        Value::Int(i) => out.extend_from_slice(i.to_string().as_bytes()),
        Value::UInt(u) => out.extend_from_slice(u.to_string().as_bytes()),
        Value::F32(f) => write_f32(out, *f),
        Value::F64(f) => write_f64(out, *f),
        Value::Str(s) => write_str(out, s),
        Value::Bin(b) => write_str(out, &String::from_utf8_lossy(b)),
        Value::Array(items) => {
            out.push(b'[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push(b',');
                }
                write(out, item);
            }
            out.push(b']');
        }
        Value::Map(map) => {
            out.push(b'{');
            for (i, (k, v)) in map.iter().enumerate() {
                if i > 0 {
                    out.push(b',');
                }
                write_str(out, k);
                out.push(b':');
                write(out, v);
            }
            out.push(b'}');
        }
    }
}

pub fn write_f64(out: &mut Vec<u8>, f: f64) {
    if !f.is_finite() {
        out.extend_from_slice(b"null");
        return;
    }
    let mut buf = ryu::Buffer::new();
    format_shortest(out, buf.format_finite(f), -5..16);
}

fn write_f32(out: &mut Vec<u8>, f: f32) {
    if !f.is_finite() {
        out.extend_from_slice(b"null");
        return;
    }
    let mut buf = ryu::Buffer::new();
    format_shortest(out, buf.format_finite(f), -6..13);
}

/// Re-lay ryu's shortest representation in orjson's notation.
fn format_shortest(out: &mut Vec<u8>, repr: &str, decimal_exps: std::ops::Range<i32>) {
    let (negative, repr) = match repr.strip_prefix('-') {
        Some(rest) => (true, rest),
        None => (false, repr),
    };
    let (mantissa, exp) = match repr.split_once(['e', 'E']) {
        Some((m, e)) => (m, e.parse::<i32>().expect("ryu exponent")),
        None => (repr, 0),
    };
    let (int_part, frac_part) = mantissa.split_once('.').unwrap_or((mantissa, ""));
    // value = 0.DIGITS * 10^point
    let mut digits: String = format!("{int_part}{frac_part}");
    let mut point = int_part.len() as i32 + exp;
    let leading = digits.len() - digits.trim_start_matches('0').len();
    digits.drain(..leading);
    point -= leading as i32;
    let trimmed = digits.trim_end_matches('0').len();
    digits.truncate(trimmed);
    if negative {
        out.push(b'-');
    }
    if digits.is_empty() {
        out.extend_from_slice(b"0.0");
        return;
    }
    let n = digits.len() as i32;
    let sci_exp = point - 1;
    if decimal_exps.contains(&sci_exp) {
        if point <= 0 {
            out.extend_from_slice(b"0.");
            out.extend(std::iter::repeat_n(b'0', (-point) as usize));
            out.extend_from_slice(digits.as_bytes());
        } else if point >= n {
            out.extend_from_slice(digits.as_bytes());
            out.extend(std::iter::repeat_n(b'0', (point - n) as usize));
            out.extend_from_slice(b".0");
        } else {
            out.extend_from_slice(&digits.as_bytes()[..point as usize]);
            out.push(b'.');
            out.extend_from_slice(&digits.as_bytes()[point as usize..]);
        }
    } else {
        out.push(digits.as_bytes()[0]);
        if n > 1 {
            out.push(b'.');
            out.extend_from_slice(&digits.as_bytes()[1..]);
        }
        out.push(b'e');
        out.push(if sci_exp > 0 { b'+' } else { b'-' });
        out.extend_from_slice(sci_exp.abs().to_string().as_bytes());
    }
}

/// `json.dumps(value, sort_keys=sort_keys)` (Python's stdlib encoder): `", "` / `": "`
/// separators, `ensure_ascii`, float `repr`, `NaN` / `Infinity`.
pub fn to_python_json(value: &Value, sort_keys: bool) -> String {
    let mut out = String::new();
    write_python(&mut out, value, sort_keys);
    out
}

fn write_python(out: &mut String, value: &Value, sort_keys: bool) {
    match value {
        Value::Nil => out.push_str("null"),
        Value::Bool(true) => out.push_str("true"),
        Value::Bool(false) => out.push_str("false"),
        Value::Int(i) => out.push_str(&i.to_string()),
        Value::UInt(u) => out.push_str(&u.to_string()),
        Value::F32(f) => out.push_str(&python_float_repr(*f as f64)),
        Value::F64(f) => out.push_str(&python_float_repr(*f)),
        Value::Str(s) => write_python_str(out, s),
        Value::Bin(b) => write_python_str(out, &String::from_utf8_lossy(b)),
        Value::Array(items) => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                write_python(out, item, sort_keys);
            }
            out.push(']');
        }
        Value::Map(map) => {
            let mut entries: Vec<(&String, &Value)> = map.iter().collect();
            if sort_keys {
                entries.sort_by(|a, b| a.0.cmp(b.0));
            }
            out.push('{');
            for (i, (k, v)) in entries.into_iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                write_python_str(out, k);
                out.push_str(": ");
                write_python(out, v, sort_keys);
            }
            out.push('}');
        }
    }
}

/// Python's `repr(float)`: shortest round-trip digits, scientific notation (with a
/// two-digit exponent) outside `1e-4 <= |x| < 1e16`.
fn python_float_repr(f: f64) -> String {
    if f.is_nan() {
        return "NaN".into();
    }
    if f.is_infinite() {
        return if f > 0.0 { "Infinity".into() } else { "-Infinity".into() };
    }
    let mut buf = ryu::Buffer::new();
    let mut out = Vec::new();
    format_shortest(&mut out, buf.format_finite(f), -4..16);
    let s = String::from_utf8(out).unwrap();
    // orjson-style exponent `e-5` / `e+16` -> Python's `e-05` / `e+16`
    match s.split_once('e') {
        Some((mantissa, exp)) => {
            let (sign, digits) = exp.split_at(1);
            format!("{mantissa}e{sign}{digits:0>2}")
        }
        None => s,
    }
}

fn write_python_str(out: &mut String, s: &str) {
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
                let mut units = [0u16; 2];
                for unit in c.encode_utf16(&mut units) {
                    out.push_str(&format!("\\u{unit:04x}"));
                }
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

pub fn write_str(out: &mut Vec<u8>, s: &str) {
    out.push(b'"');
    let bytes = s.as_bytes();
    let mut start = 0;
    for (i, &b) in bytes.iter().enumerate() {
        let escape: Option<&[u8]> = match b {
            b'"' => Some(b"\\\""),
            b'\\' => Some(b"\\\\"),
            b'\n' => Some(b"\\n"),
            b'\r' => Some(b"\\r"),
            b'\t' => Some(b"\\t"),
            0x08 => Some(b"\\b"),
            0x0c => Some(b"\\f"),
            0x00..=0x1f => None,
            _ => continue,
        };
        out.extend_from_slice(&bytes[start..i]);
        match escape {
            Some(e) => out.extend_from_slice(e),
            None => out.extend_from_slice(format!("\\u{b:04x}").as_bytes()),
        }
        start = i + 1;
    }
    out.extend_from_slice(&bytes[start..]);
    out.push(b'"');
}
