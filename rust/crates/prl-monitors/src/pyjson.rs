//! Python `json.dumps` (default separators, `ensure_ascii=True`) for the metrics JSONL, which
//! the Python file monitor writes with the stdlib encoder rather than orjson.

pub use prl_types::{Metrics as Row, Scalar};

/// A scalar as a JSON value (non-finite floats become null).
pub fn scalar_json(value: &Scalar) -> serde_json::Value {
    match value {
        Scalar::Null => serde_json::Value::Null,
        Scalar::Bool(b) => (*b).into(),
        Scalar::Int(i) => (*i).into(),
        Scalar::Float(f) => {
            serde_json::Number::from_f64(*f).map_or(serde_json::Value::Null, Into::into)
        }
        Scalar::Str(s) => s.clone().into(),
    }
}

/// `prime_rl.utils.utils.sanitize` on a flat row: drops non-finite floats, returns their keys.
pub fn sanitize(row: &Row) -> (Row, Vec<String>) {
    let mut dropped = Vec::new();
    let mut out = Row::with_capacity(row.len());
    for (key, value) in row {
        if let Scalar::Float(f) = value {
            if !f.is_finite() {
                dropped.push(key.clone());
                continue;
            }
        }
        out.insert(key.clone(), value.clone());
    }
    (out, dropped)
}

/// `repr(float)`: shortest round-trip digits, exponent form when the decimal point sits
/// at or below 1e-5 or at or above 1e16.
pub fn float_repr(x: f64) -> String {
    if x.is_nan() {
        return "NaN".into();
    }
    if x.is_infinite() {
        return if x > 0.0 {
            "Infinity".into()
        } else {
            "-Infinity".into()
        };
    }
    if x == 0.0 {
        return if x.is_sign_negative() {
            "-0.0".into()
        } else {
            "0.0".into()
        };
    }
    let sci = format!("{:e}", x); // e.g. "-1.5e-5"
    let (mantissa, exp) = sci.split_once('e').expect("{:e} has an exponent");
    let exp: i32 = exp.parse().expect("integer exponent");
    let (sign, mantissa) = mantissa
        .strip_prefix('-')
        .map_or(("", mantissa), |m| ("-", m));
    let digits: String = mantissa.chars().filter(|c| *c != '.').collect();
    let decpt = exp + 1;
    let n = digits.len() as i32;
    if decpt <= -4 || decpt > 16 {
        let mut s = String::from(sign);
        s.push_str(&digits[..1]);
        if n > 1 {
            s.push('.');
            s.push_str(&digits[1..]);
        }
        let e = decpt - 1;
        s.push_str(&format!("e{}{:02}", if e < 0 { '-' } else { '+' }, e.abs()));
        s
    } else if decpt <= 0 {
        format!("{sign}0.{}{digits}", "0".repeat((-decpt) as usize))
    } else if decpt >= n {
        format!("{sign}{digits}{}.0", "0".repeat((decpt - n) as usize))
    } else {
        let (int, frac) = digits.split_at(decpt as usize);
        format!("{sign}{int}.{frac}")
    }
}

pub fn write_str(out: &mut String, s: &str) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 || (c as u32) > 0x7e && (c as u32) != 0x7f => {
                let mut buf = [0u16; 2];
                for unit in c.encode_utf16(&mut buf) {
                    out.push_str(&format!("\\u{:04x}", unit));
                }
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

pub fn write_scalar(out: &mut String, value: &Scalar) {
    match value {
        Scalar::Null => out.push_str("null"),
        Scalar::Bool(true) => out.push_str("true"),
        Scalar::Bool(false) => out.push_str("false"),
        Scalar::Int(i) => out.push_str(&i.to_string()),
        Scalar::Float(f) => out.push_str(&float_repr(*f)),
        Scalar::Str(s) => write_str(out, s),
    }
}

/// `json.dumps(row)`.
pub fn dumps_row(row: &Row) -> String {
    let mut out = String::from("{");
    for (i, (key, value)) in row.iter().enumerate() {
        if i > 0 {
            out.push_str(", ");
        }
        write_str(&mut out, key);
        out.push_str(": ");
        write_scalar(&mut out, value);
    }
    out.push('}');
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn float_repr_matches_python() {
        let cases = [
            (0.00001, "1e-05"),
            (1e-6, "1e-06"),
            (1e16, "1e+16"),
            (1e15, "1000000000000000.0"),
            (123456789012345680000.0, "1.2345678901234568e+20"),
            (0.1, "0.1"),
            (1.0, "1.0"),
            (-0.0, "-0.0"),
            (3.14e-5, "3.14e-05"),
            (0.0001, "0.0001"),
            (1.0 / 3.0, "0.3333333333333333"),
            (-2.5, "-2.5"),
            (123.456, "123.456"),
            (1.5e300, "1.5e+300"),
            (5e-324, "5e-324"),
        ];
        for (x, want) in cases {
            assert_eq!(float_repr(x), want, "{x}");
        }
    }

    #[test]
    fn strings_are_ascii_escaped() {
        let mut s = String::new();
        write_str(&mut s, "a\"b\\c\nd\u{1}é😀\u{7f}");
        assert_eq!(
            s,
            concat!(r#""a\"b\\c\nd\u0001\u00e9\ud83d\ude00"#, "\u{7f}", "\"")
        );
    }
}
