//! `Value`: an ordered, msgpack-faithful dynamic value.
//!
//! Python dicts keep insertion order and the wire is msgpack, so this keeps map order,
//! distinguishes bytes from strings and ints from floats. Integer map keys (only the delta
//! `links` map uses them) become strings on decode.

use std::fmt;

use indexmap::IndexMap;
use serde::de::{self, Deserialize, Deserializer, MapAccess, SeqAccess, Visitor};
use serde::ser::{Serialize, SerializeMap, SerializeSeq, Serializer};

pub type Map = IndexMap<String, Value>;

#[derive(Clone, Debug, Default, PartialEq)]
pub enum Value {
    #[default]
    Nil,
    Bool(bool),
    Int(i64),
    /// Only for integers above `i64::MAX`.
    UInt(u64),
    /// Never produced by msgpack decoding (Python packs float64); used for decoded float32
    /// arrays so the JSON writer prints them the way orjson prints numpy float32.
    F32(f32),
    F64(f64),
    Str(String),
    Bin(Vec<u8>),
    Array(Vec<Value>),
    Map(Map),
}

impl Value {
    pub fn map() -> Value {
        Value::Map(Map::new())
    }

    pub fn is_nil(&self) -> bool {
        matches!(self, Value::Nil)
    }

    /// `self[key]` for a map; `None` for a missing key or a non-map.
    pub fn get(&self, key: &str) -> Option<&Value> {
        match self {
            Value::Map(map) => map.get(key),
            _ => None,
        }
    }

    pub fn get_mut(&mut self, key: &str) -> Option<&mut Value> {
        match self {
            Value::Map(map) => map.get_mut(key),
            _ => None,
        }
    }

    /// Insert into a map (panics on a non-map: a programming error).
    pub fn insert(&mut self, key: impl Into<String>, value: impl Into<Value>) -> Option<Value> {
        match self {
            Value::Map(map) => map.insert(key.into(), value.into()),
            other => panic!("insert into non-map value {other:?}"),
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::Str(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Value::Bool(b) => Some(*b),
            _ => None,
        }
    }

    pub fn as_i64(&self) -> Option<i64> {
        match self {
            Value::Int(i) => Some(*i),
            Value::UInt(u) => i64::try_from(*u).ok(),
            _ => None,
        }
    }

    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Value::Int(i) => u64::try_from(*i).ok(),
            Value::UInt(u) => Some(*u),
            _ => None,
        }
    }

    /// Any number as f64 (ints included, like a pydantic `float` field accepts them).
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Value::Int(i) => Some(*i as f64),
            Value::UInt(u) => Some(*u as f64),
            Value::F32(f) => Some(*f as f64),
            Value::F64(f) => Some(*f),
            _ => None,
        }
    }

    pub fn as_bytes(&self) -> Option<&[u8]> {
        match self {
            Value::Bin(b) => Some(b),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&Vec<Value>> {
        match self {
            Value::Array(a) => Some(a),
            _ => None,
        }
    }

    pub fn as_array_mut(&mut self) -> Option<&mut Vec<Value>> {
        match self {
            Value::Array(a) => Some(a),
            _ => None,
        }
    }

    pub fn as_map(&self) -> Option<&Map> {
        match self {
            Value::Map(m) => Some(m),
            _ => None,
        }
    }

    pub fn as_map_mut(&mut self) -> Option<&mut Map> {
        match self {
            Value::Map(m) => Some(m),
            _ => None,
        }
    }

    pub fn into_map(self) -> Option<Map> {
        match self {
            Value::Map(m) => Some(m),
            _ => None,
        }
    }

    pub fn into_array(self) -> Option<Vec<Value>> {
        match self {
            Value::Array(a) => Some(a),
            _ => None,
        }
    }

    /// Python truthiness (`bool(value)`).
    pub fn truthy(&self) -> bool {
        match self {
            Value::Nil => false,
            Value::Bool(b) => *b,
            Value::Int(i) => *i != 0,
            Value::UInt(u) => *u != 0,
            Value::F32(f) => *f != 0.0,
            Value::F64(f) => *f != 0.0,
            Value::Str(s) => !s.is_empty(),
            Value::Bin(b) => !b.is_empty(),
            Value::Array(a) => !a.is_empty(),
            Value::Map(m) => !m.is_empty(),
        }
    }
}

macro_rules! from_int {
    ($($t:ty),*) => {$(
        impl From<$t> for Value {
            fn from(v: $t) -> Self { Value::Int(v as i64) }
        }
    )*};
}
from_int!(i8, i16, i32, i64, u8, u16, u32, isize);

impl From<u64> for Value {
    fn from(v: u64) -> Self {
        i64::try_from(v).map(Value::Int).unwrap_or(Value::UInt(v))
    }
}
impl From<usize> for Value {
    fn from(v: usize) -> Self {
        Value::from(v as u64)
    }
}
impl From<f64> for Value {
    fn from(v: f64) -> Self {
        Value::F64(v)
    }
}
impl From<f32> for Value {
    fn from(v: f32) -> Self {
        Value::F32(v)
    }
}
impl From<bool> for Value {
    fn from(v: bool) -> Self {
        Value::Bool(v)
    }
}
impl From<&str> for Value {
    fn from(v: &str) -> Self {
        Value::Str(v.to_owned())
    }
}
impl From<String> for Value {
    fn from(v: String) -> Self {
        Value::Str(v)
    }
}
impl From<&String> for Value {
    fn from(v: &String) -> Self {
        Value::Str(v.clone())
    }
}
impl From<Vec<u8>> for Value {
    fn from(v: Vec<u8>) -> Self {
        Value::Bin(v)
    }
}
impl From<Vec<Value>> for Value {
    fn from(v: Vec<Value>) -> Self {
        Value::Array(v)
    }
}
impl From<Map> for Value {
    fn from(v: Map) -> Self {
        Value::Map(v)
    }
}
impl<T: Into<Value>> From<Option<T>> for Value {
    fn from(v: Option<T>) -> Self {
        v.map_or(Value::Nil, Into::into)
    }
}

impl<K: Into<String>, V: Into<Value>> FromIterator<(K, V)> for Value {
    fn from_iter<I: IntoIterator<Item = (K, V)>>(iter: I) -> Self {
        Value::Map(iter.into_iter().map(|(k, v)| (k.into(), v.into())).collect())
    }
}

// ── msgpack ────────────────────────────────────────────────────────────────

fn key_string(key: rmpv::Value) -> Result<String, DecodeError> {
    Ok(match key {
        rmpv::Value::String(s) => s.into_str().ok_or(DecodeError::InvalidUtf8)?,
        rmpv::Value::Integer(i) => i.to_string(),
        rmpv::Value::Boolean(b) => if b { "True".into() } else { "False".into() },
        rmpv::Value::Nil => "None".into(),
        rmpv::Value::F64(f) => f.to_string(),
        other => return Err(DecodeError::UnsupportedKey(format!("{other}"))),
    })
}

fn from_rmpv(value: rmpv::Value) -> Result<Value, DecodeError> {
    Ok(match value {
        rmpv::Value::Nil => Value::Nil,
        rmpv::Value::Boolean(b) => Value::Bool(b),
        rmpv::Value::Integer(i) => match i.as_i64() {
            Some(v) => Value::Int(v),
            None => Value::UInt(i.as_u64().expect("msgpack integer is i64 or u64")),
        },
        rmpv::Value::F32(f) => Value::F64(f as f64),
        rmpv::Value::F64(f) => Value::F64(f),
        // `raw=False`: str must be UTF-8
        rmpv::Value::String(s) => Value::Str(s.into_str().ok_or(DecodeError::InvalidUtf8)?),
        rmpv::Value::Binary(b) => Value::Bin(b),
        rmpv::Value::Array(items) => {
            Value::Array(items.into_iter().map(from_rmpv).collect::<Result<_, _>>()?)
        }
        rmpv::Value::Map(pairs) => {
            let mut map = Map::with_capacity(pairs.len());
            for (k, v) in pairs {
                map.insert(key_string(k)?, from_rmpv(v)?);
            }
            Value::Map(map)
        }
        rmpv::Value::Ext(code, _) => return Err(DecodeError::UnsupportedExt(code)),
    })
}

#[derive(Debug, thiserror::Error)]
pub enum DecodeError {
    #[error("invalid msgpack: {0}")]
    Msgpack(#[from] rmpv::decode::Error),
    #[error("invalid UTF-8 in msgpack str")]
    InvalidUtf8,
    #[error("unsupported msgpack map key {0}")]
    UnsupportedKey(String),
    #[error("unsupported msgpack ext type {0}")]
    UnsupportedExt(i8),
    #[error("trailing bytes after msgpack value")]
    Trailing,
}

/// `msgpack.unpackb(data, raw=False, strict_map_key=False)`.
pub fn from_msgpack(mut data: &[u8]) -> Result<Value, DecodeError> {
    let value = rmpv::decode::read_value(&mut data)?;
    if !data.is_empty() {
        return Err(DecodeError::Trailing);
    }
    from_rmpv(value)
}

/// `msgpack.packb(value, use_bin_type=True)`: smallest encodings, float64 floats.
pub fn to_msgpack(value: &Value) -> Vec<u8> {
    let mut out = Vec::new();
    write_msgpack(&mut out, value);
    out
}

fn write_msgpack(out: &mut Vec<u8>, value: &Value) {
    use rmp::encode;
    // Writes into a Vec cannot fail.
    match value {
        Value::Nil => encode::write_nil(out).unwrap(),
        Value::Bool(b) => encode::write_bool(out, *b).unwrap(),
        Value::Int(i) => {
            if *i >= 0 {
                encode::write_uint(out, *i as u64).map(|_| ()).unwrap()
            } else {
                encode::write_sint(out, *i).map(|_| ()).unwrap()
            }
        }
        Value::UInt(u) => encode::write_uint(out, *u).map(|_| ()).unwrap(),
        Value::F32(f) => encode::write_f64(out, *f as f64).unwrap(),
        Value::F64(f) => encode::write_f64(out, *f).unwrap(),
        Value::Str(s) => encode::write_str(out, s).unwrap(),
        Value::Bin(b) => encode::write_bin(out, b).unwrap(),
        Value::Array(items) => {
            encode::write_array_len(out, items.len() as u32).unwrap();
            for item in items {
                write_msgpack(out, item);
            }
        }
        Value::Map(map) => {
            encode::write_map_len(out, map.len() as u32).unwrap();
            for (k, v) in map {
                encode::write_str(out, k).unwrap();
                write_msgpack(out, v);
            }
        }
    }
}

// ── serde ──────────────────────────────────────────────────────────────────

impl Serialize for Value {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            Value::Nil => serializer.serialize_unit(),
            Value::Bool(b) => serializer.serialize_bool(*b),
            Value::Int(i) => serializer.serialize_i64(*i),
            Value::UInt(u) => serializer.serialize_u64(*u),
            Value::F32(f) => serializer.serialize_f32(*f),
            Value::F64(f) => serializer.serialize_f64(*f),
            Value::Str(s) => serializer.serialize_str(s),
            Value::Bin(b) => serializer.serialize_bytes(b),
            Value::Array(items) => {
                let mut seq = serializer.serialize_seq(Some(items.len()))?;
                for item in items {
                    seq.serialize_element(item)?;
                }
                seq.end()
            }
            Value::Map(map) => {
                let mut m = serializer.serialize_map(Some(map.len()))?;
                for (k, v) in map {
                    m.serialize_entry(k, v)?;
                }
                m.end()
            }
        }
    }
}

impl<'de> Deserialize<'de> for Value {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Value, D::Error> {
        struct ValueVisitor;

        impl<'de> Visitor<'de> for ValueVisitor {
            type Value = Value;

            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_str("any value")
            }
            fn visit_bool<E>(self, v: bool) -> Result<Value, E> {
                Ok(Value::Bool(v))
            }
            fn visit_i64<E>(self, v: i64) -> Result<Value, E> {
                Ok(Value::Int(v))
            }
            fn visit_u64<E>(self, v: u64) -> Result<Value, E> {
                Ok(Value::from(v))
            }
            fn visit_f32<E>(self, v: f32) -> Result<Value, E> {
                Ok(Value::F64(v as f64))
            }
            fn visit_f64<E>(self, v: f64) -> Result<Value, E> {
                Ok(Value::F64(v))
            }
            fn visit_str<E>(self, v: &str) -> Result<Value, E> {
                Ok(Value::Str(v.to_owned()))
            }
            fn visit_string<E>(self, v: String) -> Result<Value, E> {
                Ok(Value::Str(v))
            }
            fn visit_bytes<E>(self, v: &[u8]) -> Result<Value, E> {
                Ok(Value::Bin(v.to_vec()))
            }
            fn visit_byte_buf<E>(self, v: Vec<u8>) -> Result<Value, E> {
                Ok(Value::Bin(v))
            }
            fn visit_none<E>(self) -> Result<Value, E> {
                Ok(Value::Nil)
            }
            fn visit_some<D: Deserializer<'de>>(self, d: D) -> Result<Value, D::Error> {
                Value::deserialize(d)
            }
            fn visit_unit<E>(self) -> Result<Value, E> {
                Ok(Value::Nil)
            }
            fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Value, A::Error> {
                let mut items = Vec::with_capacity(seq.size_hint().unwrap_or(0));
                while let Some(item) = seq.next_element()? {
                    items.push(item);
                }
                Ok(Value::Array(items))
            }
            fn visit_map<A: MapAccess<'de>>(self, mut access: A) -> Result<Value, A::Error> {
                let mut map = Map::with_capacity(access.size_hint().unwrap_or(0));
                while let Some((k, v)) = access.next_entry::<KeyString, Value>()? {
                    map.insert(k.0, v);
                }
                Ok(Value::Map(map))
            }
        }

        deserializer.deserialize_any(ValueVisitor)
    }
}

/// A map key: strings as-is, integer keys stringified.
struct KeyString(String);

impl<'de> Deserialize<'de> for KeyString {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct KeyVisitor;
        impl<'de> Visitor<'de> for KeyVisitor {
            type Value = KeyString;
            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_str("a string or integer map key")
            }
            fn visit_str<E>(self, v: &str) -> Result<KeyString, E> {
                Ok(KeyString(v.to_owned()))
            }
            fn visit_string<E>(self, v: String) -> Result<KeyString, E> {
                Ok(KeyString(v))
            }
            fn visit_i64<E>(self, v: i64) -> Result<KeyString, E> {
                Ok(KeyString(v.to_string()))
            }
            fn visit_u64<E>(self, v: u64) -> Result<KeyString, E> {
                Ok(KeyString(v.to_string()))
            }
            fn visit_bytes<E: de::Error>(self, v: &[u8]) -> Result<KeyString, E> {
                String::from_utf8(v.to_vec()).map(KeyString).map_err(E::custom)
            }
        }
        deserializer.deserialize_any(KeyVisitor)
    }
}

/// Convert any `Serialize` value (e.g. a config struct) into a `Value`, keeping field order.
pub fn to_value<T: Serialize + ?Sized>(value: &T) -> Result<Value, rmp_serde::encode::Error> {
    let bytes = rmp_serde::to_vec_named(value)?;
    Ok(from_msgpack(&bytes).expect("rmp_serde output is valid msgpack"))
}
