//! `Value`: an ordered, msgpack-faithful dynamic value.
//!
//! Python dicts keep insertion order and the wire is msgpack, so this keeps map order,
//! distinguishes bytes from strings and ints from floats. Integer map keys (only the delta
//! `links` map uses them) become strings on decode.

use std::fmt;

use indexmap::IndexMap;
use serde::de::{Deserialize, Deserializer, MapAccess, SeqAccess, Visitor};
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

#[derive(Debug, thiserror::Error)]
pub enum DecodeError {
    #[error("truncated msgpack data")]
    Truncated,
    #[error("reserved msgpack marker 0xc1")]
    Reserved,
    #[error("invalid UTF-8 in msgpack str")]
    InvalidUtf8,
    #[error("unsupported msgpack map key {0}")]
    UnsupportedKey(String),
    #[error("unsupported msgpack ext type")]
    UnsupportedExt,
    #[error("msgpack nesting too deep")]
    TooDeep,
    #[error("trailing bytes after msgpack value")]
    Trailing,
}

/// `msgpack.unpackb(data, raw=False, strict_map_key=False)`.
pub fn from_msgpack(data: &[u8]) -> Result<Value, DecodeError> {
    let mut reader = Reader { data, pos: 0 };
    let value = reader.value(0)?;
    if reader.pos != data.len() {
        return Err(DecodeError::Trailing);
    }
    Ok(value)
}

const MAX_DEPTH: usize = 512;

struct Reader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], DecodeError> {
        let end = self.pos.checked_add(n).filter(|&end| end <= self.data.len()).ok_or(DecodeError::Truncated)?;
        let bytes = &self.data[self.pos..end];
        self.pos = end;
        Ok(bytes)
    }

    fn uint(&mut self, n: usize) -> Result<u64, DecodeError> {
        Ok(self.take(n)?.iter().fold(0u64, |acc, &b| acc << 8 | b as u64))
    }

    fn string(&mut self, n: usize) -> Result<String, DecodeError> {
        String::from_utf8(self.take(n)?.to_vec()).map_err(|_| DecodeError::InvalidUtf8)
    }

    fn value(&mut self, depth: usize) -> Result<Value, DecodeError> {
        if depth > MAX_DEPTH {
            return Err(DecodeError::TooDeep);
        }
        let marker = self.take(1)?[0];
        Ok(match marker {
            0x00..=0x7f => Value::Int(marker as i64),
            0x80..=0x8f => self.map((marker & 0x0f) as usize, depth)?,
            0x90..=0x9f => self.array((marker & 0x0f) as usize, depth)?,
            0xa0..=0xbf => Value::Str(self.string((marker & 0x1f) as usize)?),
            0xc0 => Value::Nil,
            0xc2 => Value::Bool(false),
            0xc3 => Value::Bool(true),
            0xc4 => { let n = self.uint(1)? as usize; Value::Bin(self.take(n)?.to_vec()) }
            0xc5 => { let n = self.uint(2)? as usize; Value::Bin(self.take(n)?.to_vec()) }
            0xc6 => { let n = self.uint(4)? as usize; Value::Bin(self.take(n)?.to_vec()) }
            0xc7..=0xc9 | 0xd4..=0xd8 => return Err(DecodeError::UnsupportedExt),
            0xca => Value::F64(f32::from_bits(self.uint(4)? as u32) as f64),
            0xcb => Value::F64(f64::from_bits(self.uint(8)?)),
            0xcc => Value::Int(self.uint(1)? as i64),
            0xcd => Value::Int(self.uint(2)? as i64),
            0xce => Value::Int(self.uint(4)? as i64),
            0xcf => Value::from(self.uint(8)?),
            0xd0 => Value::Int(self.uint(1)? as u8 as i8 as i64),
            0xd1 => Value::Int(self.uint(2)? as u16 as i16 as i64),
            0xd2 => Value::Int(self.uint(4)? as u32 as i32 as i64),
            0xd3 => Value::Int(self.uint(8)? as i64),
            0xd9 => { let n = self.uint(1)? as usize; Value::Str(self.string(n)?) }
            0xda => { let n = self.uint(2)? as usize; Value::Str(self.string(n)?) }
            0xdb => { let n = self.uint(4)? as usize; Value::Str(self.string(n)?) }
            0xdc => { let n = self.uint(2)? as usize; self.array(n, depth)? }
            0xdd => { let n = self.uint(4)? as usize; self.array(n, depth)? }
            0xde => { let n = self.uint(2)? as usize; self.map(n, depth)? }
            0xdf => { let n = self.uint(4)? as usize; self.map(n, depth)? }
            0xe0..=0xff => Value::Int(marker as i8 as i64),
            0xc1 => return Err(DecodeError::Reserved),
        })
    }

    fn array(&mut self, n: usize, depth: usize) -> Result<Value, DecodeError> {
        let mut items = Vec::with_capacity(n.min(self.data.len() - self.pos));
        for _ in 0..n {
            items.push(self.value(depth + 1)?);
        }
        Ok(Value::Array(items))
    }

    fn map(&mut self, n: usize, depth: usize) -> Result<Value, DecodeError> {
        let mut map = Map::with_capacity(n.min(self.data.len() - self.pos));
        for _ in 0..n {
            let key = match self.value(depth + 1)? {
                Value::Str(s) => s,
                Value::Int(i) => i.to_string(),
                Value::UInt(u) => u.to_string(),
                other => return Err(DecodeError::UnsupportedKey(format!("{other:?}"))),
            };
            map.insert(key, self.value(depth + 1)?);
        }
        Ok(Value::Map(map))
    }
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
                while let Some((k, v)) = access.next_entry::<String, Value>()? {
                    map.insert(k, v);
                }
                Ok(Value::Map(map))
            }
        }

        deserializer.deserialize_any(ValueVisitor)
    }
}

/// Convert any `Serialize` value (e.g. a config struct) into a `Value`, keeping field order.
pub fn to_value<T: Serialize + ?Sized>(value: &T) -> Result<Value, rmp_serde::encode::Error> {
    let bytes = rmp_serde::to_vec_named(value)?;
    Ok(from_msgpack(&bytes).expect("rmp_serde output is valid msgpack"))
}
