//! numpy arrays on the wire: `encode_ndarray` dicts (`{"__nd__": True, dtype, shape, data}`).

use crate::value::{Map, Value};

pub const ND_TAG: &str = "__nd__";

/// A raw numpy array: dtype name (`int32`, `bool`, `float32`, `uint8`, ...), shape and
/// little-endian C-order bytes.
#[derive(Clone, Debug, PartialEq)]
pub struct NdArray {
    pub dtype: String,
    pub shape: Vec<usize>,
    pub data: Vec<u8>,
}

#[derive(Debug, thiserror::Error)]
pub enum NdError {
    #[error("not an __nd__ array: {0}")]
    NotAnArray(String),
    #[error("unsupported dtype {0:?}")]
    UnsupportedDtype(String),
    #[error("array data of {len} bytes does not fit dtype {dtype} and shape {shape:?}")]
    BadSize { len: usize, dtype: String, shape: Vec<usize> },
    #[error("stream item is not a number: {0:?}")]
    NotANumber(Value),
}

pub fn is_nd(value: &Value) -> bool {
    value.get(ND_TAG).is_some_and(Value::truthy)
}

fn itemsize(dtype: &str) -> Result<usize, NdError> {
    Ok(match dtype {
        "bool" | "int8" | "uint8" => 1,
        "int16" | "uint16" | "float16" => 2,
        "int32" | "uint32" | "float32" => 4,
        "int64" | "uint64" | "float64" => 8,
        other => return Err(NdError::UnsupportedDtype(other.to_owned())),
    })
}

impl NdArray {
    pub fn from_value(value: &Value) -> Result<NdArray, NdError> {
        if !is_nd(value) {
            return Err(NdError::NotAnArray(format!("{value:?}").chars().take(200).collect()));
        }
        let dtype = value.get("dtype").and_then(Value::as_str).unwrap_or_default().to_owned();
        let shape = value
            .get("shape")
            .and_then(Value::as_array)
            .map(|s| s.iter().map(|d| d.as_u64().unwrap_or(0) as usize).collect())
            .unwrap_or_default();
        let data = value.get("data").and_then(Value::as_bytes).unwrap_or_default().to_vec();
        let array = NdArray { dtype, shape, data };
        let expected = array.len() * itemsize(&array.dtype)?;
        if expected != array.data.len() {
            return Err(NdError::BadSize { len: array.data.len(), dtype: array.dtype, shape: array.shape });
        }
        Ok(array)
    }

    pub fn to_value(&self) -> Value {
        let mut map = Map::with_capacity(4);
        map.insert(ND_TAG.into(), Value::Bool(true));
        map.insert("dtype".into(), Value::Str(self.dtype.clone()));
        map.insert("shape".into(), Value::Array(self.shape.iter().map(|&d| Value::from(d)).collect()));
        map.insert("data".into(), Value::Bin(self.data.clone()));
        Value::Map(map)
    }

    /// Number of elements.
    pub fn len(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Elements as f64, in C order (exact for every supported dtype but 64-bit ints > 2^53).
    pub fn to_f64(&self) -> Result<Vec<f64>, NdError> {
        let size = itemsize(&self.dtype)?;
        let chunks = self.data.chunks_exact(size);
        Ok(match self.dtype.as_str() {
            "bool" | "uint8" => chunks.map(|c| c[0] as f64).collect(),
            "int8" => chunks.map(|c| c[0] as i8 as f64).collect(),
            "int16" => chunks.map(|c| i16::from_le_bytes([c[0], c[1]]) as f64).collect(),
            "uint16" => chunks.map(|c| u16::from_le_bytes([c[0], c[1]]) as f64).collect(),
            "float16" => chunks.map(|c| f16_to_f64(u16::from_le_bytes([c[0], c[1]]))).collect(),
            "int32" => chunks.map(|c| i32::from_le_bytes(c.try_into().unwrap()) as f64).collect(),
            "uint32" => chunks.map(|c| u32::from_le_bytes(c.try_into().unwrap()) as f64).collect(),
            "float32" => chunks.map(|c| f32::from_le_bytes(c.try_into().unwrap()) as f64).collect(),
            "int64" => chunks.map(|c| i64::from_le_bytes(c.try_into().unwrap()) as f64).collect(),
            "uint64" => chunks.map(|c| u64::from_le_bytes(c.try_into().unwrap()) as f64).collect(),
            "float64" => chunks.map(|c| f64::from_le_bytes(c.try_into().unwrap())).collect(),
            other => return Err(NdError::UnsupportedDtype(other.to_owned())),
        })
    }

    /// Elements as `Value`s, nested by shape (what `decode_ndarray(...).tolist()` gives;
    /// float32 stays `Value::F32`).
    pub fn to_values(&self) -> Result<Value, NdError> {
        let size = itemsize(&self.dtype)?;
        let flat: Vec<Value> = match self.dtype.as_str() {
            "bool" => self.data.iter().map(|&b| Value::Bool(b != 0)).collect(),
            "float16" => self.to_f64()?.into_iter().map(|f| Value::F32(f as f32)).collect(),
            "float32" => self
                .data
                .chunks_exact(size)
                .map(|c| Value::F32(f32::from_le_bytes(c.try_into().unwrap())))
                .collect(),
            "float64" => self.to_f64()?.into_iter().map(Value::F64).collect(),
            "uint64" => self
                .data
                .chunks_exact(size)
                .map(|c| Value::from(u64::from_le_bytes(c.try_into().unwrap())))
                .collect(),
            "int64" => self
                .data
                .chunks_exact(size)
                .map(|c| Value::Int(i64::from_le_bytes(c.try_into().unwrap())))
                .collect(),
            _ => self.to_f64()?.into_iter().map(|f| Value::Int(f as i64)).collect(),
        };
        Ok(nest(flat, &self.shape))
    }
}

fn nest(flat: Vec<Value>, shape: &[usize]) -> Value {
    if shape.len() <= 1 {
        return Value::Array(flat);
    }
    let inner: usize = shape[1..].iter().product();
    if inner == 0 {
        return Value::Array((0..shape[0]).map(|_| nest(Vec::new(), &shape[1..])).collect());
    }
    let mut rows = Vec::with_capacity(shape[0]);
    let mut items = flat.into_iter();
    for _ in 0..shape[0] {
        rows.push(nest(items.by_ref().take(inner).collect(), &shape[1..]));
    }
    Value::Array(rows)
}

fn f16_to_f64(bits: u16) -> f64 {
    let sign = if bits >> 15 == 1 { -1.0 } else { 1.0 };
    let exp = ((bits >> 10) & 0x1f) as i32;
    let frac = (bits & 0x3ff) as f64;
    match exp {
        0 => sign * frac * 2f64.powi(-24),
        0x1f => if frac == 0.0 { sign * f64::INFINITY } else { f64::NAN },
        _ => sign * (1.0 + frac / 1024.0) * 2f64.powi(exp - 15),
    }
}

/// A stream field as numbers: an `__nd__` dict or a plain list (`np.asarray(value)`).
fn stream_f64(value: &Value) -> Result<Vec<f64>, NdError> {
    if is_nd(value) {
        return NdArray::from_value(value)?.to_f64();
    }
    match value {
        Value::Array(items) => items
            .iter()
            .map(|v| match v {
                Value::Bool(b) => Ok(*b as u8 as f64),
                other => other.as_f64().ok_or_else(|| NdError::NotANumber(other.clone())),
            })
            .collect(),
        other => Err(NdError::NotANumber(other.clone())),
    }
}

/// `IntStream`: `np.asarray(value, dtype=np.int32)`.
pub fn int_stream(value: &Value) -> Result<Vec<i32>, NdError> {
    if is_nd(value) {
        let array = NdArray::from_value(value)?;
        if array.dtype == "int32" {
            return Ok(array.data.chunks_exact(4).map(|c| i32::from_le_bytes(c.try_into().unwrap())).collect());
        }
    }
    Ok(stream_f64(value)?.into_iter().map(|f| f as i32).collect())
}

/// `BoolStream`: `np.asarray(value, dtype=np.bool_)`.
pub fn bool_stream(value: &Value) -> Result<Vec<bool>, NdError> {
    Ok(stream_f64(value)?.into_iter().map(|f| f != 0.0).collect())
}

/// `FloatStream`: `np.asarray(value, dtype=np.float32)`.
pub fn float_stream(value: &Value) -> Result<Vec<f32>, NdError> {
    if is_nd(value) {
        let array = NdArray::from_value(value)?;
        if array.dtype == "float32" {
            return Ok(array.data.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect());
        }
    }
    Ok(stream_f64(value)?.into_iter().map(|f| f as f32).collect())
}

pub fn encode_i32(values: &[i32]) -> NdArray {
    NdArray {
        dtype: "int32".into(),
        shape: vec![values.len()],
        data: values.iter().flat_map(|v| v.to_le_bytes()).collect(),
    }
}

pub fn encode_bool(values: &[bool]) -> NdArray {
    NdArray { dtype: "bool".into(), shape: vec![values.len()], data: values.iter().map(|&b| b as u8).collect() }
}

pub fn encode_f32(values: &[f32]) -> NdArray {
    NdArray {
        dtype: "float32".into(),
        shape: vec![values.len()],
        data: values.iter().flat_map(|v| v.to_le_bytes()).collect(),
    }
}

/// The file monitor's `_live_value`: `__nd__` dicts decoded to (nested) arrays, recursing
/// through dicts (not lists).
pub fn live_value(value: &Value) -> Result<Value, NdError> {
    match value {
        Value::Map(map) => {
            if is_nd(value) {
                return NdArray::from_value(value)?.to_values();
            }
            Ok(Value::Map(map.iter().map(|(k, v)| Ok((k.clone(), live_value(v)?))).collect::<Result<_, NdError>>()?))
        }
        other => Ok(other.clone()),
    }
}
