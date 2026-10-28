//! numpy arrays on the wire: `encode_ndarray` dicts (`{"__nd__": True, dtype, shape, data}`).

use crate::value::{Map, Value};

#[derive(Debug, thiserror::Error)]
pub enum NdError {
    #[error("unsupported __nd__ array: {0}")]
    Unsupported(String),
    #[error("array data of {len} bytes does not fit dtype {dtype} and shape {shape:?}")]
    BadSize { len: usize, dtype: String, shape: Vec<usize> },
    #[error("stream item is not a number: {0:?}")]
    NotANumber(Value),
}

fn is_nd(value: &Value) -> bool {
    value.get("__nd__").is_some_and(Value::truthy)
}

/// A decoded `__nd__` array: its dtype, shape and elements in C order (stream dtypes only).
struct NdArray {
    shape: Vec<usize>,
    values: Vec<Value>,
}

fn decode(value: &Value) -> Result<NdArray, NdError> {
    let dtype = value.get("dtype").and_then(Value::as_str).unwrap_or_default();
    let shape: Vec<usize> = value
        .get("shape")
        .and_then(Value::as_array)
        .map(|s| s.iter().map(|d| d.as_u64().unwrap_or(0) as usize).collect())
        .unwrap_or_default();
    let data = value.get("data").and_then(Value::as_bytes).unwrap_or_default();
    let size = match dtype {
        "bool" => 1,
        "int32" | "float32" => 4,
        "int64" | "float64" => 8,
        other => return Err(NdError::Unsupported(other.to_owned())),
    };
    if shape.iter().product::<usize>() * size != data.len() {
        return Err(NdError::BadSize { len: data.len(), dtype: dtype.to_owned(), shape });
    }
    let chunks = data.chunks_exact(size);
    let values = match dtype {
        "bool" => chunks.map(|c| Value::Bool(c[0] != 0)).collect(),
        "int32" => chunks.map(|c| Value::Int(i32::from_le_bytes(c.try_into().unwrap()) as i64)).collect(),
        "int64" => chunks.map(|c| Value::Int(i64::from_le_bytes(c.try_into().unwrap()))).collect(),
        "float32" => chunks.map(|c| Value::F32(f32::from_le_bytes(c.try_into().unwrap()))).collect(),
        _ => chunks.map(|c| Value::F64(f64::from_le_bytes(c.try_into().unwrap()))).collect(),
    };
    Ok(NdArray { shape, values })
}

/// A stream field as numbers: an `__nd__` dict or a plain list (`np.asarray(value)`).
fn stream_f64(value: &Value) -> Result<Vec<f64>, NdError> {
    let items = if is_nd(value) { decode(value)?.values } else {
        value.as_array().cloned().ok_or_else(|| NdError::NotANumber(value.clone()))?
    };
    items
        .iter()
        .map(|v| match v {
            Value::Bool(b) => Ok(*b as u8 as f64),
            other => other.as_f64().ok_or_else(|| NdError::NotANumber(other.clone())),
        })
        .collect()
}

/// `IntStream`: `np.asarray(value, dtype=np.int32)`.
pub fn int_stream(value: &Value) -> Result<Vec<i32>, NdError> {
    Ok(stream_f64(value)?.into_iter().map(|f| f as i32).collect())
}

/// `BoolStream`: `np.asarray(value, dtype=np.bool_)`.
pub fn bool_stream(value: &Value) -> Result<Vec<bool>, NdError> {
    Ok(stream_f64(value)?.into_iter().map(|f| f != 0.0).collect())
}

/// `FloatStream`: `np.asarray(value, dtype=np.float32)`.
pub fn float_stream(value: &Value) -> Result<Vec<f32>, NdError> {
    Ok(stream_f64(value)?.into_iter().map(|f| f as f32).collect())
}

/// The file monitor's `_live_value`: `__nd__` dicts decoded to (nested) lists, recursing
/// through dicts (not lists). float32 stays `Value::F32`, which JSON prints like numpy.
pub fn live_value(value: &Value) -> Result<Value, NdError> {
    match value {
        Value::Map(_) if is_nd(value) => {
            let NdArray { shape, values } = decode(value)?;
            Ok(nest(values, &shape))
        }
        Value::Map(map) => {
            Ok(Value::Map(map.iter().map(|(k, v)| Ok((k.clone(), live_value(v)?))).collect::<Result<Map, NdError>>()?))
        }
        other => Ok(other.clone()),
    }
}

fn nest(flat: Vec<Value>, shape: &[usize]) -> Value {
    if shape.len() <= 1 {
        return Value::Array(flat);
    }
    let inner: usize = shape[1..].iter().product();
    let mut items = flat.into_iter();
    Value::Array((0..shape[0]).map(|_| nest(items.by_ref().take(inner).collect(), &shape[1..])).collect())
}
