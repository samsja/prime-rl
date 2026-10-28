//! Extraction helpers shared by the config structs.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use serde_json::Value;

/// `obj.model_dump(mode="json")` as a JSON value: for sub-configs Rust only forwards.
pub(crate) fn model_dump(obj: &Bound<'_, PyAny>) -> PyResult<Value> {
    let kwargs = PyDict::new(obj.py());
    kwargs.set_item("mode", "json")?;
    to_json(&obj.call_method("model_dump", (), Some(&kwargs))?)
}

/// A plain JSON-able Python value (dict, list, str, number, bool, None) as a JSON value.
pub(crate) fn to_json(obj: &Bound<'_, PyAny>) -> PyResult<Value> {
    pythonize::depythonize(obj).map_err(|e| PyValueError::new_err(e.to_string()))
}

/// The `type` (or other discriminator) attribute of a discriminated-union member.
pub(crate) fn tag(obj: &Bound<'_, PyAny>, discriminator: &str) -> PyResult<String> {
    obj.getattr(discriminator)?.extract()
}

pub(crate) fn unknown_tag(union: &str, tag: &str, known: &[&str]) -> PyErr {
    PyValueError::new_err(format!(
        "unknown {union} {tag:?}, expected one of {known:?}"
    ))
}

/// A Rust enum for a pydantic `Literal[...]` of strings, extracted from the `str`.
macro_rules! str_enum {
    ($(#[$meta:meta])* $name:ident { $($variant:ident = $value:literal),+ $(,)? }) => {
        $(#[$meta])*
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        pub enum $name {
            $($variant),+
        }

        impl $name {
            pub const VALUES: &'static [&'static str] = &[$($value),+];

            pub fn as_str(&self) -> &'static str {
                match self {
                    $(Self::$variant => $value),+
                }
            }
        }

        impl std::fmt::Display for $name {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str(self.as_str())
            }
        }

        impl<'py> pyo3::FromPyObject<'py> for $name {
            fn extract_bound(obj: &pyo3::Bound<'py, pyo3::PyAny>) -> pyo3::PyResult<Self> {
                let value: String = obj.extract()?;
                match value.as_str() {
                    $($value => Ok(Self::$variant),)+
                    other => Err($crate::py::unknown_tag(stringify!($name), other, Self::VALUES)),
                }
            }
        }
    };
}

pub(crate) use str_enum;
