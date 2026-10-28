//! Minimal Python pickle codec for checkpoint payloads: protocol-2 writer (what `torch.save`
//! emits) and a reader for the opcodes CPython produces for plain containers, scalars and
//! `copyreg.__newobj__`-reduced objects such as dataclasses.

use anyhow::{anyhow, bail, ensure, Context, Result};

/// A Python value as stored in a checkpoint.
#[derive(Debug, Clone, PartialEq)]
pub enum PyValue {
    None,
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(String),
    Bytes(Vec<u8>),
    List(Vec<PyValue>),
    Tuple(Vec<PyValue>),
    /// Insertion-ordered key/value pairs.
    Dict(Vec<(PyValue, PyValue)>),
    /// `module.name` reference (`GLOBAL`).
    Global { module: String, name: String },
    /// `cls.__new__(cls, *args)` followed by `__setstate__`/`__dict__.update(state)`.
    Object { module: String, name: String, args: Vec<PyValue>, state: Option<Box<PyValue>> },
}

impl PyValue {
    pub fn str(s: impl Into<String>) -> Self {
        Self::Str(s.into())
    }

    pub fn dict<K: Into<String>>(items: impl IntoIterator<Item = (K, PyValue)>) -> Self {
        Self::Dict(items.into_iter().map(|(k, v)| (Self::Str(k.into()), v)).collect())
    }

    pub fn as_dict(&self) -> Result<&[(PyValue, PyValue)]> {
        match self {
            Self::Dict(items) => Ok(items),
            other => bail!("expected a dict, got {}", other.type_name()),
        }
    }

    /// `d[key]` for a str key.
    pub fn get(&self, key: &str) -> Result<&PyValue> {
        self.as_dict()?
            .iter()
            .find(|(k, _)| matches!(k, PyValue::Str(s) if s == key))
            .map(|(_, v)| v)
            .ok_or_else(|| anyhow!("missing key {key:?}"))
    }

    pub fn str_keys(&self) -> Result<Vec<&str>> {
        self.as_dict()?.iter().map(|(k, _)| k.as_str()).collect()
    }

    pub fn as_str(&self) -> Result<&str> {
        match self {
            Self::Str(s) => Ok(s),
            other => bail!("expected a str, got {}", other.type_name()),
        }
    }

    pub fn as_int(&self) -> Result<i64> {
        match self {
            Self::Int(v) => Ok(*v),
            Self::Bool(b) => Ok(*b as i64),
            other => bail!("expected an int, got {}", other.type_name()),
        }
    }

    pub fn as_float(&self) -> Result<f64> {
        match self {
            Self::Float(v) => Ok(*v),
            Self::Int(v) => Ok(*v as f64),
            other => bail!("expected a float, got {}", other.type_name()),
        }
    }

    pub fn as_tuple(&self) -> Result<&[PyValue]> {
        match self {
            Self::Tuple(items) | Self::List(items) => Ok(items),
            other => bail!("expected a tuple, got {}", other.type_name()),
        }
    }

    pub fn type_name(&self) -> &'static str {
        match self {
            Self::None => "NoneType",
            Self::Bool(_) => "bool",
            Self::Int(_) => "int",
            Self::Float(_) => "float",
            Self::Str(_) => "str",
            Self::Bytes(_) => "bytes",
            Self::List(_) => "list",
            Self::Tuple(_) => "tuple",
            Self::Dict(_) => "dict",
            Self::Global { .. } => "global",
            Self::Object { .. } => "object",
        }
    }
}

mod op {
    pub const MARK: u8 = b'(';
    pub const STOP: u8 = b'.';
    pub const POP: u8 = b'0';
    pub const POP_MARK: u8 = b'1';
    pub const BINFLOAT: u8 = b'G';
    pub const BININT: u8 = b'J';
    pub const BININT1: u8 = b'K';
    pub const BININT2: u8 = b'M';
    pub const NONE: u8 = b'N';
    pub const BINUNICODE: u8 = b'X';
    pub const APPEND: u8 = b'a';
    pub const BUILD: u8 = b'b';
    pub const GLOBAL: u8 = b'c';
    pub const DICT: u8 = b'd';
    pub const EMPTY_DICT: u8 = b'}';
    pub const APPENDS: u8 = b'e';
    pub const BINGET: u8 = b'h';
    pub const LONG_BINGET: u8 = b'j';
    pub const LIST: u8 = b'l';
    pub const EMPTY_LIST: u8 = b']';
    pub const BINPUT: u8 = b'q';
    pub const LONG_BINPUT: u8 = b'r';
    pub const SETITEM: u8 = b's';
    pub const TUPLE: u8 = b't';
    pub const EMPTY_TUPLE: u8 = b')';
    pub const SETITEMS: u8 = b'u';
    pub const REDUCE: u8 = b'R';
    pub const PROTO: u8 = 0x80;
    pub const NEWOBJ: u8 = 0x81;
    pub const TUPLE1: u8 = 0x85;
    pub const TUPLE2: u8 = 0x86;
    pub const TUPLE3: u8 = 0x87;
    pub const NEWTRUE: u8 = 0x88;
    pub const NEWFALSE: u8 = 0x89;
    pub const LONG1: u8 = 0x8a;
    pub const LONG4: u8 = 0x8b;
    pub const BINBYTES: u8 = b'B';
    pub const SHORT_BINBYTES: u8 = b'C';
    pub const SHORT_BINUNICODE: u8 = 0x8c;
    pub const BINUNICODE8: u8 = 0x8d;
    pub const BINBYTES8: u8 = 0x8e;
    pub const EMPTY_SET: u8 = 0x8f;
    pub const FRAME: u8 = 0x95;
    pub const MEMOIZE: u8 = 0x94;
    pub const STACK_GLOBAL: u8 = 0x93;
    pub const NEWOBJ_EX: u8 = 0x92;
}

/// Pickle `value` with protocol 2, mirroring CPython's `pickle.Pickler` opcode choices.
pub fn dumps(value: &PyValue) -> Vec<u8> {
    let mut writer = Writer { out: vec![op::PROTO, 2], memo: 0 };
    writer.write(value);
    writer.out.push(op::STOP);
    writer.out
}

struct Writer {
    out: Vec<u8>,
    memo: u32,
}

impl Writer {
    fn put(&mut self) {
        if self.memo < 256 {
            self.out.extend([op::BINPUT, self.memo as u8]);
        } else {
            self.out.push(op::LONG_BINPUT);
            self.out.extend(self.memo.to_le_bytes());
        }
        self.memo += 1;
    }

    fn write_global(&mut self, module: &str, name: &str) {
        self.out.push(op::GLOBAL);
        self.out.extend(module.as_bytes());
        self.out.push(b'\n');
        self.out.extend(name.as_bytes());
        self.out.push(b'\n');
        self.put();
    }

    fn write(&mut self, value: &PyValue) {
        match value {
            PyValue::None => self.out.push(op::NONE),
            PyValue::Bool(true) => self.out.push(op::NEWTRUE),
            PyValue::Bool(false) => self.out.push(op::NEWFALSE),
            PyValue::Int(v) => self.write_int(*v),
            PyValue::Float(v) => {
                self.out.push(op::BINFLOAT);
                self.out.extend(v.to_be_bytes());
            }
            PyValue::Str(s) => {
                self.out.push(op::BINUNICODE);
                self.out.extend((s.len() as u32).to_le_bytes());
                self.out.extend(s.as_bytes());
                self.put();
            }
            PyValue::Bytes(b) => {
                // Protocol 2 has no bytes opcode: CPython reduces to `_codecs.encode(latin1 str, "latin1")`.
                self.write_global("_codecs", "encode");
                let latin1: String = b.iter().map(|&c| c as char).collect();
                self.write(&PyValue::Str(latin1));
                self.write(&PyValue::str("latin1"));
                self.out.push(op::TUPLE2);
                self.put();
                self.out.push(op::REDUCE);
                self.put();
            }
            PyValue::Tuple(items) => {
                if items.is_empty() {
                    self.out.push(op::EMPTY_TUPLE);
                    return;
                }
                if items.len() <= 3 {
                    for item in items {
                        self.write(item);
                    }
                    self.out.push([op::TUPLE1, op::TUPLE2, op::TUPLE3][items.len() - 1]);
                } else {
                    self.out.push(op::MARK);
                    for item in items {
                        self.write(item);
                    }
                    self.out.push(op::TUPLE);
                }
                self.put();
            }
            PyValue::List(items) => {
                self.out.push(op::EMPTY_LIST);
                self.put();
                for chunk in items.chunks(1000) {
                    if chunk.len() == 1 {
                        self.write(&chunk[0]);
                        self.out.push(op::APPEND);
                    } else {
                        self.out.push(op::MARK);
                        for item in chunk {
                            self.write(item);
                        }
                        self.out.push(op::APPENDS);
                    }
                }
            }
            PyValue::Dict(items) => {
                self.out.push(op::EMPTY_DICT);
                self.put();
                self.write_items(items);
            }
            PyValue::Global { module, name } => self.write_global(module, name),
            PyValue::Object { module, name, args, state } => {
                self.write_global(module, name);
                self.write(&PyValue::Tuple(args.clone()));
                self.out.push(op::NEWOBJ);
                self.put();
                if let Some(state) = state {
                    self.write(state);
                    self.out.push(op::BUILD);
                }
            }
        }
    }

    fn write_items(&mut self, items: &[(PyValue, PyValue)]) {
        for chunk in items.chunks(1000) {
            if chunk.len() == 1 {
                self.write(&chunk[0].0);
                self.write(&chunk[0].1);
                self.out.push(op::SETITEM);
            } else {
                self.out.push(op::MARK);
                for (k, v) in chunk {
                    self.write(k);
                    self.write(v);
                }
                self.out.push(op::SETITEMS);
            }
        }
    }

    fn write_int(&mut self, v: i64) {
        if (0..=0xff).contains(&v) {
            self.out.extend([op::BININT1, v as u8]);
        } else if (0..=0xffff).contains(&v) {
            self.out.push(op::BININT2);
            self.out.extend((v as u16).to_le_bytes());
        } else if (-0x8000_0000..=0x7fff_ffff).contains(&v) {
            self.out.push(op::BININT);
            self.out.extend((v as i32).to_le_bytes());
        } else {
            let bytes = encode_long(v);
            self.out.extend([op::LONG1, bytes.len() as u8]);
            self.out.extend(bytes);
        }
    }
}

/// `pickle.encode_long`: minimal little-endian two's complement.
fn encode_long(v: i64) -> Vec<u8> {
    let mut bytes = v.to_le_bytes().to_vec();
    while bytes.len() > 1 {
        let last = bytes[bytes.len() - 1];
        let prev_high = bytes[bytes.len() - 2] & 0x80;
        if (last == 0x00 && prev_high == 0) || (last == 0xff && prev_high != 0) {
            bytes.pop();
        } else {
            break;
        }
    }
    bytes
}

fn decode_long(bytes: &[u8]) -> Result<i64> {
    if bytes.is_empty() {
        return Ok(0);
    }
    ensure!(bytes.len() <= 8, "pickled int of {} bytes does not fit in i64", bytes.len());
    let fill = if bytes[bytes.len() - 1] & 0x80 != 0 { 0xff } else { 0x00 };
    let mut buf = [fill; 8];
    buf[..bytes.len()].copy_from_slice(bytes);
    Ok(i64::from_le_bytes(buf))
}

enum Item {
    Mark,
    Value(PyValue),
}

/// Unpickle `data` (protocols 0-5 binary opcodes for plain data; no persistent ids).
pub fn loads(data: &[u8]) -> Result<PyValue> {
    let mut reader = Reader { data, pos: 0 };
    let mut stack: Vec<Item> = Vec::new();
    let mut memo: std::collections::HashMap<u32, PyValue> = std::collections::HashMap::new();

    fn pop(stack: &mut Vec<Item>) -> Result<PyValue> {
        match stack.pop() {
            Some(Item::Value(v)) => Ok(v),
            Some(Item::Mark) => bail!("unexpected MARK on the pickle stack"),
            None => bail!("pickle stack underflow"),
        }
    }
    fn pop_mark(stack: &mut Vec<Item>) -> Result<Vec<PyValue>> {
        let mut items = Vec::new();
        loop {
            match stack.pop() {
                Some(Item::Value(v)) => items.push(v),
                Some(Item::Mark) => break,
                None => bail!("pickle MARK not found"),
            }
        }
        items.reverse();
        Ok(items)
    }
    fn top(stack: &mut [Item]) -> Result<&mut PyValue> {
        match stack.last_mut() {
            Some(Item::Value(v)) => Ok(v),
            _ => bail!("expected a value on top of the pickle stack"),
        }
    }
    fn set_items(target: &mut PyValue, items: Vec<PyValue>) -> Result<()> {
        let PyValue::Dict(dict) = target else { bail!("SETITEMS on a {}", target.type_name()) };
        let mut iter = items.into_iter();
        while let (Some(k), Some(v)) = (iter.next(), iter.next()) {
            if let Some(slot) = dict.iter_mut().find(|(key, _)| *key == k) {
                slot.1 = v;
            } else {
                dict.push((k, v));
            }
        }
        Ok(())
    }

    loop {
        let code = reader.byte()?;
        match code {
            op::PROTO => {
                reader.byte()?;
            }
            op::FRAME => {
                reader.take(8)?;
            }
            op::STOP => return pop(&mut stack),
            op::MARK => stack.push(Item::Mark),
            op::POP => {
                stack.pop();
            }
            op::POP_MARK => {
                pop_mark(&mut stack)?;
            }
            op::NONE => stack.push(Item::Value(PyValue::None)),
            op::NEWTRUE => stack.push(Item::Value(PyValue::Bool(true))),
            op::NEWFALSE => stack.push(Item::Value(PyValue::Bool(false))),
            op::BININT1 => stack.push(Item::Value(PyValue::Int(reader.byte()? as i64))),
            op::BININT2 => stack.push(Item::Value(PyValue::Int(u16::from_le_bytes(reader.array()?) as i64))),
            op::BININT => stack.push(Item::Value(PyValue::Int(i32::from_le_bytes(reader.array()?) as i64))),
            op::LONG1 => {
                let n = reader.byte()? as usize;
                stack.push(Item::Value(PyValue::Int(decode_long(reader.take(n)?)?)));
            }
            op::LONG4 => {
                let n = u32::from_le_bytes(reader.array()?) as usize;
                stack.push(Item::Value(PyValue::Int(decode_long(reader.take(n)?)?)));
            }
            op::BINFLOAT => stack.push(Item::Value(PyValue::Float(f64::from_be_bytes(reader.array()?)))),
            op::BINUNICODE | op::SHORT_BINUNICODE | op::BINUNICODE8 => {
                let n = reader.len(code, op::SHORT_BINUNICODE, op::BINUNICODE, op::BINUNICODE8)?;
                let s = std::str::from_utf8(reader.take(n)?).context("pickled str is not utf-8")?;
                stack.push(Item::Value(PyValue::Str(s.to_owned())));
            }
            op::BINBYTES | op::SHORT_BINBYTES | op::BINBYTES8 => {
                let n = reader.len(code, op::SHORT_BINBYTES, op::BINBYTES, op::BINBYTES8)?;
                stack.push(Item::Value(PyValue::Bytes(reader.take(n)?.to_vec())));
            }
            op::EMPTY_TUPLE => stack.push(Item::Value(PyValue::Tuple(Vec::new()))),
            op::TUPLE1 | op::TUPLE2 | op::TUPLE3 => {
                let n = (code - op::TUPLE1 + 1) as usize;
                let mut items = (0..n).map(|_| pop(&mut stack)).collect::<Result<Vec<_>>>()?;
                items.reverse();
                stack.push(Item::Value(PyValue::Tuple(items)));
            }
            op::TUPLE => {
                let items = pop_mark(&mut stack)?;
                stack.push(Item::Value(PyValue::Tuple(items)));
            }
            op::EMPTY_LIST => stack.push(Item::Value(PyValue::List(Vec::new()))),
            op::LIST => {
                let items = pop_mark(&mut stack)?;
                stack.push(Item::Value(PyValue::List(items)));
            }
            op::APPEND => {
                let v = pop(&mut stack)?;
                match top(&mut stack)? {
                    PyValue::List(items) => items.push(v),
                    other => bail!("APPEND on a {}", other.type_name()),
                }
            }
            op::APPENDS => {
                let vs = pop_mark(&mut stack)?;
                match top(&mut stack)? {
                    PyValue::List(items) => items.extend(vs),
                    other => bail!("APPENDS on a {}", other.type_name()),
                }
            }
            op::EMPTY_DICT => stack.push(Item::Value(PyValue::Dict(Vec::new()))),
            op::DICT => {
                let items = pop_mark(&mut stack)?;
                let mut dict = PyValue::Dict(Vec::new());
                set_items(&mut dict, items)?;
                stack.push(Item::Value(dict));
            }
            op::SETITEM => {
                let v = pop(&mut stack)?;
                let k = pop(&mut stack)?;
                set_items(top(&mut stack)?, vec![k, v])?;
            }
            op::SETITEMS => {
                let items = pop_mark(&mut stack)?;
                set_items(top(&mut stack)?, items)?;
            }
            op::EMPTY_SET => bail!("sets are not supported in orchestrator checkpoints"),
            op::BINPUT => {
                let idx = reader.byte()? as u32;
                memo.insert(idx, top(&mut stack)?.clone());
            }
            op::LONG_BINPUT => {
                let idx = u32::from_le_bytes(reader.array()?);
                memo.insert(idx, top(&mut stack)?.clone());
            }
            op::MEMOIZE => {
                let idx = memo.len() as u32;
                memo.insert(idx, top(&mut stack)?.clone());
            }
            op::BINGET | op::LONG_BINGET => {
                let idx = if code == op::BINGET { reader.byte()? as u32 } else { u32::from_le_bytes(reader.array()?) };
                let v = memo.get(&idx).ok_or_else(|| anyhow!("pickle memo {idx} missing"))?.clone();
                stack.push(Item::Value(v));
            }
            op::GLOBAL => {
                let module = reader.line()?;
                let name = reader.line()?;
                stack.push(Item::Value(PyValue::Global { module, name }));
            }
            op::STACK_GLOBAL => {
                let name = pop(&mut stack)?.as_str()?.to_owned();
                let module = pop(&mut stack)?.as_str()?.to_owned();
                stack.push(Item::Value(PyValue::Global { module, name }));
            }
            op::NEWOBJ | op::NEWOBJ_EX => {
                if code == op::NEWOBJ_EX {
                    pop(&mut stack)?; // kwargs
                }
                let args = pop(&mut stack)?.as_tuple()?.to_vec();
                let PyValue::Global { module, name } = pop(&mut stack)? else { bail!("NEWOBJ without a class") };
                stack.push(Item::Value(PyValue::Object { module, name, args, state: None }));
            }
            op::REDUCE => {
                let args = pop(&mut stack)?;
                let callable = pop(&mut stack)?;
                stack.push(Item::Value(reduce(callable, args)?));
            }
            op::BUILD => {
                let state = pop(&mut stack)?;
                match top(&mut stack)? {
                    PyValue::Object { state: slot, .. } => *slot = Some(Box::new(state)),
                    other => bail!("BUILD on a {}", other.type_name()),
                }
            }
            other => bail!("unsupported pickle opcode 0x{other:02x} at byte {}", reader.pos - 1),
        }
    }
}

/// The few reductions plain-data pickles contain.
fn reduce(callable: PyValue, args: PyValue) -> Result<PyValue> {
    let PyValue::Global { module, name } = &callable else { bail!("REDUCE on a {}", callable.type_name()) };
    let args = args.as_tuple()?;
    match (module.as_str(), name.as_str()) {
        ("_codecs", "encode") => {
            let text = args.first().ok_or_else(|| anyhow!("_codecs.encode without args"))?.as_str()?;
            Ok(PyValue::Bytes(text.chars().map(|c| c as u32 as u8).collect()))
        }
        ("collections", "OrderedDict") if args.is_empty() => Ok(PyValue::Dict(Vec::new())),
        ("copyreg", "_reconstructor") => {
            let PyValue::Global { module, name } = args.first().cloned().unwrap_or(PyValue::None) else {
                bail!("copyreg._reconstructor without a class")
            };
            Ok(PyValue::Object { module, name, args: Vec::new(), state: None })
        }
        _ => bail!("unsupported pickle reduction {module}.{name}"),
    }
}

struct Reader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        ensure!(self.pos + n <= self.data.len(), "truncated pickle");
        let slice = &self.data[self.pos..self.pos + n];
        self.pos += n;
        Ok(slice)
    }

    fn byte(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }

    fn array<const N: usize>(&mut self) -> Result<[u8; N]> {
        Ok(self.take(N)?.try_into().expect("length checked"))
    }

    fn len(&mut self, code: u8, short: u8, four: u8, eight: u8) -> Result<usize> {
        Ok(match code {
            c if c == short => self.byte()? as usize,
            c if c == four => u32::from_le_bytes(self.array()?) as usize,
            c if c == eight => u64::from_le_bytes(self.array()?) as usize,
            _ => unreachable!(),
        })
    }

    fn line(&mut self) -> Result<String> {
        let start = self.pos;
        let end = self.data[start..].iter().position(|&b| b == b'\n').ok_or_else(|| anyhow!("truncated pickle line"))?;
        self.pos = start + end + 1;
        Ok(std::str::from_utf8(&self.data[start..start + end])?.to_owned())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ints_use_cpython_opcodes() {
        assert_eq!(dumps(&PyValue::Int(5)), vec![0x80, 2, b'K', 5, b'.']);
        assert_eq!(dumps(&PyValue::Int(300)), vec![0x80, 2, b'M', 0x2c, 0x01, b'.']);
        assert_eq!(dumps(&PyValue::Int(-1)), vec![0x80, 2, b'J', 0xff, 0xff, 0xff, 0xff, b'.']);
        // 2**31 -> LONG1 5 bytes (00 00 00 80 00), as pickletools shows for MT state words.
        assert_eq!(dumps(&PyValue::Int(2147483648)), vec![0x80, 2, 0x8a, 5, 0, 0, 0, 0x80, 0, b'.']);
        assert_eq!(dumps(&PyValue::Int(1 << 40)), vec![0x80, 2, 0x8a, 6, 0, 0, 0, 0, 0, 1, b'.']);
    }

    #[test]
    fn round_trip() {
        let value = PyValue::dict([
            ("a", PyValue::Tuple((0..700).map(PyValue::Int).collect())),
            ("b", PyValue::List(vec![PyValue::Float(1.5), PyValue::None, PyValue::Bool(true)])),
            ("c", PyValue::Bytes(vec![0, 200, 255])),
            ("neg", PyValue::Int(-(1 << 40))),
            (
                "o",
                PyValue::Object {
                    module: "m".into(),
                    name: "C".into(),
                    args: vec![],
                    state: Some(Box::new(PyValue::dict([("x", PyValue::Int(1))]))),
                },
            ),
        ]);
        assert_eq!(loads(&dumps(&value)).unwrap(), value);
    }
}
