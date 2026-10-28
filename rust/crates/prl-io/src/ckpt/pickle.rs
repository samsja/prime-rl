//! The pickle subset of orchestrator checkpoints: a protocol-2 writer with CPython's opcode
//! choices (what `torch.save` emits) and a reader for the opcodes CPython produces for the same
//! values (dicts, tuples, str, int, float, None and `copyreg.__newobj__`-reduced dataclasses).

use anyhow::{anyhow, bail, ensure, Context, Result};

#[derive(Debug, Clone, PartialEq)]
pub enum PyValue {
    None,
    Int(i64),
    Float(f64),
    Str(String),
    Tuple(Vec<PyValue>),
    /// Insertion-ordered key/value pairs.
    Dict(Vec<(PyValue, PyValue)>),
    /// `cls.__new__(cls)` followed by `obj.__dict__.update(state)`.
    Object { module: String, name: String, state: Box<PyValue> },
}

impl PyValue {
    pub fn dict<K: Into<String>>(items: impl IntoIterator<Item = (K, PyValue)>) -> Self {
        Self::Dict(items.into_iter().map(|(k, v)| (Self::Str(k.into()), v)).collect())
    }

    pub fn as_dict(&self) -> Result<&[(PyValue, PyValue)]> {
        match self {
            Self::Dict(items) => Ok(items),
            other => bail!("expected a dict, got {other:?}"),
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
            other => bail!("expected a str, got {other:?}"),
        }
    }

    pub fn as_int(&self) -> Result<i64> {
        match self {
            Self::Int(v) => Ok(*v),
            other => bail!("expected an int, got {other:?}"),
        }
    }

    pub fn as_float(&self) -> Result<f64> {
        match self {
            Self::Float(v) => Ok(*v),
            Self::Int(v) => Ok(*v as f64),
            other => bail!("expected a float, got {other:?}"),
        }
    }

    pub fn as_tuple(&self) -> Result<&[PyValue]> {
        match self {
            Self::Tuple(items) => Ok(items),
            other => bail!("expected a tuple, got {other:?}"),
        }
    }
}

const MARK: u8 = b'(';
const STOP: u8 = b'.';
const BINFLOAT: u8 = b'G';
const BININT: u8 = b'J';
const BININT1: u8 = b'K';
const BININT2: u8 = b'M';
const NONE: u8 = b'N';
const BINUNICODE: u8 = b'X';
const BUILD: u8 = b'b';
const GLOBAL: u8 = b'c';
const EMPTY_DICT: u8 = b'}';
const BINGET: u8 = b'h';
const LONG_BINGET: u8 = b'j';
const BINPUT: u8 = b'q';
const LONG_BINPUT: u8 = b'r';
const SETITEM: u8 = b's';
const TUPLE: u8 = b't';
const EMPTY_TUPLE: u8 = b')';
const SETITEMS: u8 = b'u';
const PROTO: u8 = 0x80;
const NEWOBJ: u8 = 0x81;
const TUPLE1: u8 = 0x85;
const TUPLE3: u8 = 0x87;
const LONG1: u8 = 0x8a;

pub fn dumps(value: &PyValue) -> Vec<u8> {
    let mut writer = Writer { out: vec![PROTO, 2], memo: 0 };
    writer.write(value);
    writer.out.push(STOP);
    writer.out
}

struct Writer {
    out: Vec<u8>,
    memo: u32,
}

impl Writer {
    fn put(&mut self) {
        if self.memo < 256 {
            self.out.extend([BINPUT, self.memo as u8]);
        } else {
            self.out.push(LONG_BINPUT);
            self.out.extend(self.memo.to_le_bytes());
        }
        self.memo += 1;
    }

    fn write(&mut self, value: &PyValue) {
        match value {
            PyValue::None => self.out.push(NONE),
            PyValue::Int(v) => self.write_int(*v),
            PyValue::Float(v) => {
                self.out.push(BINFLOAT);
                self.out.extend(v.to_be_bytes());
            }
            PyValue::Str(s) => {
                self.out.push(BINUNICODE);
                self.out.extend((s.len() as u32).to_le_bytes());
                self.out.extend(s.as_bytes());
                self.put();
            }
            PyValue::Tuple(items) if items.is_empty() => self.out.push(EMPTY_TUPLE),
            PyValue::Tuple(items) => {
                if items.len() > 3 {
                    self.out.push(MARK);
                }
                items.iter().for_each(|item| self.write(item));
                self.out.push(if items.len() > 3 { TUPLE } else { TUPLE1 + items.len() as u8 - 1 });
                self.put();
            }
            PyValue::Dict(items) => {
                self.out.push(EMPTY_DICT);
                self.put();
                for chunk in items.chunks(1000) {
                    if chunk.len() > 1 {
                        self.out.push(MARK);
                    }
                    for (k, v) in chunk {
                        self.write(k);
                        self.write(v);
                    }
                    self.out.push(if chunk.len() > 1 { SETITEMS } else { SETITEM });
                }
            }
            PyValue::Object { module, name, state } => {
                self.out.push(GLOBAL);
                self.out.extend(format!("{module}\n{name}\n").as_bytes());
                self.put();
                self.out.extend([EMPTY_TUPLE, NEWOBJ]);
                self.put();
                self.write(state);
                self.out.push(BUILD);
            }
        }
    }

    fn write_int(&mut self, v: i64) {
        if (0..=0xff).contains(&v) {
            self.out.extend([BININT1, v as u8]);
        } else if (0..=0xffff).contains(&v) {
            self.out.push(BININT2);
            self.out.extend((v as u16).to_le_bytes());
        } else if i32::try_from(v).is_ok() {
            self.out.push(BININT);
            self.out.extend((v as i32).to_le_bytes());
        } else {
            // `pickle.encode_long`: minimal little-endian two's complement.
            let mut bytes = v.to_le_bytes().to_vec();
            while bytes.len() > 1 {
                let (last, prev_high) = (bytes[bytes.len() - 1], bytes[bytes.len() - 2] & 0x80);
                if (last == 0x00 && prev_high == 0) || (last == 0xff && prev_high != 0) {
                    bytes.pop();
                } else {
                    break;
                }
            }
            self.out.extend([LONG1, bytes.len() as u8]);
            self.out.extend(bytes);
        }
    }
}

enum Item {
    Mark,
    Value(PyValue),
    Global(String, String),
}

struct Reader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        ensure!(self.pos + n <= self.data.len(), "truncated pickle");
        self.pos += n;
        Ok(&self.data[self.pos - n..self.pos])
    }

    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into()?))
    }

    fn line(&mut self) -> Result<String> {
        let end = self.data[self.pos..].iter().position(|&b| b == b'\n').ok_or_else(|| anyhow!("truncated pickle line"))?;
        let text = std::str::from_utf8(self.take(end)?)?.to_owned();
        self.pos += 1;
        Ok(text)
    }
}

pub fn loads(data: &[u8]) -> Result<PyValue> {
    let mut r = Reader { data, pos: 0 };
    let mut stack: Vec<Item> = Vec::new();
    // Memoized values are copies: checkpoints never share mutable containers.
    let mut memo: Vec<PyValue> = Vec::new();

    fn pop(stack: &mut Vec<Item>) -> Result<PyValue> {
        match stack.pop() {
            Some(Item::Value(v)) => Ok(v),
            _ => bail!("expected a value on the pickle stack"),
        }
    }
    fn pop_mark(stack: &mut Vec<Item>) -> Result<Vec<PyValue>> {
        let mark = stack.iter().rposition(|item| matches!(item, Item::Mark)).ok_or_else(|| anyhow!("pickle MARK not found"))?;
        let items = stack.split_off(mark + 1);
        stack.pop();
        items.into_iter().map(|item| match item {
            Item::Value(v) => Ok(v),
            _ => bail!("unexpected item after MARK"),
        }).collect()
    }
    fn top(stack: &mut [Item]) -> Result<&mut PyValue> {
        match stack.last_mut() {
            Some(Item::Value(v)) => Ok(v),
            _ => bail!("expected a value on top of the pickle stack"),
        }
    }
    fn set_items(stack: &mut [Item], items: Vec<PyValue>) -> Result<()> {
        let PyValue::Dict(dict) = top(stack)? else { bail!("SETITEMS on a non-dict") };
        let mut items = items.into_iter();
        while let (Some(k), Some(v)) = (items.next(), items.next()) {
            match dict.iter_mut().find(|(key, _)| *key == k) {
                Some(slot) => slot.1 = v,
                None => dict.push((k, v)),
            }
        }
        Ok(())
    }

    loop {
        let code = r.take(1)?[0];
        match code {
            PROTO => {
                r.take(1)?;
            }
            STOP => return pop(&mut stack),
            MARK => stack.push(Item::Mark),
            NONE => stack.push(Item::Value(PyValue::None)),
            BININT1 => stack.push(Item::Value(PyValue::Int(r.take(1)?[0] as i64))),
            BININT2 => stack.push(Item::Value(PyValue::Int(u16::from_le_bytes(r.take(2)?.try_into()?) as i64))),
            BININT => stack.push(Item::Value(PyValue::Int(i32::from_le_bytes(r.take(4)?.try_into()?) as i64))),
            LONG1 => {
                let n = r.take(1)?[0] as usize;
                let bytes = r.take(n)?;
                ensure!(n <= 8, "pickled int of {n} bytes does not fit in i64");
                let fill = if bytes.last().is_some_and(|b| b & 0x80 != 0) { 0xff } else { 0x00 };
                let mut buf = [fill; 8];
                buf[..n].copy_from_slice(bytes);
                stack.push(Item::Value(PyValue::Int(i64::from_le_bytes(buf))));
            }
            BINFLOAT => stack.push(Item::Value(PyValue::Float(f64::from_be_bytes(r.take(8)?.try_into()?)))),
            BINUNICODE => {
                let n = r.u32()? as usize;
                let s = std::str::from_utf8(r.take(n)?).context("pickled str is not utf-8")?;
                stack.push(Item::Value(PyValue::Str(s.to_owned())));
            }
            EMPTY_TUPLE => stack.push(Item::Value(PyValue::Tuple(Vec::new()))),
            TUPLE1..=TUPLE3 => {
                let n = (code - TUPLE1 + 1) as usize;
                let items = stack.split_off(stack.len().checked_sub(n).ok_or_else(|| anyhow!("pickle stack underflow"))?);
                let items = items.into_iter().map(|item| match item {
                    Item::Value(v) => Ok(v),
                    _ => bail!("unexpected tuple item"),
                });
                stack.push(Item::Value(PyValue::Tuple(items.collect::<Result<_>>()?)));
            }
            TUPLE => {
                let items = pop_mark(&mut stack)?;
                stack.push(Item::Value(PyValue::Tuple(items)));
            }
            EMPTY_DICT => stack.push(Item::Value(PyValue::Dict(Vec::new()))),
            SETITEM => {
                let v = pop(&mut stack)?;
                let k = pop(&mut stack)?;
                set_items(&mut stack, vec![k, v])?;
            }
            SETITEMS => {
                let items = pop_mark(&mut stack)?;
                set_items(&mut stack, items)?;
            }
            BINPUT | LONG_BINPUT => {
                let idx = if code == BINPUT { r.take(1)?[0] as usize } else { r.u32()? as usize };
                let value = match stack.last() {
                    Some(Item::Value(v)) => v.clone(),
                    // Memoized classes are only re-read by NEWOBJ, which never happens twice here.
                    _ => PyValue::None,
                };
                if memo.len() <= idx {
                    memo.resize(idx + 1, PyValue::None);
                }
                memo[idx] = value;
            }
            BINGET | LONG_BINGET => {
                let idx = if code == BINGET { r.take(1)?[0] as usize } else { r.u32()? as usize };
                stack.push(Item::Value(memo.get(idx).cloned().ok_or_else(|| anyhow!("pickle memo {idx} missing"))?));
            }
            GLOBAL => {
                let (module, name) = (r.line()?, r.line()?);
                stack.push(Item::Global(module, name));
            }
            NEWOBJ => {
                pop(&mut stack)?; // args: dataclasses take none
                let Some(Item::Global(module, name)) = stack.pop() else { bail!("NEWOBJ without a class") };
                stack.push(Item::Value(PyValue::Object { module, name, state: Box::new(PyValue::None) }));
            }
            BUILD => {
                let new_state = pop(&mut stack)?;
                let PyValue::Object { state, .. } = top(&mut stack)? else { bail!("BUILD on a non-object") };
                **state = new_state;
            }
            other => bail!("unsupported pickle opcode 0x{other:02x} at byte {}", r.pos - 1),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ints_use_cpython_opcodes() {
        assert_eq!(dumps(&PyValue::Int(5)), [0x80, 2, b'K', 5, b'.']);
        assert_eq!(dumps(&PyValue::Int(-1)), [0x80, 2, b'J', 0xff, 0xff, 0xff, 0xff, b'.']);
        // 2**31 -> LONG1 with 5 bytes, as CPython pickles MT state words.
        assert_eq!(dumps(&PyValue::Int(1 << 31)), [0x80, 2, 0x8a, 5, 0, 0, 0, 0x80, 0, b'.']);
    }
}
