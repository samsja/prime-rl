//! Minimal msgpack writer emitting exactly msgspec's encoding (smallest int/str/bin/array forms).

pub struct Writer {
    pub buf: Vec<u8>,
}

impl Writer {
    pub fn new() -> Self {
        Self { buf: Vec::new() }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self { buf: Vec::with_capacity(capacity) }
    }

    pub fn nil(&mut self) {
        self.buf.push(0xc0);
    }

    pub fn bool(&mut self, value: bool) {
        self.buf.push(if value { 0xc3 } else { 0xc2 });
    }

    pub fn int(&mut self, value: i64) {
        let b = &mut self.buf;
        if value >= 0 {
            let v = value as u64;
            if v < 128 {
                b.push(v as u8);
            } else if v <= u8::MAX as u64 {
                b.extend_from_slice(&[0xcc, v as u8]);
            } else if v <= u16::MAX as u64 {
                b.push(0xcd);
                b.extend_from_slice(&(v as u16).to_be_bytes());
            } else if v <= u32::MAX as u64 {
                b.push(0xce);
                b.extend_from_slice(&(v as u32).to_be_bytes());
            } else {
                b.push(0xcf);
                b.extend_from_slice(&v.to_be_bytes());
            }
        } else if value >= -32 {
            b.push(value as i8 as u8);
        } else if value >= i8::MIN as i64 {
            b.extend_from_slice(&[0xd0, value as i8 as u8]);
        } else if value >= i16::MIN as i64 {
            b.push(0xd1);
            b.extend_from_slice(&(value as i16).to_be_bytes());
        } else if value >= i32::MIN as i64 {
            b.push(0xd2);
            b.extend_from_slice(&(value as i32).to_be_bytes());
        } else {
            b.push(0xd3);
            b.extend_from_slice(&value.to_be_bytes());
        }
    }

    pub fn str(&mut self, value: &str) {
        let n = value.len();
        let b = &mut self.buf;
        if n < 32 {
            b.push(0xa0 | n as u8);
        } else if n < 256 {
            b.extend_from_slice(&[0xd9, n as u8]);
        } else if n < 65536 {
            b.push(0xda);
            b.extend_from_slice(&(n as u16).to_be_bytes());
        } else {
            b.push(0xdb);
            b.extend_from_slice(&(n as u32).to_be_bytes());
        }
        b.extend_from_slice(value.as_bytes());
    }

    pub fn bin_header(&mut self, n: usize) {
        let b = &mut self.buf;
        if n < 256 {
            b.extend_from_slice(&[0xc4, n as u8]);
        } else if n < 65536 {
            b.push(0xc5);
            b.extend_from_slice(&(n as u16).to_be_bytes());
        } else {
            b.push(0xc6);
            b.extend_from_slice(&(n as u32).to_be_bytes());
        }
    }

    pub fn array_header(&mut self, n: usize) {
        let b = &mut self.buf;
        if n < 16 {
            b.push(0x90 | n as u8);
        } else if n < 65536 {
            b.push(0xdc);
            b.extend_from_slice(&(n as u16).to_be_bytes());
        } else {
            b.push(0xdd);
            b.extend_from_slice(&(n as u32).to_be_bytes());
        }
    }

    /// A per-token stream as raw little-endian bytes (`encode_array`).
    pub fn stream<T: RawBytes>(&mut self, values: &[T]) {
        self.bin_header(values.len() * T::SIZE);
        for v in values {
            v.extend_le(&mut self.buf);
        }
    }
}

impl Default for Writer {
    fn default() -> Self {
        Self::new()
    }
}

pub trait RawBytes: Copy {
    const SIZE: usize;
    fn extend_le(self, buf: &mut Vec<u8>);
}

macro_rules! raw_bytes {
    ($($t:ty),*) => {$(
        impl RawBytes for $t {
            const SIZE: usize = std::mem::size_of::<$t>();
            fn extend_le(self, buf: &mut Vec<u8>) {
                buf.extend_from_slice(&self.to_le_bytes());
            }
        }
    )*};
}
raw_bytes!(i16, i32, f32);

impl RawBytes for bool {
    const SIZE: usize = 1;
    fn extend_le(self, buf: &mut Vec<u8>) {
        buf.push(self as u8);
    }
}
