//! CPython-compatible `random.Random` (MT19937): seeding, `getstate`/`setstate`, `random()` and
//! `choices(k=1)`, so a resumed Python run and a Rust run draw the same sequence.

use anyhow::{bail, ensure, Result};

use super::pickle::PyValue;

const N: usize = 624;
const M: usize = 397;

/// `random.Random.getstate()`: `(3, (*mt, index), gauss_next)`.
#[derive(Debug, Clone, PartialEq)]
pub struct PyRandomState {
    pub key: [u32; N],
    pub pos: u32,
    pub gauss_next: Option<f64>,
}

impl PyRandomState {
    pub(crate) fn to_py(&self) -> PyValue {
        let mut words: Vec<PyValue> = self.key.iter().map(|&w| PyValue::Int(w as i64)).collect();
        words.push(PyValue::Int(self.pos as i64));
        PyValue::Tuple(vec![
            PyValue::Int(3),
            PyValue::Tuple(words),
            self.gauss_next.map_or(PyValue::None, PyValue::Float),
        ])
    }

    pub(crate) fn from_py(value: &PyValue) -> Result<Self> {
        let parts = value.as_tuple()?;
        ensure!(parts.len() == 3, "random state must be a 3-tuple");
        let version = parts[0].as_int()?;
        ensure!(version == 3, "state with version {version} passed to Random.setstate() of version 3");
        let words = parts[1].as_tuple()?;
        ensure!(words.len() == N + 1, "random state tuple must hold {} ints", N + 1);
        let mut key = [0u32; N];
        for (slot, word) in key.iter_mut().zip(words) {
            *slot = u32::try_from(word.as_int()?)?;
        }
        let pos = u32::try_from(words[N].as_int()?)?;
        ensure!(pos as usize <= N, "invalid random state index {pos}");
        let gauss_next = match &parts[2] {
            PyValue::None => None,
            other => Some(other.as_float()?),
        };
        Ok(Self { key, pos, gauss_next })
    }
}

#[derive(Debug, Clone)]
pub struct PyRandom {
    mt: [u32; N],
    index: usize,
    gauss_next: Option<f64>,
}

impl PyRandom {
    /// `random.Random(seed)` for a non-negative int seed (`init_by_array` over its 32-bit words).
    pub fn new(seed: u64) -> Self {
        let key: Vec<u32> = if seed >> 32 == 0 { vec![seed as u32] } else { vec![seed as u32, (seed >> 32) as u32] };
        Self::from_key(&key)
    }

    fn from_key(key: &[u32]) -> Self {
        let mut rng = Self { mt: [0; N], index: N, gauss_next: None };
        rng.init_genrand(19650218);
        let (mut i, mut j) = (1usize, 0usize);
        let mut k = N.max(key.len());
        while k > 0 {
            let prev = rng.mt[i - 1] ^ (rng.mt[i - 1] >> 30);
            rng.mt[i] = (rng.mt[i] ^ prev.wrapping_mul(1664525)).wrapping_add(key[j]).wrapping_add(j as u32);
            i += 1;
            j += 1;
            if i >= N {
                rng.mt[0] = rng.mt[N - 1];
                i = 1;
            }
            if j >= key.len() {
                j = 0;
            }
            k -= 1;
        }
        k = N - 1;
        while k > 0 {
            let prev = rng.mt[i - 1] ^ (rng.mt[i - 1] >> 30);
            rng.mt[i] = (rng.mt[i] ^ prev.wrapping_mul(1566083941)).wrapping_sub(i as u32);
            i += 1;
            if i >= N {
                rng.mt[0] = rng.mt[N - 1];
                i = 1;
            }
            k -= 1;
        }
        rng.mt[0] = 0x8000_0000;
        rng
    }

    fn init_genrand(&mut self, seed: u32) {
        self.mt[0] = seed;
        for i in 1..N {
            self.mt[i] = 1812433253u32.wrapping_mul(self.mt[i - 1] ^ (self.mt[i - 1] >> 30)).wrapping_add(i as u32);
        }
        self.index = N;
    }

    pub fn from_state(state: &PyRandomState) -> Self {
        Self { mt: state.key, index: state.pos as usize, gauss_next: state.gauss_next }
    }

    pub fn state(&self) -> PyRandomState {
        PyRandomState { key: self.mt, pos: self.index as u32, gauss_next: self.gauss_next }
    }

    fn genrand_u32(&mut self) -> u32 {
        const UPPER: u32 = 0x8000_0000;
        const LOWER: u32 = 0x7fff_ffff;
        const MATRIX_A: u32 = 0x9908_b0df;
        if self.index >= N {
            for kk in 0..N {
                let y = (self.mt[kk] & UPPER) | (self.mt[(kk + 1) % N] & LOWER);
                let mag = if y & 1 == 1 { MATRIX_A } else { 0 };
                self.mt[kk] = self.mt[(kk + M) % N] ^ (y >> 1) ^ mag;
            }
            self.index = 0;
        }
        let mut y = self.mt[self.index];
        self.index += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^= y >> 18;
        y
    }

    /// `random.random()`: 53-bit float in [0, 1).
    pub fn random(&mut self) -> f64 {
        let a = (self.genrand_u32() >> 5) as f64;
        let b = (self.genrand_u32() >> 6) as f64;
        (a * 67108864.0 + b) * (1.0 / 9007199254740992.0)
    }

    /// Index chosen by `random.choices(population, weights=weights, k=1)`.
    pub fn choices_index(&mut self, weights: &[f64]) -> Result<usize> {
        ensure!(!weights.is_empty(), "choices() needs a non-empty population");
        let mut cum = Vec::with_capacity(weights.len());
        let mut acc = 0.0;
        for &w in weights {
            acc += w;
            cum.push(acc);
        }
        let total = acc + 0.0;
        if total <= 0.0 {
            bail!("Total of weights must be greater than zero");
        }
        if !total.is_finite() {
            bail!("Total of weights must be finite");
        }
        let x = self.random() * total;
        let hi = weights.len() - 1;
        // bisect_right(cum, x, 0, hi)
        Ok(cum[..hi].partition_point(|&c| c <= x))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_cpython() {
        // r = random.Random(42); [r.random() for _ in range(3)]
        let mut rng = PyRandom::new(42);
        assert_eq!([rng.random(), rng.random(), rng.random()], [0.6394267984578837, 0.025010755222666936, 0.27502931836911926]);
        // r = random.Random(42); [r.choices(range(4), weights=[1.0, 0.0, 2.5, 0.5])[0] for _ in range(10)]
        let mut rng = PyRandom::new(42);
        let picks: Vec<usize> = (0..10).map(|_| rng.choices_index(&[1.0, 0.0, 2.5, 0.5]).unwrap()).collect();
        assert_eq!(picks, [2, 0, 2, 0, 2, 2, 3, 0, 2, 0]);
        // A seed wider than 32 bits: random.Random(2**40 + 7).random()
        assert_eq!(PyRandom::new((1 << 40) + 7).random(), 0.6137037779936511);
        assert!(PyRandom::new(1).choices_index(&[0.0, 0.0]).is_err());
    }
}
