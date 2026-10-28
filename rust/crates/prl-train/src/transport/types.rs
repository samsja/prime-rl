//! `MicroBatch` (orchestrator -> trainer) and its msgspec-compatible encoding.

use super::msgpack::Writer;
use crate::sample::{MMRefs, PayloadSlice};

/// A packed micro batch. Field order and defaults mirror the Python `msgspec.Struct`
/// (`array_like=True, omit_defaults=True`). `routed_experts`, `sampling_mask` and `top_logprobs`
/// are filled by the trainer from `payloads` and always travel as nil from the orchestrator.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct MicroBatch {
    pub input_ids: Vec<i32>,
    pub loss_mask: Vec<bool>,
    pub advantages: Vec<f32>,
    pub inference_logprobs: Vec<f32>,
    pub position_ids: Vec<i32>,
    pub sequence_lengths: Vec<usize>,
    pub temperatures: Vec<f32>,
    pub env_names: Vec<String>,
    pub seq_lens: Vec<usize>,
    pub ref_logprobs: Option<Vec<f32>>,
    pub mm_refs: Option<MMRefs>,
    pub mm_token_type_ids: Option<Vec<i32>>,
    pub rl_weights: Option<Vec<f32>>,
    pub ce_weights: Option<Vec<f32>>,
    pub ref_kl_weights: Option<Vec<f32>>,
    pub trace_ids: Option<Vec<String>>,
    pub branch_indices: Option<Vec<i64>>,
    pub padding: usize,
    pub staleness: Option<Vec<i16>>,
    pub payloads: Option<Vec<Option<PayloadSlice>>>,
}

/// Number of struct fields (positions) in the Python `MicroBatch`.
const NUM_FIELDS: usize = 23;
/// Fields 0..=8 have no default and are always written.
const NUM_REQUIRED: usize = 9;

impl MicroBatch {
    pub fn is_multimodal(&self) -> bool {
        self.mm_refs.is_some()
    }

    /// Whether the sample's kept arrays include routing rows.
    pub fn has_routing(&self) -> bool {
        self.payloads
            .as_ref()
            .is_some_and(|ps| ps.iter().flatten().any(|p| p.has("routed_experts")))
    }

    fn is_default(&self, field: usize) -> bool {
        match field {
            9 => self.ref_logprobs.is_none(),
            10 | 16 | 19 => true, // routed_experts, sampling_mask, top_logprobs
            11 => self.mm_refs.is_none(),
            12 => self.mm_token_type_ids.is_none(),
            13 => self.rl_weights.is_none(),
            14 => self.ce_weights.is_none(),
            15 => self.ref_kl_weights.is_none(),
            17 => self.trace_ids.is_none(),
            18 => self.branch_indices.is_none(),
            20 => self.padding == 0,
            21 => self.staleness.is_none(),
            22 => self.payloads.is_none(),
            _ => false,
        }
    }

    pub fn encode_into(&self, w: &mut Writer) {
        let mut n = NUM_FIELDS;
        while n > NUM_REQUIRED && self.is_default(n - 1) {
            n -= 1;
        }
        w.array_header(n);
        for field in 0..n {
            self.encode_field(field, w);
        }
    }

    fn encode_field(&self, field: usize, w: &mut Writer) {
        fn opt_stream<T: super::msgpack::RawBytes>(w: &mut Writer, v: &Option<Vec<T>>) {
            match v {
                Some(v) => w.stream(v),
                None => w.nil(),
            }
        }
        fn ints(w: &mut Writer, v: &[usize]) {
            w.array_header(v.len());
            for &x in v {
                w.int(x as i64);
            }
        }
        fn strs(w: &mut Writer, v: &[String]) {
            w.array_header(v.len());
            for s in v {
                w.str(s);
            }
        }
        match field {
            0 => w.stream(&self.input_ids),
            1 => w.stream(&self.loss_mask),
            2 => w.stream(&self.advantages),
            3 => w.stream(&self.inference_logprobs),
            4 => w.stream(&self.position_ids),
            5 => ints(w, &self.sequence_lengths),
            6 => w.stream(&self.temperatures),
            7 => strs(w, &self.env_names),
            8 => ints(w, &self.seq_lens),
            9 => opt_stream(w, &self.ref_logprobs),
            10 | 16 | 19 => w.nil(),
            11 => match &self.mm_refs {
                Some(refs) => encode_mm_refs(refs, w),
                None => w.nil(),
            },
            12 => opt_stream(w, &self.mm_token_type_ids),
            13 => opt_stream(w, &self.rl_weights),
            14 => opt_stream(w, &self.ce_weights),
            15 => opt_stream(w, &self.ref_kl_weights),
            17 => match &self.trace_ids {
                Some(v) => strs(w, v),
                None => w.nil(),
            },
            18 => match &self.branch_indices {
                Some(v) => {
                    w.array_header(v.len());
                    for &x in v {
                        w.int(x);
                    }
                }
                None => w.nil(),
            },
            20 => w.int(self.padding as i64),
            21 => opt_stream(w, &self.staleness),
            22 => match &self.payloads {
                Some(ps) => {
                    w.array_header(ps.len());
                    for p in ps {
                        match p {
                            Some(p) => encode_payload_slice(p, w),
                            None => w.nil(),
                        }
                    }
                }
                None => w.nil(),
            },
            _ => unreachable!("MicroBatch has {NUM_FIELDS} fields"),
        }
    }
}

fn encode_mm_refs(refs: &MMRefs, w: &mut Writer) {
    w.array_header(1);
    w.array_header(refs.images.len());
    for image in &refs.images {
        w.array_header(3);
        w.str(&image.url);
        w.int(image.offset as i64);
        w.int(image.length as i64);
    }
}

fn encode_payload_slice(p: &PayloadSlice, w: &mut Writer) {
    w.array_header(6);
    w.str(&p.url);
    w.str(&p.id);
    w.int(p.num_tokens as i64);
    w.array_header(p.fields.len());
    for f in &p.fields {
        w.str(f);
    }
    match &p.routed_shape {
        Some(shape) => {
            w.array_header(shape.len());
            for &x in shape {
                w.int(x);
            }
        }
        None => w.nil(),
    }
    match &p.routed_dtype {
        Some(d) => w.str(d),
        None => w.nil(),
    }
}

/// One data rank's micro batches as the trainer decodes them (`list[MicroBatch]`).
pub fn encode_micro_batches(micro_batches: &[MicroBatch]) -> Vec<u8> {
    let tokens: usize = micro_batches.iter().map(|mb| mb.input_ids.len()).sum();
    let mut w = Writer::with_capacity(tokens * 24 + 64);
    w.array_header(micro_batches.len());
    for mb in micro_batches {
        mb.encode_into(&mut w);
    }
    w.buf
}
