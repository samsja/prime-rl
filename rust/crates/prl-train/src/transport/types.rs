//! `MicroBatch` (orchestrator -> trainer) and its msgspec-compatible encoding.

use rmp::encode::{write_array_len, write_bin_len, write_nil, write_sint, write_str};

use crate::sample::{MMRefs, PayloadSlice};

/// A packed micro batch. Field order mirrors the Python `msgspec.Struct` (`array_like=True`, so
/// msgspec writes every field). `routed_experts`, `sampling_mask` and `top_logprobs` are filled by
/// the trainer from `payloads` and always travel as nil from the orchestrator.
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

impl MicroBatch {
    pub fn is_multimodal(&self) -> bool {
        self.mm_refs.is_some()
    }

    /// Whether the kept arrays of a packed sample include routing rows.
    pub fn has_routing(&self) -> bool {
        self.payloads.as_ref().is_some_and(|ps| ps.iter().flatten().any(|p| p.has("routed_experts")))
    }
}

// Writing into a Vec cannot fail.
type Buf = Vec<u8>;

/// A per-token stream travels as msgpack bin of its raw little-endian bytes.
fn stream<T: Copy>(buf: &mut Buf, values: &[T], to_le: impl Fn(T) -> Vec<u8>) {
    write_bin_len(buf, (values.len() * size_of::<T>()) as u32).unwrap();
    for &v in values {
        buf.extend(to_le(v));
    }
}

fn opt<T>(buf: &mut Buf, value: &Option<T>, write: impl FnOnce(&mut Buf, &T)) {
    match value {
        Some(v) => write(buf, v),
        None => write_nil(buf).unwrap(),
    }
}

fn f32s(buf: &mut Buf, v: &Vec<f32>) {
    stream(buf, v, |x| x.to_le_bytes().to_vec());
}

fn i32s(buf: &mut Buf, v: &Vec<i32>) {
    stream(buf, v, |x| x.to_le_bytes().to_vec());
}

fn ints<T: Copy + Into<i64>>(buf: &mut Buf, v: &[T]) {
    write_array_len(buf, v.len() as u32).unwrap();
    for &x in v {
        write_sint(buf, x.into()).unwrap();
    }
}

fn usizes(buf: &mut Buf, v: &[usize]) {
    write_array_len(buf, v.len() as u32).unwrap();
    for &x in v {
        write_sint(buf, x as i64).unwrap();
    }
}

fn strs(buf: &mut Buf, v: &Vec<String>) {
    write_array_len(buf, v.len() as u32).unwrap();
    for s in v {
        write_str(buf, s).unwrap();
    }
}

fn mm_refs(buf: &mut Buf, refs: &MMRefs) {
    write_array_len(buf, 1).unwrap();
    write_array_len(buf, refs.images.len() as u32).unwrap();
    for image in &refs.images {
        write_array_len(buf, 3).unwrap();
        write_str(buf, &image.url).unwrap();
        write_sint(buf, image.offset as i64).unwrap();
        write_sint(buf, image.length as i64).unwrap();
    }
}

fn payload_slice(buf: &mut Buf, p: &PayloadSlice) {
    write_array_len(buf, 6).unwrap();
    write_str(buf, &p.url).unwrap();
    write_str(buf, &p.id).unwrap();
    write_sint(buf, p.num_tokens as i64).unwrap();
    strs(buf, &p.fields);
    opt(buf, &p.routed_shape, |b, s| ints(b, s));
    opt(buf, &p.routed_dtype, |b, s| write_str(b, s).unwrap());
}

fn micro_batch(buf: &mut Buf, mb: &MicroBatch) {
    write_array_len(buf, 23).unwrap();
    i32s(buf, &mb.input_ids);
    stream(buf, &mb.loss_mask, |x| vec![x as u8]);
    f32s(buf, &mb.advantages);
    f32s(buf, &mb.inference_logprobs);
    i32s(buf, &mb.position_ids);
    usizes(buf, &mb.sequence_lengths);
    f32s(buf, &mb.temperatures);
    strs(buf, &mb.env_names);
    usizes(buf, &mb.seq_lens);
    opt(buf, &mb.ref_logprobs, f32s);
    write_nil(buf).unwrap(); // routed_experts
    opt(buf, &mb.mm_refs, mm_refs);
    opt(buf, &mb.mm_token_type_ids, i32s);
    opt(buf, &mb.rl_weights, f32s);
    opt(buf, &mb.ce_weights, f32s);
    opt(buf, &mb.ref_kl_weights, f32s);
    write_nil(buf).unwrap(); // sampling_mask
    opt(buf, &mb.trace_ids, strs);
    opt(buf, &mb.branch_indices, |b, v| ints(b, v));
    write_nil(buf).unwrap(); // top_logprobs
    write_sint(buf, mb.padding as i64).unwrap();
    opt(buf, &mb.staleness, |b, v| stream(b, v, |x| x.to_le_bytes().to_vec()));
    opt(buf, &mb.payloads, |b, ps| {
        write_array_len(b, ps.len() as u32).unwrap();
        for p in ps {
            opt(b, p, payload_slice);
        }
    });
}

/// One data rank's micro batches as the trainer decodes them (`list[MicroBatch]`).
pub fn encode_micro_batches(micro_batches: &[MicroBatch]) -> Vec<u8> {
    let tokens: usize = micro_batches.iter().map(|mb| mb.input_ids.len()).sum();
    let mut buf = Vec::with_capacity(tokens * 24 + 64);
    write_array_len(&mut buf, micro_batches.len() as u32).unwrap();
    for mb in micro_batches {
        micro_batch(&mut buf, mb);
    }
    buf
}
