//! `TrainingSample` and the per-sample wire references (`transports/batch/types.py`).

/// One image of a multimodal sample: its placeholder token span.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MMImageRef {
    pub url: String,
    pub offset: usize,
    pub length: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MMRefs {
    pub images: Vec<MMImageRef>,
}

/// A sample's training arrays kept by the env-server worker at `url`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PayloadSlice {
    pub url: String,
    pub id: String,
    pub num_tokens: usize,
    pub fields: Vec<String>,
    pub routed_shape: Option<Vec<i64>>,
    pub routed_dtype: Option<String>,
}

impl PayloadSlice {
    pub fn has(&self, field: &str) -> bool {
        self.fields.iter().any(|f| f == field)
    }

    pub fn url(&self) -> String {
        format!("{}/{}", self.url, self.id)
    }
}

/// One branch of a rollout as a flat token sequence. All per-token streams share the length of
/// `token_ids`. `None` streams are absent (rl weight 1.0 on the mask, no ce/ref_kl component).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct TrainingSample {
    pub token_ids: Vec<i32>,
    pub mask: Vec<bool>,
    pub logprobs: Vec<f32>,
    pub temperatures: Vec<f32>,
    pub env_name: String,
    pub ref_logprobs: Option<Vec<f32>>,
    pub mm_refs: Option<MMRefs>,
    pub mm_token_type_ids: Option<Vec<i32>>,
    pub rl_weights: Option<Vec<f32>>,
    pub ce_weights: Option<Vec<f32>>,
    pub ref_kl_weights: Option<Vec<f32>>,
    pub advantages: Option<Vec<f32>>,
    pub trace_id: Option<String>,
    pub branch_index: Option<i64>,
    pub staleness: Option<Vec<i16>>,
    pub payload: Option<PayloadSlice>,
}

impl TrainingSample {
    /// Whether an env server keeps this array of the sample.
    pub fn kept(&self, field: &str) -> bool {
        self.payload.as_ref().is_some_and(|p| p.has(field))
    }
}

/// `sample_payload_urls`: URLs of the kept arrays of the samples.
pub fn sample_payload_urls<'a>(samples: impl IntoIterator<Item = &'a TrainingSample>) -> Vec<String> {
    samples.into_iter().filter_map(|s| s.payload.as_ref().map(PayloadSlice::url)).collect()
}
