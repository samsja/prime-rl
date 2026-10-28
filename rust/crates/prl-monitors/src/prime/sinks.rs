//! Sinks of the Prime platform uploads: Prime Traces (`prime_runs.sinks.traces` over
//! `prime_traces.TracesClient.upload_records`) and the RFT metrics rows (`sinks.metrics`).

use std::io::Write;
use std::time::Duration;

use async_trait::async_trait;
use flate2::write::GzEncoder;
use flate2::Compression;
use prl_wire::Value;

use crate::EpisodeRef;
use reqwest::multipart::{Form, Part};
use serde_json::{json, Value as Json};
use sha2::{Digest, Sha256};

use super::client::{error_for_response, retry_delay, ApiError, PlatformClient, USER_AGENT};
use super::worker::{send_each_result, Sink, SinkError};

const MIB: usize = 1024 * 1024;
/// Server-side cap on one JSONL line.
pub const MAX_LINE_BYTES: usize = 64 * MIB;
/// Server-side cap on staged rows per request.
pub const MAX_BATCH_LINES: usize = 1_000_000;
/// Chunk-close threshold, held under Cloud Run's 32 MiB request cap.
pub const DEFAULT_TARGET_BATCH_BYTES: usize = 30 * MIB;
pub const UPLOAD_MAX_ATTEMPTS: u32 = 5;

/// Python routes episodes to the legacy sample tables when Prime Traces is off for the
/// account; that fallback is not ported.
fn warn_no_legacy_samples() {
    tracing::warn!(
        "The legacy samples fallback is not available in the Rust orchestrator; episodes of this run are not uploaded"
    );
}

/// A copy of `record` keyed to this run: `run.id`/`run.type` on the envelope and every member trace.
pub fn stamp_run(record: &Value, run_id: &str, run_type: &str) -> Value {
    fn rekey(run: Option<&Value>, run_id: &str, run_type: &str) -> Value {
        let mut keyed = match run {
            Some(Value::Map(m)) => m.clone(),
            _ => Default::default(),
        };
        keyed.insert("id".into(), run_id.into());
        keyed.insert("type".into(), run_type.into());
        Value::Map(keyed)
    }
    let mut stamped = record.as_map().cloned().unwrap_or_default();
    let run = rekey(stamped.get("run"), run_id, run_type);
    stamped.insert("run".into(), run.clone());
    if let Some(Value::Array(members)) = stamped.get_mut("traces") {
        for member in members.iter_mut() {
            if let Value::Map(m) = member {
                let own = m
                    .get("run")
                    .filter(|r| r.truthy())
                    .cloned()
                    .unwrap_or_else(|| run.clone());
                m.insert("run".into(), rekey(Some(&own), run_id, run_type));
            }
        }
    }
    Value::Map(stamped)
}

/// One upload request's exact bytes and its content address.
#[derive(Debug, Clone, PartialEq)]
pub struct Batch {
    pub data: Vec<u8>,
    pub digest: String,
    pub num_lines: usize,
    pub first_line: usize,
}

impl Batch {
    fn seal(lines: &mut Vec<Vec<u8>>, first_line: usize) -> Batch {
        let data = lines.concat();
        lines.clear();
        let digest = hex::encode(Sha256::digest(&data));
        Batch {
            num_lines: 0,
            data,
            digest,
            first_line,
        }
    }
    pub fn idempotency_key(&self) -> String {
        format!("sha256:{}", self.digest)
    }
}

/// `iter_batches`: group JSONL lines into content-addressed batches, closing at
/// `target_bytes` (uncompressed) or `max_lines`; an oversized line errors once reached.
pub fn iter_batches(
    lines: Vec<Vec<u8>>,
    target_bytes: usize,
) -> impl Iterator<Item = Result<Batch, ApiError>> {
    let mut lines = lines.into_iter().enumerate();
    let mut buffer: Vec<Vec<u8>> = Vec::new();
    let mut size = 0usize;
    let mut start = 0usize;
    let mut failed = false;
    std::iter::from_fn(move || {
        if failed {
            return None;
        }
        for (index, line) in lines.by_ref() {
            let number = index + 1;
            if line.iter().all(u8::is_ascii_whitespace) {
                continue;
            }
            let content = line.len() - line.iter().rev().take_while(|b| **b == b'\n').count();
            if content > MAX_LINE_BYTES {
                failed = true;
                return Some(Err(ApiError::TraceTooLarge {
                    line: number,
                    size: content,
                    limit: MAX_LINE_BYTES,
                }));
            }
            let mut closed = None;
            if !buffer.is_empty()
                && (size + line.len() > target_bytes || buffer.len() >= MAX_BATCH_LINES)
            {
                let count = buffer.len();
                let mut batch = Batch::seal(&mut buffer, start);
                batch.num_lines = count;
                size = 0;
                closed = Some(batch);
            }
            if buffer.is_empty() {
                start = number;
            }
            size += line.len();
            buffer.push(line);
            if let Some(batch) = closed {
                return Some(Ok(batch));
            }
        }
        if buffer.is_empty() {
            return None;
        }
        let count = buffer.len();
        let mut batch = Batch::seal(&mut buffer, start);
        batch.num_lines = count;
        Some(Ok(batch))
    })
}

/// Streaming sink over Prime Traces: episodes go up as content-addressed JSONL batches.
pub struct TracesSink {
    http: reqwest::Client,
    url: String,
    run_id: String,
    run_type: &'static str,
    metadata: String,
    enabled: bool,
    /// Prime Traces is off for the account (or opted out): only then is the legacy fallback allowed.
    pub service_not_enabled: bool,
    pub receipts_received: usize,
    pub episodes_written: usize,
    target_batch_bytes: usize,
}

impl TracesSink {
    pub fn new(
        api_key: &str,
        team_id: Option<String>,
        traces_url: &str,
        run_id: &str,
        run_type: &'static str,
        context: &[(&str, String)],
    ) -> Result<Self, ApiError> {
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert(
            reqwest::header::AUTHORIZATION,
            format!("Bearer {api_key}")
                .parse()
                .map_err(|e| ApiError::Configuration(format!("invalid API key header: {e}")))?,
        );
        if let Some(team) = &team_id {
            headers.insert(
                "X-Prime-Team-ID",
                team.parse()
                    .map_err(|e| ApiError::Configuration(format!("invalid team id: {e}")))?,
            );
        }
        let http = reqwest::Client::builder()
            .default_headers(headers)
            .user_agent(USER_AGENT)
            .connect_timeout(Duration::from_secs(10))
            .read_timeout(Duration::from_secs(120))
            .build()
            .map_err(|e| ApiError::Configuration(format!("http client: {e}")))?;
        // json.dumps({"schema_version": 1, "context": {...}})
        let mut metadata = String::from("{\"schema_version\": 1");
        if !context.is_empty() {
            metadata.push_str(", \"context\": {");
            for (i, (key, value)) in context.iter().enumerate() {
                if i > 0 {
                    metadata.push_str(", ");
                }
                crate::pyjson::write_str(&mut metadata, key);
                metadata.push_str(": ");
                crate::pyjson::write_str(&mut metadata, value);
            }
            metadata.push('}');
        }
        metadata.push('}');
        Ok(TracesSink {
            http,
            url: format!("{}/api/v1/traces", traces_url.trim_end_matches('/')),
            run_id: run_id.to_string(),
            run_type,
            metadata,
            enabled: true,
            service_not_enabled: false,
            receipts_received: 0,
            episodes_written: 0,
            target_batch_bytes: DEFAULT_TARGET_BATCH_BYTES,
        })
    }

    async fn upload_batch(&self, batch: &Batch) -> Result<Json, ApiError> {
        let mut gz = GzEncoder::new(Vec::new(), Compression::new(9));
        gz.write_all(&batch.data).expect("in-memory gzip");
        let compressed = gz.finish().expect("in-memory gzip");
        let mut part_headers = reqwest::header::HeaderMap::new();
        part_headers.insert(
            reqwest::header::CONTENT_ENCODING,
            "gzip".parse().expect("static header"),
        );
        let form = Form::new()
            .part(
                "metadata",
                Part::bytes(self.metadata.clone().into_bytes())
                    .mime_str("application/json")
                    .expect("static mime"),
            )
            .part(
                "traces",
                Part::bytes(compressed)
                    .file_name("traces.jsonl")
                    .mime_str("application/x-ndjson")
                    .expect("static mime")
                    .headers(part_headers),
            );
        let response = self
            .http
            .post(&self.url)
            .header("Idempotency-Key", batch.idempotency_key())
            .header("X-Prime-Line-Format", "episode")
            .multipart(form)
            .send()
            .await
            .map_err(|e| ApiError::from_reqwest(&e, &format!("POST {}", self.url)))?;
        let status = response.status();
        let headers = response.headers().clone();
        let body = response
            .bytes()
            .await
            .map_err(|e| ApiError::from_reqwest(&e, "reading the upload receipt"))?;
        if let Some(e) = error_for_response(status, &headers, &body) {
            return Err(e);
        }
        match serde_json::from_slice::<Json>(&body) {
            Ok(receipt @ Json::Object(_)) => Ok(receipt),
            _ => Err(ApiError::Api {
                status: Some(status.as_u16()),
                message: "API response was not a dictionary".into(),
                code: None,
            }),
        }
    }

    async fn send_with_retry(&self, batch: &Batch) -> Result<Json, ApiError> {
        let mut attempt = 0;
        loop {
            match self.upload_batch(batch).await {
                Err(e) if e.is_transient() && attempt + 1 < UPLOAD_MAX_ATTEMPTS => {
                    tokio::time::sleep(retry_delay(&e, attempt)).await;
                    attempt += 1;
                }
                other => return other,
            }
        }
    }

    /// Leave the sink as an account-level denial would, without contacting Prime Traces.
    pub fn opt_out(&mut self) {
        self.service_not_enabled = true;
        self.retire_quietly("opted out of Prime Traces; uploading to legacy samples");
        warn_no_legacy_samples();
    }

    fn retire_quietly(&mut self, reason: &str) {
        if self.enabled {
            tracing::info!("Traces sink off: {reason}");
        }
        self.enabled = false;
    }

    async fn upload(&mut self, records: &[EpisodeRef]) -> Result<(), ApiError> {
        let episodes = records.to_vec();
        let (run_id, run_type) = (self.run_id.clone(), self.run_type);
        // Record serialization is heavy: off the reactor.
        let (has_traces, lines) = tokio::task::spawn_blocking(move || {
            let mut has_traces = Vec::with_capacity(episodes.len());
            let mut lines = Vec::with_capacity(episodes.len());
            for episode in &episodes {
                let stamped = stamp_run(
                    &episode.to_record(Some(crate::RECORD_FLOAT_DECIMALS)),
                    &run_id,
                    run_type,
                );
                has_traces.push(stamped.get("traces").is_some_and(Value::truthy));
                let mut line = prl_wire::json::to_vec(&stamped);
                line.push(b'\n');
                lines.push(line);
            }
            (has_traces, lines)
        })
        .await
        .expect("record serialization panicked");
        let mut offset = 0;
        for batch in iter_batches(lines, self.target_batch_bytes) {
            let batch = batch?;
            self.send_with_retry(&batch).await?;
            self.episodes_written += has_traces[offset..offset + batch.num_lines]
                .iter()
                .filter(|t| **t)
                .count();
            offset += batch.num_lines;
            self.receipts_received += 1;
        }
        Ok(())
    }
}

#[async_trait]
impl Sink<EpisodeRef> for TracesSink {
    fn name(&self) -> &'static str {
        "traces"
    }
    fn enabled(&self) -> bool {
        self.enabled
    }
    fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }
    fn episodes_written(&self) -> Option<usize> {
        Some(self.episodes_written)
    }

    async fn write(&mut self, records: &[EpisodeRef]) -> Result<(), SinkError> {
        if !self.enabled || records.is_empty() {
            return Ok(());
        }
        match self.upload(records).await {
            Ok(()) => Ok(()),
            Err(ApiError::Forbidden { message, code }) => {
                if code.as_deref() == Some("service_not_enabled") && self.receipts_received == 0 {
                    // Not enabled for the account: there was never anywhere for these records to go.
                    self.service_not_enabled = true;
                    self.retire_quietly(&format!(
                        "Prime Traces is not enabled for this account ({message}); continuing with legacy samples"
                    ));
                    warn_no_legacy_samples();
                    return Ok(());
                }
                if self.enabled {
                    tracing::warn!(
                        "Traces sink disabled: this credential cannot write traces ({message})"
                    );
                }
                self.enabled = false;
                Err(SinkError {
                    cause: ApiError::Forbidden { message, code },
                    failed_records: records.len(),
                })
            }
            Err(cause) => Err(SinkError {
                cause,
                failed_records: records.len(),
            }),
        }
    }
}

/// Training metrics: one `POST /rft/metrics` per logged row, replayed after an ambiguous
/// failure (the platform keeps one row per step).
pub struct RftMetricsSink {
    client: PlatformClient,
    run_id: String,
    enabled: bool,
    pub steps_written: usize,
}

impl RftMetricsSink {
    pub fn new(client: PlatformClient, run_id: &str) -> Self {
        RftMetricsSink {
            client,
            run_id: run_id.to_string(),
            enabled: true,
            steps_written: 0,
        }
    }
}

#[async_trait]
impl Sink<Json> for RftMetricsSink {
    fn name(&self) -> &'static str {
        "rft_metrics"
    }
    fn enabled(&self) -> bool {
        self.enabled
    }
    fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    async fn write(&mut self, records: &[Json]) -> Result<(), SinkError> {
        if !self.enabled || records.is_empty() {
            return Ok(());
        }
        let mut results = Vec::with_capacity(records.len());
        for record in records {
            let body = json!({ "run_id": self.run_id, "metrics": record });
            let result = self
                .client
                .post("/rft/metrics", &body, true)
                .await
                .map(drop);
            let fatal = matches!(&result, Err(e) if !(e.is_record_rejection() || e.is_transient()));
            if result.is_ok() {
                self.steps_written += 1;
            }
            results.push(result);
            if fatal {
                break;
            }
        }
        send_each_result(results, records.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn batches_close_at_the_target_and_hash_their_bytes() {
        let lines: Vec<Vec<u8>> = vec![
            b"aaaa\n".to_vec(),
            b"  \n".to_vec(),
            b"bbbb\n".to_vec(),
            b"cc\n".to_vec(),
        ];
        let batches: Vec<Batch> = iter_batches(lines, 8).map(Result::unwrap).collect();
        assert_eq!(batches.len(), 2);
        assert_eq!(batches[0].data, b"aaaa\n");
        assert_eq!((batches[0].num_lines, batches[0].first_line), (1, 1));
        assert_eq!(batches[1].data, b"bbbb\ncc\n");
        assert_eq!((batches[1].num_lines, batches[1].first_line), (2, 3));
        assert_eq!(batches[0].digest, hex::encode(Sha256::digest(b"aaaa\n")));
    }

    #[test]
    fn stamp_run_rekeys_envelope_and_members() {
        let record: Value = [
            ("id", Value::from("e")),
            (
                "run",
                [("id", Value::from("local")), ("name", Value::from("n"))]
                    .into_iter()
                    .collect(),
            ),
            (
                "traces",
                Value::Array(vec![[("id", Value::from("t"))].into_iter().collect()]),
            ),
        ]
        .into_iter()
        .collect();
        let stamped = stamp_run(&record, "R", "train");
        assert_eq!(
            prl_wire::json::to_vec(&stamped),
            br#"{"id":"e","run":{"id":"R","name":"n","type":"train"},"traces":[{"id":"t","run":{"id":"R","name":"n","type":"train"}}]}"#
        );
    }
}
