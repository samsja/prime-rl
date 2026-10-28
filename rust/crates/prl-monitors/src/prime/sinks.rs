//! The Prime platform upload sinks: Prime Traces (`prime_runs.sinks.traces` over
//! `prime_traces.TracesClient.upload_records`) and the RFT metrics rows (`sinks.metrics`).

use std::io::Write;
use std::time::Duration;

use async_trait::async_trait;
use flate2::write::GzEncoder;
use flate2::Compression;
use prl_wire::Value;
use reqwest::header::HeaderMap;
use reqwest::multipart::{Form, Part};
use serde_json::{json, Value as Json};
use sha2::{Digest, Sha256};

use super::client::{header, http_client, retry_delay, ApiError, PlatformClient};
use super::worker::{Sink, SinkError};
use crate::EpisodeRef;

/// Server-side cap on one JSONL line (one episode).
const MAX_LINE_BYTES: usize = 64 << 20;
/// Batch-close threshold in uncompressed bytes, held under Cloud Run's 32 MiB request cap.
const TARGET_BATCH_BYTES: usize = 30 << 20;
const UPLOAD_ATTEMPTS: u32 = 5;
/// Precision of per-token float streams in uploaded records (`RECORD_FLOAT_DECIMALS`).
const RECORD_FLOAT_DECIMALS: u32 = 4;

/// A copy of `record` keyed to this run: `run.id`/`run.type` on the envelope and on every
/// member trace (the traces service reads the run off each trace).
pub fn stamp_run(record: &Value, run_id: &str, run_type: &str) -> Value {
    let rekey = |run: Option<&Value>| {
        let mut keyed = run.and_then(Value::as_map).cloned().unwrap_or_default();
        keyed.insert("id".into(), run_id.into());
        keyed.insert("type".into(), run_type.into());
        Value::Map(keyed)
    };
    let mut stamped = record.as_map().cloned().unwrap_or_default();
    let run = rekey(stamped.get("run"));
    if let Some(Value::Array(traces)) = stamped.get_mut("traces") {
        for trace in traces.iter_mut().filter_map(Value::as_map_mut) {
            let own = trace.get("run").filter(|r| r.truthy()).unwrap_or(&run);
            let keyed = rekey(Some(own));
            trace.insert("run".into(), keyed);
        }
    }
    stamped.insert("run".into(), run);
    Value::Map(stamped)
}

/// Python routes episodes to the legacy sample tables when Prime Traces is off for the
/// account; that fallback is not ported.
fn warn_no_legacy_samples() {
    tracing::warn!("The legacy samples fallback is not available in the Rust orchestrator; episodes of this run are not uploaded");
}

/// Episodes go up as content-addressed JSONL batches: the idempotency key is the SHA-256 of
/// the exact uncompressed bytes, so a retry after a lost response replays its receipt.
pub struct TracesSink {
    http: reqwest::Client,
    url: String,
    run_id: String,
    run_type: &'static str,
    metadata: String,
    enabled: bool,
    receipts: usize,
    episodes_written: usize,
}

impl TracesSink {
    pub fn new(
        api_key: &str,
        team_id: Option<&str>,
        traces_url: &str,
        run_id: &str,
        run_type: &'static str,
        context: &[(&str, String)],
    ) -> Result<Self, ApiError> {
        let mut headers = HeaderMap::new();
        headers.insert(
            reqwest::header::AUTHORIZATION,
            header(&format!("Bearer {api_key}"))?,
        );
        if let Some(team) = team_id {
            headers.insert("X-Prime-Team-ID", header(team)?);
        }
        let context: Value = context
            .iter()
            .map(|(k, v)| (*k, Value::from(v.as_str())))
            .collect();
        let metadata: Value = [("schema_version", Value::Int(1)), ("context", context)]
            .into_iter()
            .collect();
        Ok(TracesSink {
            http: http_client(headers, Duration::from_secs(120))?,
            url: format!("{traces_url}/api/v1/traces"),
            run_id: run_id.to_string(),
            run_type,
            metadata: prl_wire::json::to_python_json(&metadata, false),
            enabled: true,
            receipts: 0,
            episodes_written: 0,
        })
    }

    /// Leave the sink as an account-level denial would, without contacting Prime Traces.
    pub fn opt_out(&mut self) {
        tracing::info!("Traces sink off: opted out of Prime Traces; uploading to legacy samples");
        self.enabled = false;
        warn_no_legacy_samples();
    }

    async fn post(&self, data: &[u8]) -> Result<(), ApiError> {
        let mut gz = GzEncoder::new(Vec::new(), Compression::new(9));
        gz.write_all(data).expect("in-memory gzip");
        let traces = Part::bytes(gz.finish().expect("in-memory gzip"))
            .file_name("traces.jsonl")
            .mime_str("application/x-ndjson")
            .expect("static mime")
            .headers(
                [(reqwest::header::CONTENT_ENCODING, header("gzip")?)]
                    .into_iter()
                    .collect(),
            );
        let metadata = Part::bytes(self.metadata.clone().into_bytes())
            .mime_str("application/json")
            .expect("static mime");
        let response = self
            .http
            .post(&self.url)
            .header(
                "Idempotency-Key",
                format!("sha256:{}", hex::encode(Sha256::digest(data))),
            )
            .header("X-Prime-Line-Format", "episode")
            .multipart(
                Form::new()
                    .part("metadata", metadata)
                    .part("traces", traces),
            )
            .send()
            .await
            .map_err(|e| ApiError::transport(&e, "POST /traces"))?;
        let (status, headers) = (response.status(), response.headers().clone());
        let body = response
            .bytes()
            .await
            .map_err(|e| ApiError::transport(&e, "POST /traces"))?;
        ApiError::from_response(status, &headers, &body).map_or(Ok(()), Err)
    }

    /// Upload one batch, retrying transient failures with the same bytes.
    async fn send(&mut self, data: &[u8], with_traces: usize) -> Result<(), ApiError> {
        let mut attempt = 0;
        while let Err(e) = self.post(data).await {
            attempt += 1;
            if !e.is_transient() || attempt == UPLOAD_ATTEMPTS {
                return Err(e);
            }
            tokio::time::sleep(retry_delay(&e, attempt - 1)).await;
        }
        // empty episodes produce no samples on the platform
        self.episodes_written += with_traces;
        self.receipts += 1;
        Ok(())
    }

    async fn upload(&mut self, episodes: &[EpisodeRef]) -> Result<(), ApiError> {
        let episodes = episodes.to_vec();
        let (run_id, run_type) = (self.run_id.clone(), self.run_type);
        // record serialization is heavy: off the reactor
        let lines = tokio::task::spawn_blocking(move || {
            episodes
                .iter()
                .map(|episode| {
                    let stamped = stamp_run(
                        &episode.to_record(Some(RECORD_FLOAT_DECIMALS)),
                        &run_id,
                        run_type,
                    );
                    let mut line = prl_wire::json::to_vec(&stamped);
                    line.push(b'\n');
                    (line, stamped.get("traces").is_some_and(Value::truthy))
                })
                .collect::<Vec<_>>()
        })
        .await
        .expect("record serialization panicked");
        let (mut batch, mut with_traces) = (Vec::new(), 0);
        for (number, (line, has_traces)) in lines.into_iter().enumerate() {
            let size = line.len() - 1;
            if size > MAX_LINE_BYTES {
                return Err(ApiError::TraceTooLarge {
                    line: number + 1,
                    size,
                    limit: MAX_LINE_BYTES,
                });
            }
            if !batch.is_empty() && batch.len() + line.len() > TARGET_BATCH_BYTES {
                self.send(&batch, with_traces).await?;
                (batch, with_traces) = (Vec::new(), 0);
            }
            batch.extend(line);
            with_traces += usize::from(has_traces);
        }
        if !batch.is_empty() {
            self.send(&batch, with_traces).await?;
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

    async fn write(&mut self, episodes: &[EpisodeRef]) -> Result<(), SinkError> {
        if !self.enabled || episodes.is_empty() {
            return Ok(());
        }
        let Err(cause) = self.upload(episodes).await else {
            return Ok(());
        };
        if let ApiError::Http {
            status: 403,
            code,
            message,
            ..
        } = &cause
        {
            // nothing at runtime fixes a 403: the sink retires either way
            self.enabled = false;
            if code.as_deref() == Some("service_not_enabled") && self.receipts == 0 {
                // there was never anywhere for these records to go
                tracing::info!("Traces sink off: Prime Traces is not enabled for this account ({message}); continuing with legacy samples");
                warn_no_legacy_samples();
                return Ok(());
            }
            tracing::warn!("Traces sink disabled: this credential cannot write traces ({message})");
        }
        Err(SinkError {
            cause,
            failed_records: episodes.len(),
        })
    }
}

/// Training metrics: one `POST /rft/metrics` per row, replayed after an ambiguous failure
/// (the platform keeps one row per step).
pub struct RftMetricsSink {
    client: PlatformClient,
    run_id: String,
    enabled: bool,
}

impl RftMetricsSink {
    pub fn new(client: PlatformClient, run_id: &str) -> Self {
        RftMetricsSink {
            client,
            run_id: run_id.to_string(),
            enabled: true,
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

    /// Rows go one at a time (`send_each`): a rejected or transient row costs only itself, a
    /// sink-wide failure stops the batch and costs the rest too. A transient error is the one
    /// reported, so the worker's strike count is not fooled by a row-local one.
    async fn write(&mut self, rows: &[Json]) -> Result<(), SinkError> {
        if !self.enabled {
            return Ok(());
        }
        let mut failed = 0;
        let mut reported: Option<ApiError> = None;
        for (position, row) in rows.iter().enumerate() {
            let body = json!({ "run_id": self.run_id, "metrics": row });
            let Err(e) = self.client.post("/rft/metrics", &body, true).await else {
                continue;
            };
            failed += 1;
            if !(e.is_record_rejection() || e.is_transient()) {
                failed += rows.len() - position - 1;
                reported = Some(e);
                break;
            }
            if reported.is_none() || e.is_transient() {
                reported = Some(e);
            }
        }
        reported.map_or(Ok(()), |cause| {
            Err(SinkError {
                cause,
                failed_records: failed,
            })
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stamp_run_rekeys_envelope_and_members() {
        let run: Value = [("id", Value::from("local")), ("name", Value::from("n"))]
            .into_iter()
            .collect();
        let trace: Value = [("id", Value::from("t"))].into_iter().collect();
        let record: Value = [
            ("id", Value::from("e")),
            ("run", run),
            ("traces", Value::Array(vec![trace])),
        ]
        .into_iter()
        .collect();
        assert_eq!(
            String::from_utf8(prl_wire::json::to_vec(&stamp_run(&record, "R", "train"))).unwrap(),
            r#"{"id":"e","run":{"id":"R","name":"n","type":"train"},"traces":[{"id":"t","run":{"id":"R","name":"n","type":"train"}}]}"#
        );
    }
}
