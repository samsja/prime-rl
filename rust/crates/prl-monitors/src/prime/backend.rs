//! Run lifecycles (`prime_runs.backend`): create, attach, update, finalize. Training runs speak
//! `/rft/external-runs`, evaluations speak `/evaluations`.

use serde_json::{json, Map, Value as Json};

use super::client::{ApiError, PlatformClient};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RunKind {
    Train,
    Eval,
}

impl RunKind {
    pub fn as_str(self) -> &'static str {
        match self {
            RunKind::Train => "train",
            RunKind::Eval => "eval",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RunStatus {
    Completed,
    Failed,
    Cancelled,
    Crashed,
}

impl RunStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            RunStatus::Completed => "completed",
            RunStatus::Failed => "failed",
            RunStatus::Cancelled => "cancelled",
            RunStatus::Crashed => "crashed",
        }
    }
}

/// Display fields a training run is registered with (`TrainingSpec`); 0 max steps is unknown.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct TrainingSpec {
    pub max_steps: u64,
    pub batch_size: Option<u64>,
    pub rollouts_per_example: Option<u64>,
    pub seq_len: Option<u64>,
    pub wandb_project: Option<String>,
}

/// What `prime_runs.init` registers. Environments are hub references: an `owner/name` slug or
/// a bare name (passed through as ids for training runs).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct RunSpec {
    pub name: Option<String>,
    pub environments: Vec<String>,
    pub model: Option<String>,
    pub framework: Option<String>,
    pub team_id: Option<String>,
    pub config: Map<String, Json>,
    pub training: TrainingSpec,
}

pub struct RunHandle {
    pub id: String,
    pub url: String,
}

/// A request body without its unset (null) and empty-mapping fields.
fn body(value: Json) -> Json {
    let Json::Object(fields) = value else {
        unreachable!("bodies are objects")
    };
    Json::Object(
        fields
            .into_iter()
            .filter(|(_, v)| !v.is_null() && v.as_object().is_none_or(|m| !m.is_empty()))
            .collect(),
    )
}

fn id_of(value: Option<&Json>) -> Option<String> {
    match value {
        Some(Json::String(s)) if !s.is_empty() => Some(s.clone()),
        Some(Json::Number(n)) => Some(n.to_string()),
        _ => None,
    }
}

pub struct Backend {
    client: PlatformClient,
    kind: RunKind,
    frontend_url: String,
    team_id: Option<String>,
}

impl Backend {
    pub fn new(
        client: PlatformClient,
        kind: RunKind,
        frontend_url: String,
        team_id: Option<String>,
    ) -> Self {
        Backend {
            client,
            kind,
            frontend_url,
            team_id,
        }
    }

    fn url_for(&self, run_id: &str) -> String {
        let page = if self.kind == RunKind::Train {
            "training"
        } else {
            "evaluations"
        };
        format!("{}/dashboard/{page}/{run_id}", self.frontend_url)
    }

    /// A handle for a run a launcher already created; no request.
    pub fn attach(&self, run_id: &str) -> RunHandle {
        RunHandle {
            id: run_id.to_string(),
            url: self.url_for(run_id),
        }
    }

    pub async fn create(&self, spec: &RunSpec) -> Result<RunHandle, ApiError> {
        let team_id = spec.team_id.clone().or_else(|| self.team_id.clone());
        if self.kind == RunKind::Train {
            let model = spec.model.as_ref().ok_or_else(|| {
                ApiError::Configuration(
                    "A training run needs model= — the base model being trained.".into(),
                )
            })?;
            let team_id = team_id.ok_or_else(|| {
                ApiError::Configuration(
                    "A training run needs a team: pass team_id= or set PRIME_TEAM_ID. External training runs are \
                     created for a team, and the platform enables them per team."
                        .into(),
                )
            })?;
            let t = &spec.training;
            let payload = body(json!({
                "base_model": model,
                "max_steps": t.max_steps,
                "environments": spec.environments.iter().map(|env| json!({ "id": env })).collect::<Vec<_>>(),
                "team_id": team_id,
                "name": spec.name,
                "batch_size": t.batch_size,
                "rollouts_per_example": t.rollouts_per_example,
                "seq_len": t.seq_len,
                "run_config": spec.config,
                "wandb_project": t.wandb_project,
            }));
            // Not replayable: a retry after a lost response would register a second run.
            let response = self
                .client
                .post("/rft/external-runs", &payload, false)
                .await?;
            let run_id = id_of(response.get("run").and_then(|r| r.get("id"))).ok_or_else(|| {
                ApiError::Configuration(format!(
                    "POST /rft/external-runs returned no run id: {response}"
                ))
            })?;
            return Ok(self.attach(&run_id));
        }
        let mut environments = Vec::new();
        for env in &spec.environments {
            environments.push(json!({ "id": self.resolve_environment(env).await? }));
        }
        let Some(first) = spec.environments.first() else {
            return Err(ApiError::Configuration(
                "An eval run needs at least one environment. Pass environments=[\"my-env\"] to init().".into(),
            ));
        };
        let dataset = first
            .split_once('/')
            .map_or(first.as_str(), |(_, name)| name);
        let name = spec.name.clone().unwrap_or_else(|| {
            format!(
                "{dataset}-{}",
                &uuid::Uuid::new_v4().simple().to_string()[..8]
            )
        });
        let payload = body(json!({
            "name": name,
            "environments": environments,
            "tags": [],
            "model_name": spec.model,
            "dataset": dataset,
            "framework": spec.framework,
            "metadata": spec.config,
            "team_id": team_id,
        }));
        let response = self.client.post("/evaluations/", &payload, false).await?;
        let run_id = id_of(response.get("evaluation_id")).ok_or_else(|| {
            ApiError::Configuration(format!(
                "POST /evaluations/ returned no evaluation_id: {response}"
            ))
        })?;
        let url = response
            .get("viewer_url")
            .and_then(Json::as_str)
            .map_or_else(|| self.url_for(&run_id), str::to_string);
        Ok(RunHandle { id: run_id, url })
    }

    /// A hub reference as its environment id: a slug is looked up, a name goes through the
    /// hub's get-or-create.
    async fn resolve_environment(&self, env: &str) -> Result<String, ApiError> {
        let id = match env.split_once('/') {
            Some((owner, name)) => self
                .client
                .get(&format!("/environmentshub/{owner}/{name}/@latest"))
                .await
                .map(|response| {
                    let data = response
                        .get("data")
                        .filter(|d| !d.is_null())
                        .unwrap_or(&response);
                    id_of(data.get("id"))
                }),
            None => {
                let request = body(json!({ "name": env, "team_id": self.team_id }));
                let response = self
                    .client
                    .post("/environmentshub/resolve", &request, true)
                    .await;
                response.map(|response| id_of(response.get("data").and_then(|d| d.get("id"))))
            }
        };
        let id = id.map_err(|e| {
            ApiError::Configuration(format!("Could not resolve environment {env:?}: {e}"))
        })?;
        id.ok_or_else(|| {
            ApiError::Configuration(format!("Hub returned no id for environment {env:?}"))
        })
    }

    /// Persist the run's config and summary (evaluations only; the RFT API has neither).
    pub async fn update(
        &self,
        run_id: &str,
        config: &Map<String, Json>,
        summary: &Map<String, Json>,
    ) -> Result<(), ApiError> {
        let payload = body(json!({ "metadata": config, "metrics": summary }));
        if self.kind == RunKind::Eval && payload.as_object().is_some_and(|m| !m.is_empty()) {
            self.client
                .put(&format!("/evaluations/{run_id}"), &payload)
                .await?;
        }
        Ok(())
    }

    /// Close the run out, once per run.
    pub async fn finalize(
        &self,
        run_id: &str,
        status: RunStatus,
        summary: &Map<String, Json>,
        error: Option<&str>,
        config: &Map<String, Json>,
    ) -> Result<(), ApiError> {
        match (self.kind, status) {
            (RunKind::Train, RunStatus::Completed) => {
                // compare-and-set server-side: "already finalized" is a success
                let finalize = self
                    .client
                    .post(
                        "/rft/finalize",
                        &json!({ "run_id": run_id, "exit_code": 0 }),
                        true,
                    )
                    .await;
                if let Err(e) = finalize {
                    tracing::warn!("Run {run_id} could not be finalized ({e}); falling back to a status update");
                    self.set_terminal_status(run_id, status, None).await?;
                }
                Ok(())
            }
            (RunKind::Train, _) => {
                let message = error.unwrap_or(status.as_str());
                let message = if status == RunStatus::Failed {
                    message.to_string()
                } else {
                    format!("{}: {message}", status.as_str())
                };
                self.set_terminal_status(run_id, status, Some(&message))
                    .await
            }
            (RunKind::Eval, RunStatus::Completed) => {
                // Not replayable: finalization enqueues the platform's statistics task.
                self.client
                    .post(
                        &format!("/evaluations/{run_id}/finalize"),
                        &json!({ "metrics": summary }),
                        false,
                    )
                    .await?;
                Ok(())
            }
            (RunKind::Eval, _) => {
                // the status update below closes the run out even when this one is rejected
                if let Err(e) = self.update(run_id, config, summary).await {
                    tracing::warn!(
                        "Run {run_id}: config/summary update failed ({e}); still marking it {}",
                        status.as_str()
                    );
                }
                let message = match (status, error) {
                    (RunStatus::Crashed, Some(e)) => Some(format!("crashed: {e}")),
                    (RunStatus::Crashed, None) => Some("crashed".to_string()),
                    (_, e) => e.map(str::to_string),
                };
                self.set_terminal_status(run_id, status, message.as_deref())
                    .await
            }
        }
    }

    async fn set_terminal_status(
        &self,
        run_id: &str,
        status: RunStatus,
        message: Option<&str>,
    ) -> Result<(), ApiError> {
        let (path, payload) = match self.kind {
            RunKind::Train => (
                format!("/rft/external-runs/{run_id}/status"),
                body(json!({
                    "status": if status == RunStatus::Completed { "completed" } else { "failed" },
                    "error_message": message,
                })),
            ),
            RunKind::Eval => (
                format!("/evaluations/{run_id}"),
                body(json!({
                    "status": if status == RunStatus::Cancelled { "CANCELLED" } else { "FAILED" },
                    "error_message": message.filter(|m| !m.is_empty()).map(|m| m.chars().take(4096).collect::<String>()),
                })),
            ),
        };
        match self.client.put(&path, &payload).await {
            Err(ApiError::Http { status: 409, .. }) => {
                tracing::info!(
                    "Run {run_id} is already closed on the platform; not marking it {}",
                    status.as_str()
                );
                Ok(())
            }
            other => other.map(drop),
        }
    }
}
