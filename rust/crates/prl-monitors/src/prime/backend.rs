//! Run lifecycles (`prime_runs.backend`): create, attach, update, finalize. `Rft` speaks
//! `/rft/external-runs`, `Evals` speaks `/evaluations`.

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

/// Display fields a training run is registered with (`TrainingSpec`).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct TrainingSpec {
    pub max_steps: u64,
    pub batch_size: Option<u64>,
    pub rollouts_per_example: Option<u64>,
    pub seq_len: Option<u64>,
    pub wandb_project: Option<String>,
}

/// What `prime_runs.init` registers (`RunSpec`). Environments are hub references: a
/// `owner/name` slug or a bare name (or id for training runs).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct RunSpec {
    pub name: Option<String>,
    pub environments: Vec<String>,
    pub model: Option<String>,
    pub framework: Option<String>,
    pub team_id: Option<String>,
    pub config: Map<String, Json>,
    pub training: Option<TrainingSpec>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RunHandle {
    pub id: String,
    pub name: Option<String>,
    pub url: Option<String>,
}

pub struct Backend {
    client: PlatformClient,
    kind: RunKind,
    frontend_url: String,
    team_id: Option<String>,
}

fn set_if(payload: &mut Map<String, Json>, key: &str, value: Option<impl Into<Json>>) {
    if let Some(value) = value {
        payload.insert(key.into(), value.into());
    }
}

fn nonempty(map: &Map<String, Json>) -> Option<Json> {
    (!map.is_empty()).then(|| Json::Object(map.clone()))
}

impl Backend {
    pub fn new(
        client: PlatformClient,
        kind: RunKind,
        frontend_url: &str,
        team_id: Option<String>,
    ) -> Self {
        Backend {
            client,
            kind,
            frontend_url: frontend_url.trim_end_matches('/').to_string(),
            team_id,
        }
    }

    pub fn url_for(&self, run_id: &str) -> String {
        match self.kind {
            RunKind::Train => format!("{}/dashboard/training/{run_id}", self.frontend_url),
            RunKind::Eval => format!("{}/dashboard/evaluations/{run_id}", self.frontend_url),
        }
    }

    /// A handle for a run a launcher already created; no request.
    pub fn attach(&self, run_id: &str) -> RunHandle {
        RunHandle {
            id: run_id.to_string(),
            name: None,
            url: Some(self.url_for(run_id)),
        }
    }

    pub async fn create(&self, spec: &RunSpec) -> Result<RunHandle, ApiError> {
        match self.kind {
            RunKind::Train => self.create_train(spec).await,
            RunKind::Eval => self.create_eval(spec).await,
        }
    }

    async fn create_train(&self, spec: &RunSpec) -> Result<RunHandle, ApiError> {
        let model = spec.model.as_ref().ok_or_else(|| {
            ApiError::Configuration(
                "A training run needs model= — the base model being trained.".into(),
            )
        })?;
        let team_id = spec.team_id.clone().or_else(|| self.team_id.clone()).ok_or_else(|| {
            ApiError::Configuration(
                "A training run needs a team: pass team_id= or set PRIME_TEAM_ID. External training runs are created \
                 for a team, and the platform enables them per team."
                    .into(),
            )
        })?;
        let training = spec.training.clone().unwrap_or_default();
        let mut payload = Map::new();
        payload.insert("base_model".into(), model.clone().into());
        payload.insert("max_steps".into(), training.max_steps.into());
        let environments: Vec<Json> = spec
            .environments
            .iter()
            .map(|env| json!({ "id": env }))
            .collect();
        payload.insert("environments".into(), environments.into());
        payload.insert("team_id".into(), team_id.into());
        set_if(&mut payload, "name", spec.name.clone());
        set_if(&mut payload, "batch_size", training.batch_size);
        set_if(
            &mut payload,
            "rollouts_per_example",
            training.rollouts_per_example,
        );
        set_if(&mut payload, "seq_len", training.seq_len);
        set_if(&mut payload, "run_config", nonempty(&spec.config));
        set_if(&mut payload, "wandb_project", training.wandb_project);
        // Not replayable: a retry after a lost response would register a second run.
        let response = self
            .client
            .post("/rft/external-runs", &Json::Object(payload), false)
            .await?;
        let run = response.get("run").cloned().unwrap_or(Json::Null);
        let run_id = match run.get("id") {
            Some(Json::String(s)) if !s.is_empty() => s.clone(),
            Some(Json::Number(n)) => n.to_string(),
            _ => {
                let keys: Vec<&String> = response
                    .as_object()
                    .map(|m| m.keys().collect())
                    .unwrap_or_default();
                return Err(ApiError::Api {
                    status: None,
                    message: format!("POST /rft/external-runs returned no run id (keys: {keys:?})"),
                    code: None,
                });
            }
        };
        let name = run
            .get("name")
            .and_then(Json::as_str)
            .map(str::to_string)
            .or_else(|| spec.name.clone())
            .filter(|n| !n.is_empty());
        Ok(RunHandle {
            url: Some(self.url_for(&run_id)),
            id: run_id,
            name,
        })
    }

    async fn create_eval(&self, spec: &RunSpec) -> Result<RunHandle, ApiError> {
        let mut environments = Vec::new();
        for env in &spec.environments {
            environments.push(json!({ "id": self.lookup_environment(env).await? }));
        }
        if environments.is_empty() {
            return Err(ApiError::Configuration(
                "An eval run needs at least one environment. Pass environments=[\"my-env\"] to init().".into(),
            ));
        }
        let first_name = spec
            .environments
            .first()
            .map(|e| e.split_once('/').map_or(e.as_str(), |(_, n)| n).to_string());
        let run_name = spec.name.clone().unwrap_or_else(|| {
            let stem = first_name
                .clone()
                .or_else(|| spec.framework.clone())
                .unwrap_or_else(|| "eval".into());
            format!("{stem}-{}", &uuid::Uuid::new_v4().simple().to_string()[..8])
        });
        let mut payload = Map::new();
        payload.insert("name".into(), run_name.clone().into());
        payload.insert("environments".into(), environments.into());
        payload.insert("tags".into(), Json::Array(vec![]));
        set_if(&mut payload, "model_name", spec.model.clone());
        set_if(&mut payload, "dataset", first_name);
        set_if(&mut payload, "framework", spec.framework.clone());
        set_if(&mut payload, "metadata", nonempty(&spec.config));
        set_if(
            &mut payload,
            "team_id",
            spec.team_id.clone().or_else(|| self.team_id.clone()),
        );
        let response = self
            .client
            .post("/evaluations/", &Json::Object(payload), false)
            .await?;
        let run_id = response
            .get("evaluation_id")
            .and_then(Json::as_str)
            .filter(|s| !s.is_empty())
            .ok_or_else(|| ApiError::Api {
                status: None,
                message: "POST /evaluations/ returned no evaluation_id".into(),
                code: None,
            })?;
        let url = response
            .get("viewer_url")
            .and_then(Json::as_str)
            .map_or_else(|| self.url_for(run_id), str::to_string);
        let name = response
            .get("name")
            .and_then(Json::as_str)
            .map_or(run_name, str::to_string);
        Ok(RunHandle {
            id: run_id.to_string(),
            name: Some(name),
            url: Some(url),
        })
    }

    async fn lookup_environment(&self, env: &str) -> Result<String, ApiError> {
        let id = if let Some((owner, name)) = env.split_once('/') {
            let response = self
                .client
                .get(&format!("/environmentshub/{owner}/{name}/@latest"))
                .await
                .map_err(|e| {
                    ApiError::Configuration(format!("Could not resolve environment {env:?}: {e}"))
                })?;
            let data = response
                .get("data")
                .filter(|d| !d.is_null())
                .unwrap_or(&response);
            data.get("id").cloned()
        } else {
            let mut body = Map::new();
            body.insert("name".into(), env.into());
            set_if(&mut body, "team_id", self.team_id.clone());
            // Get-or-create: a replay returns the same environment.
            let response = self
                .client
                .post("/environmentshub/resolve", &Json::Object(body), true)
                .await
                .map_err(|e| {
                    ApiError::Configuration(format!("Could not resolve environment {env:?}: {e}"))
                })?;
            response.get("data").and_then(|d| d.get("id")).cloned()
        };
        match id {
            Some(Json::String(s)) if !s.is_empty() => Ok(s),
            Some(Json::Number(n)) => Ok(n.to_string()),
            _ => Err(ApiError::Configuration(format!(
                "Hub returned no id for environment {env:?}"
            ))),
        }
    }

    /// Persist the run's config and/or summary (evaluations only; the RFT API has none).
    pub async fn update(
        &self,
        run_id: &str,
        config: Option<&Map<String, Json>>,
        summary: Option<&Map<String, Json>>,
    ) -> Result<(), ApiError> {
        if self.kind == RunKind::Train {
            return Ok(());
        }
        let mut payload = Map::new();
        set_if(&mut payload, "metadata", config.and_then(nonempty));
        set_if(&mut payload, "metrics", summary.and_then(nonempty));
        if !payload.is_empty() {
            self.client
                .put(&format!("/evaluations/{run_id}"), &Json::Object(payload))
                .await?;
        }
        Ok(())
    }

    /// Close the run out, once per run.
    pub async fn finalize(
        &self,
        run_id: &str,
        status: RunStatus,
        summary: Option<&Map<String, Json>>,
        error: Option<&str>,
        config: Option<&Map<String, Json>>,
    ) -> Result<(), ApiError> {
        match self.kind {
            RunKind::Train => {
                if status == RunStatus::Completed {
                    let body = json!({ "run_id": run_id, "exit_code": 0 });
                    if let Err(e) = self.client.post("/rft/finalize", &body, true).await {
                        if !e.is_api_error() {
                            return Err(e);
                        }
                        tracing::warn!("Run {run_id} could not be finalized ({e}); falling back to a status update");
                        self.set_terminal_status(run_id, status, None).await?;
                    }
                    return Ok(());
                }
                let mut message = error.map_or_else(|| status.as_str().to_string(), str::to_string);
                if status != RunStatus::Failed {
                    message = format!("{}: {message}", status.as_str());
                }
                self.set_terminal_status(run_id, status, Some(&message))
                    .await
            }
            RunKind::Eval => {
                if status == RunStatus::Completed {
                    let mut body = Map::new();
                    body.insert(
                        "metrics".into(),
                        summary.and_then(nonempty).unwrap_or_else(|| json!({})),
                    );
                    // Not replayable: finalization enqueues the platform's statistics task.
                    self.client
                        .post(
                            &format!("/evaluations/{run_id}/finalize"),
                            &Json::Object(body),
                            false,
                        )
                        .await?;
                    return Ok(());
                }
                if let Err(e) = self.update(run_id, config, summary).await {
                    if !e.is_api_error() {
                        return Err(e);
                    }
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
        error_message: Option<&str>,
    ) -> Result<(), ApiError> {
        let (path, wire_status, message) = match self.kind {
            RunKind::Train => (
                format!("/rft/external-runs/{run_id}/status"),
                if status == RunStatus::Completed {
                    "completed"
                } else {
                    "failed"
                },
                error_message.map(str::to_string),
            ),
            RunKind::Eval => (
                format!("/evaluations/{run_id}"),
                if status == RunStatus::Cancelled {
                    "CANCELLED"
                } else {
                    "FAILED"
                },
                error_message
                    .filter(|m| !m.is_empty())
                    .map(|m| m.chars().take(4096).collect()),
            ),
        };
        let mut payload = Map::new();
        payload.insert("status".into(), wire_status.into());
        set_if(&mut payload, "error_message", message);
        match self.client.put(&path, &Json::Object(payload)).await {
            Err(ApiError::Conflict { .. }) => {
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
