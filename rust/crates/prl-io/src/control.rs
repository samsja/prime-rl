//! Run control (`control/`): a pause submitted over HTTP is queued and applied by the step loop at
//! a step boundary.
//!
//! The controller state is shared between the HTTP handlers and the orchestrator (the Python
//! version guards it with a lock across the API thread and the event loop); `submitted()` wakes the
//! orchestrator, replacing `on_submit`.

use std::collections::{HashMap, HashSet};
use std::net::SocketAddr;
use std::sync::{Arc, Mutex};

use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};
use tokio::sync::Notify;
use tokio_util::sync::CancellationToken;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ActionType {
    Pause,
}

impl ActionType {
    pub fn as_str(self) -> &'static str {
        match self {
            ActionType::Pause => "pause",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ActionState {
    Pending,
    Accepted,
    Committed,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Action {
    pub id: String,
    #[serde(rename = "type")]
    pub type_: ActionType,
    pub state: ActionState,
    pub step: Option<u64>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ActionRequest {
    #[serde(rename = "type")]
    pub type_: ActionType,
    #[serde(default)]
    pub step: Option<u64>,
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum ControlError {
    #[error("{0}")]
    NotFound(String),
    #[error("{0}")]
    Conflict(String),
    #[error("{0}")]
    Invalid(String),
}

#[derive(Debug, Default)]
struct ControllerState {
    id_to_action: HashMap<String, Action>,
    pending: Option<String>,
    saved_steps: HashSet<u64>,
    step: Option<u64>,
    stop_step: Option<u64>,
}

/// Cheap-to-clone handle; all clones share one controller.
#[derive(Debug, Clone)]
pub struct Controller {
    requires_step: bool,
    state: Arc<Mutex<ControllerState>>,
    submitted: Arc<Notify>,
}

impl Controller {
    /// `requires_step`: whether a pause names its step (orchestrator) or applies at the next one (trainer).
    pub fn new(requires_step: bool) -> Self {
        Self { requires_step, state: Arc::default(), submitted: Arc::new(Notify::new()) }
    }

    /// Notified after every accepted submission.
    pub fn submitted(&self) -> Arc<Notify> {
        self.submitted.clone()
    }

    pub fn get(&self, action_id: &str) -> Result<Action, ControlError> {
        let mut state = self.state.lock().expect("controller lock");
        let saved = state.saved_steps.clone();
        let action = state.id_to_action.get_mut(action_id).ok_or_else(|| {
            ControlError::NotFound(format!(
                "Unknown action '{action_id}': the run may have restarted since it was submitted; submit it again."
            ))
        })?;
        if action.state == ActionState::Accepted && action.step.is_some_and(|s| saved.contains(&s)) {
            action.state = ActionState::Committed;
        }
        Ok(action.clone())
    }

    /// Record that this process saved `step`, the step its pause applied to.
    pub fn mark_saved(&self, step: u64) {
        self.state.lock().expect("controller lock").saved_steps.insert(step);
    }

    pub fn submit(&self, action_type: ActionType, step: Option<u64>) -> Result<Action, ControlError> {
        if self.requires_step && step.is_none() {
            return Err(ControlError::Invalid(
                "The orchestrator pauses at a step you name: pass the trainer's accepted 'step'.".into(),
            ));
        }
        if !self.requires_step && step.is_some() {
            return Err(ControlError::Invalid("The trainer pauses at its next step and takes no 'step'.".into()));
        }
        let action = {
            let mut state = self.state.lock().expect("controller lock");
            if let Some(stop_step) = state.stop_step {
                return Err(ControlError::Conflict(format!(
                    "The run paused at step {stop_step} and takes no more actions; resume it to submit new ones."
                )));
            }
            if state.pending.is_some() {
                return Err(ControlError::Conflict("A pause is already queued and the run takes no more actions.".into()));
            }
            if let (Some(step), Some(current)) = (step, state.step) {
                if step < current {
                    return Err(ControlError::Conflict(format!(
                        "Step {step} has passed, this component is at step {current}."
                    )));
                }
            }
            let action = Action {
                id: uuid::Uuid::new_v4().simple().to_string(),
                type_: action_type,
                state: ActionState::Pending,
                step,
            };
            state.id_to_action.insert(action.id.clone(), action.clone());
            state.pending = Some(action.id.clone());
            action
        };
        self.submitted.notify_one();
        Ok(action)
    }

    /// Apply a pause due at `step`. Returns whether the run pauses there.
    pub fn update(&self, step: u64) -> bool {
        let (id, type_) = {
            let mut state = self.state.lock().expect("controller lock");
            state.step = Some(step);
            let Some(id) = state.pending.clone() else { return false };
            let action = state.id_to_action.get_mut(&id).expect("pending action is registered");
            if action.step.is_some_and(|s| s != step) {
                return false;
            }
            action.state = ActionState::Accepted;
            action.step = Some(step);
            let type_ = action.type_;
            state.pending = None;
            state.stop_step = Some(step);
            (id, type_)
        };
        tracing::info!("Control action {id}: {} at step {step}", type_.as_str());
        true
    }
}

fn detail(status: StatusCode, message: String) -> Response {
    (status, Json(serde_json::json!({ "detail": message }))).into_response()
}

impl IntoResponse for ControlError {
    fn into_response(self) -> Response {
        let status = match &self {
            ControlError::NotFound(_) => StatusCode::NOT_FOUND,
            ControlError::Conflict(_) => StatusCode::CONFLICT,
            ControlError::Invalid(_) => StatusCode::UNPROCESSABLE_ENTITY,
        };
        detail(status, self.to_string())
    }
}

async fn get_action(State(controller): State<Controller>, Path(action_id): Path<String>) -> Result<Json<Action>, ControlError> {
    controller.get(&action_id).map(Json)
}

async fn submit_action(State(controller): State<Controller>, body: axum::body::Bytes) -> Response {
    let request: ActionRequest = match serde_json::from_slice(&body) {
        Ok(request) => request,
        // FastAPI answers malformed bodies with 422 and a `detail` list of validation errors.
        Err(error) => {
            return (
                StatusCode::UNPROCESSABLE_ENTITY,
                Json(serde_json::json!({ "detail": [{ "type": "value_error", "loc": ["body"], "msg": error.to_string() }] })),
            )
                .into_response()
        }
    };
    match controller.submit(request.type_, request.step) {
        Ok(action) => (StatusCode::ACCEPTED, Json(action)).into_response(),
        Err(error) => error.into_response(),
    }
}

pub fn router(controller: Controller) -> Router {
    Router::new()
        .route("/v1/actions/{action_id}", get(get_action))
        .route("/v1/actions", post(submit_action))
        .with_state(controller)
}

/// Serve the control API on `host:port` until `cancel` fires.
pub async fn serve(controller: Controller, host: &str, port: u16, cancel: CancellationToken) -> anyhow::Result<()> {
    let listener = tokio::net::TcpListener::bind((host, port)).await?;
    serve_on(controller, listener, cancel).await
}

pub async fn serve_on(controller: Controller, listener: tokio::net::TcpListener, cancel: CancellationToken) -> anyhow::Result<()> {
    axum::serve(listener, router(controller).into_make_service_with_connect_info::<SocketAddr>())
        .with_graceful_shutdown(cancel.cancelled_owned())
        .await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn update_pauses_at_due_step() {
        // (requires_step, requested_step, steps_before, steps_after, expected)
        type Case = (bool, Option<u64>, &'static [u64], &'static [u64], &'static [bool]);
        let cases: [Case; 3] = [
            (false, None, &[], &[16], &[true]),
            (true, Some(17), &[16], &[16, 17], &[false, true]),
            (true, Some(16), &[16], &[16], &[true]),
        ];
        for (requires_step, requested, before, after, expected) in cases {
            let controller = Controller::new(requires_step);
            for &step in before {
                controller.update(step);
            }
            controller.submit(ActionType::Pause, requested).unwrap();
            let paused: Vec<bool> = after.iter().map(|&s| controller.update(s)).collect();
            assert_eq!(paused, expected);
        }
    }

    #[test]
    fn update_without_pending_pause() {
        assert!(!Controller::new(false).update(16));
    }

    #[test]
    fn submit_while_pausing() {
        for applied in [false, true] {
            let controller = Controller::new(false);
            controller.submit(ActionType::Pause, None).unwrap();
            if applied {
                controller.update(16);
            }
            assert!(matches!(controller.submit(ActionType::Pause, None), Err(ControlError::Conflict(_))));
        }
    }

    #[test]
    fn submit_passed_step() {
        let controller = Controller::new(true);
        controller.update(16);
        assert!(matches!(controller.submit(ActionType::Pause, Some(15)), Err(ControlError::Conflict(_))));
    }

    #[test]
    fn submit_invalid_step() {
        assert!(matches!(Controller::new(true).submit(ActionType::Pause, None), Err(ControlError::Invalid(_))));
        assert!(matches!(Controller::new(false).submit(ActionType::Pause, Some(16)), Err(ControlError::Invalid(_))));
    }

    #[test]
    fn get_commits_once_saved() {
        for (saved, expected) in [(true, ActionState::Committed), (false, ActionState::Accepted)] {
            let controller = Controller::new(false);
            let action = controller.submit(ActionType::Pause, None).unwrap();
            controller.update(6);
            if saved {
                controller.mark_saved(6);
            }
            let observed = controller.get(&action.id).unwrap();
            assert_eq!(observed, Action { id: action.id.clone(), type_: ActionType::Pause, state: expected, step: Some(6) });
        }
    }

    #[test]
    fn get_unknown_action() {
        assert!(matches!(Controller::new(false).get("missing"), Err(ControlError::NotFound(_))));
    }
}
