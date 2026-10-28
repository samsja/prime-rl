//! What the orchestrator knows about an episode, recorded as it learns it
//! (`prime_rl/orchestrator/annotations.py`): arrival facts are stamped onto the traces before
//! they are logged, and what a batch adds becomes an append-only trace update.

use prl_types::Kind;
use prl_wire::{Episode, Map, RunInfo, Value};

use crate::file::unix_time;
use crate::file::update::make_update;

fn stamp(step: i64, time: f64) -> Value {
    [("step", Value::Int(step)), ("time", Value::F64(time))]
        .into_iter()
        .collect()
}

/// Stamp the kind of work, the step that dispatched each trace and the step it came back at.
pub fn stamp_arrival(episodes: &mut [Episode], kind: Kind, step: u64) {
    let now = unix_time();
    for episode in episodes {
        let work_step = match &episode.run {
            Some(RunInfo::Train(run)) => Some(run.work.step()),
            _ => None,
        };
        for trace in &mut episode.traces {
            trace.info.insert("kind".into(), kind.as_str().into());
            if let Some(work_step) = work_step {
                let dispatch = stamp(work_step, trace.timing.start);
                trace.info.insert("dispatch".into(), dispatch);
            }
            trace.info.insert("arrival".into(), stamp(step as i64, now));
        }
    }
}

/// The updates a batched cohort adds over its arrival records: membership, the step it ties
/// to, the scalar advantage and the per-token advantage streams of trainable branches.
pub fn stamp_batch<'a>(episodes: impl IntoIterator<Item = &'a Episode>, step: u64) -> Vec<Value> {
    let now = unix_time();
    let mut updates = Vec::new();
    for episode in episodes {
        for trace in &episode.traces {
            let mut info = Map::new();
            info.insert("effective".into(), Value::Bool(true));
            info.insert("ship".into(), stamp(step as i64, now));
            if let Some(advantage) = trace.info.get("advantage").filter(|a| !a.is_nil()) {
                info.insert("advantage".into(), advantage.clone());
            }
            let branches = trace
                .branches()
                .iter()
                .filter(|b| b.trainable)
                .filter_map(|b| {
                    let advantages = b.advantages()?;
                    let stream: Vec<Value> = advantages.into_iter().map(Value::F32).collect();
                    Some((
                        b.index as i64,
                        [("advantages".to_string(), Value::Array(stream))]
                            .into_iter()
                            .collect(),
                    ))
                })
                .collect();
            updates.push(make_update(&trace.id, Some(info), branches));
        }
    }
    updates
}
