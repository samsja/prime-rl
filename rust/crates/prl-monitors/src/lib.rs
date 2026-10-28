//! Monitors: the orchestrator's metrics/episode sinks (file traces, Prime platform, W&B).

pub mod file;
pub mod pyjson;
pub mod records;

pub use file::live::{stage, LiveEvent, STAGES};
pub use pyjson::{Row, Scalar};

/// An episode as the monitors serialize it: `Episode.to_record(float_decimals)`.
pub trait EpisodeRecord: Send + Sync + 'static {
    fn to_record(&self, float_decimals: Option<u32>) -> prl_wire::Value;
}

/// An already-serialized record (tests, replays).
impl EpisodeRecord for prl_wire::Value {
    fn to_record(&self, _float_decimals: Option<u32>) -> prl_wire::Value {
        self.clone()
    }
}
