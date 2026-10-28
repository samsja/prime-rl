//! Inference clients and admin plane, inference metrics, weight receivers, run control,
//! heartbeat, orchestrator checkpoints, gc gauges and loguru-compatible logging.
//! See `rust/docs/specs/io.md`.

pub mod ckpt;
pub mod control;
pub mod gc;
pub mod heartbeat;
pub mod logging;
pub mod pathing;
