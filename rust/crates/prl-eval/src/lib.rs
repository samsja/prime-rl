//! `uv run eval` and the SFT online-eval process on the Rust core (port of `src/prime_rl/eval`).

pub mod control;
mod eval;
mod online;
pub mod resume;
mod runner;

pub use eval::run_eval;
pub use online::run_online_eval;
