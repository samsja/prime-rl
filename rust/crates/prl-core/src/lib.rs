//! Orchestrator core: dispatcher, concurrency, sources, curriculum, watcher, sinks and the
//! supervisor that wires the actors together.

pub mod concurrency;
pub mod eval_source;
pub mod eval_sink;
pub mod watcher;
pub mod lag;
pub mod curriculum;
pub mod train_source;
pub mod live;
pub mod dispatcher;
