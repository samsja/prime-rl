//! GC gauges (`orchestrator/gc.py`). Rust has no tracing garbage collector, so there is nothing
//! to freeze, schedule or pause; the pipeline view keeps the same gauge keys, always zero.

use prl_types::Metrics;

#[derive(Debug, Clone)]
pub struct GarbageCollector {
    pub interval: u64,
}

impl GarbageCollector {
    pub fn new(interval: u64) -> Self {
        assert!(interval > 0, "gc interval must be a positive integer");
        Self { interval }
    }

    pub fn gauges(&self) -> Metrics {
        let mut gauges = Metrics::new();
        gauges.insert("gc/collections".into(), 0.0.into());
        gauges.insert("gc/pause_max".into(), 0.0.into());
        gauges.insert("gc/pause_total".into(), 0.0.into());
        gauges
    }
}
