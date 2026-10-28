//! Train side of the orchestrator: train sink, algorithms, trajectories, metrics, batch packer and
//! the trainer-facing batch transport.

pub mod algo;
pub mod episodes;
pub mod metrics;
pub mod packer;
pub mod partition;
pub mod sample;
pub mod sink;
pub mod staleness;
pub mod trajectories;
pub mod transport;

pub use episodes::EpisodeRef;
pub use metrics::{dispatch_failure_metrics, EvalEpisodes, MetricWindow, RolloutWindow, Stat, TrainEpisodes};
pub use packer::{BatchPacker, FlopTable};
pub use sample::{sample_payload_urls, MMImageRef, MMRefs, PayloadSlice, TrainingSample, PAYLOAD_INFO_KEY};
pub use sink::{AdmitAll, Admission, PayloadReleaser, SinkConfig, TrainBatch, TrainEnvSpec, TrainSink};
pub use transport::{encode_micro_batches, BatchSender, MicroBatch};
