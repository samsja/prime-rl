//! Train side of the orchestrator: train sink, algorithms, trajectories, metrics, batch packer and
//! the trainer-facing batch transport.

pub mod packer;
pub mod partition;
pub mod sample;
pub mod transport;

pub use packer::{BatchPacker, BinCost, FlopTable};
pub use sample::{sample_payload_urls, MMImageRef, MMRefs, PayloadSlice, TrainingSample, PAYLOAD_INFO_KEY};
pub use transport::{encode_micro_batches, BatchSender, MicroBatch, TransportConfig};
