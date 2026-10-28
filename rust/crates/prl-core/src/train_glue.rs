//! Builders for the prl-train pieces the orchestrator wires (algorithms, packer, sink).

use std::sync::Arc;

use anyhow::Result;
use indexmap::IndexMap;
use prl_config::{ModelReference, OrchestratorConfig, TrainSourceConfig};
use prl_io::clients::{connect_frozen_client, InferenceClient};
use prl_train::algo::{build_algorithm as build, Algorithm, AlgorithmDeps};
use prl_train::{BatchPacker, FlopTable, PayloadReleaser, SinkConfig, TrainEnvSpec, TrainSink};
use prl_types::{Bridge, Scorer};

/// The env's runtime algorithm (`build_algorithm(config.algo, clients)`).
pub fn build_algorithm(
    _config: &OrchestratorConfig,
    source: &TrainSourceConfig,
    policy: &Arc<InferenceClient>,
    bridge: &Arc<dyn Bridge>,
) -> Result<Box<dyn Algorithm>> {
    let deps = AlgorithmDeps {
        policy: policy.clone() as Arc<dyn Scorer>,
        policy_model_name: policy.model_name.clone(),
        connect_frozen: Arc::new(|model| {
            Box::pin(async move { Ok(Arc::new(connect_frozen_client(&model, None).await?) as Arc<dyn Scorer>) })
        }),
        bridge: Some(bridge.clone()),
    };
    build(&source.algo, &deps)
}

/// The micro-batch packer with the trainer model's FLOP cost table (from the bridge).
pub async fn build_packer(config: &OrchestratorConfig, bridge: &Arc<dyn Bridge>) -> Result<BatchPacker> {
    let (seq_len, multiple) = (config.seq_len as usize, config.pad_to_multiple_of as usize);
    let max_len = if config.pad_to_seq_len { seq_len } else { FlopTable::max_len(seq_len, multiple) };
    let table = FlopTable::new(bridge.flop_cost_table(max_len).await?);
    Ok(BatchPacker::new(seq_len, config.num_train_workers as usize, multiple, config.pad_to_seq_len, Arc::new(table)))
}

pub fn train_env_specs(
    config: &OrchestratorConfig,
    algorithms: Vec<Box<dyn Algorithm>>,
) -> Result<IndexMap<String, TrainEnvSpec>> {
    Ok(config
        .train
        .source
        .iter()
        .zip(algorithms)
        .map(|(source, algorithm)| {
            let live = matches!(source.algo.sampling.source, ModelReference::Policy);
            let spec = TrainEnvSpec {
                group_size: source.group_size as usize,
                temperature: source.sampling.temperature as f32,
                requires_sampling_masks: source.sampling.truncates_distribution && live,
                requires_top_logprobs: source.sampling.logprobs.is_some() && live,
                algorithm,
            };
            (source.resolved_name.clone(), spec)
        })
        .collect())
}

pub fn build_sink(config: &OrchestratorConfig, specs: IndexMap<String, TrainEnvSpec>) -> Result<TrainSink> {
    let cfg = SinkConfig {
        batch_size: config.batch_size as usize,
        max_staleness: config.max_staleness as i64,
        constant_trainer_batch_size: config.constant_trainer_batch_size,
    };
    Ok(TrainSink::new(cfg, specs, PayloadReleaser::new(prl_wire::payload::delete_payloads)))
}
