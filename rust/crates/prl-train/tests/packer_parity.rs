//! Packer + encoder parity with Python: random sample sets packed by `prepare_batch` and encoded
//! by `BatchSender.encode` (fixtures from `tests/fixtures/gen_packer.py`).

use base64::Engine;
use prl_train::packer::prepare_batch;
use prl_train::{encode_micro_batches, FlopTable, MMImageRef, MMRefs, PayloadSlice, TrainingSample};
use serde_json::Value;
use sha2::{Digest, Sha256};

fn f32s(v: &Value) -> Option<Vec<f32>> {
    v.as_array().map(|a| a.iter().map(|x| x.as_f64().unwrap() as f32).collect())
}

fn i32s(v: &Value) -> Option<Vec<i32>> {
    v.as_array().map(|a| a.iter().map(|x| x.as_i64().unwrap() as i32).collect())
}

fn sample(v: &Value) -> TrainingSample {
    TrainingSample {
        token_ids: i32s(&v["token_ids"]).unwrap(),
        mask: v["mask"].as_array().unwrap().iter().map(|x| x.as_bool().unwrap()).collect(),
        logprobs: f32s(&v["logprobs"]).unwrap(),
        temperatures: f32s(&v["temperatures"]).unwrap(),
        env_name: v["env_name"].as_str().unwrap().into(),
        ref_logprobs: f32s(&v["ref_logprobs"]),
        mm_refs: v["mm_refs"].as_array().map(|images| MMRefs {
            images: images
                .iter()
                .map(|i| MMImageRef {
                    url: i[0].as_str().unwrap().into(),
                    offset: i[1].as_u64().unwrap() as usize,
                    length: i[2].as_u64().unwrap() as usize,
                })
                .collect(),
        }),
        mm_token_type_ids: i32s(&v["mm_token_type_ids"]),
        rl_weights: f32s(&v["rl_weights"]),
        ce_weights: f32s(&v["ce_weights"]),
        ref_kl_weights: f32s(&v["ref_kl_weights"]),
        advantages: f32s(&v["advantages"]),
        trace_id: v["trace_id"].as_str().map(Into::into),
        branch_index: v["branch_index"].as_i64(),
        staleness: v["staleness"].as_array().map(|a| a.iter().map(|x| x.as_i64().unwrap() as i16).collect()),
        payload: v["payload"].as_object().map(|p| PayloadSlice {
            url: p["url"].as_str().unwrap().into(),
            id: p["id"].as_str().unwrap().into(),
            num_tokens: p["num_tokens"].as_u64().unwrap() as usize,
            fields: p["fields"].as_array().unwrap().iter().map(|f| f.as_str().unwrap().into()).collect(),
            routed_shape: p["routed_shape"].as_array().map(|a| a.iter().map(|x| x.as_i64().unwrap()).collect()),
            routed_dtype: p["routed_dtype"].as_str().map(Into::into),
        }),
    }
}

fn cost(name: &str) -> FlopTable {
    let per: fn(i128) -> i128 = match name {
        "sum" => |n| n,
        "flop" => |n| 64 * n + n * n,
        "big" => |n| n * 4 * 10i128.pow(16) + n * n * 10i128.pow(12),
        other => panic!("unknown cost {other}"),
    };
    FlopTable::new((0..200).map(per).collect())
}

#[test]
fn packer_matches_python_bytes() {
    let cases: Vec<Value> = serde_json::from_str(include_str!("fixtures/packer_cases.json")).unwrap();
    for (index, case) in cases.iter().enumerate() {
        let samples: Vec<TrainingSample> = case["samples"].as_array().unwrap().iter().map(sample).collect();
        let bin_cost = cost(case["cost"].as_str().unwrap());
        let grid = prepare_batch(
            samples,
            case["seq_len"].as_u64().unwrap() as usize,
            case["num_train_workers"].as_u64().unwrap() as usize,
            &bin_cost,
            case["pad_to_multiple_of"].as_u64().unwrap() as usize,
        )
        .unwrap();
        let digests = case["rank_sha256"].as_array().unwrap();
        assert_eq!(grid.len(), digests.len(), "case {index}: rank count");
        for (rank, micro_batches) in grid.iter().enumerate() {
            let encoded = encode_micro_batches(micro_batches);
            if let Some(ranks) = case["ranks"].as_array() {
                let want = base64::engine::general_purpose::STANDARD.decode(ranks[rank].as_str().unwrap()).unwrap();
                assert_eq!(encoded, want, "case {index} rank {rank}: bytes differ");
            }
            let digest = format!("{:x}", Sha256::digest(&encoded));
            assert_eq!(digest, digests[rank].as_str().unwrap(), "case {index} rank {rank}: digest differs");
        }
    }
}
