use prl_wire::sampling::{train_sampling, with_cache_salt, TrainSampling};
use prl_wire::{json, task_hash, value, Value};

const FIXTURES: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures");

fn fixture(name: &str) -> Vec<u8> {
    std::fs::read(format!("{FIXTURES}/{name}")).unwrap()
}

#[test]
fn train_sampling_matches_vf_sampling_config() {
    let cases = value::from_msgpack(&fixture("sampling.msgpack")).unwrap();
    for case in cases.as_array().unwrap() {
        let input = case.get("input").unwrap();
        let sampling = TrainSampling {
            temperature: input.get("temperature").unwrap().as_f64().unwrap(),
            top_p: input.get("top_p").unwrap().as_f64().unwrap(),
            top_k: input.get("top_k").unwrap().as_i64(),
            max_completion_tokens: input.get("max_completion_tokens").unwrap().as_i64(),
            logprobs: input.get("logprobs").unwrap().as_i64(),
            extra_body: input.get("extra_body").unwrap().as_map().unwrap().clone(),
        };
        let live = case.get("live").unwrap().as_bool().unwrap();
        let dumped = train_sampling(&sampling, live);
        // key order too: the dict goes on the wire as is
        assert_eq!(json::to_vec(&Value::Map(dumped.clone())), json::to_vec(case.get("expected").unwrap()), "{case:?}");
        let salted = with_cache_salt(&dumped, Some("abc-3"));
        assert_eq!(json::to_vec(&Value::Map(salted)), json::to_vec(case.get("salted").unwrap()));
    }
}

#[test]
fn task_hash_matches_task_key() {
    let cases = value::from_msgpack(&fixture("task_hashes.msgpack")).unwrap();
    for case in cases.as_array().unwrap() {
        assert_eq!(task_hash(case.get("data").unwrap().as_map().unwrap()), case.get("hash").unwrap().as_str().unwrap());
    }
}

#[test]
fn python_json_matches_json_dumps() {
    let input = value::from_msgpack(&fixture("python_json_input.msgpack")).unwrap();
    assert_eq!(json::to_python_json(&input, true), String::from_utf8(fixture("python_json_expected.json")).unwrap());
}
