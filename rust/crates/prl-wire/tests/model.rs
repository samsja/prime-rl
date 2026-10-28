use prl_wire::protocol::Response;
use prl_wire::{json, value, Episode, EpisodeAssembly, Error, GroupInfo, PolicySpan, RunInfo, TrainRunInfo, Value, WorkInfo};

const FIXTURES: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures");

fn fixture(name: &str) -> Vec<u8> {
    std::fs::read(format!("{FIXTURES}/{name}")).unwrap()
}

fn text(bytes: Vec<u8>) -> String {
    String::from_utf8(bytes).unwrap()
}

/// `assert_eq!` on long strings, reporting the neighbourhood of the first difference.
fn assert_same(actual: String, expected: String) {
    if actual == expected {
        return;
    }
    let at = actual.bytes().zip(expected.bytes()).take_while(|(a, b)| a == b).count();
    let window = |s: &str| s.get(at.saturating_sub(150)..(at + 150).min(s.len())).unwrap_or("").to_owned();
    panic!("differ at byte {at}\n  actual: {}\nexpected: {}", window(&actual), window(&expected));
}

/// Apply the fixture's stream like `EnvClient.run`, returning the assembled record.
fn assemble() -> Value {
    let stream = value::from_msgpack(&fixture("episode_stream.msgpack")).unwrap();
    let mut assembly = EpisodeAssembly::new();
    for frame in stream.get("deltas").unwrap().as_array().unwrap() {
        assembly.apply(&value::from_msgpack(frame.as_bytes().unwrap()).unwrap()).unwrap();
    }
    let reply = Response::from_value(&value::from_msgpack(stream.get("reply").unwrap().as_bytes().unwrap()).unwrap()).unwrap();
    assert!(reply.success);
    assembly.finish(reply.head.as_ref().unwrap(), &reply.traces).unwrap()
}

fn received() -> Episode {
    Episode::from_wire(&assemble()).unwrap()
}

#[test]
fn assembly_matches_python() {
    let expected = value::from_msgpack(&fixture("episode_assembled.msgpack")).unwrap();
    assert_eq!(assemble(), expected);
}

#[test]
fn records_match_python_byte_for_byte() {
    let episode = received();
    assert_same(text(episode.to_json(Some(4))), text(fixture("episode_record_4.json")));
    assert_same(text(episode.to_json(None)), text(fixture("episode_record_none.json")));
    assert_same(text(json::to_vec(&episode.traces[0].to_record(Some(2)))), text(fixture("episode_trace0_record_2.json")));
}

fn derived(episode: &Episode) -> serde_json::Value {
    let traces: Vec<serde_json::Value> = episode
        .traces
        .iter()
        .map(|t| {
            let branches: Vec<serde_json::Value> = t
                .branches()
                .iter()
                .map(|b| {
                    serde_json::json!({
                        "index": b.index,
                        "trainable": b.trainable,
                        "token_ids": b.token_ids(),
                        "sampled_mask": b.sampled_mask(),
                        "logprobs": b.logprobs().iter().map(|&x| x as f64).collect::<Vec<_>>(),
                        "advantages": b.advantages().map(|a| a.iter().map(|&x| x as f64).collect::<Vec<_>>()),
                        "loss_weights_rl": b.loss_weights("rl").unwrap().map(|a| a.iter().map(|&x| x as f64).collect::<Vec<_>>()),
                        "mm_token_type_ids": b.mm_token_type_ids(),
                        "num_input_tokens": b.num_input_tokens(),
                        "num_output_tokens": b.num_output_tokens(),
                        "num_total_tokens": b.num_total_tokens(),
                    })
                })
                .collect();
            serde_json::json!({
                "reward": t.reward(),
                "has_error": t.has_error(),
                "is_truncated": t.is_truncated(),
                "num_turns": t.num_turns(),
                "num_branches": t.num_branches(),
                "num_input_tokens": t.num_input_tokens(),
                "num_output_tokens": t.num_output_tokens(),
                "num_total_tokens": t.num_total_tokens(),
                "last_reply": t.last_reply(),
                "branches": branches,
            })
        })
        .collect();
    serde_json::json!({
        "num_input_tokens": episode.num_input_tokens(),
        "num_output_tokens": episode.num_output_tokens(),
        "num_total_tokens": episode.num_total_tokens(),
        "traces": traces,
    })
}

#[test]
fn derived_values_match_python() {
    let expected: serde_json::Value = serde_json::from_slice(&fixture("episode_derived.json")).unwrap();
    assert_eq!(derived(&received()), expected);
}

/// The mutations `gen_episode.py` applies, as the orchestrator does them.
fn mutate(episode: &mut Episode) {
    episode.env.name = Some("my-env-name".into());
    episode.group = Some(GroupInfo { id: "g1".into() });
    episode.record_run(RunInfo::Train(TrainRunInfo {
        id: "run1".into(),
        name: None,
        work: WorkInfo::Train { step: 3, policy: Some(PolicySpan::new(1, 2).unwrap()) },
    }));
    let a = &mut episode.traces[0];
    a.info.insert("kind".into(), "train".into());
    a.info.insert("arrival".into(), [("step", Value::from(3)), ("time", Value::from(1.25))].into_iter().collect());
    a.nodes[4].advantages = Some(vec![0.7; 2]);
    a.nodes[2].advantages = Some(vec![-1.0 / 3.0; 3]);
    a.nodes[4].loss_weights = Some([("rl".to_owned(), vec![1.0f32 / 3.0; 3])].into_iter().collect());
    a.nodes[4].reference_logprobs = Some(vec![-0.3, -0.123456]);
    let b = &mut episode.traces[1];
    b.errors.push(Error::new("Timeout", "Trace stopped by error"));
    b.ok = false;
    episode.ok = false;
}

#[test]
fn mutated_episode_matches_python() {
    let mut episode = received();
    mutate(&mut episode);
    assert_same(text(episode.to_json(Some(4))), text(fixture("episode_mutated_record_4.json")));
    let expected: serde_json::Value = serde_json::from_slice(&fixture("episode_mutated_derived.json")).unwrap();
    assert_eq!(derived(&episode), expected);
}

#[test]
fn a_record_reloads_into_the_same_record() {
    let mut episode = received();
    mutate(&mut episode);
    let record = episode.to_json(Some(4));
    let reloaded = Episode::from_record_json(&record).unwrap();
    assert_same(text(reloaded.to_json(Some(4))), text(record));
}

#[test]
fn record_validation_is_strict_where_pydantic_is() {
    assert!(Episode::from_record_json(br#"{"task": {"type": "T", "data": {}}}"#).unwrap().traces.is_empty());
    assert!(Episode::from_record_json(br#"{"task": {"type": "T", "data": {}}, "traces": null}"#).is_err());
    assert!(Episode::from_record_json(br#"{"task": {"type": "T", "data": {}}, "traces": [{}]}"#).is_err());
    assert!(Episode::from_record_json(br#"{"traces": []}"#).is_err());
}
