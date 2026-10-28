use prl_wire::protocol::Response;
use prl_wire::sampling::{train_sampling, with_cache_salt, TrainSampling};
use prl_wire::{json, value, Episode, EpisodeAssembly, Error, GroupInfo, PolicySpan, RunInfo, TrainRunInfo, Value, WorkInfo};

/// A two-trace episode streamed by the Python env server (`DeltaStreamer`), and its Python
/// `to_json()` record after the orchestrator-side mutations below.
fn fixture() -> Value {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/episode.msgpack");
    value::from_msgpack(&std::fs::read(path).unwrap()).unwrap()
}

fn received(fixture: &Value) -> Episode {
    let mut assembly = EpisodeAssembly::new();
    for frame in fixture.get("deltas").unwrap().as_array().unwrap() {
        assembly.apply(&value::from_msgpack(frame.as_bytes().unwrap()).unwrap()).unwrap();
    }
    let reply = value::from_msgpack(fixture.get("reply").unwrap().as_bytes().unwrap()).unwrap();
    let reply = Response::from_value(&reply).unwrap();
    Episode::from_wire(&assembly.finish(reply.head.as_ref().unwrap(), &reply.traces).unwrap()).unwrap()
}

/// What the dispatcher, the algorithm and the sink do to a received episode.
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
fn a_streamed_episode_records_like_python() {
    let fixture = fixture();
    let mut episode = received(&fixture);
    mutate(&mut episode);
    let record = episode.to_json(Some(4));
    assert_eq!(String::from_utf8_lossy(&record), String::from_utf8_lossy(fixture.get("record").unwrap().as_bytes().unwrap()));
    // a record reloads into the same record
    assert_eq!(Episode::from_record_json(&record).unwrap().to_json(Some(4)), record);
}

#[test]
fn derived_trace_values_match_python() {
    let mut episode = received(&fixture());
    mutate(&mut episode);
    assert_eq!((episode.num_input_tokens(), episode.num_output_tokens(), episode.num_total_tokens()), (12, 6, 18));
    let (a, b) = (&episode.traces[0], &episode.traces[1]);
    assert_eq!((a.reward(), a.has_error(), a.is_truncated(), a.num_turns()), (1.0, false, true, 3));
    assert_eq!((b.reward(), b.has_error(), b.is_truncated(), b.num_turns()), (0.0, true, false, 0));
    let branches = a.branches();
    // the compaction attempt (leaf 5) is trainable: node 4 links back to it
    assert_eq!(branches.iter().map(|b| (b.trainable, b.token_ids().len())).collect::<Vec<_>>(), [(true, 14), (true, 8)]);
    assert_eq!(branches[0].token_ids()[3..6], [151655, 151655, 4]);
    assert_eq!(branches[0].mm_token_type_ids().unwrap()[3..6], [1, 1, 0]);
    assert_eq!(branches[1].logprobs()[7], -0.75);
    let advantages = branches[0].advantages().unwrap();
    assert_eq!((advantages[6], advantages[7], advantages[13]), (0.0, -1.0 / 3.0, 0.7));
    assert!(branches[1].advantages().is_none());
    assert_eq!(
        branches.iter().map(|b| (b.num_input_tokens(), b.num_output_tokens(), b.num_total_tokens())).collect::<Vec<_>>(),
        [(6, 5, 10), (6, 1, 7)]
    );
}

#[test]
fn record_validation_is_strict_where_pydantic_is() {
    assert!(Episode::from_record_json(br#"{"task": {"type": "T", "data": {}}}"#).unwrap().traces.is_empty());
    assert!(Episode::from_record_json(br#"{"task": {"type": "T", "data": {}}, "traces": null}"#).is_err());
    assert!(Episode::from_record_json(br#"{"task": {"type": "T", "data": {}}, "traces": [{}]}"#).is_err());
    assert!(Episode::from_record_json(br#"{"traces": []}"#).is_err());
}

#[test]
fn json_matches_orjson() {
    let floats = [1e-5, 0.1, 1e16, 1e15, 123456789012345678.0, 100.0, 1.5e-7, -0.0, f64::NAN, 2.5e300, 0.1 + 0.2, 1e-4];
    let floats = Value::Array(floats.into_iter().map(Value::F64).collect());
    assert_eq!(
        String::from_utf8(json::to_vec(&floats)).unwrap(),
        "[0.00001,0.1,1e+16,1000000000000000.0,1.2345678901234568e+17,100.0,1.5e-7,-0.0,null,2.5e+300,0.30000000000000004,0.0001]"
    );
    // numpy float32 arrays print float32-shortest, decimal for exponents -6..13
    let f32s = Value::Array([1e-7f32, 1e-6, 3.937215e-6, 1e12, 1e13, 0.1].into_iter().map(Value::F32).collect());
    assert_eq!(String::from_utf8(json::to_vec(&f32s)).unwrap(), "[1e-7,0.000001,0.000003937215,1000000000000.0,1e+13,0.1]");
    let strings: Value = [("s", Value::from("a\nb\tc\u{8}\u{c}\r\"\\ \u{0} \u{1b} é\u{2028}"))].into_iter().collect();
    assert_eq!(String::from_utf8(json::to_vec(&strings)).unwrap(), "{\"s\":\"a\\nb\\tc\\b\\f\\r\\\"\\\\ \\u0000 \\u001b é\u{2028}\"}");
}

#[test]
fn python_json_matches_json_dumps() {
    let list = Value::Array(vec![1.0.into(), 1e16.into(), 0.1.into(), "é\n".into(), Value::Nil, true.into()]);
    let value: Value = [("b", Value::from(1e-5)), ("a", list)].into_iter().collect();
    assert_eq!(json::to_python_json(&value, true), r#"{"a": [1.0, 1e+16, 0.1, "\u00e9\n", null, true], "b": 1e-05}"#);
}

#[test]
fn train_sampling_matches_vf_sampling_config() {
    let extra_body = [("max_tokens", Value::from(10)), ("repetition_penalty", Value::from(1.1))].into_iter().collect::<Value>();
    let sampling = TrainSampling {
        temperature: 0.7,
        top_p: 0.9,
        top_k: Some(20),
        max_completion_tokens: Some(64),
        logprobs: None,
        extra_body: extra_body.into_map().unwrap(),
    };
    let salted = with_cache_salt(&train_sampling(&sampling, true), Some("s-1"));
    assert_eq!(
        String::from_utf8(json::to_vec(&Value::Map(salted))).unwrap(),
        r#"{"temperature":0.7,"top_p":0.9,"reasoning_effort":null,"max_tokens":64,"repetition_penalty":1.1,"top_k":20,"logprobs":true,"cache_salt":"s-1"}"#
    );
    let frozen = TrainSampling { temperature: 1.0, top_p: 1.0, top_k: None, max_completion_tokens: None, logprobs: None, extra_body: Default::default() };
    assert_eq!(
        String::from_utf8(json::to_vec(&Value::Map(train_sampling(&frozen, false)))).unwrap(),
        r#"{"temperature":1.0,"top_p":1.0,"reasoning_effort":null,"max_tokens":null,"logprobs":null}"#
    );
}

#[test]
fn task_hash_matches_task_key() {
    let data: Value = serde_json::from_str(
        r#"{"idx":3,"id":null,"name":null,"description":null,"prompt":"héllo","system_prompt":null,"scoring":{"mode":"shared","runtime":null,"env":null,"retries":{"max_retries":2,"rules":[]},"network":null,"skip_on_truncated":false,"skip_on_timeout":false},"image":null,"workdir":null,"agent_user":null,"skills":[],"network":{"allow":["*"],"block":[]},"artifacts":[],"artifact_max_bytes":33554432,"timeout":{"setup":null,"agent":null,"finalize":null,"scoring":null},"resources":{"cpu":null,"memory":null,"gpu":null,"disk":null},"answer":"olléh"}"#,
    )
    .unwrap();
    assert_eq!(prl_wire::task_hash(data.as_map().unwrap()), "f990e3529502cbafd59f3bd3801b35a0f01c8775129fcc6c291d8324739e87bc");
}
