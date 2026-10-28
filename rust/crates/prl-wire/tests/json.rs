use prl_wire::{json, value, Value};

const FIXTURES: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures");

fn fixture(name: &str) -> Vec<u8> {
    std::fs::read(format!("{FIXTURES}/{name}")).unwrap()
}

#[test]
fn floats_match_orjson() {
    let input = value::from_msgpack(&fixture("json_input.msgpack")).unwrap();
    let f64s = input.get("f64").unwrap().clone();
    assert_eq!(
        String::from_utf8(json::to_vec(&f64s)).unwrap(),
        String::from_utf8(fixture("json_expected_f64.json")).unwrap()
    );
    let f32s = Value::Array(
        input.get("f32").unwrap().as_array().unwrap().iter()
            .map(|v| Value::F32(v.as_f64().unwrap() as f32)).collect(),
    );
    assert_eq!(
        String::from_utf8(json::to_vec(&f32s)).unwrap(),
        String::from_utf8(fixture("json_expected_f32.json")).unwrap()
    );
}

#[test]
fn strings_ints_and_nesting_match_orjson() {
    let input = value::from_msgpack(&fixture("json_input.msgpack")).unwrap();
    let mut rest = input.as_map().unwrap().clone();
    rest.shift_remove("f64");
    rest.shift_remove("f32");
    assert_eq!(json::to_vec(&Value::Map(rest)), fixture("json_expected_rest.json"));
}

#[test]
fn msgpack_roundtrip_is_identical() {
    let data = fixture("json_input.msgpack");
    let value = value::from_msgpack(&data).unwrap();
    assert_eq!(value::to_msgpack(&value), data);
}
