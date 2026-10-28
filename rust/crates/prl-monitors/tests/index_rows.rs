//! Index rows of random records against Python's `index_row` (`tests/fixtures/gen_index_rows.py`).

use std::path::Path;

use prl_monitors::file::index::index_row;
use prl_wire::Value;

fn lines(name: &str) -> Vec<Vec<u8>> {
    std::fs::read(
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures")
            .join(name),
    )
    .unwrap()
    .split(|b| *b == b'\n')
    .filter(|l| !l.is_empty())
    .map(<[u8]>::to_vec)
    .collect()
}

#[test]
fn index_rows_match_python() {
    let records = lines("index_records.jsonl");
    let rows = lines("index_rows.jsonl");
    assert_eq!(records.len(), rows.len());
    for (i, (record, want)) in records.iter().zip(rows).enumerate() {
        let record: Value = serde_json::from_slice(record).unwrap();
        let got = prl_wire::json::to_vec(&Value::Map(index_row(
            i as u64 + 1,
            &record,
            i as u64 / 10,
            1000 * i as u64,
        )));
        assert_eq!(
            String::from_utf8(got).unwrap(),
            String::from_utf8(want).unwrap(),
            "record {i}"
        );
    }
}
