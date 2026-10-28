"""Fixtures for prl-wire tests: run `uv run --no-sync python rust/crates/prl-wire/tests/fixtures/gen_json.py`."""

from pathlib import Path

import msgpack
import numpy as np
import orjson

OUT = Path(__file__).parent
rng = np.random.default_rng(0)

f64 = np.concatenate(
    [
        rng.standard_normal(2000) * 10.0 ** rng.integers(-30, 30, 2000),
        np.round(rng.standard_normal(500), 4),
        np.array([0.0, -0.0, 1.0, 100.0, 1e15, 1e16, 1e-5, 9.99e-6, 1e-6, 5e-324, 1.7976931348623157e308, 0.1, 1 / 3]),
        10.0 ** np.arange(-25, 25),
        -(10.0 ** np.arange(-25, 25)),
    ]
)
f32 = np.concatenate(
    [
        rng.standard_normal(2000).astype(np.float32)
        * np.float32(10.0) ** rng.integers(-20, 20, 2000).astype(np.float32),
        (10.0 ** np.arange(-20, 20)).astype(np.float32),
    ]
)
strings = ["plain", "é\u2028<\x7f", 'a\nb\tc\x08\x0c\r"\\ \x00 \x1b \x1f', "日本語", "emoji 🎉", "/slash"]
ints = [0, 1, -1, 2**31, -(2**63), 2**63 - 1, 2**64 - 1]

(OUT / "json_input.msgpack").write_bytes(
    msgpack.packb(
        {
            "f64": f64.tolist(),
            "f32": [float(x) for x in f32],
            "strings": strings,
            "ints": ints,
            "nested": {"a": [None, True, False, {"b": []}], "c": {}},
        },
        use_bin_type=True,
    )
)
(OUT / "json_expected_f64.json").write_bytes(orjson.dumps(f64.tolist()))
(OUT / "json_expected_f32.json").write_bytes(orjson.dumps(f32, option=orjson.OPT_SERIALIZE_NUMPY))
(OUT / "json_expected_rest.json").write_bytes(
    orjson.dumps({"strings": strings, "ints": ints, "nested": {"a": [None, True, False, {"b": []}], "c": {}}})
)
