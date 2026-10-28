"""Generate the record schema of `verifiers.v1.WireEpisode` for prl-wire.

Walks the pydantic models a `WireEpisode` is made of and writes, per model, its fields
in dump order with the facts a JSON-mode dump depends on (exclusions, extra fields,
computed fields, union discriminators). prl-wire interprets this to dump records
exactly like `Episode.to_record`.

    uv run python rust/crates/prl-wire/schema/gen_schema.py
"""

import dataclasses
import json
import sys
import types
import typing
from pathlib import Path

import numpy as np
from pydantic import BaseModel
from pydantic.fields import FieldInfo

from verifiers.v1.episode import WireEpisode

OUT = Path(__file__).with_name("wire_episode.json")

models: dict[str, dict] = {}
names: dict[type, str] = {}


def model_name(cls: type) -> str:
    if cls in names:
        return names[cls]
    name = cls.__name__
    taken = set(names.values())
    i = 2
    base = name
    while name in taken:
        name = f"{base}_{i}"
        i += 1
    names[cls] = name
    models[name] = {}  # reserve before recursing
    models[name] = describe_model(cls)
    return name


def discriminator_of(metadata) -> str | None:
    for item in metadata:
        if isinstance(item, FieldInfo) and isinstance(item.discriminator, str):
            return item.discriminator
        disc = getattr(item, "discriminator", None)
        if isinstance(disc, str):
            return disc
        if disc is not None:
            raise TypeError(f"unsupported callable discriminator {disc!r}")
    return None


def describe(tp, discriminator: str | None = None) -> dict:
    origin = typing.get_origin(tp)
    if origin is typing.Annotated:
        base, *metadata = typing.get_args(tp)
        disc = discriminator_of(metadata) or discriminator
        # SerializeAsAny and validators wrap the base type; the stream types are arrays
        if base is np.ndarray:
            return {"k": "ndarray"}
        return describe(base, disc)
    if tp is typing.Any or tp is object:
        return {"k": "any"}
    if tp is type(None):
        return {"k": "none"}
    if tp in (str, int, float, bool, bytes):
        return {"k": tp.__name__}
    if tp is np.ndarray:
        return {"k": "ndarray"}
    if origin in (typing.Union, types.UnionType):
        return {
            "k": "union",
            "disc": discriminator,
            "options": [describe(arg) for arg in typing.get_args(tp)],
        }
    if origin is typing.Literal:
        return {"k": "literal", "values": list(typing.get_args(tp))}
    if origin in (list, set, frozenset, typing.Sequence) or tp is list:
        args = typing.get_args(tp)
        return {"k": "list", "item": describe(args[0]) if args else {"k": "any"}}
    if origin is tuple:
        return {"k": "list", "item": {"k": "any"}}
    if origin is dict or tp is dict:
        args = typing.get_args(tp)
        return {"k": "dict", "value": describe(args[1]) if args else {"k": "any"}}
    if isinstance(tp, type) and issubclass(tp, BaseModel):
        return {"k": "model", "ref": model_name(tp)}
    if isinstance(tp, type) and dataclasses.is_dataclass(tp):
        return {"k": "any"}
    # SerializeAsAny[X] and similar markers
    if hasattr(tp, "__origin__"):
        return describe(tp.__origin__, discriminator)
    if isinstance(tp, typing.TypeVar):
        bound = tp.__bound__
        return describe(bound) if bound is not None else {"k": "any"}
    if isinstance(tp, type):
        # Path, UUID, Enum, datetime, ...: plain values on the wire and in records
        return {"k": "any"}
    raise TypeError(f"unsupported annotation {tp!r}")


def describe_model(cls: type[BaseModel]) -> dict:
    # A plain field serializer returns a plain value (dict, list, ...), which a dump
    # writes as-is: no `exclude_none` inside it.
    serialized = {
        name for decorator in cls.__pydantic_decorators__.field_serializers.values() for name in decorator.info.fields
    }
    fields = []
    for name, field in cls.model_fields.items():
        tp = field.annotation
        if field.metadata:
            tp = typing.Annotated[(tp, *field.metadata)]
        if name in serialized:
            schema = {"k": "any"}
        else:
            disc = field.discriminator if isinstance(field.discriminator, str) else None
            schema = describe(tp, disc)
        entry = {"name": name, "schema": schema}
        if field.exclude:
            entry["exclude"] = True
        if field.exclude_if is not None:
            # every exclude_if in verifiers is "exclude when empty"
            entry["exclude_if_empty"] = True
        fields.append(entry)
    return {
        "fields": fields,
        "extra": cls.model_config.get("extra") == "allow",
        "computed": list(cls.model_computed_fields),
    }


model_name(WireEpisode)
OUT.write_text(json.dumps({"models": models}, separators=(",", ":")) + "\n")
print(f"wrote {len(models)} models to {OUT}", file=sys.stderr)
