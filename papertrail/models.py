"""Core data types. Small on purpose: everything the AI says must fit into these."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum


class Status(str, Enum):
    REPRODUCED = "reproduced under the defined criteria"
    PARTIAL = "partially reproduced"
    NOT_REPRODUCED = "not reproduced"
    UNABLE = "unable to evaluate"


class Origin(str, Enum):
    STATED = "stated"    # the paper says it, and we can point at the quote
    ASSUMED = "assumed"  # we filled a gap; a human is told about it
    MISSING = "missing"  # unknown; we refuse to guess


@dataclass
class Evidence:
    page: int
    quote: str


@dataclass
class Claim:
    metric: str            # e.g. "accuracy"
    dataset: str           # e.g. "Digits"
    printed_value: str     # exactly as printed in the paper, e.g. "96.2%"
    value: float           # converted to percentage points, e.g. 96.2
    evidence: Evidence


@dataclass
class PlanField:
    value: str | None
    origin: Origin
    evidence: list[Evidence] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.origin is Origin.STATED and not self.evidence:
            raise ValueError("A STATED field must cite evidence.")
        if self.origin is Origin.MISSING and self.value is not None:
            raise ValueError("A MISSING field must not carry a value.")
        if self.origin is not Origin.MISSING and self.value is None:
            raise ValueError("STATED/ASSUMED fields need a value.")


def to_jsonable(obj):
    """Turn our dataclasses/enums into plain JSON-friendly data."""
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if hasattr(obj, "__dataclass_fields__"):
        return {k: to_jsonable(getattr(obj, k)) for k in obj.__dataclass_fields__}
    return obj


def canonical_json(obj) -> str:
    return json.dumps(to_jsonable(obj), sort_keys=True, separators=(",", ":"))


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()
