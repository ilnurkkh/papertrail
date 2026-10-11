"""Step 3: the experiment plan + the human approval gate."""
from __future__ import annotations

import re

from .ingest import normalize
from .models import Claim, Evidence, Origin, PlanField, canonical_json, sha256_text

SUPPORTED = {("accuracy", "digits")}  # the one experiment this MVP can actually run
TOLERANCE_POINTS = 1.0                # chosen BEFORE we run anything


def find_seed(pages: list[str]) -> PlanField:
    for page_no, text in enumerate(pages, 1):
        for sentence in re.split(r"(?<=[.!?])\s+", normalize(text)):
            match = re.search(r"\bseed\s*(?:=|:|of)?\s*(\d+)", sentence, re.IGNORECASE)
            if match:
                return PlanField(match.group(1), Origin.STATED, [Evidence(page_no, sentence)])
    return PlanField(None, Origin.MISSING)          # not in the paper -> we do NOT guess


def build_plan(claim: Claim, pages: list[str]) -> dict[str, PlanField]:
    return {
        "metric": PlanField(claim.metric, Origin.STATED, [claim.evidence]),
        "dataset": PlanField(claim.dataset, Origin.STATED, [claim.evidence]),
        "reported_value": PlanField(str(claim.value), Origin.STATED, [claim.evidence]),
        "seed": find_seed(pages),
        "command": PlanField("python experiment/run.py --seed <seed>", Origin.ASSUMED),
        "tolerance": PlanField(f"absolute +/-{TOLERANCE_POINTS} percentage points", Origin.ASSUMED),
    }


def plan_hash(plan: dict[str, PlanField]) -> str:
    return sha256_text(canonical_json(plan))


def missing_fields(plan: dict[str, PlanField]) -> list[str]:
    return [name for name, f in plan.items() if f.origin is Origin.MISSING]


def fill_missing(plan: dict[str, PlanField], name: str, value: str) -> None:
    """A human supplies a missing value. It is recorded as ASSUMED, never as STATED."""
    plan[name] = PlanField(value, Origin.ASSUMED)


def has_deviations(plan: dict[str, PlanField]) -> bool:
    """True if a human had to fill a value the paper did not state (not counting our defaults)."""
    return plan["seed"].origin is Origin.ASSUMED


def approval_record(plan: dict[str, PlanField], approved_by: str) -> dict:
    return {"plan_sha256": plan_hash(plan), "approved_by": approved_by}


def is_approved(plan: dict[str, PlanField], approval: dict | None) -> bool:
    """Approval is bound to the exact plan. Edit the plan after approving and this turns False."""
    return bool(approval) and approval.get("plan_sha256") == plan_hash(plan)
