"""Step 5: compare reported vs measured, then assign a status. No AI, ever."""
from __future__ import annotations

from dataclasses import dataclass

from .models import Status


@dataclass
class Comparison:
    metric: str
    reported: float
    measured: float | None
    difference: float | None
    allowed: float
    within_tolerance: bool | None   # None = could not be evaluated


def compare_metric(metric: str, reported: float, measured: float | None, allowed: float) -> Comparison:
    if measured is None:
        return Comparison(metric, reported, None, None, allowed, None)
    diff = measured - reported
    return Comparison(metric, reported, measured, round(diff, 4), allowed, abs(diff) <= allowed + 1e-9)


def classify(comparisons: list[Comparison], run_succeeded: bool, has_deviations: bool = False) -> Status:
    if not run_succeeded or not comparisons or any(c.within_tolerance is None for c in comparisons):
        return Status.UNABLE            # "could not run" is NOT the same as "ran and disagreed"
    hits = [c.within_tolerance for c in comparisons]
    if all(hits):
        return Status.PARTIAL if has_deviations else Status.REPRODUCED
    return Status.PARTIAL if any(hits) else Status.NOT_REPRODUCED
