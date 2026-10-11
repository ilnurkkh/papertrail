"""Step 2b: the 'never trust the AI' layer. Pure code, no AI.

Every claim the AI proposes must pass these checks or it is rejected with a reason.
"""
from __future__ import annotations

import re

from .ingest import normalize
from .models import Claim, Evidence

BASELINE_WORDS = ("baseline", "prior work", "previous", "earlier work", "state-of-the-art")


def parse_value(printed: str) -> float | None:
    """'96.2%' -> 96.2 ; '0.962' -> 96.2 ; '96.2 percent' -> 96.2"""
    match = re.search(r"-?\d+(?:\.\d+)?", str(printed))
    if not match:
        return None
    number = float(match.group(0))
    is_percent = "%" in str(printed) or "percent" in str(printed).lower()
    if not is_percent and 0 <= number <= 1:
        number *= 100
    return round(number, 4)


def validate_proposal(proposal: dict, page_no: int, page_text: str) -> tuple[Claim | None, str | None]:
    """Return (claim, None) if the proposal is trustworthy, else (None, reason_code)."""
    try:
        metric = str(proposal["metric_name"]).strip().lower()
        dataset = str(proposal["dataset"]).strip()
        printed = str(proposal["value"]).strip()
        quote = str(proposal["quote"]).strip()
    except KeyError:
        return None, "missing_fields"
    if not (metric and dataset and printed and quote):
        return None, "missing_fields"

    page_norm = normalize(page_text).casefold()
    quote_norm = normalize(quote).casefold()
    if quote_norm not in page_norm:
        return None, "quote_not_on_page"          # the AI invented or altered a quote
    if printed.casefold().replace(" ", "") not in quote_norm.replace(" ", ""):
        return None, "value_not_in_quote"         # the number isn't actually in that sentence
    if dataset.casefold() not in quote_norm:
        return None, "dataset_not_in_quote"
    value = parse_value(printed)
    if value is None:
        return None, "value_unparseable"
    return Claim(metric, dataset, printed, value, Evidence(page_no, quote)), None


def validate_all(proposals: list[tuple[int, dict]], pages: list[str]):
    """Check every (page_no, proposal). Returns (accepted_claims, rejections)."""
    accepted: list[Claim] = []
    rejected: list[dict] = []
    seen: set[tuple] = set()
    for page_no, proposal in proposals:
        claim, reason = validate_proposal(proposal, page_no, pages[page_no - 1])
        if claim is None:
            rejected.append({"page": page_no, "proposal": proposal, "reason": reason})
            continue
        key = (claim.metric, claim.dataset.casefold(), claim.value)
        if key not in seen:                       # same number stated twice = one claim
            seen.add(key)
            accepted.append(claim)
    return accepted, rejected


def pick_main_claim(claims: list[Claim]) -> Claim | None:
    """Auto-pick the paper's OWN result (skip baselines / prior work)."""
    for claim in claims:
        if not any(word in claim.evidence.quote.casefold() for word in BASELINE_WORDS):
            return claim
    return claims[0] if claims else None
