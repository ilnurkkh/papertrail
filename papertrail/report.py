"""Step 6: the evidence-linked report and the hash ledger (tamper-evident, not tamper-proof)."""
from __future__ import annotations

import json
from pathlib import Path

from .models import Status, sha256_file, to_jsonable


def write_ledger(run_dir: Path, files: list[str]) -> dict:
    entries = {name: sha256_file(run_dir / name) for name in files if (run_dir / name).exists()}
    ledger = {"note": "SHA-256 of every artifact. Change any file and `verify` will fail.", "files": entries}
    (run_dir / "ledger.json").write_text(json.dumps(ledger, indent=2))
    return ledger


def verify_ledger(run_dir: Path) -> list[str]:
    """Return a list of problems (empty list = everything matches)."""
    ledger_path = run_dir / "ledger.json"
    if not ledger_path.exists():
        return ["ledger.json is missing"]
    problems = []
    for name, expected in json.loads(ledger_path.read_text())["files"].items():
        path = run_dir / name
        if not path.exists():
            problems.append(f"{name}: file missing")
        elif sha256_file(path) != expected:
            problems.append(f"{name}: CONTENT CHANGED")
    return problems


def render_report(*, paper: str, status: Status, reason: str, claim, plan, comparison, outcome: dict,
                  rejected: list[dict], llm_mode: str, approval: dict | None) -> str:
    L = [f"# PaperTrail report: {paper}", "", f"## Verdict: **{status.value}**", "", f"_{reason}_", ""]
    if claim:
        L += ["## 1. The claim (from the paper)",
              f"- **{claim.metric}** = **{claim.printed_value}** on **{claim.dataset}**",
              f"- Evidence: page {claim.evidence.page}: \"{claim.evidence.quote}\"", ""]
    if plan:
        L += ["## 2. Experiment plan (every field says where it came from)", "",
              "| Field | Value | Origin | Evidence |", "|---|---|---|---|"]
        for name, f in plan.items():
            ev = "; ".join(f"p{e.page}: \"{e.quote}\"" for e in f.evidence) or "-"
            L.append(f"| {name} | {f.value if f.value is not None else '(missing)'} | {f.origin.value.upper()} | {ev} |")
        L += ["", f"Approved by: **{approval['approved_by']}**, bound to plan hash `{approval['plan_sha256'][:16]}...`"
              if approval else "Not approved: nothing was run.", ""]
    if comparison:
        L += ["## 3. Result (measured by ordinary code, not by the AI)", "",
              "| Reported | Measured | Difference | Allowed | Within tolerance |", "|---|---|---|---|---|",
              f"| {comparison.reported} | {comparison.measured} | {comparison.difference} | "
              f"+/-{comparison.allowed} | {comparison.within_tolerance} |", ""]
    L += ["## 4. Run", f"- Outcome: `{outcome.get('outcome', 'not run')}`", ""]
    L += ["## 5. What the AI got wrong (caught by validators)", ""]
    if rejected:
        for r in rejected:
            L.append(f"- REJECTED `{r['reason']}` (page {r['page']}): \"{r['proposal'].get('quote', '?')}\"")
    else:
        L.append("- Nothing rejected in this run.")
    L += ["", f"Claim extraction mode: `{llm_mode}`" + (" (open-source model on the local machine, no data sent anywhere)." if llm_mode == "local" else "."), "",
          "## Honest limits",
          "- Matching one number does not prove a whole paper is right; a miss may come from seeds or library versions.",
          "- Process-level isolation only (timeout, clean environment); production would use a Podman container.",
          "- The hash ledger is tamper-evident, not tamper-proof."]
    return "\n".join(L) + "\n"
