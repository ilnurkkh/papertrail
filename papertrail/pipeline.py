"""The whole story in one function: PDF in -> verified report out."""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

from . import llm, plan as planmod
from .compare import classify, compare_metric
from .ingest import IngestError, read_pdf
from .models import Status, to_jsonable
from .report import render_report, write_ledger
from .runner import run_experiment
from .validate import pick_main_claim, validate_all

WORKSPACE = Path(__file__).resolve().parent.parent / "workspace"
C = {"g": "\033[92m", "r": "\033[91m", "y": "\033[93m", "b": "\033[96m", "d": "\033[2m", "x": "\033[0m", "B": "\033[1m"}


def say(color: str, text: str) -> None:
    print(f"{C[color]}{text}{C['x']}")


def step(n: int, title: str) -> None:
    print()
    say("B", f"━━ Step {n}: {title}")


def save(run_dir: Path, name: str, data) -> None:
    (run_dir / name).write_text(json.dumps(to_jsonable(data), indent=2))


def run_pipeline(pdf: str, llm_mode: str | None = None, auto_yes: bool = False, claim_index: int | None = None) -> dict:
    pdf_path = Path(pdf)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = WORKSPACE / f"{stamp}_{re.sub(r'[^A-Za-z0-9]+', '-', pdf_path.stem)}"
    run_dir.mkdir(parents=True, exist_ok=True)
    result = {"run_dir": run_dir, "paper": pdf_path.name}
    ctx = dict(paper=pdf_path.name, claim=None, plan=None, comparison=None, outcome={}, rejected=[],
               llm_mode=llm_mode or llm.default_mode(), approval=None)

    def finish(status: Status, reason: str) -> dict:
        say("B" if status is not Status.NOT_REPRODUCED else "r", f"\n  VERDICT: {status.value.upper()}")
        say("d", f"  {reason}")
        report = render_report(status=status, reason=reason, **ctx)
        (run_dir / "report.md").write_text(report)
        write_ledger(run_dir, [str(p.relative_to(run_dir)) for p in sorted(run_dir.rglob("*"))
                               if p.is_file() and p.name != "ledger.json"])
        say("d", f"  Report: {run_dir / 'report.md'}")
        result.update(status=status, reason=reason)
        return result

    say("b", f"\n▶ Paper: {pdf_path.name}")

    # 1. INGEST
    step(1, "Read the PDF")
    try:
        pages = read_pdf(pdf_path)
    except IngestError as error:
        say("r", f"  ✘ {error}")
        return finish(Status.UNABLE, str(error))
    (run_dir / "pages").mkdir(exist_ok=True)
    for i, text in enumerate(pages, 1):
        (run_dir / "pages" / f"{i:04d}.txt").write_text(text)
    say("g", f"  ✔ {len(pages)} pages of text saved")

    # 2. PROPOSE (AI) then VERIFY (code)
    step(2, "AI proposes claims, code verifies each quote")
    proposals, used_mode, warning = llm.propose_all(pages, ctx["llm_mode"])
    ctx["llm_mode"] = used_mode
    if warning:
        say("y", f"  ⚠ {warning}")
    model = f" ({llm.local_settings()['model']}, running locally)" if used_mode == "local" else ""
    say("d", f"  Extraction mode: {used_mode}{model}; the AI proposed {len(proposals)} claim(s)")
    claims, rejected = validate_all(proposals, pages)
    ctx["rejected"] = rejected
    for c in claims:
        say("g", f"  ✔ ACCEPTED  {c.metric} {c.printed_value} on {c.dataset}  (page {c.evidence.page})")
    for r in rejected:
        say("r", f"  ✘ REJECTED  [{r['reason']}]  \"{r['proposal'].get('quote', '?')[:80]}\"")
    save(run_dir, "claims.json", claims)
    save(run_dir, "rejected.json", rejected)
    if not claims:
        return finish(Status.UNABLE, "No claim could be verified against the paper text.")
    claim = claims[claim_index - 1] if claim_index else pick_main_claim(claims)
    ctx["claim"] = claim
    say("b", f"  → Selected: {claim.metric} = {claim.printed_value} on {claim.dataset} (baselines/prior work skipped)")
    if (claim.metric, claim.dataset.casefold()) not in planmod.SUPPORTED:
        return finish(Status.UNABLE, f"This demo can only run 'accuracy on Digits'; no experiment for {claim.metric} on {claim.dataset}.")

    # 3. PLAN (every field labelled)
    step(3, "Build the experiment plan (STATED / ASSUMED / MISSING)")
    plan = planmod.build_plan(claim, pages)
    colors = {"stated": "g", "assumed": "y", "missing": "r"}
    for name, f in plan.items():
        say(colors[f.origin.value], f"  {f.origin.value.upper():8} {name:15} = {f.value}")
    ctx["plan"] = plan

    # 4. APPROVAL GATE
    step(4, "Human approval gate")
    for name in planmod.missing_fields(plan):
        if auto_yes:
            planmod.fill_missing(plan, name, "0")
            say("y", f"  ⚠ '{name}' missing in paper. Filled with 0 and marked ASSUMED (--yes mode).")
        else:
            value = input(f"  '{name}' is MISSING from the paper. Enter a value (or blank to abort): ").strip()
            if not value:
                return finish(Status.UNABLE, f"Plan has a missing field ({name}); a human must supply it.")
            planmod.fill_missing(plan, name, value)
    if auto_yes:
        approval = planmod.approval_record(plan, "auto (--yes demo mode)")
    else:
        if input("  Type 'yes' to approve running this plan: ").strip().lower() != "yes":
            return finish(Status.UNABLE, "Plan was not approved; nothing was run.")
        approval = planmod.approval_record(plan, "human")
    save(run_dir, "plan.json", plan)
    save(run_dir, "approval.json", approval)
    ctx["approval"] = approval
    if not planmod.is_approved(plan, approval):  # the runner re-checks this itself
        return finish(Status.UNABLE, "Approval does not match plan.")
    say("g", f"  ✔ Approved. Plan hash {approval['plan_sha256'][:16]}...")

    # 5. RUN
    step(5, "Run the experiment in a restricted subprocess")
    outcome = run_experiment(plan["seed"].value, run_dir / "run")
    ctx["outcome"] = outcome
    save(run_dir, "outcome.json", outcome)
    if outcome["outcome"] != "succeeded":
        say("r", f"  ✘ Run outcome: {outcome['outcome']}")
        return finish(Status.UNABLE, f"The run did not produce a trustworthy number ({outcome['outcome']}).")
    say("g", f"  ✔ Run finished. Measured accuracy = {outcome['measured']}%")

    # 6. COMPARE (no AI)
    step(6, "Compare with the paper (plain code, no AI)")
    comparison = compare_metric("accuracy", claim.value, outcome["measured"], planmod.TOLERANCE_POINTS)
    ctx["comparison"] = comparison
    save(run_dir, "comparison.json", comparison)
    say("b", f"  Reported {comparison.reported}%  vs  measured {comparison.measured}%  (difference {comparison.difference}, allowed ±{comparison.allowed})")
    status = classify([comparison], True, planmod.has_deviations(plan))
    reason = {
        Status.REPRODUCED: "The measured value is within the tolerance that was fixed before the run.",
        Status.PARTIAL: "The value matches, but a human had to fill a detail the paper never stated (see plan).",
        Status.NOT_REPRODUCED: "The run was valid but the measured value is outside the tolerance.",
        Status.UNABLE: "No trustworthy comparison was possible.",
    }[status]
    return finish(status, reason)
