"""Command line:  python -m papertrail <command>"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import llm
from .pipeline import C, run_pipeline, say
from .report import verify_ledger

SAMPLES = Path(__file__).resolve().parent.parent / "samples"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="papertrail", description="Check if a paper's result reproduces.")
    sub = parser.add_subparsers(dest="cmd")

    run = sub.add_parser("run", help="Run the full pipeline on one PDF")
    run.add_argument("pdf")
    run.add_argument("--llm", choices=["local", "regex", "liar"], help="claim extraction brain")
    run.add_argument("--yes", action="store_true", help="auto-approve (demo only)")
    run.add_argument("--claim", type=int, help="pick the Nth accepted claim instead of auto-pick")

    demo = sub.add_parser("demo", help="Run all sample papers back to back")
    demo.add_argument("--llm", choices=["local", "regex", "liar"])

    ver = sub.add_parser("verify", help="Check a finished run folder has not been tampered with")
    ver.add_argument("run_dir")

    sub.add_parser("llm-check", help="Test that your local Ollama model answers")

    args = parser.parse_args(argv)

    if args.cmd == "run":
        run_pipeline(args.pdf, args.llm, args.yes, args.claim)
        return 0
    if args.cmd == "demo":
        rows = []
        for pdf in sorted(SAMPLES.glob("*.pdf")):
            res = run_pipeline(str(pdf), args.llm, auto_yes=True)
            rows.append((pdf.name, res["status"].value))
        print()
        say("B", "━━ SUMMARY")
        for name, status in rows:
            print(f"  {name:32} {status}")
        return 0
    if args.cmd == "verify":
        problems = verify_ledger(Path(args.run_dir))
        if problems:
            for p in problems:
                say("r", f"  ✘ {p}")
            return 1
        say("g", "  ✔ Every file matches the ledger. Nothing was changed.")
        return 0
    if args.cmd == "llm-check":
        try:
            cfg = llm.local_settings()
            print(f"Asking {cfg['model']} at {cfg['url']} ...")
            print("Model replied:", llm.local_chat([{"role": "user", "content": 'Reply with the JSON {"ok": true}'}], schema=None))
            return 0
        except llm.LLMUnavailable as error:
            say("r", f"  ✘ {error}")
            return 1
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
