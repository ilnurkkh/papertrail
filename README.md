# PaperTrail

Give it a research paper PDF. A **local open-source model (Qwen via Ollama, on your own machine)** reads it and
proposes the headline result; ordinary code verifies it, runs the experiment, and gives an honest verdict.
**The AI proposes. Ordinary code verifies.** Nothing leaves your computer.

## Setup (Fedora / Linux laptop)
```bash
ollama serve &                 # skip if Ollama already runs as a service
ollama pull qwen3.5:4b         # one-time, ~3.4 GB
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m papertrail llm-check # should print: Model replied: {"ok": true}
```

## Run
```bash
python -m papertrail demo                                        # all 4 sample papers, local model
python -m papertrail run samples/2_paper_inflated.pdf --yes      # one paper
python -m papertrail run samples/2_paper_inflated.pdf --llm liar --yes   # fake lying AI; validators catch it
python -m papertrail run samples/1_paper_match.pdf --llm regex --yes     # no AI at all (baseline)
python -m papertrail verify workspace/<run-folder>               # tamper check
python -m unittest discover -s tests                             # 24 tests
```
Use another local model: `export PT_MODEL=phi4-mini` (any model you have pulled).
If Ollama is not running, the tool warns and falls back to regex mode so the demo never dies.

## Pipeline
PDF -> text per page -> local LLM proposes claims -> code checks every quote is on the page -> plan labelled
STATED / ASSUMED / MISSING -> human approval bound to a plan hash -> restricted run -> plain-code comparison
-> report.md + SHA-256 ledger.

## Honest limits
Only "accuracy on Digits" is supported. Isolation is process-level, not a container. Sample papers are synthetic.
