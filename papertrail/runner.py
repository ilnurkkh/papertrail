"""Step 4: run the experiment in a restricted subprocess. Never through a shell.

HONEST LIMIT: this is process-level isolation (clean env, own folder, timeout, size limits).
A production version would run inside a rootless Podman/Docker container with no network.
"""
from __future__ import annotations

import json
import math
import re
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXPERIMENT_DIR = PROJECT_ROOT / "experiment"
TIMEOUT_S = 120
MAX_LOG = 100_000


def _limits():  # runs in the child process (Linux/macOS only)
    import resource
    resource.setrlimit(resource.RLIMIT_CPU, (TIMEOUT_S, TIMEOUT_S))
    resource.setrlimit(resource.RLIMIT_FSIZE, (10_000_000, 10_000_000))


def safe_command(script: str, seed: str, out_file: Path) -> list[str]:
    """Build the command as a LIST (no shell). Reject anything outside experiment/."""
    path = (EXPERIMENT_DIR / script).resolve()
    if EXPERIMENT_DIR not in path.parents or not path.is_file():
        raise ValueError("Script must live inside experiment/")
    if not re.fullmatch(r"\d{1,9}", str(seed)):
        raise ValueError("Seed must be a plain number")
    return [sys.executable, str(path), "--seed", str(seed), "--out", str(out_file)]


def run_experiment(seed: str, run_dir: Path, script: str = "run.py") -> dict:
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_file = run_dir / "metrics.json"
    cmd = safe_command(script, seed, metrics_file)
    outcome = "succeeded"
    try:
        done = subprocess.run(
            cmd, cwd=run_dir, capture_output=True, text=True, timeout=TIMEOUT_S,
            env={"PATH": "/usr/bin:/bin", "PYTHONHASHSEED": "0", "OMP_NUM_THREADS": "1"},
            preexec_fn=_limits, check=False,
        )
        stdout, stderr = done.stdout, done.stderr
        if done.returncode != 0:
            outcome = "nonzero_exit"
    except subprocess.TimeoutExpired as error:
        stdout, stderr, outcome = (error.stdout or ""), (error.stderr or ""), "timeout"
        stdout = stdout if isinstance(stdout, str) else stdout.decode(errors="replace")
        stderr = stderr if isinstance(stderr, str) else stderr.decode(errors="replace")
    (run_dir / "stdout.txt").write_text(stdout[-MAX_LOG:])
    (run_dir / "stderr.txt").write_text(stderr[-MAX_LOG:])

    metric = None
    if outcome == "succeeded":
        metric = read_metric(metrics_file)
        if metric is None:
            outcome = "no_metric_found"
    return {"outcome": outcome, "command": cmd[:1] + ["<script>"] + cmd[2:4], "measured": metric}


def read_metric(metrics_file: Path, key: str = "accuracy") -> float | None:
    """The measured number is read by plain code. The AI is never asked."""
    try:
        value = float(json.loads(metrics_file.read_text())[key])
    except (OSError, ValueError, KeyError, TypeError):
        return None
    return None if math.isnan(value) or math.isinf(value) else value
