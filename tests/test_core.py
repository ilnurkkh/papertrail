import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from papertrail import llm, plan as planmod
from papertrail.compare import classify, compare_metric
from papertrail.models import Evidence, Origin, PlanField, Status
from papertrail.report import verify_ledger, write_ledger
from papertrail.runner import safe_command
from papertrail.validate import parse_value, validate_proposal

PAGE = "Our model achieves 96.2% accuracy on the Digits dataset. Seed 42 was used."
GOOD = {"metric_name": "accuracy", "value": "96.2%", "dataset": "Digits",
        "quote": "Our model achieves 96.2% accuracy on the Digits dataset."}


class Validators(unittest.TestCase):
    def test_good_claim_accepted(self):
        claim, reason = validate_proposal(GOOD, 1, PAGE)
        self.assertIsNone(reason)
        self.assertEqual(claim.value, 96.2)

    def test_invented_quote_rejected(self):
        bad = {**GOOD, "quote": "Our model reaches 99.9% accuracy on Digits.", "value": "99.9%"}
        self.assertEqual(validate_proposal(bad, 1, PAGE)[1], "quote_not_on_page")

    def test_wrong_value_rejected(self):
        self.assertEqual(validate_proposal({**GOOD, "value": "98.7%"}, 1, PAGE)[1], "value_not_in_quote")

    def test_missing_fields_rejected(self):
        self.assertEqual(validate_proposal({"quote": "x"}, 1, PAGE)[1], "missing_fields")

    def test_units(self):
        self.assertEqual(parse_value("0.962"), 96.2)
        self.assertEqual(parse_value("96.2 percent"), 96.2)
        self.assertEqual(parse_value("96.2%"), 96.2)


class CompareTests(unittest.TestCase):
    def test_match(self):
        self.assertIs(classify([compare_metric("a", 96.2, 96.0, 1.0)], True), Status.REPRODUCED)

    def test_mismatch(self):
        self.assertIs(classify([compare_metric("a", 99.5, 96.0, 1.0)], True), Status.NOT_REPRODUCED)

    def test_deviation_makes_partial(self):
        self.assertIs(classify([compare_metric("a", 96.2, 96.0, 1.0)], True, True), Status.PARTIAL)

    def test_failed_run_is_unable_not_mismatch(self):
        self.assertIs(classify([compare_metric("a", 96.2, None, 1.0)], False), Status.UNABLE)

    def test_boundary_is_inclusive(self):
        self.assertTrue(compare_metric("a", 95.0, 96.0, 1.0).within_tolerance)


class PlanTests(unittest.TestCase):
    def test_stated_needs_evidence(self):
        with self.assertRaises(ValueError):
            PlanField("42", Origin.STATED)

    def test_missing_seed_detected(self):
        self.assertEqual(planmod.find_seed(["no seeds here"]).origin, Origin.MISSING)

    def test_seed_found_with_evidence(self):
        field = planmod.find_seed(["We use random seed 42."])
        self.assertEqual((field.value, field.origin), ("42", Origin.STATED))

    def test_editing_plan_after_approval_blocks_run(self):
        plan = {"seed": PlanField("42", Origin.STATED, [Evidence(1, "seed 42")])}
        approval = planmod.approval_record(plan, "human")
        self.assertTrue(planmod.is_approved(plan, approval))
        plan["seed"] = PlanField("7", Origin.STATED, [Evidence(1, "seed 42")])
        self.assertFalse(planmod.is_approved(plan, approval))


class RunnerSafety(unittest.TestCase):
    def test_path_traversal_rejected(self):
        with self.assertRaises(ValueError):
            safe_command("../papertrail/models.py", "1", Path("m.json"))

    def test_bad_seed_rejected(self):
        with self.assertRaises(ValueError):
            safe_command("run.py", "1; rm -rf /", Path("m.json"))


class LedgerTests(unittest.TestCase):
    def test_tamper_detected(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "a.txt").write_text("hello")
            write_ledger(d, ["a.txt"])
            self.assertEqual(verify_ledger(d), [])
            (d / "a.txt").write_text("changed")
            self.assertEqual(verify_ledger(d), ["a.txt: CONTENT CHANGED"])


class LLMTests(unittest.TestCase):
    def test_regex_finds_claims(self):
        self.assertEqual(len(llm.regex_propose(1, PAGE)), 1)

    def test_bad_json_retries_then_gives_up(self):
        calls = []
        def chat(messages):
            calls.append(1)
            return "not json"
        self.assertEqual(llm.local_propose(1, PAGE, chat=chat, max_retries=2), [])
        self.assertEqual(len(calls), 3)

    def test_fenced_json_is_accepted(self):
        out = llm.local_propose(1, PAGE, chat=lambda m: "```json\n" + json.dumps({"claims": [GOOD]}) + "\n```")
        self.assertEqual(out[0]["value"], "96.2%")

    def test_ollama_http_path_with_mock_server(self):
        seen = {}
        class H(BaseHTTPRequestHandler):
            def do_POST(self):
                seen["path"] = self.path
                seen["body"] = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                body = json.dumps({"message": {"content": json.dumps({"claims": [GOOD]})}}).encode()
                self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers()
                self.wfile.write(body)
            def log_message(self, *a): pass
        server = HTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        import os
        os.environ["PT_OLLAMA_URL"] = f"http://127.0.0.1:{server.server_port}"
        try:
            self.assertEqual(llm.local_propose(1, PAGE)[0]["dataset"], "Digits")
        finally:
            server.shutdown(); server.server_close()
            del os.environ["PT_OLLAMA_URL"]
        self.assertEqual(seen["path"], "/api/chat")
        self.assertIs(seen["body"]["think"], False)
        self.assertEqual(seen["body"]["options"]["temperature"], 0)
        self.assertEqual(seen["body"]["model"], "qwen3.5:4b")
        self.assertIn("claims", seen["body"]["format"]["properties"])

    def test_ollama_down_falls_back_to_regex(self):
        import os
        os.environ["PT_OLLAMA_URL"] = "http://127.0.0.1:9"
        try:
            proposals, mode, warning = llm.propose_all([PAGE], "local")
        finally:
            del os.environ["PT_OLLAMA_URL"]
        self.assertEqual(mode, "regex")
        self.assertIsNotNone(warning)
        self.assertEqual(len(proposals), 1)


class EndToEnd(unittest.TestCase):
    def test_all_four_samples(self):
        from papertrail.pipeline import run_pipeline
        root = Path(__file__).resolve().parent.parent / "samples"
        expected = {"1_paper_match.pdf": Status.REPRODUCED, "2_paper_inflated.pdf": Status.NOT_REPRODUCED,
                    "3_paper_injection.pdf": Status.REPRODUCED, "4_paper_no_seed.pdf": Status.PARTIAL}
        import contextlib, io
        for name, status in expected.items():
            with contextlib.redirect_stdout(io.StringIO()):
                result = run_pipeline(str(root / name), "regex", auto_yes=True)
            self.assertIs(result["status"], status, name)

    def test_non_pdf_is_unable(self):
        from papertrail.pipeline import run_pipeline
        import contextlib, io
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(b"this is not a pdf")
        with contextlib.redirect_stdout(io.StringIO()):
            result = run_pipeline(f.name, "regex", auto_yes=True)
        self.assertIs(result["status"], Status.UNABLE)


if __name__ == "__main__":
    unittest.main()
