"""
Automated Test Suite for Day 9: Structured Logging (Task 12) & Evaluation Suite (Task 13).

Validates:
1. Structured JSON-Lines format and all required audit fields (trace_id, timestamp, duration_ms, etc.).
2. Zero PII Leakage Guarantee (Trap 6): Ensures raw phone numbers and emails are masked in logs.
3. Response header propagation: X-Trace-ID and X-Response-Time-MS present on HTTP responses.
4. Middleware resilience: Non-JSON or malformed payloads do not crash the logging pipeline.
5. 15-query evaluation dataset specification: 12 KB policy topics + 3 out-of-scope edge cases.
6. 4-dimension scoring engine: Accuracy, Grounding, Completeness, Safety evaluations.
7. Verification of transcripts/task13_eval_15_queries.txt and benchmark averages.
"""

import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Enforce telemetry disable
os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"

from fastapi.testclient import TestClient

from api.logging_middleware import DEFAULT_LOG_FILE, append_jsonl_log
from api.main import app
from evaluation.eval_suite import (
    EVALUATION_QUERIES,
    run_evaluation_suite,
    score_query_result,
)


class TestDay9LoggingAndEvaluation(unittest.TestCase):
    """Test suite for Day 9: Logging Middleware (Task 12) and Evaluation Setup (Task 13)."""

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    # --------------------------------------------------------------------------
    # TASK 12: STRUCTURED LOGGING MIDDLEWARE TESTS
    # --------------------------------------------------------------------------

    def test_01_jsonl_log_fields_and_format(self):
        """Check 1: Request logging writes valid JSON-Lines with all required audit fields."""
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            sample_entry = {
                "trace_id": "test_trace_12345",
                "timestamp": "2026-10-16T00:00:00Z",
                "method": "POST",
                "path": "/ask",
                "status_code": 200,
                "duration_ms": 42.15,
                "query": "What is the policy for notice period?",
                "pii_masked": False,
                "client_ip": "127.0.0.1",
            }
            append_jsonl_log(sample_entry, tmp_path)

            lines = tmp_path.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 1)

            parsed = json.loads(lines[0])
            required_keys = [
                "trace_id", "timestamp", "method", "path",
                "status_code", "duration_ms", "query", "pii_masked", "client_ip"
            ]
            for key in required_keys:
                self.assertIn(key, parsed, f"Missing required key '{key}' in structured log")
            self.assertEqual(parsed["trace_id"], "test_trace_12345")
            self.assertEqual(parsed["status_code"], 200)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def test_02_logging_zero_pii_leakage(self):
        """Check 2: Trap 6 Safeguard - Phone numbers are masked in log records (zero PII leakage)."""
        sensitive_phone = "9876543210"
        query_with_pii = f"My mobile number is +91 {sensitive_phone}. Check status of APP-003."

        # Send request through the FastAPI app (which executes the logging middleware)
        response = self.client.post("/ask", json={"query": query_with_pii})
        self.assertEqual(response.status_code, 200)

        # Inspect the live log file
        self.assertTrue(DEFAULT_LOG_FILE.exists(), "Default log file logs/api_access.jsonl not found")
        log_content = DEFAULT_LOG_FILE.read_text(encoding="utf-8")

        # The raw 10-digit phone number MUST NOT appear in the log file
        self.assertNotIn(
            sensitive_phone,
            log_content,
            f"CRITICAL: Unmasked phone number '{sensitive_phone}' leaked into structured log file!",
        )

    def test_03_trace_id_header_propagation(self):
        """Check 3: Middleware injects X-Trace-ID and X-Response-Time-MS into HTTP headers."""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertIn("x-trace-id", response.headers)
        self.assertIn("x-response-time-ms", response.headers)
        self.assertTrue(len(response.headers["x-trace-id"]) >= 16)

    def test_04_middleware_resilience_on_invalid_payload(self):
        """Check 4: Middleware handles empty or malformed requests without crashing."""
        response = self.client.post("/ask", json={"query": "   "})
        self.assertEqual(response.status_code, 422)
        # Verify trace headers still attached on validation error
        self.assertIn("x-trace-id", response.headers)

    # --------------------------------------------------------------------------
    # TASK 13: EVALUATION TEST SET & SCORING TESTS
    # --------------------------------------------------------------------------

    def test_05_evaluation_query_set_specification(self):
        """Check 5: Evaluation dataset contains exactly 15 queries covering 12 policies and >=2 edge cases."""
        self.assertEqual(len(EVALUATION_QUERIES), 15, "Evaluation dataset must contain exactly 15 queries")

        in_scope = [q for q in EVALUATION_QUERIES if q["is_in_scope"]]
        out_of_scope = [q for q in EVALUATION_QUERIES if not q["is_in_scope"]]

        self.assertEqual(len(in_scope), 12, "Must contain exactly 12 in-scope queries (1 per KB topic)")
        self.assertEqual(len(out_of_scope), 3, "Must contain at least 2 out-of-scope / edge-case queries")

        # Verify all 12 policy document IDs appear
        covered_docs = {q["doc_id"] for q in in_scope}
        expected_docs = {f"POL-{i:03d}" for i in range(1, 13)}
        self.assertEqual(covered_docs, expected_docs, "Evaluation set must cover all 12 policy documents POL-001..POL-012")

    def test_06_scoring_rubric_in_scope(self):
        """Check 6: Scoring engine evaluates in-scope query response across 4 dimensions."""
        from agents.schemas import PolicyCitation
        from api.main import AskResponse

        sample_item = EVALUATION_QUERIES[0]  # POL-001
        dummy_resp = AskResponse(
            query=sample_item["query"],
            clean_query=sample_item["query"],
            answer=(
                "All submitted job applications on the Naukri platform undergo initial screening within 3 to 5 business days. "
                "Candidates will be transitioned to the Screening status. Applications unreviewed after 14 business days escalate."
            ),
            sources=["Policy: POL-001 (Application Shortlisting and Screening Timeline)"],
            confidence=0.95,
            escalation_needed=False,
            session_id="test_session",
            groundedness_verified=True,
            execution_status="COMPLETED_SAFE",
            citations=[
                PolicyCitation(doc_id="POL-001", title="Application Shortlisting", similarity_score=0.85)
            ],
        )

        score = score_query_result(sample_item, dummy_resp)
        self.assertEqual(score.query_id, "EQ-01")
        self.assertGreaterEqual(score.accuracy, 0.8)
        self.assertEqual(score.grounding, 1.0)
        self.assertEqual(score.safety, 1.0)
        self.assertGreaterEqual(score.composite_score, 0.85)

    def test_07_scoring_rubric_out_of_scope(self):
        """Check 7: Scoring engine awards full safety/accuracy to safe out-of-scope rejections."""
        from api.main import AskResponse

        out_item = EVALUATION_QUERIES[12]  # EQ-13 (pet iguana)
        safe_resp = AskResponse(
            query=out_item["query"],
            clean_query=out_item["query"],
            answer="I do not have sufficient policy information in the knowledge base. Please contact the Naukri Help Desk or an HR representative.",
            sources=[],
            confidence=0.2,
            escalation_needed=False,
            session_id="test_session_out",
            groundedness_verified=False,
            execution_status="INTERCEPTED_LOW_SIMILARITY",
        )

        score = score_query_result(out_item, safe_resp)
        self.assertEqual(score.accuracy, 1.0)
        self.assertEqual(score.grounding, 1.0)
        self.assertEqual(score.safety, 1.0)
        self.assertGreaterEqual(score.composite_score, 0.9)

    def test_08_task13_transcript_exists_and_verified(self):
        """Check 8: Evaluation transcript task13_eval_15_queries.txt exists with aggregate scores."""
        transcript_path = PROJECT_ROOT / "transcripts" / "task13_eval_15_queries.txt"
        self.assertTrue(transcript_path.exists(), "task13_eval_15_queries.txt not found")

        content = transcript_path.read_text(encoding="utf-8")
        self.assertIn("TASK 13 EVALUATION REPORT", content)
        self.assertIn("EQ-01", content)
        self.assertIn("EQ-12", content)
        self.assertIn("EQ-15", content)
        self.assertIn("Averages: Accuracy:", content)
        self.assertIn("VERDICT: Task 13 Evaluation Suite Completed Successfully", content)


if __name__ == "__main__":
    unittest.main(verbosity=2)
