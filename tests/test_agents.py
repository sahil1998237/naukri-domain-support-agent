"""
Automated Test Suite for Day 4: Status Tool (Task 6) and MockLLM (Task 7 Setup).

Validates:
1. Continuous escalation score formula produces values strictly within [0.0, 1.0].
2. Recency and priority flag signals contribute expected continuous increments.
3. 80th percentile threshold is dynamically computed from dataset.
4. check_job_application_status returns correct record data and recommendations.
5. Missing record IDs are handled gracefully with found=False.
6. MockLLM subclasses BaseLLM and instantiates without errors.
7. MockLLM dispatches tool actions by argument schema (e.g. APP-XXX).
8. MockLLM synthesizes Final Answer after real tool observation.
9. MockLLM tracks per-request token usage.
10. CREWAI_DISABLE_TELEMETRY is confirmed active.
"""

import os
from pathlib import Path
import sys
import unittest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.mock_llm import MockLLM
from agents.tools import (
    DEFAULT_ESCALATION_THRESHOLD,
    calculate_escalation_score,
    check_job_application_status,
    compute_80th_percentile_threshold,
)


class TestDay4AgentsAndTools(unittest.TestCase):
    """Test suite for Escalation Tool (Task 6) and MockLLM (Task 7 Setup)."""

    def test_01_escalation_score_formula_bounds(self):
        """Check 1: Escalation scores are strictly within [0.0, 1.0]."""
        test_cases = [
            (False, 0),
            (False, 15),
            (False, 30),
            (False, 60),
            (True, 0),
            (True, 15),
            (True, 30),
            (True, 90),
        ]
        for flagged, days in test_cases:
            score = calculate_escalation_score(flagged, days)
            self.assertGreaterEqual(score, 0.0, f"Score {score} < 0.0 for flagged={flagged}, days={days}")
            self.assertLessEqual(score, 1.0, f"Score {score} > 1.0 for flagged={flagged}, days={days}")

    def test_02_escalation_score_continuous_gradations(self):
        """Check 2: Escalation scores provide continuous, expected signal increments."""
        # Unflagged: 0 days -> 0.0, 15 days -> 0.25, 30 days -> 0.50
        self.assertEqual(calculate_escalation_score(False, 0), 0.0)
        self.assertEqual(calculate_escalation_score(False, 15), 0.25)
        self.assertEqual(calculate_escalation_score(False, 30), 0.50)

        # Flagged: 0 days -> 0.50, 15 days -> 0.75, 30 days -> 1.00
        self.assertEqual(calculate_escalation_score(True, 0), 0.50)
        self.assertEqual(calculate_escalation_score(True, 15), 0.75)
        self.assertEqual(calculate_escalation_score(True, 30), 1.00)

    def test_03_80th_percentile_threshold_calculated(self):
        """Check 3: Empirical 80th percentile threshold is derived from data."""
        threshold = compute_80th_percentile_threshold()
        self.assertGreater(threshold, 0.40)
        self.assertLess(threshold, 0.80)
        self.assertEqual(threshold, DEFAULT_ESCALATION_THRESHOLD)

    def test_04_check_job_application_status_valid(self):
        """Check 4: Valid record lookup returns full details and escalation recommendation."""
        # APP-003 is flagged=True, days=37 -> score=1.00 -> should escalate
        result = check_job_application_status("APP-003")
        self.assertTrue(result["found"])
        self.assertEqual(result["record_id"], "APP-003")
        self.assertEqual(result["status"], "Interview Scheduled")
        self.assertEqual(result["escalation_score"], 1.0)
        self.assertTrue(result["escalation_recommended"])
        self.assertIn("ESCALATION RECOMMENDED", result["summary"])

        # APP-001 is flagged=False, days=9 -> score=0.15 -> should not escalate
        res_normal = check_job_application_status("APP-001")
        self.assertTrue(res_normal["found"])
        self.assertFalse(res_normal["escalation_recommended"])
        self.assertIn("Normal Processing", res_normal["summary"])

    def test_05_check_job_application_status_not_found(self):
        """Check 5: Nonexistent record ID returns found=False and informative error."""
        result = check_job_application_status("APP-999")
        self.assertFalse(result["found"])
        self.assertIn("not found", result["error"])
        self.assertFalse(result["escalation_recommended"])

    def test_06_mock_llm_instantiation(self):
        """Check 6: MockLLM instantiates with default model name and methods."""
        mock = MockLLM()
        self.assertEqual(mock.model, "mock-naukri-support-llm")
        self.assertEqual(mock.get_context_window_size(), 8192)
        self.assertTrue(mock.supports_stop_words())

    def test_07_mock_llm_tool_dispatch(self):
        """Check 7: MockLLM dispatches check_job_application_status for record ID query."""
        mock = MockLLM()
        prompt = "Candidate asks: What is the status of my application APP-003?"
        response = mock.call(prompt)

        self.assertIn("Action: check_job_application_status", response)
        self.assertIn('"record_id": "APP-003"', response)

    def test_08_mock_llm_observation_synthesis(self):
        """Check 8: MockLLM parses observation and generates Final Answer."""
        mock = MockLLM()
        prompt = (
            "Action: check_job_application_status\n"
            "Action Input: {\"record_id\": \"APP-003\"}\n"
            "Observation: Application APP-003 is currently Interview Scheduled.\n"
            "Thought: I have the status.\n"
        )
        response = mock.call(prompt)
        self.assertIn("Final Answer:", response)
        self.assertIn("APP-003", response)

    def test_09_mock_llm_token_tracking(self):
        """Check 9: MockLLM accumulates token counts and tracks request count."""
        mock = MockLLM()
        self.assertEqual(mock.total_tokens, 0)
        self.assertEqual(mock.request_count, 0)

        mock.call("What is the status of APP-001?")
        self.assertGreater(mock.total_tokens, 0)
        self.assertEqual(mock.request_count, 1)

    def test_10_telemetry_disabled(self):
        """Check 10: CREWAI_DISABLE_TELEMETRY environment variable is active."""
        self.assertEqual(os.environ.get("CREWAI_DISABLE_TELEMETRY"), "true")


if __name__ == "__main__":
    unittest.main(verbosity=2)
