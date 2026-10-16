"""
Automated Test Suite for Day 14: Capstone Integrity Audit & Submission Readiness.

Validates:
1. Master Capstone Audit executes cleanly and reports 100% compliance.
2. All 16 capstone tasks are validated and pass.
3. All 11 mandatory demonstration transcripts exist and are non-empty.
4. Telemetry is strictly disabled (CREWAI_DISABLE_TELEMETRY=true).
5. Zero external API keys are required for full offline local execution.
"""

import os
from pathlib import Path
import sys
import unittest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_capstone_audit import run_capstone_audit


class TestDay14CapstoneIntegrity(unittest.TestCase):
    """Test suite for Day 14 final capstone submission readiness."""

    def test_01_master_audit_clean_execution(self):
        """Check 1: run_capstone_audit executes and reports 100% compliance across all checks."""
        res = run_capstone_audit()
        self.assertTrue(res["is_fully_compliant"], "Capstone audit must be 100% compliant")
        self.assertEqual(res["passed_checks"], res["total_checks"])
        self.assertEqual(res["total_checks"], 16)
        self.assertIn("ALL CHECKS PASSED", res["report_text"])

    def test_02_all_transcripts_verified(self):
        """Check 2: All 11 mandatory demonstration transcripts exist and exceed minimum byte size."""
        required_transcripts = [
            "task4_grounded_gen.txt",
            "task5_eval_comparison.txt",
            "task7_crew_demo.txt",
            "task8_memory_multi_turn.txt",
            "task8_memory_fresh.txt",
            "task10_guardrails.txt",
            "task13_eval_15_queries.txt",
            "task14_review_approve.txt",
            "task14_review_revise.txt",
            "task15_budget_reject.txt",
            "task16_cache_hit.txt",
        ]

        transcripts_dir = PROJECT_ROOT / "transcripts"
        for t_name in required_transcripts:
            t_path = transcripts_dir / t_name
            self.assertTrue(t_path.exists(), f"Missing required transcript: {t_name}")
            self.assertGreater(t_path.stat().st_size, 500, f"Transcript too small: {t_name}")

    def test_03_telemetry_protection(self):
        """Check 3: Telemetry disable flag is strictly active."""
        self.assertEqual(os.environ.get("CREWAI_DISABLE_TELEMETRY"), "true")

    def test_04_zero_external_api_keys_needed(self):
        """Check 4: System operates with MOCK_LLM without external API keys."""
        from agents.mock_llm import MockLLM
        mock_llm = MockLLM()
        res = mock_llm.call(messages="What is the notice period policy?")
        self.assertTrue(len(str(res)) > 0)


if __name__ == "__main__":
    unittest.main()
