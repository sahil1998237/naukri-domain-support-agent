"""
Automated Test Suite for Grounded Generation (Task 4) and Strategy Evaluation (Task 5).

Validates:
1. In-scope queries produce grounded responses (is_grounded = True).
2. Out-of-scope queries are rejected with fallback text (is_grounded = False).
3. The empirical similarity threshold of 0.40 correctly separates in-scope from noise.
4. The comparative evaluation suite runs and confirms 100% target document recall.
"""

from pathlib import Path
import sys
import unittest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag.evaluation import run_full_evaluation
from rag.retriever import CALIBRATED_SIMILARITY_THRESHOLD, generate_grounded_response


class TestGroundedGenerationAndEvaluation(unittest.TestCase):
    """Test suite verifying Task 4 and Task 5 requirements."""

    def test_01_in_scope_query_grounded(self):
        """Check 1: In-scope interview reschedule query produces grounded answer with POL-003 citation."""
        query = "What is the policy for rescheduling an interview?"
        res = generate_grounded_response(query, strategy="sentence")

        self.assertTrue(res["is_grounded"], f"Expected query to be grounded, got {res}")
        self.assertEqual(res["doc_id"], "POL-003")
        self.assertGreaterEqual(res["similarity"], CALIBRATED_SIMILARITY_THRESHOLD)
        self.assertIn("POL-003", res["answer"])
        self.assertIn("Interview Scheduling", res["answer"])

    def test_02_out_of_scope_query_rejected(self):
        """Check 2: Out-of-scope weather query is rejected without hallucination."""
        query = "What is the weather forecast for Bangalore today?"
        res = generate_grounded_response(query, strategy="sentence")

        self.assertFalse(res["is_grounded"], f"Expected query to be rejected as out-of-scope, got {res}")
        self.assertIsNone(res["doc_id"])
        self.assertLess(res["similarity"], CALIBRATED_SIMILARITY_THRESHOLD)
        self.assertIn("I do not have sufficient policy information", res["answer"])

    def test_03_out_of_scope_unrelated_rejected(self):
        """Check 3: Out-of-scope pet in office query is rejected."""
        query = "Can I bring my pet dog to the office on Fridays?"
        res = generate_grounded_response(query, strategy="sentence")

        self.assertFalse(res["is_grounded"])
        self.assertIsNone(res["doc_id"])
        self.assertLess(res["similarity"], CALIBRATED_SIMILARITY_THRESHOLD)

    def test_04_calibrated_threshold_sanity(self):
        """Check 4: Calibrated threshold is strictly set between 0.35 and 0.45."""
        self.assertGreaterEqual(CALIBRATED_SIMILARITY_THRESHOLD, 0.35)
        self.assertLessEqual(CALIBRATED_SIMILARITY_THRESHOLD, 0.45)
        self.assertEqual(CALIBRATED_SIMILARITY_THRESHOLD, 0.40)

    def test_05_comparative_evaluation_metrics(self):
        """Check 5: Both chunking strategies achieve 100% recall on target documents."""
        eval_data = run_full_evaluation()

        self.assertEqual(
            eval_data["sentence"]["avg_recall"],
            1.0,
            "Sentence strategy should achieve 100% recall on target documents",
        )
        self.assertEqual(
            eval_data["fixed"]["avg_recall"],
            1.0,
            "Fixed-size strategy should achieve 100% recall on target documents",
        )
        self.assertGreaterEqual(
            eval_data["sentence"]["avg_precision"],
            0.70,
            "Sentence strategy precision should be >= 0.70",
        )
        self.assertTrue(len(eval_data["recommendation"]) > 50)


if __name__ == "__main__":
    unittest.main(verbosity=2)
