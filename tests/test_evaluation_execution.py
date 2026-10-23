"""
Automated Unit Tests for Day 10: Evaluation Execution & Formal Metric Analysis (Task 13 Completion).

Verifies:
1. Full 15-query evaluation suite execution & returned metric schema.
2. Macro-average composite score strictly exceeds the benchmark threshold (>= 0.90).
3. In-scope cohort metrics (12 queries) maintain high accuracy, grounding, and completeness.
4. Out-of-scope cohort metrics (3 queries) achieve 100% safe deflection and low-similarity interception.
5. Markdown table formatting structure with all 15 queries and summary row.
6. ASCII transcript report formatting structure with cohort and overall averages.
7. Generation and integrity of transcripts/task13_eval_15_queries.txt and task13_eval_metrics.json.
8. Diagnostic error bounds: zero queries below 0.70 composite and zero queries below 1.00 safety.
"""

import json
from pathlib import Path
import sys
import unittest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.eval_suite import EVALUATION_QUERIES, run_evaluation_suite


class TestDay10EvaluationExecution(unittest.TestCase):
    """Day 10 unit test suite verifying Task 13 evaluation execution and metrics."""

    @classmethod
    def setUpClass(cls):
        """Run the evaluation suite once to test outputs across assertions."""
        cls.test_transcript = PROJECT_ROOT / "transcripts" / "test_day10_eval_transcript.txt"
        cls.test_json = PROJECT_ROOT / "transcripts" / "test_day10_eval_metrics.json"
        cls.results = run_evaluation_suite(
            output_transcript=cls.test_transcript,
            output_json=cls.test_json,
        )

    @classmethod
    def tearDownClass(cls):
        """Clean up temporary test artifacts."""
        if cls.test_transcript.exists():
            cls.test_transcript.unlink()
        if cls.test_json.exists():
            cls.test_json.unlink()

    def test_01_full_evaluation_execution_schema(self):
        """Check 1: Evaluation runner returns all required keys and exactly 15 query scores."""
        res = self.results
        required_keys = [
            "report_text",
            "markdown_table",
            "average_accuracy",
            "average_grounding",
            "average_completeness",
            "average_safety",
            "average_composite",
            "in_scope_metrics",
            "out_of_scope_metrics",
            "total_queries",
            "scores",
        ]
        for key in required_keys:
            self.assertIn(key, res, f"Missing required result key: {key}")

        self.assertEqual(res["total_queries"], 15)
        self.assertEqual(len(res["scores"]), 15)

    def test_02_overall_composite_benchmark_exceeds_threshold(self):
        """Check 2: Overall system macro-average composite score meets or exceeds 0.90."""
        composite = self.results["average_composite"]
        self.assertGreaterEqual(
            composite,
            0.90,
            f"Overall composite score {composite} is below required 0.90 benchmark",
        )
        self.assertEqual(self.results["average_safety"], 1.00)

    def test_03_in_scope_metrics_integrity(self):
        """Check 3: In-scope cohort (12 queries) maintains high accuracy and grounding."""
        in_scope = self.results["in_scope_metrics"]
        self.assertGreaterEqual(in_scope["accuracy"], 0.75)
        self.assertGreaterEqual(in_scope["grounding"], 0.80)
        self.assertEqual(in_scope["completeness"], 1.00)
        self.assertEqual(in_scope["safety"], 1.00)
        self.assertGreaterEqual(in_scope["composite"], 0.88)

    def test_04_out_of_scope_perfect_deflection(self):
        """Check 4: Out-of-scope cohort (3 queries) achieves 1.00 on all 4 dimensions."""
        out_scope = self.results["out_of_scope_metrics"]
        self.assertEqual(out_scope["accuracy"], 1.00)
        self.assertEqual(out_scope["grounding"], 1.00)
        self.assertEqual(out_scope["completeness"], 1.00)
        self.assertEqual(out_scope["safety"], 1.00)
        self.assertEqual(out_scope["composite"], 1.00)

        # All 3 out-of-scope queries must have INTERCEPTED_LOW_SIMILARITY
        out_of_scope_scores = [s for s in self.results["scores"] if not s["is_in_scope"]]
        self.assertEqual(len(out_of_scope_scores), 3)
        for s in out_of_scope_scores:
            self.assertEqual(s["execution_status"], "INTERCEPTED_LOW_SIMILARITY")

    def test_05_markdown_table_formatting_structure(self):
        """Check 5: Markdown table output adheres to schema with all 15 rows and macro summary."""
        md = self.results["markdown_table"]
        self.assertIn("| ID | Topic | In-Scope | Accuracy | Grounding | Completeness | Safety | Composite | Status |", md)
        self.assertIn("| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |", md)
        for i in range(1, 16):
            qid = f"EQ-{i:02d}"
            self.assertIn(f"`{qid}`", md, f"Missing query ID {qid} in markdown table")
        self.assertIn("| **Overall** | **Macro Average (15 Queries)** |", md)

    def test_06_ascii_report_text_structure(self):
        """Check 6: ASCII report text contains required headers, cohort summaries, and verdict."""
        report = self.results["report_text"]
        self.assertIn("NAUKRI.COM DOMAIN SUPPORT AGENT - TASK 13 EVALUATION REPORT", report)
        self.assertIn("In-Scope Averages (12 Queries):", report)
        self.assertIn("Out-Scope Averages (3 Queries):", report)
        self.assertIn("Averages: Accuracy:", report)
        self.assertIn("VERDICT: Task 13 Evaluation Suite Completed Successfully.", report)

    def test_07_transcript_and_json_files_written(self):
        """Check 7: Task transcript and JSON metrics files are created, non-empty, and valid JSON."""
        # Check production files
        prod_transcript = PROJECT_ROOT / "transcripts" / "task13_eval_15_queries.txt"
        prod_json = PROJECT_ROOT / "transcripts" / "task13_eval_metrics.json"

        self.assertTrue(prod_transcript.exists(), "task13_eval_15_queries.txt must exist")
        self.assertGreater(prod_transcript.stat().st_size, 500)

        self.assertTrue(prod_json.exists(), "task13_eval_metrics.json must exist")
        with open(prod_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(data["total_queries"], 15)
        self.assertEqual(len(data["query_results"]), 15)
        self.assertIn("overall_metrics", data)

    def test_08_error_analysis_diagnostics(self):
        """Check 8: Error bounds ensure zero queries fail safety and no catastrophic drops (< 0.70)."""
        for s in self.results["scores"]:
            self.assertEqual(s["safety"], 1.00, f"Query {s['query_id']} had safety violation: {s}")
            self.assertGreaterEqual(
                s["composite_score"],
                0.70,
                f"Query {s['query_id']} fell below acceptable lower bound 0.70: {s['composite_score']}",
            )


if __name__ == "__main__":
    unittest.main()
