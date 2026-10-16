"""
Automated Test Suite for Job Applications Dataset.

Validates all 11 requirements for the Day 1 Naukri.com support agent capstone:
1. JSON file can be loaded successfully.
2. There are exactly 50 records.
3. Every record has all required fields.
4. Record IDs are unique.
5. All five required categories appear.
6. All five required statuses appear.
7. Every salary is within the specified range (INR 300,000 to INR 3,000,000).
8. Days since creation are valid non-negative integers (<= 60 days).
9. Priority-review flags are actual Boolean values.
10. Actual priority-review rate is between 10% and 30% (target 20%).
11. Running the generator repeatedly with the same seed produces identical results.
"""

import json
from pathlib import Path
import sys
import unittest

# Ensure the scripts directory is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.generate_dataset import (  # noqa: E402
    CATEGORIES,
    MAX_DAYS_SINCE_CREATED,
    MAX_SALARY_INR,
    MIN_SALARY_INR,
    RANDOM_SEED,
    STATUSES,
    TOTAL_RECORDS,
    generate_dataset,
)


class TestJobApplicationsDataset(unittest.TestCase):
    """Test suite verifying synthetic dataset integrity, constraints, and reproducibility."""

    @classmethod
    def setUpClass(cls):
        """Load the generated dataset JSON once before running test cases."""
        cls.data_file = PROJECT_ROOT / "data" / "job_applications.json"
        if not cls.data_file.exists():
            # If the file does not exist yet, generate it for testing
            generate_dataset(output_path=cls.data_file)

        with open(cls.data_file, "r", encoding="utf-8") as f:
            cls.records = json.load(f)

    def test_01_json_file_loadable(self):
        """Check 1: The JSON file exists and can be parsed as a non-empty list."""
        self.assertTrue(self.data_file.exists(), f"Data file missing at {self.data_file}")
        self.assertIsInstance(self.records, list, "Expected root JSON structure to be a list")
        self.assertGreater(len(self.records), 0, "Dataset should not be empty")

    def test_02_record_count(self):
        """Check 2: There are exactly 50 records."""
        self.assertEqual(
            len(self.records),
            TOTAL_RECORDS,
            f"Expected exactly {TOTAL_RECORDS} records, found {len(self.records)}",
        )

    def test_03_required_fields(self):
        """Check 3: Every record contains all required fields."""
        required_fields = {
            "record_id",
            "category",
            "status",
            "expected_salary_inr",
            "days_since_created",
            "flagged_priority_review",
        }
        for idx, record in enumerate(self.records):
            missing = required_fields - set(record.keys())
            self.assertFalse(
                missing,
                f"Record at index {idx} ({record.get('record_id', 'unknown')}) missing required fields: {missing}",
            )

    def test_04_record_ids_unique(self):
        """Check 4: Record IDs are unique across all records."""
        record_ids = [r["record_id"] for r in self.records]
        unique_ids = set(record_ids)
        self.assertEqual(
            len(record_ids),
            len(unique_ids),
            f"Duplicate record IDs found! Total={len(record_ids)}, Unique={len(unique_ids)}",
        )

    def test_05_all_categories_appear(self):
        """Check 5: All five required categories appear at least once."""
        observed_categories = {r["category"] for r in self.records}
        expected_categories = set(CATEGORIES)
        missing = expected_categories - observed_categories
        self.assertEqual(
            len(missing),
            0,
            f"Missing required categories in dataset: {missing}",
        )

    def test_06_all_statuses_appear(self):
        """Check 6: All five required statuses appear at least once."""
        observed_statuses = {r["status"] for r in self.records}
        expected_statuses = set(STATUSES)
        missing = expected_statuses - observed_statuses
        self.assertEqual(
            len(missing),
            0,
            f"Missing required statuses in dataset: {missing}",
        )

    def test_07_salary_in_range(self):
        """Check 7: Every salary is an integer within INR 300,000 to INR 3,000,000."""
        for record in self.records:
            salary = record["expected_salary_inr"]
            self.assertIsInstance(
                salary,
                int,
                f"Salary must be an integer, got {type(salary)} in record {record['record_id']}",
            )
            self.assertGreaterEqual(
                salary,
                MIN_SALARY_INR,
                f"Salary {salary} below minimum {MIN_SALARY_INR} in record {record['record_id']}",
            )
            self.assertLessEqual(
                salary,
                MAX_SALARY_INR,
                f"Salary {salary} exceeds maximum {MAX_SALARY_INR} in record {record['record_id']}",
            )

    def test_08_days_since_created_valid(self):
        """Check 8: Days since creation are valid non-negative integers <= MAX_DAYS_SINCE_CREATED."""
        for record in self.records:
            days = record["days_since_created"]
            self.assertIsInstance(
                days,
                int,
                f"days_since_created must be an int, got {type(days)} in {record['record_id']}",
            )
            self.assertGreaterEqual(
                days,
                0,
                f"days_since_created cannot be negative in record {record['record_id']}",
            )
            self.assertLessEqual(
                days,
                MAX_DAYS_SINCE_CREATED,
                f"days_since_created ({days}) exceeds max ({MAX_DAYS_SINCE_CREATED}) in {record['record_id']}",
            )

    def test_09_priority_review_is_boolean(self):
        """Check 9: Priority-review flags are actual Boolean values (True/False)."""
        for record in self.records:
            flag = record["flagged_priority_review"]
            # In Python, bool is a subclass of int, so type(x) is bool strictly ensures True/False
            self.assertIs(
                type(flag),
                bool,
                f"flagged_priority_review must be strict bool, got {type(flag)} in {record['record_id']}",
            )

    def test_10_priority_review_rate_in_target_range(self):
        """Check 10: Actual priority-review rate is between 10% and 30% (target 20%)."""
        flagged_count = sum(1 for r in self.records if r["flagged_priority_review"])
        rate = flagged_count / len(self.records)

        self.assertGreaterEqual(
            rate,
            0.10,
            f"Priority review rate {rate:.1%} is below minimum threshold of 10%",
        )
        self.assertLessEqual(
            rate,
            0.30,
            f"Priority review rate {rate:.1%} is above maximum threshold of 30%",
        )
        # Verify exact target (10 records out of 50 = 20.0%)
        self.assertEqual(
            flagged_count,
            10,
            f"Expected exactly 10 flagged records (20%), found {flagged_count}",
        )

    def test_11_generator_is_reproducible(self):
        """Check 11: Running the generator repeatedly produces identical results."""
        run_a = generate_dataset(seed=RANDOM_SEED, total_records=TOTAL_RECORDS)
        run_b = generate_dataset(seed=RANDOM_SEED, total_records=TOTAL_RECORDS)

        self.assertEqual(
            run_a,
            run_b,
            "Re-running generate_dataset with seed=42 produced different results!",
        )
        self.assertEqual(
            run_a,
            self.records,
            "In-memory generation with seed=42 differs from data/job_applications.json!",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
