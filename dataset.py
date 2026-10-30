"""
Final Capstone — Part 1 Task 1: Job Applications Dataset Generator (Naukri.com Domain).

Generates a seeded, deterministic JOB_APPLICATIONS list of >= 40 records (default: 50 records)
satisfying every required constraint:
- Category vocabulary: 'Software Engineer', 'Data Analyst', 'Product Manager', 'HR Executive', 'Sales Associate'
  (every category >= 3 records).
- Status vocabulary: 'Applied', 'Screening', 'Interview Scheduled', 'Offered', 'Rejected'
  (every status >= 1 record).
- expected_salary_inr: INR 300,000 to INR 3,000,000.
  Reasoning: Annual salary range of INR 300,000 to INR 3,000,000 reflects realistic entry-level
  to senior professional compensation benchmarks across Indian tech, sales, and HR roles on Naukri.com.
- days_since_created: integer between 0 and 30 inclusive.
- flagged_priority_review: boolean, landing between 10% and 30% (strictly 20% = 10 of 50).
- Deterministic reproduction with seed=42.
"""

from collections import Counter
import json
from pathlib import Path
import random
from typing import Any, Dict, List, Optional

# --- Configuration & Specification Constants ---
RANDOM_SEED: int = 42
TOTAL_RECORDS: int = 50

# Salary bounds in INR (annual)
# Reasoning: Annual salary range of INR 300,000 to INR 3,000,000 reflects realistic entry-level
# to senior professional compensation benchmarks across Indian tech, sales, and HR roles on Naukri.com.
MIN_SALARY_INR: int = 300_000
MAX_SALARY_INR: int = 3_000_000
SALARY_STEP_INR: int = 25_000

# Application timeline window: integer, 0-30 days
MIN_DAYS_SINCE_CREATED: int = 0
MAX_DAYS_SINCE_CREATED: int = 30

# Priority review target: exactly 10 of 50 = 20% (strictly lands within 10% - 30%)
PRIORITY_REVIEW_COUNT: int = 10

# Category vocabulary (all 5 required categories)
CATEGORIES: List[str] = [
    "Software Engineer",
    "Data Analyst",
    "Product Manager",
    "HR Executive",
    "Sales Associate",
]
CATEGORY_WEIGHTS: List[float] = [0.20, 0.20, 0.20, 0.20, 0.20]

# Status vocabulary (all 5 required statuses)
STATUSES: List[str] = [
    "Applied",
    "Screening",
    "Interview Scheduled",
    "Offered",
    "Rejected",
]
STATUS_WEIGHTS: List[float] = [0.35, 0.25, 0.20, 0.10, 0.10]


def generate_dataset(
    seed: int = RANDOM_SEED,
    total_records: int = TOTAL_RECORDS,
    output_path: Optional[Path | str] = None,
) -> List[Dict[str, Any]]:
    """
    Generate a seeded, deterministic list of job application records.
    """
    random.seed(seed)

    # Guarantee category representation: include each category at least once,
    # then sample remaining records with weights
    initial_categories = list(CATEGORIES)
    remaining_cat_count = max(0, total_records - len(initial_categories))
    sampled_categories = random.choices(
        CATEGORIES, weights=CATEGORY_WEIGHTS, k=remaining_cat_count
    )
    all_categories = initial_categories + sampled_categories
    random.shuffle(all_categories)

    # Guarantee status representation: include each status at least once,
    # then sample remaining records with weights
    initial_statuses = list(STATUSES)
    remaining_status_count = max(0, total_records - len(initial_statuses))
    sampled_statuses = random.choices(
        STATUSES, weights=STATUS_WEIGHTS, k=remaining_status_count
    )
    all_statuses = initial_statuses + sampled_statuses
    random.shuffle(all_statuses)

    # Priority review selection: exact count using seeded sampling without replacement
    priority_indices = set(random.sample(range(total_records), PRIORITY_REVIEW_COUNT))

    # Assemble records
    records: List[Dict[str, Any]] = []
    for i in range(total_records):
        record_id = f"APP-{i + 1:03d}"
        salary = random.randrange(MIN_SALARY_INR, MAX_SALARY_INR + 1, SALARY_STEP_INR)
        days = random.randint(MIN_DAYS_SINCE_CREATED, MAX_DAYS_SINCE_CREATED)
        flagged = i in priority_indices

        record: Dict[str, Any] = {
            "record_id": record_id,
            "category": all_categories[i],
            "status": all_statuses[i],
            "expected_salary_inr": salary,
            "days_since_created": days,
            "flagged_priority_review": flagged,
        }
        records.append(record)

    if output_path is not None:
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)
            f.write("\n")

    return records


# Pre-generated dataset instance as required by Part 1 Task 1
JOB_APPLICATIONS: List[Dict[str, Any]] = generate_dataset(
    seed=RANDOM_SEED,
    total_records=TOTAL_RECORDS,
)


def print_dataset_report(records: List[Dict[str, Any]]) -> None:
    """Print the required counts and distribution metrics as per Part 1 Task 1."""
    print("=" * 60)
    print("NAUKRI.COM JOB APPLICATIONS DATASET REPORT (PART 1 TASK 1)")
    print("=" * 60)
    print(f"Total Records: {len(records)} (Requirement: >= 40)")
    print(f"Seed: {RANDOM_SEED} (Deterministic PRNG)")
    print("Salary Reasoning: Annual salary range of INR 300,000 to INR 3,000,000 reflects realistic entry-level to senior professional compensation benchmarks across Indian tech, sales, and HR roles on Naukri.com.")
    print("-" * 60)

    # Category counts
    cat_counts = Counter(r["category"] for r in records)
    print("Category Counts (Requirement: each given category >= 3):")
    for cat in CATEGORIES:
        count = cat_counts.get(cat, 0)
        status_symbol = "[PASS]" if count >= 3 else "[FAIL]"
        print(f"  {status_symbol} {cat:20s}: {count:2d} records")

    print("-" * 60)
    # Status counts
    stat_counts = Counter(r["status"] for r in records)
    print("Status Counts (Requirement: each given status >= 1):")
    for stat in STATUSES:
        count = stat_counts.get(stat, 0)
        status_symbol = "[PASS]" if count >= 1 else "[FAIL]"
        print(f"  {status_symbol} {stat:20s}: {count:2d} records")

    print("-" * 60)
    # Flagged percentage
    flagged_count = sum(1 for r in records if r["flagged_priority_review"])
    percentage = (flagged_count / len(records)) * 100.0
    pct_status = "[PASS]" if 10.0 <= percentage <= 30.0 else "[FAIL]"
    print(f"Flagged Priority Review: {flagged_count}/{len(records)} ({percentage:.1f}%) {pct_status} (Requirement: 10% - 30%)")

    # Days range
    min_days = min(r["days_since_created"] for r in records)
    max_days = max(r["days_since_created"] for r in records)
    days_status = "[PASS]" if min_days >= 0 and max_days <= 30 else "[FAIL]"
    print(f"Days Since Created Range: {min_days} to {max_days} days {days_status} (Requirement: integer, 0-30)")
    print("=" * 60)


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent
    default_output = project_root / "data" / "job_applications.json"
    records = generate_dataset(output_path=default_output)
    print_dataset_report(records)

