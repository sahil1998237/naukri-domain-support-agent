"""
Synthetic Job Applications Dataset Generator for Naukri.com Domain Support Agent.

This script generates reproducible, fictional job-application data satisfying
all Day 1 capstone requirements:
- Fixed random seed (seed=42) for deterministic output
- 50 records
- INR 300,000 to INR 3,000,000 salary range
- 5 distinct categories with equal 20% target weights (all guaranteed >= 1)
- 5 distinct statuses with weighted distribution (all guaranteed >= 1)
- Non-negative days since created (1 to 60 days, documented max = 60)
- Exactly 10 records (20%) flagged for priority review (target: 10% - 30%)
- Output saved as structured JSON
"""

import json
import os
from pathlib import Path
import random
from typing import Any, Dict, List, Optional

# --- Configuration & Assumptions ---
RANDOM_SEED: int = 42
TOTAL_RECORDS: int = 50

# Salary bounds in INR (annual)
MIN_SALARY_INR: int = 300_000
MAX_SALARY_INR: int = 3_000_000
SALARY_STEP_INR: int = 25_000  # Multiples of 25,000 for realistic salary figures

# Application timeline window
# Documented maximum: 60 days (standard active application window on job boards)
MIN_DAYS_SINCE_CREATED: int = 1
MAX_DAYS_SINCE_CREATED: int = 60

# Priority review target: exactly 10 of 50 = 20% (satisfies 10% - 30% range)
PRIORITY_REVIEW_COUNT: int = 10

# Category definitions (20% target weight per category)
CATEGORIES: List[str] = [
    "Software Engineer",
    "Data Analyst",
    "Product Manager",
    "HR Executive",
    "Sales Associate",
]
CATEGORY_WEIGHTS: List[float] = [0.20, 0.20, 0.20, 0.20, 0.20]

# Status definitions and target weights
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
    Generate a deterministic list of job-application records.

    Args:
        seed: Random seed for reproducibility.
        total_records: Total number of records to generate (default 50).
        output_path: Optional filesystem path to write the JSON file.

    Returns:
        List of generated job application dictionaries.
    """
    # 1. Initialize PRNG with fixed seed
    random.seed(seed)

    # 2. Guarantee category representation: include each category at least once,
    # then sample the remaining records using category weights.
    initial_categories = list(CATEGORIES)
    remaining_cat_count = max(0, total_records - len(initial_categories))
    sampled_categories = random.choices(
        CATEGORIES, weights=CATEGORY_WEIGHTS, k=remaining_cat_count
    )
    all_categories = initial_categories + sampled_categories
    random.shuffle(all_categories)

    # 3. Guarantee status representation: include each status at least once,
    # then sample the remaining records using status weights.
    initial_statuses = list(STATUSES)
    remaining_status_count = max(0, total_records - len(initial_statuses))
    sampled_statuses = random.choices(
        STATUSES, weights=STATUS_WEIGHTS, k=remaining_status_count
    )
    all_statuses = initial_statuses + sampled_statuses
    random.shuffle(all_statuses)

    # 4. Priority review selection: select exactly PRIORITY_REVIEW_COUNT records
    # using seeded sampling without replacement to hit 20% exactly.
    priority_indices = set(random.sample(range(total_records), PRIORITY_REVIEW_COUNT))

    # 5. Assemble record dictionaries
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

    # 6. Optionally write to JSON file
    if output_path is not None:
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)
            f.write("\n")

    return records


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    default_output = project_root / "data" / "job_applications.json"
    generated_records = generate_dataset(output_path=default_output)
    print(f"Generated {len(generated_records)} records -> {default_output}")
