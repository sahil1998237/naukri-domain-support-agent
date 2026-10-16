"""
Job Application Status and Escalation Score Tool (Task 6).

Implements:
- check_job_application_status(record_id)
- Continuous escalation score formula:
    escalation_score = 0.5 * int(flagged_priority_review) + 0.5 * min(1.0, days_since_created / 30.0)
- Empirically calibrated 80th percentile threshold derived from the 50-record dataset.
"""

import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_FILE = PROJECT_ROOT / "data" / "job_applications.json"


def load_application_records(file_path: Optional[Path | str] = None) -> List[Dict[str, Any]]:
    """Load application records from the generated JSON dataset."""
    path = Path(file_path) if file_path else DATA_FILE
    if not path.exists():
        raise FileNotFoundError(f"Job applications dataset not found at: {path}")

    with open(path, "r", encoding="utf-8") as f:
        records = json.load(f)
    return records


def calculate_escalation_score(
    flagged_priority_review: bool,
    days_since_created: int,
    recency_window_days: float = 30.0,
    weight_flag: float = 0.5,
    weight_recency: float = 0.5,
) -> float:
    """
    Compute a continuous escalation score in the range [0.0, 1.0].

    Formula:
        recency_signal = min(1.0, days_since_created / recency_window_days)
        escalation_score = weight_flag * int(flagged) + weight_recency * recency_signal

    Args:
        flagged_priority_review: Boolean indicator of priority triage.
        days_since_created: Non-negative integer of days elapsed since application.
        recency_window_days: Days after which recency signal saturates at 1.0 (default 30).
        weight_flag: Weight for priority flag (default 0.5).
        weight_recency: Weight for recency signal (default 0.5).

    Returns:
        Continuous escalation score in [0.0, 1.0].
    """
    recency_signal = min(1.0, max(0.0, days_since_created / recency_window_days))
    flag_signal = 1.0 if flagged_priority_review else 0.0
    score = (weight_flag * flag_signal) + (weight_recency * recency_signal)
    return round(score, 4)


def compute_80th_percentile_threshold(records: Optional[List[Dict[str, Any]]] = None) -> float:
    """
    Compute the empirical 80th percentile threshold across all application records.

    Returns:
        The score cutoff at the 80th percentile of the distribution.
    """
    if records is None:
        records = load_application_records()

    scores = sorted(
        [
            calculate_escalation_score(
                r["flagged_priority_review"],
                r["days_since_created"],
            )
            for r in records
        ]
    )

    if not scores:
        return 0.70

    # 80th percentile index
    idx = int(0.80 * len(scores))
    threshold = scores[min(idx, len(scores) - 1)]
    return round(threshold, 4)


# Precompute dataset 80th percentile threshold
DEFAULT_ESCALATION_THRESHOLD: float = compute_80th_percentile_threshold()


def check_job_application_status(
    record_id: str,
    threshold: float = DEFAULT_ESCALATION_THRESHOLD,
    file_path: Optional[Path | str] = None,
) -> Dict[str, Any]:
    """
    Look up job application status and compute escalation recommendation for an applicant.

    Args:
        record_id: The unique identifier (e.g. 'APP-003'). Case-insensitive.
        threshold: The escalation cutoff score (default: empirical 80th percentile).
        file_path: Optional custom path to job_applications.json.

    Returns:
        Dictionary containing application record details, continuous escalation score,
        and binary escalation recommendation with rationale.
    """
    records = load_application_records(file_path)
    clean_id = record_id.strip().upper()

    target_record: Optional[Dict[str, Any]] = None
    for r in records:
        if r["record_id"].upper() == clean_id:
            target_record = r
            break

    if target_record is None:
        return {
            "found": False,
            "record_id": clean_id,
            "error": f"Application record '{clean_id}' was not found in the Naukri database.",
            "escalation_recommended": False,
        }

    score = calculate_escalation_score(
        target_record["flagged_priority_review"],
        target_record["days_since_created"],
    )
    is_escalated = score >= threshold

    status = target_record["status"]
    days = target_record["days_since_created"]
    flagged = target_record["flagged_priority_review"]
    category = target_record["category"]

    reason_parts = []
    if flagged:
        reason_parts.append("Candidate was designated for priority review")
    if days >= 30:
        reason_parts.append(f"Application has been pending for {days} days (>= 30-day escalation window)")
    elif days > 0:
        reason_parts.append(f"Application was created {days} days ago")

    reason = "; ".join(reason_parts) if reason_parts else "Standard timeline"

    summary = (
        f"Application {clean_id} ({category}): Status is '{status}' "
        f"(created {days} days ago, priority={flagged}). "
        f"Escalation Score: {score:.2f} (Threshold: {threshold:.2f} -> "
        f"{'ESCALATION RECOMMENDED' if is_escalated else 'Normal Processing'}). "
        f"Rationale: {reason}."
    )

    return {
        "found": True,
        "record_id": target_record["record_id"],
        "category": category,
        "status": status,
        "expected_salary_inr": target_record["expected_salary_inr"],
        "days_since_created": days,
        "flagged_priority_review": flagged,
        "escalation_score": score,
        "escalation_threshold": threshold,
        "escalation_recommended": is_escalated,
        "rationale": reason,
        "summary": summary,
    }


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print(f"=== Job Application Status & Escalation Tool (Task 6) ===")
    print(f"Dataset 80th Percentile Escalation Threshold: {DEFAULT_ESCALATION_THRESHOLD}\n")

    test_ids = ["APP-001", "APP-003", "APP-005", "APP-999"]
    for tid in test_ids:
        result = check_job_application_status(tid)
        if result["found"]:
            print(f"[{result['record_id']}] Score: {result['escalation_score']} | Escalated: {result['escalation_recommended']}")
            print(f"  Summary: {result['summary']}\n")
        else:
            print(f"[{result['record_id']}] Error: {result['error']}\n")
