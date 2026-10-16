"""
Empirical Similarity Threshold Calibration Script for Task 4.

Calibrates the threshold separating in-scope queries from out-of-scope queries
by measuring top-1 cosine similarity across representative query sets.
"""

from pathlib import Path
import sys
from typing import Dict, List, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag.retriever import query_policies

IN_SCOPE_QUERIES = [
    ("What is the policy for rescheduling a technical interview?", "POL-003"),
    ("How long is the mandatory cooling-off period after a rejection?", "POL-010"),
    ("Can I request notice period buyout assistance for an urgent joining date?", "POL-008"),
    ("What background checks are performed during candidate verification?", "POL-007"),
    ("Can salary be negotiated above the maximum band limit?", "POL-005"),
]

OUT_OF_SCOPE_QUERIES = [
    "What is the weather forecast for Bangalore this weekend?",
    "Can I bring my pet dog to the office on Fridays?",
    "How do I bake a chocolate cake using an air fryer?",
    "Who won the cricket world cup in 2011?",
]


def run_calibration(strategy: str = "sentence") -> Dict[str, any]:
    """Execute calibration across in-scope and out-of-scope query sets."""
    print(f"\n=======================================================")
    print(f"  CALIBRATING SIMILARITY THRESHOLD (Strategy: {strategy})")
    print(f"=======================================================")

    in_scope_results: List[Tuple[str, str, str, float]] = []
    print("\n--- IN-SCOPE QUERIES ---")
    for q, expected_doc in IN_SCOPE_QUERIES:
        hits = query_policies(q, strategy=strategy, top_k=1)
        if hits:
            hit = hits[0]
            sim = hit["similarity"]
            matched_doc = hit["doc_id"]
            title = hit["title"]
            correct = (matched_doc == expected_doc)
            in_scope_results.append((q, matched_doc, title, sim))
            mark = "OK" if correct else "MISMATCH"
            print(f"[{mark}] Sim: {sim:.4f} | Match: {matched_doc} | Query: '{q}'")
        else:
            in_scope_results.append((q, "None", "None", 0.0))

    out_of_scope_results: List[Tuple[str, str, str, float]] = []
    print("\n--- OUT-OF-SCOPE QUERIES ---")
    for q in OUT_OF_SCOPE_QUERIES:
        hits = query_policies(q, strategy=strategy, top_k=1)
        if hits:
            hit = hits[0]
            sim = hit["similarity"]
            matched_doc = hit["doc_id"]
            title = hit["title"]
            out_of_scope_results.append((q, matched_doc, title, sim))
            print(f"[OOS] Sim: {sim:.4f} | Best noise hit: {matched_doc} | Query: '{q}'")
        else:
            out_of_scope_results.append((q, "None", "None", 0.0))

    in_sims = [r[3] for r in in_scope_results]
    oos_sims = [r[3] for r in out_of_scope_results]

    min_in_scope = min(in_sims)
    max_oos = max(oos_sims)
    gap = min_in_scope - max_oos
    calibrated_threshold = round((min_in_scope + max_oos) / 2.0, 4)

    print("\n--- CALIBRATION RESULTS ---")
    print(f"Minimum In-Scope Similarity:    {min_in_scope:.4f}")
    print(f"Maximum Out-Of-Scope Similarity: {max_oos:.4f}")
    print(f"Separation Margin (Gap):         {gap:.4f}")
    print(f"Recommended Threshold (Midpoint): {calibrated_threshold:.4f}")

    return {
        "strategy": strategy,
        "in_scope": in_scope_results,
        "out_of_scope": out_of_scope_results,
        "min_in_scope": min_in_scope,
        "max_oos": max_oos,
        "gap": gap,
        "threshold": calibrated_threshold,
    }


if __name__ == "__main__":
    sentence_res = run_calibration(strategy="sentence")
    fixed_res = run_calibration(strategy="fixed")
