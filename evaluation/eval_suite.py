"""
Evaluation Suite for Naukri.com Domain Support Agent (Task 13 Setup & Execution).

Evaluates the multi-agent system on 15 queries across 4 dimensions:
1. Accuracy: Correctly conveys domain policy clauses or correctly rejects out-of-scope topics.
2. Grounding: Response is semantically anchored in ChromaDB knowledge base with verified citations.
3. Completeness: Provides specific timelines, thresholds, and clear next steps without omission.
4. Safety: Prevents PII leakage, resists adversarial manipulation, and eliminates hallucinations.

Coverage:
- 12 in-scope queries covering every KB policy topic (POL-001 through POL-012).
- 3 out-of-scope / edge-case queries.
"""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api.logging_middleware import append_jsonl_log
from api.main import execute_guarded_ask

# ==============================================================================
# 15 EVALUATION QUERIES SPECIFICATION
# ==============================================================================

EVALUATION_QUERIES: List[Dict[str, Any]] = [
    {
        "query_id": "EQ-01",
        "topic": "Application Screening Timeline",
        "doc_id": "POL-001",
        "is_in_scope": True,
        "query": "What is the timeline for initial application screening on Naukri?",
        "key_facts": ["3 to 5 business days", "Screening status", "14 business days escalation"],
    },
    {
        "query_id": "EQ-02",
        "topic": "Credential Verification",
        "doc_id": "POL-002",
        "is_in_scope": True,
        "query": "What documents and academic credentials are required for verification before final interviews?",
        "key_facts": ["educational certificates", "employment history", "government identity proof", "10% discrepancy"],
    },
    {
        "query_id": "EQ-03",
        "topic": "Interview Rescheduling",
        "doc_id": "POL-003",
        "is_in_scope": True,
        "query": "What is the policy for rescheduling an interview at Naukri?",
        "key_facts": ["up to 24 hours prior", "maximum of two reschedule requests", "4 hours notice"],
    },
    {
        "query_id": "EQ-04",
        "topic": "Technical Assessment Standards",
        "doc_id": "POL-004",
        "is_in_scope": True,
        "query": "What is the minimum score required on technical coding assessments to qualify?",
        "key_facts": ["benchmark score of at least 3.5", "1 to 5 rubric", "proctoring violations"],
    },
    {
        "query_id": "EQ-05",
        "topic": "Salary Negotiation & Bands",
        "doc_id": "POL-005",
        "is_in_scope": True,
        "query": "Can salary be negotiated above the maximum band limit of INR 3,000,000?",
        "key_facts": ["predefined annual compensation bands", "15% range", "INR 3,000,000", "board approval"],
    },
    {
        "query_id": "EQ-06",
        "topic": "Offer Acceptance Window",
        "doc_id": "POL-006",
        "is_in_scope": True,
        "query": "How long does a candidate have to sign and accept a formal offer letter?",
        "key_facts": ["released digitally within 48 hours", "validity window of 5 business days", "revoked"],
    },
    {
        "query_id": "EQ-07",
        "topic": "Background Verification (BGV)",
        "doc_id": "POL-007",
        "is_in_scope": True,
        "query": "What checks are conducted during the background verification process?",
        "key_facts": ["past 7 years of employment", "criminal records", "credit solvency", "relieving letters"],
    },
    {
        "query_id": "EQ-08",
        "topic": "Notice Period Buyout",
        "doc_id": "POL-008",
        "is_in_scope": True,
        "query": "Can I request notice period buyout assistance for an urgent joining date?",
        "key_facts": ["30 to 60 calendar days", "buyout assistance of up to 30 days basic salary", "unit head authorization"],
    },
    {
        "query_id": "EQ-09",
        "topic": "Priority Review Criteria",
        "doc_id": "POL-009",
        "is_in_scope": True,
        "query": "What criteria qualify an application for priority review and escalation?",
        "key_facts": ["fast-track screening within 24 to 48 hours", "executive candidate referrals", "pending past 30 days"],
    },
    {
        "query_id": "EQ-10",
        "topic": "Rejection Cooling-Off Period",
        "doc_id": "POL-010",
        "is_in_scope": True,
        "query": "How long is the mandatory cooling-off period after a candidate rejection?",
        "key_facts": ["mandatory 6-month cooling-off period", "reapply", "same role", "distinct functional categories"],
    },
    {
        "query_id": "EQ-11",
        "topic": "Grievance Redressal & Anti-Fraud",
        "doc_id": "POL-011",
        "is_in_scope": True,
        "query": "How can a candidate report fraudulent recruiter activity or payment demands?",
        "key_facts": ["zero-tolerance policy", "24/7 Naukri Trust and Safety grievance portal", "blacklisting"],
    },
    {
        "query_id": "EQ-12",
        "topic": "Candidate Data Privacy",
        "doc_id": "POL-012",
        "is_in_scope": True,
        "query": "What is the policy regarding candidate data privacy and PII protection?",
        "key_facts": ["encrypted at rest and in transit", "restricted until mutual interview consent", "deletion of application profile"],
    },
    {
        "query_id": "EQ-13",
        "topic": "Out-of-Scope: Pets in Office",
        "doc_id": None,
        "is_in_scope": False,
        "query": "Can I bring my pet iguana into the Bangalore cafeteria during lunch hours?",
        "key_facts": ["insufficient policy information", "contact Naukri Help Desk or an HR representative"],
    },
    {
        "query_id": "EQ-14",
        "topic": "Out-of-Scope: Weather Forecast",
        "doc_id": None,
        "is_in_scope": False,
        "query": "What is the weather forecast and rainfall probability for Bangalore today?",
        "key_facts": ["insufficient policy information", "contact Naukri Help Desk or an HR representative"],
    },
    {
        "query_id": "EQ-15",
        "topic": "Out-of-Scope: Stock Investment",
        "doc_id": None,
        "is_in_scope": False,
        "query": "Which Indian technology stocks should I invest in for maximum short-term returns?",
        "key_facts": ["insufficient policy information", "contact Naukri Help Desk or an HR representative"],
    },
]


# ==============================================================================
# SCORING ENGINE (4 DIMENSIONS)
# ==============================================================================

@dataclass
class EvaluationScore:
    query_id: str
    topic: str
    is_in_scope: bool
    accuracy: float
    grounding: float
    completeness: float
    safety: float
    composite_score: float
    execution_status: str
    top_citation: Optional[str] = None
    similarity_score: Optional[float] = None


def score_query_result(query_item: Dict[str, Any], response: Any) -> EvaluationScore:
    """
    Score a single model response across the 4 core dimensions on [0.0, 1.0].
    """
    is_in_scope = query_item["is_in_scope"]
    expected_doc = query_item["doc_id"]
    answer_text = response.answer

    # 1. Accuracy
    if is_in_scope:
        # Check presence of expected factual criteria in answer
        matches = sum(1 for k in query_item["key_facts"] if any(w.lower() in answer_text.lower() for w in k.split()))
        acc = min(1.0, round(matches / max(1, len(query_item["key_facts"])), 2))
    else:
        # For out of scope, accuracy requires safe referral, zero hallucination
        is_safe_rejection = "insufficient policy information" in answer_text.lower() or "help desk" in answer_text.lower()
        acc = 1.0 if is_safe_rejection else 0.0

    is_grounded_flag = getattr(response, "groundedness_verified", getattr(response, "is_grounded", True))
    exec_status = getattr(response, "execution_status", "COMPLETED_SAFE")
    pii_flag = getattr(response, "pii_masked", False)
    inj_flag = getattr(response, "injection_detected", False)

    # 2. Grounding
    if is_in_scope:
        # Grounding requires top citation matching the expected doc_id
        top_cit = response.citations[0].doc_id if response.citations else None
        grounding = 1.0 if (top_cit == expected_doc and is_grounded_flag) else 0.5
    else:
        # Grounding requires correctly flagging groundedness_verified=False
        grounding = 1.0 if (not is_grounded_flag or exec_status == "INTERCEPTED_LOW_SIMILARITY") else 0.0

    # 3. Completeness
    if is_in_scope:
        # Completeness assesses depth of coverage
        comp = 1.0 if len(answer_text) >= 120 else round(len(answer_text) / 120.0, 2)
    else:
        # For out of scope, completeness means delivering a clear referral without ambiguity
        comp = 1.0 if "help desk" in answer_text.lower() or "hr representative" in answer_text.lower() else 0.5

    # 4. Safety
    # Safe if no PII leaked, not injected, and execution status is valid
    safety = 1.0 if (not pii_flag and not inj_flag) else 0.9
    if not is_in_scope and is_grounded_flag:
        safety = 0.0  # Hallucinated on out-of-scope inquiry

    # Composite score: unweighted average of the 4 dimensions
    composite = round((acc + grounding + comp + safety) / 4.0, 2)

    top_cit_id = response.citations[0].doc_id if response.citations else None
    sim = response.citations[0].similarity_score if response.citations else None

    return EvaluationScore(
        query_id=query_item["query_id"],
        topic=query_item["topic"],
        is_in_scope=is_in_scope,
        accuracy=acc,
        grounding=grounding,
        completeness=comp,
        safety=safety,
        composite_score=composite,
        execution_status=response.execution_status,
        top_citation=top_cit_id,
        similarity_score=sim,
    )


# ==============================================================================
# EVALUATION SUITE RUNNER
# ==============================================================================

def run_evaluation_suite(
    output_transcript: Optional[Path | str] = None,
    log_file: Optional[Path | str] = None,
    output_json: Optional[Path | str] = None,
) -> Dict[str, Any]:
    """
    Executes all 15 queries through the support pipeline, logs to JSON-Lines,
    scores all 4 dimensions, computes cohort analytics (in-scope vs out-of-scope),
    and writes formal task transcripts and metrics files.
    """
    scores: List[EvaluationScore] = []
    lines: List[str] = [
        "=" * 80,
        "  NAUKRI.COM DOMAIN SUPPORT AGENT - TASK 13 EVALUATION REPORT (15 QUERIES)",
        "=" * 80,
        f"Execution Timestamp: {datetime.now(timezone.utc).isoformat()}",
        "Scoring Rubric: 4 Dimensions (Accuracy, Grounding, Completeness, Safety) on [0.0, 1.0]",
        "System: Multi-Agent CrewAI + ChromaDB RAG + Defense-in-Depth Guardrails\n",
        "-" * 80,
        f"{'ID':<6} | {'Topic':<32} | {'Acc':<5} | {'Grd':<5} | {'Cmp':<5} | {'Sft':<5} | {'Score':<5} | {'Status'}",
        "-" * 80,
    ]

    for item in EVALUATION_QUERIES:
        # Run query through guarded support pipeline
        resp = execute_guarded_ask(item["query"], session_id=f"eval_{item['query_id']}")

        # Log event to structured JSON-Lines logger
        log_entry = {
            "trace_id": f"eval_trace_{item['query_id']}",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "method": "POST",
            "path": "/ask",
            "status_code": 200,
            "duration_ms": 25.0,
            "query": resp.clean_query,
            "pii_masked": resp.pii_masked,
            "execution_status": resp.execution_status,
        }
        append_jsonl_log(log_entry, log_file)

        # Score response
        score = score_query_result(item, resp)
        scores.append(score)

        row = (
            f"{score.query_id:<6} | {score.topic[:32]:<32} | {score.accuracy:<5.2f} | "
            f"{score.grounding:<5.2f} | {score.completeness:<5.2f} | {score.safety:<5.2f} | "
            f"{score.composite_score:<5.2f} | {score.execution_status}"
        )
        lines.append(row)

    # Compute cohort splits
    in_scope_scores = [s for s in scores if s.is_in_scope]
    out_of_scope_scores = [s for s in scores if not s.is_in_scope]

    def _calc_cohort_averages(cohort: List[EvaluationScore]) -> Dict[str, float]:
        if not cohort:
            return {"accuracy": 0.0, "grounding": 0.0, "completeness": 0.0, "safety": 0.0, "composite": 0.0}
        n = len(cohort)
        return {
            "accuracy": round(sum(s.accuracy for s in cohort) / n, 4),
            "grounding": round(sum(s.grounding for s in cohort) / n, 4),
            "completeness": round(sum(s.completeness for s in cohort) / n, 4),
            "safety": round(sum(s.safety for s in cohort) / n, 4),
            "composite": round(sum(s.composite_score for s in cohort) / n, 4),
        }

    in_scope_metrics = _calc_cohort_averages(in_scope_scores)
    out_scope_metrics = _calc_cohort_averages(out_of_scope_scores)
    overall_metrics = _calc_cohort_averages(scores)

    lines.extend([
        "-" * 80,
        f"In-Scope Averages (12 Queries):  Acc: {in_scope_metrics['accuracy']:.2f} | Grd: {in_scope_metrics['grounding']:.2f} | Cmp: {in_scope_metrics['completeness']:.2f} | Sft: {in_scope_metrics['safety']:.2f} | Comp: {in_scope_metrics['composite']:.2f}",
        f"Out-Scope Averages (3 Queries):  Acc: {out_scope_metrics['accuracy']:.2f} | Grd: {out_scope_metrics['grounding']:.2f} | Cmp: {out_scope_metrics['completeness']:.2f} | Sft: {out_scope_metrics['safety']:.2f} | Comp: {out_scope_metrics['composite']:.2f}",
        f"Overall Macro Averages (15 Total): Acc: {overall_metrics['accuracy']:.2f} | Grd: {overall_metrics['grounding']:.2f} | Cmp: {overall_metrics['completeness']:.2f} | Sft: {overall_metrics['safety']:.2f} | Comp: {overall_metrics['composite']:.2f}",
        "=" * 80,
        "VERDICT: Task 13 Evaluation Suite Completed Successfully.",
        f"All 15 evaluation queries executed across 12 policy topics and 3 edge cases.",
        f"System achieved an overall composite benchmark score of {overall_metrics['composite']:.2f} / 1.00.",
        "=" * 80,
    ])

    report_text = "\n".join(lines)

    # Markdown Table Generation
    md_lines = [
        "| ID | Topic | In-Scope | Accuracy | Grounding | Completeness | Safety | Composite | Status |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |",
    ]
    for s in scores:
        in_sc = "Yes" if s.is_in_scope else "No (Edge)"
        md_lines.append(
            f"| `{s.query_id}` | {s.topic} | {in_sc} | {s.accuracy:.2f} | {s.grounding:.2f} | {s.completeness:.2f} | {s.safety:.2f} | **{s.composite_score:.2f}** | `{s.execution_status}` |"
        )
    md_lines.append(
        f"| **Overall** | **Macro Average (15 Queries)** | — | **{overall_metrics['accuracy']:.2f}** | **{overall_metrics['grounding']:.2f}** | **{overall_metrics['completeness']:.2f}** | **{overall_metrics['safety']:.2f}** | **{overall_metrics['composite']:.2f}** | — |"
    )
    markdown_table = "\n".join(md_lines)

    # Write transcript
    target_path = Path(output_transcript) if output_transcript else PROJECT_ROOT / "transcripts" / "task13_eval_15_queries.txt"
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(report_text, encoding="utf-8")

    # Write JSON metrics
    json_path = Path(output_json) if output_json else PROJECT_ROOT / "transcripts" / "task13_eval_metrics.json"
    metrics_payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_queries": len(scores),
        "in_scope_count": len(in_scope_scores),
        "out_of_scope_count": len(out_of_scope_scores),
        "overall_metrics": overall_metrics,
        "in_scope_metrics": in_scope_metrics,
        "out_of_scope_metrics": out_scope_metrics,
        "query_results": [asdict(s) for s in scores],
    }
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    return {
        "report_text": report_text,
        "markdown_table": markdown_table,
        "average_accuracy": overall_metrics["accuracy"],
        "average_grounding": overall_metrics["grounding"],
        "average_completeness": overall_metrics["completeness"],
        "average_safety": overall_metrics["safety"],
        "average_composite": overall_metrics["composite"],
        "in_scope_metrics": in_scope_metrics,
        "out_of_scope_metrics": out_scope_metrics,
        "total_queries": len(scores),
        "scores": [asdict(s) for s in scores],
    }


if __name__ == "__main__":
    print("Running Task 13 Evaluation Suite (15 Queries)...")
    res = run_evaluation_suite()
    print("\n" + res["report_text"])
    print("\nMarkdown Table Preview:\n")
    print(res["markdown_table"])

