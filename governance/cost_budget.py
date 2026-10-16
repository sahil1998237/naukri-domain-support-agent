"""
Task 15: Token & Cost Budget Governance (Day 12).

Implements cost-control governance mechanisms:
1. TokenBudgetManager: Enforces a strict per-request token ceiling (default 2,000 tokens)
   to prevent runaway recursive agent loops, denial-of-wallet attacks, and memory exhaustion.
2. TokenBudgetExceededError: Custom exception raised when an incoming prompt exceeds
   the allowable token allocation.
3. Automated Budget Demonstration:
   - Evaluates a standard in-budget candidate inquiry.
   - Rejects an oversized prompt (2,500+ tokens) before invoking model resources.
   - Writes the formal transcript to transcripts/task15_budget_reject.txt.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class TokenBudgetExceededError(ValueError):
    """Raised when an incoming request exceeds the configured per-request token budget."""

    def __init__(self, requested_tokens: int, budget_limit: int, message: Optional[str] = None):
        self.requested_tokens = requested_tokens
        self.budget_limit = budget_limit
        default_msg = (
            f"GOVERNANCE VIOLATION: Request exceeds maximum token budget. "
            f"Estimated tokens: {requested_tokens} > Allowable limit: {budget_limit}. "
            f"Execution halted to prevent financial runaway and resource exhaustion."
        )
        super().__init__(message or default_msg)


class TokenBudgetManager:
    """
    Monitors and enforces token consumption against enterprise cost controls.
    """

    def __init__(self, budget_limit: int = 2000):
        self.budget_limit = budget_limit
        self.cumulative_tokens = 0
        self.accepted_requests = 0
        self.rejected_requests = 0
        self.history: List[Dict[str, Any]] = []

    def estimate_tokens(self, text: str) -> int:
        """Heuristic token estimation based on whitespace and punctuation tokens."""
        if not text:
            return 0
        words = len(text.split())
        return max(1, int(words * 1.35))

    def evaluate_request(self, prompt_text: str, record_usage: bool = True) -> Dict[str, Any]:
        """
        Validates prompt against token budget ceiling.
        Raises TokenBudgetExceededError if over budget.
        """
        tokens = self.estimate_tokens(prompt_text)
        is_allowed = tokens <= self.budget_limit

        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "estimated_tokens": tokens,
            "budget_limit": self.budget_limit,
            "is_allowed": is_allowed,
        }

        if not is_allowed:
            self.rejected_requests += 1
            event["status"] = "REJECTED_OVER_BUDGET"
            self.history.append(event)
            raise TokenBudgetExceededError(
                requested_tokens=tokens,
                budget_limit=self.budget_limit
            )

        if record_usage:
            self.accepted_requests += 1
            self.cumulative_tokens += tokens
            event["status"] = "ACCEPTED"
            self.history.append(event)

        return event

    def get_metrics(self) -> Dict[str, Any]:
        """Return cumulative token and request governance metrics."""
        return {
            "budget_limit_per_request": self.budget_limit,
            "accepted_requests": self.accepted_requests,
            "rejected_requests": self.rejected_requests,
            "cumulative_tokens": self.cumulative_tokens,
            "total_requests": self.accepted_requests + self.rejected_requests,
        }

    def reset(self) -> None:
        """Reset running counters."""
        self.cumulative_tokens = 0
        self.accepted_requests = 0
        self.rejected_requests = 0
        self.history.clear()


# ==============================================================================
# DEMONSTRATION & TRANSCRIPT GENERATOR (TASK 15)
# ==============================================================================

def demonstrate_token_budget_enforcement(
    output_transcript: Optional[Path | str] = None,
    budget_limit: int = 2000,
) -> Dict[str, Any]:
    """
    Executes comparative demonstration of token budget enforcement:
    1. Normal in-budget candidate query (e.g., 200 tokens) -> Accepted.
    2. Runaway oversized prompt (e.g., 2,500+ tokens) -> Rejected with error.
    Writes formal transcript to transcripts/task15_budget_reject.txt.
    """
    manager = TokenBudgetManager(budget_limit=budget_limit)

    # 1. Normal candidate inquiry (In-Budget)
    normal_prompt = (
        "Candidate Inquiry regarding application status APP-003 and policy rescheduling. "
        "Candidate asks: What is the official timeline for rescheduling a technical coding interview slot, "
        "and is priority review escalation recommended for my pending application?"
    )
    normal_tokens = manager.estimate_tokens(normal_prompt)
    normal_status = manager.evaluate_request(normal_prompt)

    # 2. Oversized runaway prompt (Denial-of-Wallet / recursive expansion payload)
    # Generate ~2,600 token prompt by repeating verbose instruction blocks
    repeating_block = (
        "Please provide an exhaustive paragraph-by-paragraph historical legal breakdown of all Indian labor laws, "
        "IT service agreements, provident fund contribution standards, notice period buyout disputes, "
        "and arbitration proceedings across all 28 states and union territories. "
    )
    oversized_prompt = "ADVERSARIAL_PAYLOAD: " + (repeating_block * 70)
    oversized_tokens = manager.estimate_tokens(oversized_prompt)

    rejection_details: Dict[str, Any] = {}
    try:
        manager.evaluate_request(oversized_prompt)
    except TokenBudgetExceededError as exc:
        rejection_details = {
            "status": "INTERCEPTED_AND_REJECTED",
            "error_type": type(exc).__name__,
            "requested_tokens": exc.requested_tokens,
            "budget_limit": exc.budget_limit,
            "message": str(exc),
        }

    # Format transcript
    lines = [
        "=" * 80,
        "  NAUKRI.COM DOMAIN SUPPORT AGENT - TASK 15 TOKEN BUDGET GOVERNANCE TRANSCRIPT",
        "=" * 80,
        f"Execution Timestamp:       {datetime.now(timezone.utc).isoformat()}",
        f"Configured Budget Cap:      {budget_limit} tokens per request",
        "Governance Objective:       Prevent denial-of-wallet attacks and recursive agent runaway\n",
        "-" * 80,
        "SCENARIO 1: NORMAL CANDIDATE INQUIRY (WITHIN BUDGET)",
        "-" * 80,
        f"Prompt Snippet:             '{normal_prompt[:120]}...'",
        f"Estimated Tokens:           {normal_tokens} tokens",
        f"Budget Comparison:          {normal_tokens} <= {budget_limit} tokens",
        f"Governance Decision:        ACCEPTED (Execution allowed to proceed)",
        f"Cost Impact:                Normal nominal token consumption recorded\n",
        "-" * 80,
        "SCENARIO 2: OVERSIZED RUNAWAY REQUEST (EXCEEDING BUDGET CAP)",
        "-" * 80,
        f"Prompt Snippet:             '{oversized_prompt[:120]}...'",
        f"Estimated Tokens:           {oversized_tokens} tokens",
        f"Budget Comparison:          {oversized_tokens} > {budget_limit} tokens",
        f"Governance Decision:        REJECTED IMMEDIATELY BEFORE LLM INVOCATION",
        f"Exception Raised:           {rejection_details.get('error_type')}",
        f"Diagnostic Message:         {rejection_details.get('message')}",
        f"Cost Incurred:              0 TOKENS (Protected from financial runaway)\n",
        "-" * 80,
        "CUMULATIVE GOVERNANCE AUDIT METRICS:",
        "-" * 80,
        f"  Total Requests Evaluated:  {manager.get_metrics()['total_requests']}",
        f"  Accepted Requests:         {manager.get_metrics()['accepted_requests']}",
        f"  Rejected Requests:         {manager.get_metrics()['rejected_requests']}",
        f"  Cumulative Billed Tokens:  {manager.get_metrics()['cumulative_tokens']}",
        "=" * 80,
        "VERDICT: Token Budget Cap Enforced Successfully. Oversized request rejected.",
        "=" * 80,
    ]

    transcript_text = "\n".join(lines)
    target_path = Path(output_transcript) if output_transcript else PROJECT_ROOT / "transcripts" / "task15_budget_reject.txt"
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(transcript_text, encoding="utf-8")

    return {
        "transcript_path": target_path,
        "transcript_text": transcript_text,
        "normal_tokens": normal_tokens,
        "oversized_tokens": oversized_tokens,
        "rejection_details": rejection_details,
        "metrics": manager.get_metrics(),
    }


if __name__ == "__main__":
    res = demonstrate_token_budget_enforcement()
    print(f"Generated: {res['transcript_path']}")
    print("\n" + res["transcript_text"])
