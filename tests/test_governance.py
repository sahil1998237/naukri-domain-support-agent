"""
Unit Tests for Day 12 / Task 15: Governance and Cost Control.

Validates:
1. Least-Autonomy role isolation across CrewAI agents.
2. Programmatic Least-Autonomy audit and report generation.
3. Medium-to-High Risk classification document completeness and regulatory references.
4. TokenBudgetManager validation of within-budget prompts.
5. TokenBudgetManager rejection and exception raising for oversized prompts.
6. MockLLM integration with per-request token budget caps.
7. Verification of transcripts/task15_budget_reject.txt contents.
8. Execution of demonstrate_token_budget_enforcement().
"""

import os
from pathlib import Path
import sys
import unittest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.crew import (
    build_naukri_support_crew,
    create_composer_agent,
    create_lookup_agent,
    create_retrieval_agent,
)
from agents.mock_llm import MockLLM
from governance.autonomy import (
    LEAST_AUTONOMY_EXPLANATION,
    audit_least_autonomy,
    format_autonomy_report,
)
from governance.cost_budget import (
    TokenBudgetExceededError,
    TokenBudgetManager,
    demonstrate_token_budget_enforcement,
)


class TestDay12GovernanceAndCostControl(unittest.TestCase):
    """Test suite for Task 15 Governance, Least Autonomy, and Token Budget Caps."""

    def setUp(self):
        self.mock_llm = MockLLM()
        self.crew = build_naukri_support_crew(self.mock_llm)

    def test_01_least_autonomy_tool_matrix(self):
        """Check 1: Each agent possesses strictly only the tools authorized for its role."""
        retrieval_agent = create_retrieval_agent(self.mock_llm)
        lookup_agent = create_lookup_agent(self.mock_llm)
        composer_agent = create_composer_agent(self.mock_llm)

        # Retrieval Agent: only retrieve_policy_context (1 tool)
        retrieval_tools = [getattr(t, "name", str(t)) for t in (retrieval_agent.tools or [])]
        self.assertEqual(len(retrieval_tools), 1, "Retrieval agent must have exactly 1 tool")
        self.assertIn("retrieve_policy_context", retrieval_tools)
        self.assertNotIn("check_job_application_status", retrieval_tools)

        # Lookup Agent: only check_job_application_status (1 tool)
        lookup_tools = [getattr(t, "name", str(t)) for t in (lookup_agent.tools or [])]
        self.assertEqual(len(lookup_tools), 1, "Lookup agent must have exactly 1 tool")
        self.assertIn("check_job_application_status", lookup_tools)
        self.assertNotIn("retrieve_policy_context", lookup_tools)

        # Response Composer Agent: zero tools
        composer_tools = [getattr(t, "name", str(t)) for t in (composer_agent.tools or [])]
        self.assertEqual(len(composer_tools), 0, "Composer agent must have zero tools assigned")

    def test_02_least_autonomy_audit_function(self):
        """Check 2: Programmatic audit verifies full system compliance and formats report."""
        audit_res = audit_least_autonomy(self.crew)
        self.assertTrue(audit_res["is_overall_compliant"], "Audit must pass for standard crew")
        self.assertEqual(audit_res["total_agents"], 3, "Must audit exactly 3 agents")

        report = format_autonomy_report(audit_res)
        self.assertIn("PASSED (COMPLIANT)", report)
        self.assertIn("LEAST-AUTONOMY ARCHITECTURAL JUSTIFICATION", report)
        self.assertIn("Naukri Policy Retrieval Specialist", report)
        self.assertIn("Application Status & Escalation Speciali", report)
        self.assertIn("Senior Candidate Support Composer", report)

    def test_03_risk_classification_document_integrity(self):
        """Check 3: Risk classification markdown document exists and contains statutory citations."""
        risk_doc_path = PROJECT_ROOT / "governance" / "risk_classification.md"
        self.assertTrue(risk_doc_path.exists(), "risk_classification.md must exist in governance/")

        content = risk_doc_path.read_text(encoding="utf-8")
        self.assertIn("Medium-to-High Risk", content)
        self.assertIn("EU AI Act", content)
        self.assertIn("Annex III, Section 4", content)
        self.assertIn("DPDP Act 2023", content)
        self.assertIn("NIST AI Risk Management Framework", content)
        self.assertIn("Threat Modeling", content)

    def test_04_token_budget_manager_under_budget_accepted(self):
        """Check 4: TokenBudgetManager permits nominal requests within the per-request ceiling."""
        manager = TokenBudgetManager(budget_limit=2000)
        query = "What is the policy regarding technical interview rescheduling for application APP-003?"
        event = manager.evaluate_request(query)

        self.assertEqual(event["status"], "ACCEPTED")
        self.assertTrue(event["is_allowed"])
        self.assertGreater(event["estimated_tokens"], 0)
        self.assertLess(event["estimated_tokens"], 2000)

        metrics = manager.get_metrics()
        self.assertEqual(metrics["accepted_requests"], 1)
        self.assertEqual(metrics["rejected_requests"], 0)
        self.assertEqual(metrics["cumulative_tokens"], event["estimated_tokens"])

    def test_05_token_budget_manager_oversized_rejected(self):
        """Check 5: TokenBudgetManager rejects oversized prompts with TokenBudgetExceededError."""
        manager = TokenBudgetManager(budget_limit=2000)
        # Create an oversized payload of ~2,500+ tokens
        repeating_block = "Legal policy notice period background verification compensation analysis " * 40
        oversized_query = repeating_block * 15

        with self.assertRaises(TokenBudgetExceededError) as ctx:
            manager.evaluate_request(oversized_query)

        self.assertIn("GOVERNANCE VIOLATION", str(ctx.exception))
        self.assertGreater(ctx.exception.requested_tokens, 2000)
        self.assertEqual(ctx.exception.budget_limit, 2000)

        # Verify usage counters
        metrics = manager.get_metrics()
        self.assertEqual(metrics["accepted_requests"], 0)
        self.assertEqual(metrics["rejected_requests"], 1)
        self.assertEqual(metrics["cumulative_tokens"], 0)

    def test_06_mock_llm_token_budget_enforcement(self):
        """Check 6: MockLLM natively enforces max_token_budget during text generation."""
        strict_llm = MockLLM(max_token_budget=50)

        # Short prompt passes
        short_prompt = "Hello Naukri support"
        resp = strict_llm.call(messages=short_prompt)
        self.assertTrue(len(str(resp)) > 0)

        # Long prompt fails budget
        long_prompt = "Detailed candidate request with extensive context. " * 30
        with self.assertRaises(TokenBudgetExceededError):
            strict_llm.call(messages=long_prompt)

    def test_07_transcript_task15_budget_reject_exists(self):
        """Check 7: Official Task 15 transcript exists and records budget rejection."""
        transcript_path = PROJECT_ROOT / "transcripts" / "task15_budget_reject.txt"
        self.assertTrue(transcript_path.exists(), "task15_budget_reject.txt must exist")

        text = transcript_path.read_text(encoding="utf-8")
        self.assertIn("TASK 15 TOKEN BUDGET GOVERNANCE TRANSCRIPT", text)
        self.assertIn("SCENARIO 1: NORMAL CANDIDATE INQUIRY (WITHIN BUDGET)", text)
        self.assertIn("SCENARIO 2: OVERSIZED RUNAWAY REQUEST (EXCEEDING BUDGET CAP)", text)
        self.assertIn("REJECTED IMMEDIATELY BEFORE LLM INVOCATION", text)
        self.assertIn("TokenBudgetExceededError", text)
        self.assertIn("VERDICT: Token Budget Cap Enforced Successfully", text)

    def test_08_demonstration_function_execution(self):
        """Check 8: demonstrate_token_budget_enforcement runs cleanly and produces audit dict."""
        demo_res = demonstrate_token_budget_enforcement()
        self.assertIn("transcript_path", demo_res)
        self.assertIn("normal_tokens", demo_res)
        self.assertIn("oversized_tokens", demo_res)
        self.assertIn("rejection_details", demo_res)
        self.assertEqual(demo_res["rejection_details"]["status"], "INTERCEPTED_AND_REJECTED")
        self.assertEqual(demo_res["metrics"]["rejected_requests"], 1)
        self.assertEqual(demo_res["metrics"]["accepted_requests"], 1)


if __name__ == "__main__":
    unittest.main()
