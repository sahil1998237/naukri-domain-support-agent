"""
Automated Unit Tests for Day 11: AutoGen Multi-Agent Policy Review Stage (Task 14).

Verifies:
1. VerdictModel Pydantic schema validation and continuous score bounds [0.0, 1.0].
2. RoundRobinGroupChat initialization with custom_message_types=[StructuredMessage[VerdictModel]] (Trap 5 Defeated).
3. Demonstration of Trap 5 avoidance (ensuring custom message typing protects against runtime schema errors).
4. Full execution of the Review-Approve workflow (verifying grounded policy adherence).
5. Full execution of the Review-Revise workflow (verifying interception and excision of ungrounded bonus claim).
6. Existence and structural integrity of transcripts/task14_review_approve.txt.
7. Existence and structural integrity of transcripts/task14_review_revise.txt.
8. Deterministic local execution with zero external API keys.
"""

import asyncio
from pathlib import Path
import unittest

from pydantic import ValidationError

PROJECT_ROOT = Path(__file__).resolve().parent.parent

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import StructuredMessage
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_ext.models.replay import ReplayChatCompletionClient

from review.autogen_review import (
    EDITOR_APPROVAL_VERDICT,
    EDITOR_REVISION_VERDICT,
    GROUNDED_DRAFT_TEXT,
    UNGROUNDED_DRAFT_TEXT,
    VerdictModel,
    create_autogen_review_team,
    generate_task14_transcripts,
    run_autogen_review_stage,
)


class TestDay11AutoGenReviewStage(unittest.TestCase):
    """Unit test suite verifying Task 14 AutoGen review multi-agent pipeline."""

    @classmethod
    def setUpClass(cls):
        """Ensure both transcripts are generated before running assertion suite."""
        generate_task14_transcripts()

    def test_01_verdict_model_schema_validation(self):
        """Check 1: VerdictModel validates required fields and strictly enforces compliance_score bounds."""
        model = VerdictModel(
            is_approved=True,
            policy_citations=["POL-001"],
            review_comments="All clauses verified against official policies.",
            revised_answer="Approved text.",
            compliance_score=0.95,
            has_hallucinations_removed=False,
        )
        self.assertTrue(model.is_approved)
        self.assertTrue(model.approved)
        self.assertEqual(model.final_answer, "Approved text.")
        self.assertEqual(model.reason, "All clauses verified against official policies.")
        self.assertEqual(model.compliance_score, 0.95)
        self.assertEqual(model.policy_citations, ["POL-001"])

        # Direct instantiation with spec-mandated fields (approved, final_answer, reason)
        spec_model = VerdictModel(
            approved=True,
            final_answer="Direct answer.",
            reason="Verified successfully.",
            compliance_score=1.0,
        )
        self.assertTrue(spec_model.approved)
        self.assertTrue(spec_model.is_approved)
        self.assertEqual(spec_model.final_answer, "Direct answer.")
        self.assertEqual(spec_model.reason, "Verified successfully.")

        # Out-of-bounds score rejection (< 0.0 or > 1.0)
        with self.assertRaises(ValidationError):
            VerdictModel(
                is_approved=True,
                policy_citations=[],
                review_comments="Invalid score test",
                revised_answer="Text",
                compliance_score=1.5,  # Exceeds 1.0
            )

    def test_02_team_initialization_with_custom_message_types(self):
        """Check 2: Trap 5 Defeated - RoundRobinGroupChat configures custom_message_types with StructuredMessage[VerdictModel]."""
        team = create_autogen_review_team(is_approval_case=True)
        self.assertIsInstance(team, RoundRobinGroupChat)
        self.assertEqual(len(team._participants), 2)
        
        # Verify custom message type is registered on team's message factory
        is_registered = team._message_factory.is_registered(StructuredMessage[VerdictModel])
        self.assertTrue(
            is_registered,
            "StructuredMessage[VerdictModel] must be registered in team._message_factory",
        )

    def test_03_trap5_contract_avoidance_verification(self):
        """Check 3: Verifies that StructuredMessage[VerdictModel] is compatible with team message dispatch."""
        sample_verdict = EDITOR_APPROVAL_VERDICT
        structured_msg = StructuredMessage(
            source="Final_Editor",
            content=sample_verdict,
        )
        self.assertIsInstance(structured_msg.content, VerdictModel)
        self.assertEqual(structured_msg.content.compliance_score, 1.00)

    def test_04_review_approve_workflow_execution(self):
        """Check 4: Approval workflow approves grounded draft and retains clean compliance status."""
        result = asyncio.run(run_autogen_review_stage(GROUNDED_DRAFT_TEXT, is_approval_case=True))
        self.assertTrue(result["is_approval_case"])
        verdict = result["verdict"]
        self.assertIsNotNone(verdict)
        self.assertTrue(verdict.is_approved)
        self.assertEqual(verdict.compliance_score, 1.00)
        self.assertFalse(verdict.has_hallucinations_removed)
        self.assertIn("POL-001", verdict.policy_citations)
        self.assertIn("3 to 5 business days", verdict.revised_answer)

    def test_05_review_revise_workflow_execution(self):
        """Check 5: Revision workflow intercepts ungrounded claim and excises fabricated clauses."""
        result = asyncio.run(run_autogen_review_stage(UNGROUNDED_DRAFT_TEXT, is_approval_case=False))
        self.assertFalse(result["is_approval_case"])
        verdict = result["verdict"]
        self.assertIsNotNone(verdict)
        self.assertFalse(verdict.is_approved)
        self.assertTrue(verdict.has_hallucinations_removed)
        self.assertLess(verdict.compliance_score, 0.60)
        
        # Critical assertion: Fabricated clauses must NOT exist in the revised answer
        self.assertNotIn("50,000", verdict.revised_answer)
        self.assertNotIn("unlimited", verdict.revised_answer.lower())
        self.assertNotIn("wallet", verdict.revised_answer.lower())

    def test_06_transcript_task14_approve_exists_and_verified(self):
        """Check 6: Verification transcript task14_review_approve.txt exists and contains evidence."""
        approve_path = PROJECT_ROOT / "transcripts" / "task14_review_approve.txt"
        self.assertTrue(approve_path.exists(), "task14_review_approve.txt does not exist")
        content = approve_path.read_text(encoding="utf-8")
        self.assertIn("TASK 14 AUTOGEN REVIEW (SCENARIO 1: APPROVAL)", content)
        self.assertIn("Approved:                  True", content)
        self.assertIn("Compliance Score:          1.00 / 1.00", content)
        self.assertIn("Hallucinations Excised:    False", content)
        self.assertIn("StructuredMessage[VerdictModel]", content)

    def test_07_transcript_task14_revise_exists_and_verified(self):
        """Check 7: Verification transcript task14_review_revise.txt exists and contains evidence."""
        revise_path = PROJECT_ROOT / "transcripts" / "task14_review_revise.txt"
        self.assertTrue(revise_path.exists(), "task14_review_revise.txt does not exist")
        content = revise_path.read_text(encoding="utf-8")
        self.assertIn("TASK 14 AUTOGEN REVIEW (SCENARIO 2: REVISION)", content)
        self.assertIn("Approved:                  False", content)
        self.assertIn("Hallucinations Excised:    True", content)
        self.assertIn("CRITICAL VIOLATION DETECTED", content)
        self.assertIn("StructuredMessage[VerdictModel]", content)

    def test_08_deterministic_mock_execution_zero_api_keys(self):
        """Check 8: Review pipeline executes completely offline in < 2.0s with zero external API keys."""
        import time
        t0 = time.perf_counter()
        asyncio.run(run_autogen_review_stage("Short test draft", is_approval_case=True))
        elapsed = time.perf_counter() - t0
        self.assertLess(elapsed, 2.0, f"AutoGen mock execution took {elapsed:.2f}s (should be near-instantaneous)")


if __name__ == "__main__":
    unittest.main(verbosity=2)
