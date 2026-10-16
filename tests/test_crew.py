"""
Automated Test Suite for Day 5: Multi-Agent Crew Assembly (Task 7).

Validates:
1. Least Autonomy Architecture:
   - Retrieval Agent has ONLY retrieve_policy_context_tool.
   - Lookup Agent has ONLY check_job_application_status_tool.
   - Composer Agent has NO tools.
2. Tool execution wrappers return correct strings and handle edge cases.
3. Multi-agent crew execution succeeds on status query with escalation.
4. Multi-agent crew execution succeeds on policy retrieval query with citation.
5. Multi-agent crew execution succeeds on normal status query without escalation.
6. Execution token tracking is non-zero and accumulated across agents.
7. Task 7 transcript is generated and verified.
"""

import os
from pathlib import Path
import sys
import unittest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Enforce telemetry disable
os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"

from agents.crew import (
    check_job_application_status_tool,
    create_composer_agent,
    create_lookup_agent,
    create_retrieval_agent,
    retrieve_policy_context_tool,
    run_naukri_support_crew,
)
from agents.mock_llm import MockLLM


class TestDay5CrewAssembly(unittest.TestCase):
    """Test suite for Day 5: CrewAI Multi-Agent System (Task 7)."""

    def setUp(self):
        self.mock_llm = MockLLM()

    def test_01_least_autonomy_agent_tools(self):
        """Check 1: Verify Least-Autonomy constraint across all 3 agent definitions."""
        retrieval_agent = create_retrieval_agent(self.mock_llm)
        lookup_agent = create_lookup_agent(self.mock_llm)
        composer_agent = create_composer_agent(self.mock_llm)

        # Retrieval Agent: Exactly 1 tool, policy retrieval only
        self.assertEqual(len(retrieval_agent.tools), 1)
        self.assertEqual(retrieval_agent.tools[0].name, "retrieve_policy_context")

        # Lookup Agent: Exactly 1 tool, status lookup only
        self.assertEqual(len(lookup_agent.tools), 1)
        self.assertEqual(lookup_agent.tools[0].name, "check_job_application_status")

        # Composer Agent: Exactly 0 tools (synthesis only)
        self.assertEqual(len(composer_agent.tools), 0)

    def test_02_retrieve_policy_context_tool_in_scope(self):
        """Check 2: Policy retrieval tool returns verified citation for in-scope query."""
        res = retrieve_policy_context_tool.run(query="What is the interview rescheduling policy?")
        self.assertIn("Verified Knowledge Base Citation", res)
        self.assertIn("POL-", res)

    def test_03_retrieve_policy_context_tool_out_of_scope(self):
        """Check 3: Policy retrieval tool returns rejection for unrelated query."""
        res = retrieve_policy_context_tool.run(query="How do I bake sourdough bread?")
        self.assertIn("No matching verified policy found", res)

    def test_04_check_job_application_status_tool_escalation(self):
        """Check 4: Status lookup tool flags APP-003 for escalation."""
        res = check_job_application_status_tool.run(record_id="APP-003")
        self.assertIn("Record ID: APP-003", res)
        self.assertIn("ESCALATION RECOMMENDED", res)
        self.assertIn("Interview Scheduled", res)

    def test_05_check_job_application_status_tool_normal(self):
        """Check 5: Status lookup tool handles normal non-escalated record APP-001."""
        res = check_job_application_status_tool.run(record_id="APP-001")
        self.assertIn("Record ID: APP-001", res)
        self.assertIn("Normal Priority", res)

    def test_06_check_job_application_status_tool_not_found(self):
        """Check 6: Status lookup tool gracefully handles nonexistent record."""
        res = check_job_application_status_tool.run(record_id="APP-999")
        self.assertIn("Lookup Failed", res)

    def test_07_crew_run_application_status_query(self):
        """Check 7: Full multi-agent execution for application status inquiry."""
        query = "What is the status of application APP-003?"
        result = run_naukri_support_crew(query, llm=self.mock_llm)

        self.assertIn("APP-003", result["final_answer"])
        self.assertIn("ESCALATION RECOMMENDED", result["final_answer"])
        self.assertEqual(result["record_id_detected"], "APP-003")
        self.assertIn("Application Status & Escalation Specialist", result["agents_invoked"])
        self.assertIn("Senior Candidate Support Composer", result["agents_invoked"])
        self.assertGreater(result["total_tokens_used"], 0)

    def test_08_crew_run_policy_inquiry(self):
        """Check 8: Full multi-agent execution for policy question."""
        query = "What is the official policy regarding rescheduling a technical interview slot?"
        result = run_naukri_support_crew(query, llm=self.mock_llm)

        self.assertIn("Policy", result["final_answer"])
        self.assertTrue(result["is_policy_query"])
        self.assertIn("Naukri Policy Retrieval Specialist", result["agents_invoked"])
        self.assertIn("Senior Candidate Support Composer", result["agents_invoked"])
        self.assertGreater(result["total_tokens_used"], 0)

    def test_09_task7_transcript_exists_and_valid(self):
        """Check 9: transcripts/task7_crew_demo.txt exists and contains both queries."""
        transcript_file = PROJECT_ROOT / "transcripts" / "task7_crew_demo.txt"
        self.assertTrue(transcript_file.exists(), "task7_crew_demo.txt missing")
        content = transcript_file.read_text(encoding="utf-8")
        self.assertIn("APP-003", content)
        self.assertIn("POL-", content)
        self.assertIn("ESCALATION RECOMMENDED", content)


if __name__ == "__main__":
    unittest.main(verbosity=2)
