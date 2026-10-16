"""
Automated Test Suite for Day 6: Session Memory (Task 8) & Structured Outputs (Task 9).

Validates:
1. Pydantic schema validation, defaults, field bounds, and rejection of blank answers.
2. Serialization to JSON dictionary and string.
3. LangChain InMemoryChatMessageHistory message retention and ordering.
4. Entity extraction (APP-XXX) across conversation turns.
5. Session isolation (Session A does not leak into Session B).
6. Session wipe and reset via clear_session().
7. Multi-turn pipeline state carryover across turns.
8. Fresh session clean initialization without carryover.
9. Verification of Task 8 demonstration transcripts.
"""

from datetime import datetime
import os
from pathlib import Path
import sys
import unittest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Enforce telemetry disable
os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"

from langchain_core.messages import AIMessage, HumanMessage
from pydantic import ValidationError

from agents.memory import SessionMemoryManager, process_support_query
from agents.mock_llm import MockLLM
from agents.schemas import ApplicationStatusInfo, NaukriSupportResponse, PolicyCitation


class TestDay6MemoryAndSchemas(unittest.TestCase):
    """Test suite for Day 6: Memory Layer (Task 8) and Pydantic Schemas (Task 9)."""

    def setUp(self):
        self.memory = SessionMemoryManager()
        self.mock_llm = MockLLM()

    # --------------------------------------------------------------------------
    # TASK 9: PYDANTIC STRUCTURED OUTPUTS TESTS
    # --------------------------------------------------------------------------

    def test_01_schema_validation_success(self):
        """Check 1: NaukriSupportResponse instantiates and validates all core fields."""
        resp = NaukriSupportResponse(
            query="What is the status of APP-003?",
            answer="Application APP-003 is Interview Scheduled.",
            sources=["Application Record: APP-003"],
            confidence=0.95,
            escalation_needed=True,
            record_id="APP-003",
        )
        self.assertEqual(resp.query, "What is the status of APP-003?")
        self.assertEqual(resp.confidence, 0.95)
        self.assertTrue(resp.escalation_needed)
        self.assertEqual(resp.record_id, "APP-003")
        self.assertIn("Application Record: APP-003", resp.sources)
        self.assertIsInstance(resp.timestamp, str)

        # Test dictionary and JSON serialization
        data_dict = resp.model_dump()
        self.assertEqual(data_dict["record_id"], "APP-003")
        json_str = resp.model_dump_json()
        self.assertIn('"record_id":"APP-003"', json_str)

    def test_02_schema_validation_blank_answer_rejected(self):
        """Check 2: Pydantic rejects blank or whitespace-only answers."""
        with self.assertRaises(ValidationError):
            NaukriSupportResponse(
                query="Hello",
                answer="   ",  # Blank whitespace must fail validation
            )

    def test_03_schema_confidence_bounds(self):
        """Check 3: Confidence score must strictly reside in [0.0, 1.0]."""
        with self.assertRaises(ValidationError):
            NaukriSupportResponse(
                query="Query",
                answer="Answer",
                confidence=1.5,  # Exceeds 1.0
            )
        with self.assertRaises(ValidationError):
            NaukriSupportResponse(
                query="Query",
                answer="Answer",
                confidence=-0.1,  # Sub-zero
            )

    # --------------------------------------------------------------------------
    # TASK 8: LANGCHAIN SESSION MEMORY TESTS
    # --------------------------------------------------------------------------

    def test_04_session_memory_add_and_retrieve(self):
        """Check 4: InMemoryChatMessageHistory stores and retrieves HumanMessage and AIMessage."""
        session_id = "test_sess_001"
        self.memory.add_user_message(session_id, "Can I reschedule my interview?")
        self.memory.add_ai_message(session_id, "Yes, according to POL-003 up to two times.")

        history = self.memory.get_history(session_id)
        messages = history.messages
        self.assertEqual(len(messages), 2)
        self.assertIsInstance(messages[0], HumanMessage)
        self.assertIsInstance(messages[1], AIMessage)
        self.assertIn("reschedule", messages[0].content)

    def test_05_session_entity_extraction(self):
        """Check 5: Memory manager extracts APP-XXX entity from user message."""
        session_id = "test_sess_002"
        self.memory.add_user_message(session_id, "Checking application APP-003 status.")
        active_id = self.memory.get_active_record_id(session_id)
        self.assertEqual(active_id, "APP-003")

    def test_06_session_isolation(self):
        """Check 6: Sessions are completely isolated and do not cross-contaminate."""
        self.memory.add_user_message("session_alpha", "Inquiry for APP-001")
        self.memory.add_user_message("session_beta", "Inquiry for APP-003")

        self.assertEqual(self.memory.get_active_record_id("session_alpha"), "APP-001")
        self.assertEqual(self.memory.get_active_record_id("session_beta"), "APP-003")

        self.assertEqual(len(self.memory.get_history("session_alpha").messages), 1)
        self.assertEqual(len(self.memory.get_history("session_beta").messages), 1)

    def test_07_session_clear(self):
        """Check 7: clear_session() wipes message history and metadata."""
        session_id = "test_sess_clear"
        self.memory.add_user_message(session_id, "My application is APP-003")
        self.assertTrue(self.memory.has_session(session_id))

        self.memory.clear_session(session_id)
        self.assertFalse(self.memory.has_session(session_id))
        self.assertIsNone(self.memory.get_active_record_id(session_id))

    def test_08_multi_turn_entity_carryover(self):
        """Check 8: Multi-turn pipeline carries application entity across conversational turns."""
        session_id = "test_sess_multi_turn"

        # Turn 1: Explicit application ID
        turn1_resp = process_support_query(
            "What is the status of APP-003?",
            session_id=session_id,
            memory=self.memory,
            llm=self.mock_llm,
        )
        self.assertEqual(turn1_resp.record_id, "APP-003")
        self.assertTrue(turn1_resp.escalation_needed)

        # Turn 2: Follow-up question WITHOUT repeating application ID
        turn2_resp = process_support_query(
            "What is the official policy regarding rescheduling an interview for this role?",
            session_id=session_id,
            memory=self.memory,
            llm=self.mock_llm,
        )
        # Entity carried over from session memory
        self.assertEqual(turn2_resp.record_id, "APP-003")
        self.assertIn("POL-003", str(turn2_resp.sources))

    def test_09_fresh_session_no_carryover(self):
        """Check 9: Fresh session has no prior entity carryover."""
        fresh_session = "test_sess_clean_fresh"
        resp = process_support_query(
            "What is the official policy regarding rescheduling an interview slot?",
            session_id=fresh_session,
            memory=self.memory,
            llm=self.mock_llm,
        )
        self.assertIsNone(resp.record_id)
        self.assertFalse(resp.escalation_needed)

    def test_10_task8_transcripts_exist_and_verified(self):
        """Check 10: Task 8 multi-turn and fresh transcripts exist and contain required evidence."""
        multi_file = PROJECT_ROOT / "transcripts" / "task8_memory_multi_turn.txt"
        fresh_file = PROJECT_ROOT / "transcripts" / "task8_memory_fresh.txt"

        self.assertTrue(multi_file.exists(), "task8_memory_multi_turn.txt missing")
        self.assertTrue(fresh_file.exists(), "task8_memory_fresh.txt missing")

        multi_content = multi_file.read_text(encoding="utf-8")
        fresh_content = fresh_file.read_text(encoding="utf-8")

        self.assertIn("APP-003", multi_content)
        self.assertIn("POL-003", multi_content)
        self.assertIn("VERDICT: Multi-Turn State Retention Confirmed", multi_content)

        self.assertIn("VERDICT: Fresh Session Isolation & Reset Confirmed", fresh_content)


if __name__ == "__main__":
    unittest.main(verbosity=2)
