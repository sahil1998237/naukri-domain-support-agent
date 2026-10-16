"""
Deterministic MockLLM for Naukri.com Multi-Agent Crew (Day 4 / Task 7 setup).

Extends crewai.llms.base_llm.BaseLLM to allow the entire capstone system
to execute 100% locally with zero external API keys and zero cost.

Architectural Safeguards Implemented:
1. ReAct Parser Trap Avoidance: Inspects actual message history without confusing
   system prompt template instructions ('Observation: ...') with real execution output.
2. Argument-Schema-Based Tool Dispatch: Routes tools based on argument signatures rather
   than ambiguous name substrings (e.g., distinguishing 'APP-XXX' from policy questions).
3. Telemetry Protection: Disables CrewAI telemetry on module import.
4. Token Counter: Tracks per-request and cumulative token usage for governance budget control.
"""

import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Dict, List, Optional, Type, Union

# Set telemetry disable flag before crewai initializes
os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from crewai.llms.base_llm import BaseLLM
from pydantic import BaseModel, Field

from agents.tools import check_job_application_status
from rag.retriever import generate_grounded_response


class MockLLM(BaseLLM):
    """
    Deterministic rule-based LLM extending CrewAI's BaseLLM interface.
    Operates without API keys while supporting CrewAI's ReAct execution loop,
    structured tool calls, and Pydantic schema validation.
    """

    total_tokens: int = Field(default=0)
    request_count: int = Field(default=0)
    max_token_budget: Optional[int] = Field(default=None)

    def __init__(
        self,
        model: str = "mock-naukri-support-llm",
        total_tokens: int = 0,
        request_count: int = 0,
        max_token_budget: Optional[int] = None,
        **kwargs: Any,
    ):
        super().__init__(model=model, **kwargs)
        self.total_tokens = total_tokens
        self.request_count = request_count
        self.max_token_budget = max_token_budget

    def get_context_window_size(self) -> int:
        """Return context window token capacity."""
        return 8192

    def supports_stop_words(self) -> bool:
        return True

    def supports_multimodal(self) -> bool:
        return False

    def _extract_prompt_text(self, messages: Union[str, List[Any]]) -> str:
        """Convert input messages into a single normalized string for inspection."""
        if isinstance(messages, str):
            return messages
        if isinstance(messages, list):
            parts = []
            for m in messages:
                if isinstance(m, str):
                    parts.append(m)
                elif hasattr(m, "content"):
                    parts.append(str(m.content))
                elif isinstance(m, dict):
                    parts.append(str(m.get("content", "")))
                else:
                    parts.append(str(m))
            return "\n".join(parts)
        return str(messages)

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count using whitespace and punctuation heuristics."""
        words = len(text.split())
        return max(1, int(words * 1.3))

    def call(
        self,
        messages: Union[str, List[Any]],
        tools: Optional[List[Any]] = None,
        callbacks: Optional[List[Any]] = None,
        available_functions: Optional[Dict[str, Any]] = None,
        from_task: Optional[Any] = None,
        from_agent: Optional[Any] = None,
        response_model: Optional[Type[BaseModel]] = None,
        **kwargs: Any,
    ) -> Union[str, Any]:
        """
        Execute deterministic mock generation.

        Handles:
        1. Tool invocation generation (Action & Action Input).
        2. Final Answer generation after tool observation.
        3. Pydantic structured output validation.
        """
        self.request_count += 1
        prompt = self._extract_prompt_text(messages)
        tokens_in = self._estimate_tokens(prompt)

        # Token Budget Cap Check (Task 15 Governance)
        if self.max_token_budget is not None and tokens_in > self.max_token_budget:
            from governance.cost_budget import TokenBudgetExceededError
            raise TokenBudgetExceededError(
                requested_tokens=tokens_in,
                budget_limit=self.max_token_budget,
            )

        # -------------------------------------------------------------
        # Trap 1 Safeguard: Parse only actual execution history
        # Do not mistake system prompt template 'Observation: result' for a real tool result.
        # A real tool execution occurs when 'Observation:' is followed by tool output
        # AFTER the Action was invoked.
        # -------------------------------------------------------------
        has_real_observation = False
        last_observation_content = ""

        # Check if there is an Action followed by an Observation in the conversation
        action_match = re.search(r"Action:\s*(\w+)", prompt, re.IGNORECASE)
        obs_match = re.search(r"Observation:\s*(.+?)(?:\nThought:|\Z)", prompt, re.DOTALL | re.IGNORECASE)

        if action_match and obs_match:
            obs_start = obs_match.start()
            action_start = action_match.start()
            if obs_start > action_start:
                has_real_observation = True
                last_observation_content = obs_match.group(1).strip()

        # Check for record ID pattern (e.g. APP-001)
        record_id_match = re.search(r"\bAPP-\d{3}\b", prompt, re.IGNORECASE)
        record_id = record_id_match.group(0).upper() if record_id_match else None

        # Check if the calling agent is the Composer or an agent with no tools
        is_composer = (
            (from_agent is not None and hasattr(from_agent, "role") and "Composer" in from_agent.role)
            or ("Composer" in prompt)
            or (from_agent is not None and tools is not None and len(tools) == 0)
        )

        # Extract clean candidate query from prompt or task description
        task_desc = getattr(from_task, "description", "") if from_task else ""
        q_match = re.search(r"(?:candidate query|inquiry|query):\s*['\"](.+?)['\"]", prompt + " " + task_desc, re.IGNORECASE)
        search_query = q_match.group(1).strip() if q_match else prompt.split("\n")[-1].replace("Question:", "").replace("User:", "").strip()
        search_query = re.sub(r"\(Context:.*?\)", "", search_query).strip()

        # Policy keywords check against candidate query text (not the tool descriptions in system prompt)
        policy_keywords = ["policy", "reschedule", "interview", "cooling", "salary", "offer", "bgv", "notice", "buyout", "screening", "grievance", "privacy"]
        is_policy_query = any(k in search_query.lower() for k in policy_keywords)

        # -------------------------------------------------------------
        # Case A: Tool has already executed and returned an Observation,
        # OR the agent is a Composer with no tools (synthesis mode).
        # -------------------------------------------------------------
        if has_real_observation or is_composer:
            # Subcase A1: Composite inquiry (Policy inquiry relating to an application)
            if is_policy_query and record_id:
                policy_res = generate_grounded_response(search_query)
                status_data = check_job_application_status(record_id)
                status_prefix = (
                    f"Application {record_id} ({status_data.get('category', 'Role')}) is currently in '{status_data.get('status', 'Active')}' status."
                    if status_data["found"]
                    else f"Regarding application {record_id}:"
                )
                if policy_res["is_grounded"]:
                    policy_part = policy_res["answer"]
                else:
                    policy_part = (
                        "According to Naukri.com Policy [POL-003: Interview Scheduling, Rescheduling, and Cancellation]: "
                        "Candidates may request a reschedule for technical or managerial interviews up to 24 hours prior to the scheduled slot via the Naukri candidate portal. "
                        "A maximum of two reschedule requests are permitted per application lifecycle before the interview slot is forfeit."
                    )
                final_text = (
                    f"Thought: I will provide the official policy answer in the context of application {record_id}.\n"
                    f"Final Answer: {status_prefix} {policy_part}"
                )
            elif record_id:
                status_data = check_job_application_status(record_id)
                if status_data["found"]:
                    esc_str = "ESCALATION RECOMMENDED" if status_data["escalation_recommended"] else "Normal Priority"
                    final_text = (
                        f"Thought: I have reviewed the application status.\n"
                        f"Final Answer: Application {status_data['record_id']} ({status_data['category']}) "
                        f"is in '{status_data['status']}' status (created {status_data['days_since_created']} days ago). "
                        f"Escalation score is {status_data['escalation_score']} ({esc_str}). "
                        f"Rationale: {status_data['rationale']}."
                    )
                else:
                    final_text = (
                        f"Thought: The record ID was not found.\n"
                        f"Final Answer: {status_data['error']}"
                    )
            else:
                # If we have an observation from the retrieval agent, use it
                if last_observation_content and "Citation:" in last_observation_content:
                    final_text = (
                        f"Thought: I have reviewed the retrieved policy documentation.\n"
                        f"Final Answer: {last_observation_content}"
                    )
                else:
                    # Look up grounded policy response directly using clean query
                    policy_res = generate_grounded_response(search_query)
                    final_text = (
                        f"Thought: I will provide the official policy answer.\n"
                        f"Final Answer: {policy_res['answer']}"
                    )

            tokens_out = self._estimate_tokens(final_text)
            self.total_tokens += tokens_in + tokens_out
            return final_text

        # -------------------------------------------------------------
        # Case B: Initial Tool Dispatch (Trap 2 Safeguard)
        # Dispatch by schema/intent or specific agent role assignment.
        # -------------------------------------------------------------
        agent_role = getattr(from_agent, "role", "") if from_agent else ""

        # If from Retrieval Agent OR pure policy query
        if "Retrieval" in agent_role or (is_policy_query and not record_id):
            task_desc = getattr(from_task, "description", "") if from_task else ""
            q_match = re.search(r"(?:query|inquiry):\s*['\"](.+?)['\"]", prompt + " " + task_desc, re.IGNORECASE)
            if q_match:
                clean_q = q_match.group(1).strip()
            else:
                clean_q = "What is the policy for rescheduling an interview?"

            action_text = (
                f"Thought: The candidate has a policy question. I need to search the knowledge base.\n"
                f"Action: retrieve_policy_context\n"
                f"Action Input: {{\"query\": \"{clean_q}\"}}"
            )
            tokens_out = self._estimate_tokens(action_text)
            self.total_tokens += tokens_in + tokens_out
            return action_text

        # If from Lookup Agent OR application record ID present
        if "Lookup" in agent_role or record_id:
            target_id = record_id or "APP-001"
            action_text = (
                f"Thought: The candidate is inquiring about job application {target_id}. "
                f"I must look up the application details and calculate the escalation score.\n"
                f"Action: check_job_application_status\n"
                f"Action Input: {{\"record_id\": \"{target_id}\"}}"
            )
            tokens_out = self._estimate_tokens(action_text)
            self.total_tokens += tokens_in + tokens_out
            return action_text

        if is_policy_query:
            clean_q = prompt.split("\n")[-1].replace("Question:", "").replace("User:", "").strip()
            action_text = (
                f"Thought: The candidate has a policy question. I need to search the knowledge base.\n"
                f"Action: retrieve_policy_context\n"
                f"Action Input: {{\"query\": \"{clean_q}\"}}"
            )
            tokens_out = self._estimate_tokens(action_text)
            self.total_tokens += tokens_in + tokens_out
            return action_text

        # -------------------------------------------------------------
        # Case C: Out-of-Scope or General Synthesis (No tools required)
        # -------------------------------------------------------------
        final_text = (
            "Thought: This inquiry does not match any internal application record or policy document.\n"
            "Final Answer: I do not have sufficient policy information in the knowledge base to answer this question. "
            "Please contact the Naukri Help Desk or an HR representative."
        )
        tokens_out = self._estimate_tokens(final_text)
        self.total_tokens += tokens_in + tokens_out
        return final_text


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("=== Testing MockLLM (Day 4 / Task 7 Setup) ===")
    mock = MockLLM()

    # Test 1: Tool invocation generation for status inquiry
    prompt1 = "Candidate asks: What is the status of my application APP-003?"
    response1 = mock.call(prompt1)
    print(f"\n[Test 1: Status Lookup Dispatch]")
    print(response1)

    # Test 2: Synthesis after Observation
    prompt2 = (
        "Action: check_job_application_status\n"
        "Action Input: {\"record_id\": \"APP-003\"}\n"
        "Observation: Application APP-003 is currently Interview Scheduled (created 37 days ago, priority=True).\n"
        "Thought: I now have the status.\n"
    )
    response2 = mock.call(prompt2)
    print(f"\n[Test 2: ReAct Observation Handling]")
    print(response2)

    # Test 3: RAG Policy Query Dispatch
    prompt3 = "Candidate asks: What is the policy for rescheduling an interview?"
    response3 = mock.call(prompt3)
    print(f"\n[Test 3: Policy Retrieval Dispatch]")
    print(response3)

    print(f"\nTotal Mock Tokens Tracked: {mock.total_tokens}")
