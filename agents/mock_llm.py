"""
Deterministic MockLLM for Naukri.com Multi-Agent Crew (Day 4 / Task 7 setup).

Extends crewai.llms.base_llm.BaseLLM to allow the entire capstone system
to execute 100% locally with zero external API keys and zero cost.

ARCHITECTURAL CONSTRAINT (Least Autonomy Compliance):
This LLM does NOT import or directly invoke any tool backend functions
(check_job_application_status, generate_grounded_response, etc.).
It operates purely through CrewAI's ReAct execution loop:
1. Specialist agents (Turn 1): Emits Action + Action Input for CrewAI to execute.
2. Specialist agents (Turn 2): Reads the Observation text placed by CrewAI after
   tool execution and synthesizes a Final Answer from it — no backdoor tool calls.
3. Composer agent (zero tools): Reads the upstream context block forwarded by
   CrewAI from specialist agents and synthesizes a Final Answer purely from text.

Safeguards Implemented:
1. ReAct Parser Trap Avoidance: Distinguishes CrewAI's instructional template
   text ("Observation: the result of the action") from actual tool output by
   only parsing the execution history section after "Begin!".
2. Schema-Based Tool Dispatch: Routes Action requests based on agent role and
   query intent (APP-XXX pattern vs policy keywords).
3. Telemetry Protection: CREWAI_DISABLE_TELEMETRY=true enforced on import.
4. Token Budget Governance: Tracks per-request and cumulative token usage,
   raising TokenBudgetExceededError if the ceiling is exceeded.
"""

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

# NOTE: This module intentionally does NOT import check_job_application_status
# or generate_grounded_response. All tool execution flows exclusively through
# CrewAI's Action -> Tool Executor -> Observation loop.


class MockLLM(BaseLLM):
    """
    Deterministic rule-based LLM extending CrewAI's BaseLLM interface.
    Operates without API keys while supporting CrewAI's ReAct execution loop,
    structured tool calls, and Pydantic schema validation.

    This LLM contains ZERO imports of tool backend functions. All tool execution
    flows exclusively through CrewAI's Action -> Observation loop, ensuring that
    the Least-Autonomy agent architecture is not violated at the LLM layer.
    """

    total_tokens: int = Field(default=0)
    request_count: int = Field(default=0)
    max_token_budget: Optional[int] = Field(default=2000)

    def __init__(
        self,
        model: str = "mock-naukri-support-llm",
        total_tokens: int = 0,
        request_count: int = 0,
        max_token_budget: Optional[int] = 2000,
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

    # ------------------------------------------------------------------
    # Observation Parsing — Trap 1 Safeguard
    # ------------------------------------------------------------------

    def _parse_real_observation(self, prompt: str) -> tuple:
        """
        Parse the prompt for real tool observations, distinguishing them from
        CrewAI's instructional template text.

        CrewAI injects a format block into every specialist agent prompt:
            ```
            Action: the action to take, only one name of [tool_name]...
            Action Input: ...
            Observation: the result of the action
            ```
        These are INSTRUCTIONS, not real tool output. Real observations appear
        after "Begin!" in the execution history section when CrewAI appends
        actual tool output following an agent's emitted Action.

        Returns:
            (has_real_observation: bool, observation_content: str)
        """
        # Isolate the execution history section (after "Begin!")
        begin_idx = prompt.rfind("Begin!")
        if begin_idx != -1:
            exec_section = prompt[begin_idx:]
        else:
            # Direct test call without CrewAI wrapper — use full prompt
            exec_section = prompt

        # Find Action: lines with an actual tool name (single identifier, not
        # template text like "the action to take, only one name of ...")
        action_pattern = re.compile(r"^Action:\s*(\w+)\s*$", re.MULTILINE)
        action_matches = list(action_pattern.finditer(exec_section))

        if not action_matches:
            return False, ""

        # Look for an Observation after the last real Action
        last_action = action_matches[-1]
        after_action = exec_section[last_action.end():]

        obs_match = re.search(
            r"Observation:\s*(.+?)(?=\nThought:|\Z)",
            after_action,
            re.DOTALL,
        )

        if not obs_match:
            return False, ""

        obs_content = obs_match.group(1).strip()

        # Filter out template placeholder text
        if obs_content.lower().startswith("the result of the action"):
            return False, ""

        # Must have substantive content (real tool output is always > 10 chars)
        if len(obs_content) < 10:
            return False, ""

        return True, obs_content

    # ------------------------------------------------------------------
    # Context Extraction (for Composer Agent)
    # ------------------------------------------------------------------

    def _extract_context_block(self, prompt: str) -> str:
        """
        Extract the context block that CrewAI forwards from upstream agents.

        CrewAI sequential process provides context as:
            This is the context you're working with:
            <accumulated output from previous specialist tasks>
        """
        ctx_match = re.search(
            r"This is the context you're working with:\s*\n(.+?)(?:\n\nProvide|\Z)",
            prompt,
            re.DOTALL,
        )
        if ctx_match:
            return ctx_match.group(1).strip()
        return ""

    # ------------------------------------------------------------------
    # Query Extraction
    # ------------------------------------------------------------------

    def _extract_query(self, prompt: str, task_desc: str = "") -> str:
        """Extract the candidate's original query from the prompt."""
        combined = prompt + " " + task_desc
        q_match = re.search(
            r"(?:candidate query|inquiry|query):\s*['\"](.+?)['\"]",
            combined,
            re.IGNORECASE,
        )
        if q_match:
            return q_match.group(1).strip()
        # Fallback: task description
        task_match = re.search(r"Current Task:\s*(.+?)(?:\n|$)", prompt)
        if task_match:
            return task_match.group(1).strip()
        return prompt.split("\n")[-1].replace("Thought:", "").strip()

    # ------------------------------------------------------------------
    # Main Generation Entry Point
    # ------------------------------------------------------------------

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
        Execute deterministic mock generation through CrewAI's ReAct loop.

        Flow:
        PATH A: Composer agent (zero tools) → synthesize from upstream context only.
        PATH B: Specialist agent Turn 2 (observation present) → synthesize from observation.
        PATH C: Specialist agent Turn 1 (no observation) → emit Action + Action Input.
        PATH D: Out-of-scope query → emit Final Answer with safe fallback.
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

        # Determine agent role and task description
        agent_role = getattr(from_agent, "role", "") if from_agent else ""
        task_desc = getattr(from_task, "description", "") if from_task else ""

        # Detect if this is the Composer agent (zero-tool synthesis agent)
        is_composer = (
            "Composer" in agent_role
            or (
                from_agent is not None
                and hasattr(from_agent, "tools")
                and len(getattr(from_agent, "tools", []) or []) == 0
                and "Composer" in prompt
            )
        )

        # Parse real observation (not template text)
        has_real_observation, observation_text = self._parse_real_observation(prompt)

        # Extract clean query and detect policy intent
        search_query = self._extract_query(prompt, task_desc)

        # Extract record ID with strict precedence to avoid matching examples in tool docstrings (e.g. 'APP-001')
        record_id = None
        if task_desc:
            m = re.search(r"\bAPP-\d{3}\b", task_desc, re.IGNORECASE)
            if m:
                record_id = m.group(0).upper()
        if not record_id and search_query:
            m = re.search(r"\bAPP-\d{3}\b", search_query, re.IGNORECASE)
            if m:
                record_id = m.group(0).upper()
        if not record_id:
            task_section = re.search(r"Current Task:\s*(.+)", prompt, re.DOTALL)
            if task_section:
                m = re.search(r"\bAPP-\d{3}\b", task_section.group(1), re.IGNORECASE)
                if m:
                    record_id = m.group(0).upper()
        if not record_id:
            # Fallback: take the LAST occurrence in prompt (past any tool descriptions)
            all_matches = list(re.finditer(r"\bAPP-\d{3}\b", prompt, re.IGNORECASE))
            if all_matches:
                record_id = all_matches[-1].group(0).upper()
        policy_keywords = [
            "policy", "reschedule", "interview", "cooling", "salary",
            "offer", "bgv", "notice", "buyout", "screening", "grievance", "privacy",
        ]
        is_policy_query = (not record_id) or any(k in search_query.lower() for k in policy_keywords)

        # -----------------------------------------------------------------
        # PATH A: Composer Agent (Zero Tools)
        # Synthesize purely from the upstream context block forwarded by CrewAI.
        # The Composer NEVER executes tools — it reads specialist findings only.
        # -----------------------------------------------------------------
        if is_composer:
            context = self._extract_context_block(prompt)
            if context:
                final_text = (
                    f"Thought: I have reviewed the findings from the specialist agents.\n"
                    f"Final Answer: {context}"
                )
            elif has_real_observation:
                # Fallback for edge cases where context is embedded as observation
                final_text = (
                    f"Thought: I have reviewed the specialist findings.\n"
                    f"Final Answer: {observation_text}"
                )
            else:
                final_text = (
                    "Thought: No specialist findings were provided.\n"
                    "Final Answer: I do not have sufficient information to answer "
                    "this question. Please contact the Naukri Help Desk or an HR "
                    "representative for assistance."
                )
            tokens_out = self._estimate_tokens(final_text)
            self.total_tokens += tokens_in + tokens_out
            return final_text

        # -----------------------------------------------------------------
        # PATH B: Specialist Agent Turn 2 — Observation Already Received
        # CrewAI executed the tool and placed the output as an Observation.
        # Synthesize Final Answer from the observation text ONLY.
        # -----------------------------------------------------------------
        if has_real_observation:
            answer_text = self._synthesize_from_observation(observation_text)
            final_text = (
                f"Thought: I have reviewed the tool observation and gathered the required information.\n"
                f"Final Answer: {answer_text}"
            )
            tokens_out = self._estimate_tokens(final_text)
            self.total_tokens += tokens_in + tokens_out
            return final_text

        # -----------------------------------------------------------------
        # PATH C: Specialist Agent Turn 1 — Emit Action + Action Input
        # This is the first call for a specialist agent. Emit the appropriate
        # tool invocation for CrewAI to execute.
        # -----------------------------------------------------------------

        # Retrieval Agent or policy query without record ID
        if "Retrieval" in agent_role or "Policy" in agent_role or (is_policy_query and not record_id):
            clean_q = search_query if search_query else "What is the policy for rescheduling an interview?"
            # Remove any surrounding quotes from extracted query
            clean_q = clean_q.strip("'\"")
            action_text = (
                f"Thought: The candidate has a policy question. I need to search the knowledge base.\n"
                f"Action: retrieve_policy_context\n"
                f'Action Input: {{"query": "{clean_q}"}}'
            )
            tokens_out = self._estimate_tokens(action_text)
            self.total_tokens += tokens_in + tokens_out
            return action_text

        # Lookup Agent or application record ID present
        if "Lookup" in agent_role or "Status" in agent_role or "Escalation" in agent_role or record_id:
            target_id = record_id or "APP-001"
            action_text = (
                f"Thought: The candidate is inquiring about job application {target_id}. "
                f"I must look up the application details and calculate the escalation score.\n"
                f"Action: check_job_application_status\n"
                f'Action Input: {{"record_id": "{target_id}"}}'
            )
            tokens_out = self._estimate_tokens(action_text)
            self.total_tokens += tokens_in + tokens_out
            return action_text

        # Policy query fallback (agent role not detected but keywords present)
        if is_policy_query:
            clean_q = search_query or "What is the relevant policy?"
            action_text = (
                f"Thought: The candidate has a policy question. I need to search the knowledge base.\n"
                f"Action: retrieve_policy_context\n"
                f'Action Input: {{"query": "{clean_q}"}}'
            )
            tokens_out = self._estimate_tokens(action_text)
            self.total_tokens += tokens_in + tokens_out
            return action_text

        # -----------------------------------------------------------------
        # PATH D: Out-of-Scope or General (No tools required)
        # -----------------------------------------------------------------
        final_text = (
            "Thought: This inquiry does not match any internal application record or policy document.\n"
            "Final Answer: I do not have sufficient policy information in the knowledge base to answer this question. "
            "Please contact the Naukri Help Desk or an HR representative."
        )
        tokens_out = self._estimate_tokens(final_text)
        self.total_tokens += tokens_in + tokens_out
        return final_text

    # ------------------------------------------------------------------
    # Observation Synthesis (Pure Text Parsing — No Tool Imports)
    # ------------------------------------------------------------------

    def _synthesize_from_observation(self, observation_text: str) -> str:
        """
        Compose a Final Answer by reading the observation text that CrewAI placed
        into the prompt after executing a tool. This method does NOT call any
        backend functions — it parses the text output only.

        Handles two observation formats:
        1. Application Status (from check_job_application_status_tool):
           Contains "Record ID:", "Summary:", "Escalation Score:", etc.
        2. Policy Retrieval (from retrieve_policy_context_tool):
           Contains "[Verified Knowledge Base Citation: ...]", "Policy Text:", etc.
        3. Error / Not Found results.
        """
        # --- Application Status Observations ---
        # The tool returns structured fields with a Summary line at the end.
        summary_match = re.search(r"Summary:\s*(.+)", observation_text)
        if summary_match:
            summary = summary_match.group(1).strip()
            # Extract escalation verdict from the observation
            if "ESCALATION RECOMMENDED" in observation_text:
                esc_match = re.search(r"Escalation Score:\s*([\d.]+)", observation_text)
                score = esc_match.group(1) if esc_match else ""
                record_match = re.search(r"Record ID:\s*(\S+)", observation_text)
                rid = record_match.group(1) if record_match else ""
                role_match = re.search(r"Role:\s*(.+?)$", observation_text, re.MULTILINE)
                role = role_match.group(1).strip() if role_match else ""
                status_match = re.search(r"Status:\s*(.+?)$", observation_text, re.MULTILINE)
                status = status_match.group(1).strip() if status_match else ""
                days_match = re.search(r"Days Active:\s*(\d+)", observation_text)
                days = days_match.group(1) if days_match else ""

                return (
                    f"Application {rid} ({role}) is in '{status}' status "
                    f"(created {days} days ago). Escalation score is {score} "
                    f"(ESCALATION RECOMMENDED). "
                    f"Rationale: {self._extract_rationale(summary)}."
                )
            else:
                # Normal priority — extract key fields from summary
                record_match = re.search(r"Record ID:\s*(\S+)", observation_text)
                rid = record_match.group(1) if record_match else ""
                role_match = re.search(r"Role:\s*(.+?)$", observation_text, re.MULTILINE)
                role = role_match.group(1).strip() if role_match else ""
                status_match = re.search(r"Status:\s*(.+?)$", observation_text, re.MULTILINE)
                status = status_match.group(1).strip() if status_match else ""
                days_match = re.search(r"Days Active:\s*(\d+)", observation_text)
                days = days_match.group(1) if days_match else ""
                esc_match = re.search(r"Escalation Score:\s*([\d.]+)", observation_text)
                score = esc_match.group(1) if esc_match else ""

                return (
                    f"Application {rid} ({role}) is in '{status}' status "
                    f"(created {days} days ago). Escalation score is {score} "
                    f"(Normal Priority). "
                    f"Rationale: {self._extract_rationale(summary)}."
                )

        # --- Policy Retrieval Observations ---
        cite_match = re.search(
            r"\[Verified Knowledge Base Citation:\s*(.+?)\]",
            observation_text,
        )
        policy_match = re.search(
            r"Policy Text:\s*(.+?)(?:\nSimilarity|\Z)",
            observation_text,
            re.DOTALL,
        )

        if cite_match and policy_match:
            citation = cite_match.group(1).strip()
            policy_text = policy_match.group(1).strip()
            return f"According to Naukri.com Policy [{citation}]: {policy_text}"

        # --- Error / Not Found ---
        if "No matching verified policy found" in observation_text:
            return observation_text

        if "Lookup Failed" in observation_text:
            return observation_text

        # --- Generic Fallback: Pass observation text through ---
        return observation_text

    def _extract_rationale(self, summary: str) -> str:
        """Extract the Rationale portion from a tool summary string."""
        rat_match = re.search(r"Rationale:\s*(.+?)\.?\s*$", summary)
        if rat_match:
            return rat_match.group(1).strip().rstrip(".")
        return "See application details above"


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

    # Test 2: Synthesis after Observation (reads observation text, no backend call)
    prompt2 = (
        "Action: check_job_application_status\n"
        "Action Input: {\"record_id\": \"APP-003\"}\n"
        "Observation: Record ID: APP-003\n"
        "Role: Software Engineer\n"
        "Status: Interview Scheduled\n"
        "Days Active: 37\n"
        "Priority Flagged: True\n"
        "Escalation Score: 1.0 (Threshold: 0.57 -> ESCALATION RECOMMENDED)\n"
        "Summary: Application APP-003 (Software Engineer): Status is 'Interview Scheduled' "
        "(created 37 days ago, priority=True). Escalation Score: 1.00 "
        "(Threshold: 0.57 -> ESCALATION RECOMMENDED). "
        "Rationale: Candidate was designated for priority review; "
        "Application has been pending for 37 days (>= 30-day escalation window).\n"
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
