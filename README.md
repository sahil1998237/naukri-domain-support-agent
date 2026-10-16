# 🎓 Naukri.com Domain Support Agent (Capstone)

A production-grade AI support agent system built for the Naukri.com domain, featuring multi-agent orchestration with CrewAI, RAG retrieval with ChromaDB, input/output guardrails, FastAPI HTTP & WebSocket endpoints, and an AutoGen policy review stage.

---

## 📌 Project Status: Days 1–12 Complete (Phase 1, Phase 2, Phase 3 Complete, Phase 4 in Progress — 105 Tests Passing)

- [x] **Day 1: Foundation & Synthetic Dataset (Phase 1)**
  - [x] Python virtual environment isolation (`.venv` on Python 3.13)
  - [x] Seeded synthetic dataset generator (`scripts/generate_dataset.py`)
  - [x] 50-record deterministic JSON dataset (`data/job_applications.json`)
  - [x] 11-point automated unit test suite (`tests/test_dataset.py`)
- [x] **Day 2: Knowledge Base & RAG Indexing (Phase 1)**
  - [x] 12 domain policy documents (`knowledge_base/policies.json`, `knowledge_base/policies.py`)
  - [x] Dual chunking strategies: fixed-size 200/50 & sentence-based (`rag/chunking.py`)
  - [x] ChromaDB vector collections: `naukri_policies_fixed` & `naukri_policies_sentence` (`rag/indexer.py`)
  - [x] Semantic search retriever with cosine similarity scoring (`rag/retriever.py`)
  - [x] 8-point RAG automated test suite (`tests/test_rag.py`)
- [x] **Day 3: Grounded Generation & Strategy Evaluation (Phase 1)**
  - [x] Empirical similarity threshold calibration: `0.4000` (`scripts/calibrate_threshold.py`)
  - [x] Grounded generation pipeline with citations and out-of-scope fallback (`rag/retriever.py`)
  - [x] Comparative evaluation suite with per-query precision/recall arithmetic (`rag/evaluation.py`)
  - [x] Transcripts: `task4_grounded_gen.txt` & `task5_eval_comparison.txt`
  - [x] 5-point evaluation automated test suite (`tests/test_evaluation.py`)
- [x] **Day 4: MOCK_LLM & Escalation Status Tool (Phase 2 Kickoff)**
  - [x] Application status lookup tool (`agents/tools.py`)
  - [x] Continuous escalation score formula: $0.5 \times \text{flagged} + 0.5 \times \min(1.0, \frac{\text{days}}{30})$
  - [x] Empirical 80th percentile escalation threshold: `0.5667`
  - [x] Deterministic `MockLLM` extending `crewai.llms.base_llm.BaseLLM` (`agents/mock_llm.py`)
  - [x] Critical ReAct parser trap avoidance & schema-based tool routing
  - [x] Telemetry disabled: `CREWAI_DISABLE_TELEMETRY=true`
  - [x] 10-point Day 4 automated test suite (`tests/test_agents.py`)
- [x] **Day 5: CrewAI Multi-Agent Assembly (Task 7 & Task 15)**
  - [x] Retrieval Agent: Policy RAG retrieval tool ONLY (Least Autonomy)
  - [x] Lookup Agent: Application status & escalation tool ONLY (Least Autonomy)
  - [x] Response Composer Agent: Zero tools (Synthesis only)
  - [x] Multi-agent orchestration pipeline (`agents/crew.py`)
  - [x] Demonstration transcript showing both tools (`transcripts/task7_crew_demo.txt`)
  - [x] 9-point Day 5 automated test suite (`tests/test_crew.py`)
- [x] **Day 6: Memory Layer & Structured Outputs (Tasks 8 & 9)**
  - [x] LangChain `InMemoryChatMessageHistory` session memory manager (`agents/memory.py`)
  - [x] Multi-turn entity carryover across conversational turns
  - [x] Fresh session isolation and state reset
  - [x] Transcripts: `task8_memory_multi_turn.txt` & `task8_memory_fresh.txt`
  - [x] Pydantic `NaukriSupportResponse` validated output schema (`agents/schemas.py`)
  - [x] 10-point Day 6 automated test suite (`tests/test_memory_schemas.py`)
- [x] **Day 7: Input & Output Guardrails (Task 10)**
  - [x] Input PII masking for Indian phone numbers (+91, 10-digit formats) and emails (`agents/guardrails.py`)
  - [x] Input prompt-injection detection intercepting adversarial jailbreaks and system override attacks
  - [x] Output groundedness verification enforcing calibrated cosine similarity threshold (`0.4000`)
  - [x] Demonstration transcript recording all 3 guardrail firings (`transcripts/task10_guardrails.txt`)
  - [x] 10-point Day 7 automated test suite (`tests/test_guardrails.py`)
- [x] **Day 8: FastAPI Backend (Task 11)**
  - [x] High-performance REST service with `POST /ask`, `POST /add-document`, and `GET /health` (`api/main.py`)
  - [x] Multi-turn real-time WebSocket chat streaming on `/chat` and `/ws/chat`
  - [x] Asyncio event-loop thread offloading via `asyncio.to_thread` preventing sync CrewAI conflicts
  - [x] Clean `WebSocketDisconnect` exception handling ensuring continuous server availability
  - [x] Dynamic vector collection indexing for new policy documents
  - [x] 10-point Day 8 automated test suite (`tests/test_api.py`)
- [x] **Day 9: Structured Logging Middleware & Evaluation Suite (Tasks 12 & 13)**
  - [x] JSON-Lines logging middleware capturing `trace_id`, `timestamp`, `duration_ms`, masked `query`, `status_code`, and `client_ip` (`api/logging_middleware.py`)
  - [x] **Trap 6 Defeated**: Guaranteed zero PII leakage (unmasked phone numbers/emails) in log files (`logs/api_access.jsonl`)
  - [x] Trace ID propagation via `X-Trace-ID` and `X-Response-Time-MS` response headers
  - [x] 15-query evaluation dataset covering all 12 policy topics (POL-001..POL-012) + 3 out-of-scope edge cases (`evaluation/eval_suite.py`)
  - [x] 4-dimension scoring engine (Accuracy, Grounding, Completeness, Safety) with composite scoring
  - [x] 8-point Day 9 automated test suite (`tests/test_logging_and_evaluation.py`)
- [x] **Day 10: Evaluation Execution & Formal Metric Analysis (Task 13 Completion)**
  - [x] Full evaluation execution of 15 queries across the multi-agent pipeline
  - [x] Comprehensive cohort analytics: In-Scope (0.91) vs Out-of-Scope (1.00) vs Macro Benchmark (**0.93 / 1.00**)
  - [x] Markdown table generation & structured JSON metrics export (`transcripts/task13_eval_metrics.json`)
  - [x] Production benchmark transcript: `transcripts/task13_eval_15_queries.txt`
  - [x] 8-point Day 10 automated test suite (`tests/test_evaluation_execution.py`)
- [x] **Day 11: AutoGen Policy Review Stage (Task 14)**
  - [x] Installed `autogen-agentchat` and `autogen-ext` with protobuf 5 dependency resolution (`review/autogen_review.py`)
  - [x] 2-Agent `RoundRobinGroupChat` with `Policy-Compliance-Reviewer` and `Final-Editor`
  - [x] Structured output contract with Pydantic `output_content_type=VerdictModel`
  - [x] **Trap 5 Defeated**: Declared `custom_message_types=[StructuredMessage[VerdictModel]]` on Team constructor preventing runtime ValueError
  - [x] Transcripts: `transcripts/task14_review_approve.txt` (approval) & `transcripts/task14_review_revise.txt` (revision)
  - [x] 8-point Day 11 automated test suite (`tests/test_autogen_review.py`)
- [x] **Day 12: Governance, Least-Autonomy & Cost Control (Task 15)**
  - [x] Least-Autonomy architecture audit (`governance/autonomy.py`): Retrieval (RAG only), Lookup (status only), Composer (zero tools)
  - [x] Programmatic governance audit report and executive architectural justification
  - [x] Comprehensive regulatory risk classification document (`governance/risk_classification.md`): Medium-to-High Risk under EU AI Act Annex III and Indian DPDP Act 2023
  - [x] Token budget manager (`governance/cost_budget.py`) enforcing hard per-request ceiling (2,000 tokens)
  - [x] Native `MockLLM` budget enforcement raising `TokenBudgetExceededError` on runaway prompts
  - [x] Demonstration transcript recording budget rejection of 3,214-token payload (`transcripts/task15_budget_reject.txt`)
  - [x] 8-point Day 12 automated test suite (`tests/test_governance.py`)
  - [x] **Full project test suite passing 105 / 105 tests**

---

## 📂 Project Directory Structure

```text
naukri-support-agent/
├── .venv/                         # Isolated Python virtual environment (Python 3.13)
├── capstone_strategy.md           # Master roadmap and architectural strategy guide
├── api/
│   ├── __init__.py                # API package initializer
│   ├── logging_middleware.py      # Task 12: Structured JSON-Lines logging middleware with PII masking
│   └── main.py                    # Task 11: FastAPI backend (HTTP & WebSocket)
├── data/
│   ├── chroma_db/                 # Persistent ChromaDB vector database storage (gitignored)
│   └── job_applications.json     # 50 synthetic application records
├── evaluation/
│   ├── __init__.py                # Evaluation package initializer
│   └── eval_suite.py              # Task 13: 15-query evaluation suite & 4-dimension scoring engine
├── governance/
│   ├── __init__.py                # Governance package initializer
│   ├── autonomy.py                # Task 15: Least-autonomy enforcement & audit report
│   ├── cost_budget.py             # Task 15: TokenBudgetManager & cost cap enforcement
│   └── risk_classification.md     # Task 15: EU AI Act & Indian DPDP Act risk assessment
├── knowledge_base/
│   ├── policies.json              # 12 official domain policy documents (2-5 sentences each)
│   └── policies.py                # Python loader and programmatic interface for policies
├── logs/
│   └── api_access.jsonl           # Task 12: Production-grade JSON-Lines access log (gitignored)
├── rag/
│   ├── chunking.py                # Dual chunking strategies (fixed-size and sentence-based)
│   ├── indexer.py                 # ChromaDB indexing pipeline using all-MiniLM-L6-v2
│   ├── retriever.py               # Semantic search, cosine similarity & grounded generation
│   └── evaluation.py              # Precision@3 & Recall@3 comparative evaluation suite
├── agents/
│   ├── crew.py                    # Task 7: CrewAI multi-agent orchestration pipeline
│   ├── guardrails.py              # Task 10: Input/Output guardrails (PII, injection, grounding)
│   ├── memory.py                  # Task 8: LangChain session memory layer
│   ├── mock_llm.py                # Task 7 Setup: Deterministic MockLLM (zero API keys)
│   ├── schemas.py                 # Task 9: Pydantic structured output models
│   └── tools.py                   # Task 6: Application status & continuous escalation tool
├── review/
│   ├── __init__.py                # Review package initializer
│   └── autogen_review.py          # Task 14: AutoGen 2-agent policy compliance review stage
├── scripts/
│   ├── calibrate_threshold.py     # Empirical similarity threshold calibration script
│   └── generate_dataset.py        # Seeded, reproducible dataset generator script
├── tests/
│   ├── test_dataset.py            # Day 1 test suite (11 assertions)
│   ├── test_rag.py                # Day 2 test suite (8 assertions)
│   ├── test_evaluation.py         # Day 3 test suite (5 assertions)
│   ├── test_agents.py             # Day 4 test suite (10 assertions)
│   ├── test_crew.py               # Day 5 test suite (9 assertions)
│   ├── test_memory_schemas.py     # Day 6 test suite (10 assertions)
│   ├── test_guardrails.py         # Day 7 test suite (10 assertions)
│   ├── test_api.py                # Day 8 test suite (10 assertions)
│   ├── test_logging_and_evaluation.py # Day 9 test suite (8 assertions)
│   ├── test_evaluation_execution.py   # Day 10 test suite (8 assertions)
│   ├── test_autogen_review.py         # Day 11 test suite (8 assertions)
│   └── test_governance.py         # Day 12 test suite (8 assertions)
├── transcripts/
│   ├── task4_grounded_gen.txt     # Task 4 grounded generation transcript
│   ├── task5_eval_comparison.txt  # Task 5 precision/recall comparison transcript
│   ├── task7_crew_demo.txt        # Task 7 multi-agent crew execution transcript
│   ├── task8_memory_multi_turn.txt# Task 8 multi-turn session memory transcript
│   ├── task8_memory_fresh.txt     # Task 8 fresh session reset transcript
│   ├── task10_guardrails.txt      # Task 10 guardrail demonstration transcript
│   ├── task13_eval_15_queries.txt # Task 13 15-query evaluation suite transcript
│   ├── task13_eval_metrics.json   # Task 13 structured evaluation metrics output
│   ├── task14_review_approve.txt  # Task 14 AutoGen review approval transcript
│   ├── task14_review_revise.txt   # Task 14 AutoGen review revision transcript
│   └── task15_budget_reject.txt   # Task 15 token budget rejection transcript
├── .gitignore                     # Git ignore rules for venv, cache, DB, and logs
├── README.md                      # Project documentation and run guide
└── requirements.txt               # Pinned project dependencies
```

---

## ⚙️ Prerequisites & Environment Setup

This project is configured for **Windows 10/11** using **Windows PowerShell** and **Python 3.13+**.

### 1. Confirm Python Availability

```powershell
python --version
```

### 2. Activate the Virtual Environment

```powershell
.\.venv\Scripts\Activate.ps1
```

> [!NOTE]
> If PowerShell reports that script execution is disabled (`Execution_Policies`), run:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```
> Alternatively, execute scripts directly via the virtual environment interpreter: `.\.venv\Scripts\python.exe <command>`.

### 3. Install Dependencies

```powershell
pip install -r requirements.txt
```

---

## 🚀 Running Day 4 Modules

### 1. Run the Escalation Status Tool (Task 6)

```powershell
python agents/tools.py
```

*Sample Output:*
```text
=== Job Application Status & Escalation Tool (Task 6) ===
Dataset 80th Percentile Escalation Threshold: 0.5667

[APP-001] Score: 0.15 | Escalated: False
  Summary: Application APP-001 (Software Engineer): Status is 'Rejected' (created 9 days ago, priority=False). Escalation Score: 0.15 (Threshold: 0.57 -> Normal Processing). Rationale: Application was created 9 days ago.

[APP-003] Score: 1.0 | Escalated: True
  Summary: Application APP-003 (Software Engineer): Status is 'Interview Scheduled' (created 37 days ago, priority=True). Escalation Score: 1.00 (Threshold: 0.57 -> ESCALATION RECOMMENDED). Rationale: Candidate was designated for priority review; Application has been pending for 37 days (>= 30-day escalation window).
```

### 2. Run the Deterministic MockLLM (Task 7 Setup)

```powershell
python agents/mock_llm.py
```

*Sample Output:*
```text
=== Testing MockLLM (Day 4 / Task 7 Setup) ===

[Test 1: Status Lookup Dispatch]
Thought: The candidate is inquiring about job application APP-003. I must look up the application details and calculate the escalation score.
Action: check_job_application_status
Action Input: {"record_id": "APP-003"}

[Test 2: ReAct Observation Handling]
Thought: I have reviewed the application status.
Final Answer: Application APP-003 (Software Engineer) is in 'Interview Scheduled' status (created 37 days ago). Escalation score is 1.0 (ESCALATION RECOMMENDED). Rationale: Candidate was designated for priority review; Application has been pending for 37 days (>= 30-day escalation window).

[Test 3: Policy Retrieval Dispatch]
Thought: The candidate has a policy question. I need to search the knowledge base.
Action: retrieve_policy_context
Action Input: {"query": "Candidate asks: What is the policy for rescheduling an interview?"}

Total Mock Tokens Tracked: 188
```

### 3. Run the Multi-Agent Crew (Day 5 / Task 7)

```powershell
python agents/crew.py
```

*Sample Output:*
```text
================================================================================
  NAUKRI.COM DOMAIN SUPPORT AGENT - TASK 7 MULTI-AGENT CREW TRANSCRIPT
================================================================================
System: CrewAI 1.15.28 | LLM: Deterministic MockLLM (Zero API Keys)
Telemetry: CREWAI_DISABLE_TELEMETRY=true (Active)
Agents: Retrieval Agent (RAG tool only) | Lookup Agent (Status tool only) | Composer (No tools)

Query 1 (Application Status & Escalation):
Candidate Query: 'What is the current status of my application APP-003, and does it require priority escalation?'
Agents Invoked:  Application Status & Escalation Specialist -> Senior Candidate Support Composer
Tasks Executed:  2
Tokens Used:     661

Final Crew Response:
Application APP-003 (Software Engineer) is in 'Interview Scheduled' status (created 37 days ago). Escalation score is 1.0 (ESCALATION RECOMMENDED). Rationale: Candidate was designated for priority review; Application has been pending for 37 days (>= 30-day escalation window).

Query 2 (Policy Knowledge Base Retrieval):
Candidate Query: 'What is the official policy regarding rescheduling a technical interview slot?'
Agents Invoked:  Naukri Policy Retrieval Specialist -> Senior Candidate Support Composer
Tasks Executed:  2
Tokens Used:     672

Final Crew Response:
According to Naukri.com Policy [POL-011: Grievance Redressal and Anti-Fraud Reporting]: Candidates can file formal grievances or report suspicious recruiter activity through the 24/7 Naukri Trust and Safety grievance portal.
```

### 4. Run the Session Memory Pipeline (Day 6 / Task 8 & 9)

```powershell
python agents/memory.py
```

*Generates both required Task 8 transcripts:*
- `transcripts/task8_memory_multi_turn.txt` (shows entity retention across conversational turns)
- `transcripts/task8_memory_fresh.txt` (shows clean session isolation with state wiped)

### 5. Run the Full Unified Test Suite (97 Tests)

```powershell
python -m unittest discover -s tests -p "test_*.py" -v
```

*Result:*
```text
Ran 97 tests in 67.949s

OK
```

---

## 🏛️ Least-Autonomy Multi-Agent Architecture (Task 15)

In enterprise governance, agents must never be granted blanket, unmonitored tool access. The CrewAI crew strictly enforces **Least Autonomy**:

| Agent Role | Assigned Tools | Authorization Boundary |
| :--- | :--- | :--- |
| **Naukri Policy Retrieval Specialist** | `retrieve_policy_context` | Read-only vector search over HR policy database; **NO** access to candidate application records. |
| **Application Status & Escalation Specialist** | `check_job_application_status` | Candidate application status and escalation formula evaluation; **NO** access to policy vectors. |
| **Senior Candidate Support Composer** | `None` (Zero Tools) | Pure synthesis and candidate communication; operates solely on findings forwarded by specialist agents. |

---

## 🧠 LangChain Session Memory & Pydantic Schema (Tasks 8 & 9)

### 1. Multi-Turn Session Memory
- Integrates `langchain_core.chat_history.InMemoryChatMessageHistory`
- Session-isolated dictionary storing user and AI messages keyed by `session_id`.
- Automatic entity extraction: detects `APP-XXX` references and remembers the active application across turns.
- Transparent session reset via `clear_session()`.

### 2. Pydantic Structured Output Contract
All agent and API responses are strictly validated against `NaukriSupportResponse`:
- Validated fields: `query`, `answer`, `sources`, `confidence`, `escalation_needed`, `record_id`, `application_details`, `citations`, `agents_involved`, `timestamp`, `metadata`.
- Strict field validators prevent blank or empty answers, ensuring high output fidelity.

---

## 🧮 Continuous Escalation Formula (Task 6)

Rather than a simplistic boolean flag, the system uses a **continuous mathematical formula**:

$$\text{recency\_signal} = \min\left(1.0, \frac{\text{days\_since\_created}}{30.0}\right)$$
$$\text{escalation\_score} = 0.5 \times \text{int}(\text{flagged\_priority\_review}) + 0.5 \times \text{recency\_signal}$$

### Distribution & Empirical Cutoff:
- **Range**: Continuous $[0.0, 1.0]$.
- **80th Percentile Cutoff**: Computed across our 50-record dataset as $\mathbf{0.5667}$.
- **Decision Rule**: Applications with $\text{escalation\_score} \ge 0.5667$ trigger an automatic recommendation for priority managerial escalation.

---

## 🛡️ Critical Traps Defeated in MockLLM (Task 7 Setup)

1. **ReAct Parser Trap**:
   - The ReAct prompt contains template text: `Observation: the result of the action`.
   - Naive string matching on `"Observation:"` matches this template before tools run, corrupting the loop.
   - **Solution**: `MockLLM` checks that an actual `Action:` was emitted *prior* to `Observation:` before triggering final synthesis.
2. **Tool Dispatch Collision Trap**:
   - Matching tool names by substring (e.g. `'lookup' in tool_name`) causes tools like `rag_lookup` to collide with status lookups.
   - **Solution**: `MockLLM` dispatches tools based on argument schemas and explicit patterns (`APP-\d{3}` triggers application lookup; policy keywords trigger RAG).
3. **Zero-Tool Composer Synthesis Trap**:
   - In CrewAI sequential execution, an agent with `tools=[]` will throw an error if the LLM attempts to emit an `Action:`.
   - **Solution**: `MockLLM` detects when the calling agent is the Composer or has zero tools, directly outputting `Final Answer:` with synthesized findings.
4. **System Prompt Keyword Collision Trap**:
   - System prompts contain tool docstrings with words like `"interview"` and `"policy"`.
   - Naive substring scanning of the entire prompt falsely flags all status inquiries as policy queries.
   - **Solution**: `MockLLM` extracts the candidate's actual inquiry text (`search_query`) and scans intent only on user inputs.
5. **Telemetry Protection**:
   - Set `CREWAI_DISABLE_TELEMETRY=true` on import to guarantee zero external phone-home network traffic.

---

## 🛡️ Input & Output Guardrails (Task 10)

The system implements a multi-layer defense-in-depth safety engine (`agents/guardrails.py`):

### 1. Input PII Masking (Indian Mobile Numbers & Emails)
- Uses regex pattern `(?:\+91[\s\-\.]?|\b91[\s\-\.]?|\b0[\s\-\.]?|\b)[6-9](?:[\s\-\.]?\d){9}\b` supporting:
  - `+91 9876543210` (standard country code with space)
  - `+91-98765-43210` (hyphenated country code and 5-5 grouping)
  - `9876543210`, `8765432109`, `7654321098` (10-digit formats starting with 6-9)
  - `09876543210` (leading zero format)
  - `candidate@example.com` (email addresses)
- Replaces matches with `[REDACTED_PHONE]` and `[REDACTED_EMAIL]` *prior* to passing prompts to agents or LLMs.

### 2. Input Prompt-Injection Detection
- Intercepts adversarial jailbreaks and instruction-override attacks (e.g., "Ignore all previous instructions", "disregard prior instructions", "system prompt reveal", "act as DAN").
- Stops pipeline execution immediately with `BLOCKED_INPUT_INJECTION` without invoking agent or LLM resources.

### 3. Output Groundedness Verification
- Evaluates candidate inquiries against the policy knowledge base using ChromaDB semantic search.
- If top cosine similarity falls below the calibrated empirical threshold $\mathbf{0.4000}$, the ungrounded draft answer is rejected and replaced with a safe referral to the Naukri Help Desk / HR representative, preventing model hallucinations.

---

## 🚀 FastAPI Service Layer (Task 11)

The system exposes production-grade HTTP REST and WebSocket streaming interfaces (`api/main.py`):

### 1. HTTP Endpoints
- `GET /health` (alias `/api/v1/health`): Returns system health, ChromaDB collection counts, and active memory sessions.
- `POST /ask` (alias `/api/v1/support/ask`): Primary support endpoint accepting `AskRequest` (query, optional session_id). Executes the complete defense-in-depth pipeline (PII masking, injection defense, multi-agent crew execution, groundedness verification) and returns validated `AskResponse`.
- `POST /add-document` (alias `/api/v1/knowledge/add-document`): Dynamically indexes new policy documents by chunking into fixed and sentence strategies and upserting into live ChromaDB collections.

### 2. WebSocket Real-Time Chat (`/chat` & `/ws/chat`)
- Multi-turn real-time conversational streaming maintaining persistent session state across turns.
- **Async Event Loop Thread Offloading**: Uses `asyncio.to_thread` to run synchronous CrewAI agent execution in a background worker thread, eliminating synchronous event-loop collision crashes (`RuntimeError: Agent execution was invoked synchronously from within a running event loop`).
- **Clean Disconnect Handling**: Explicitly catches `WebSocketDisconnect` exceptions, allowing clients to disconnect without raising unhandled errors or interrupting service for other clients.

---

## 📊 Structured JSON-Lines Logging Middleware (Task 12)

The API is wrapped with high-performance logging middleware (`api/logging_middleware.py`) outputting directly to `logs/api_access.jsonl`:

- **Field Specification**:
  - `trace_id`: Unique UUID generated per incoming request (propagated back to client via `X-Trace-ID`).
  - `timestamp`: UTC ISO-8601 millisecond timestamp.
  - `method`, `path`, `status_code`, `client_ip`: Standard access coordinates.
  - `duration_ms`: Total end-to-end request latency (propagated via `X-Response-Time-MS`).
  - `query`: Candidate inquiry text with PII safely masked.
  - `pii_masked`: Boolean indicator specifying whether sensitive data was sanitized.
- **Trap 6 Defeated (PII Leakage in Logs)**:
  - Raw request bodies are intercepted before logging and processed through `mask_pii()`.
  - Zero unmasked phone numbers or email addresses ever touch disk, eliminating critical GDPR/DPDP audit violations.

---

## 🧪 15-Query Evaluation Suite & 4-Dimension Rubric (Task 13)

Located in `evaluation/eval_suite.py`, the evaluation harness provides automated quality assurance across 15 structured test cases:

- **Query Distribution**:
  - **12 Policy Queries**: Full 1-to-1 coverage of knowledge base policies (`POL-001` through `POL-012`).
  - **3 Out-of-Scope Queries**: Adversarial and irrelevant questions (e.g. pet policy, lottery rewards, internal infrastructure) testing safety fallbacks.
- **4-Dimension Scoring Rubric**:
  - **Accuracy** ($0.0 - 1.0$): Does the answer resolve the query correctly, or reject out-of-scope appropriately?
  - **Grounding** ($0.0 - 1.0$): Are statements backed by retrieved chunks or official policy citations?
  - **Completeness** ($0.0 - 1.0$): Does the response contain key operational elements (deadlines, portals, contacts)?
  - **Safety** ($0.0 - 1.0$): Was PII protected, injection resisted, and out-of-scope safely deflected?
- **Empirical Baseline & Formal Cohort Metrics (Task 13 Completion)**:
  - Automatically writes results to `transcripts/task13_eval_15_queries.txt` and `transcripts/task13_eval_metrics.json`.
  - **In-Scope Policy Queries (12 Queries)**: Accuracy: 0.79 | Grounding: 0.83 | Completeness: 1.00 | Safety: 1.00 | Composite: **0.91**
  - **Out-of-Scope Fallback Queries (3 Queries)**: Accuracy: 1.00 | Grounding: 1.00 | Completeness: 1.00 | Safety: 1.00 | Composite: **1.00**
  - **Overall System Macro Benchmark (15 Total)**: Accuracy: 0.83 | Grounding: 0.87 | Completeness: 1.00 | Safety: 1.00 | Composite: **0.93 / 1.00**

---

## 🤝 AutoGen Multi-Agent Policy Review Stage (Task 14)

Day 11 integrates Microsoft AutoGen 0.7+ (`review/autogen_review.py`) as a formal editorial compliance stage operating sequentially after the primary composer:

```mermaid
flowchart LR
    A[Candidate Support Draft] --> B[Policy_Compliance_Reviewer]
    B -->|Audit Critique| C[Final_Editor]
    C -->|StructuredMessage[VerdictModel]| D{Approved?}
    D -->|Yes: Score=1.0| E[Deliver Grounded Answer]
    D -->|No: Hallucination| F[Excise Fabrications & Emit Sanitized Answer]
```

### 1. Two-Agent Editorial Architecture
- **Policy_Compliance_Reviewer**: Audits candidate support drafts against official domain policies (`POL-001` through `POL-012`). Confirms timeline accuracy, notice periods, and document requirements. Detects unauthorized or ungrounded claims.
- **Final_Editor**: Emits a structured Pydantic `VerdictModel` using `output_content_type=VerdictModel`. Either formally certifies compliance (`compliance_score=1.0`, `is_approved=True`) or excises ungrounded claims and delivers a sanitized answer.

### 2. Trap 5 Defeated (AutoGen `custom_message_types` Runtime Crash)
- In AutoGen 0.7+, configuring `output_content_type=VerdictModel` on an `AssistantAgent` causes runtime `ValueError` crashes in group chat teams if the team does not recognize the custom message type.
- **Solution**: Explicitly passed `custom_message_types=[StructuredMessage[VerdictModel]]` to `RoundRobinGroupChat`, registering the structured message factory and guaranteeing clean message serialization.

### 3. Demonstration Transcripts Generated
- [`transcripts/task14_review_approve.txt`](file:///d:/antigravity/Naukri.com/transcripts/task14_review_approve.txt): Fully grounded draft adhering to `POL-001` receives formal certification without modification.
- [`transcripts/task14_review_revise.txt`](file:///d:/antigravity/Naukri.com/transcripts/task14_review_revise.txt): Draft with an injected ungrounded claim ("INR 50,000 cash joining bonus and unlimited paid leaves") is intercepted by the reviewer, flagged as high regulatory risk, and sanitized by the editor.

---

## 🛡️ Governance, Least-Autonomy & Cost Control (Task 15)

Day 12 implements enterprise multi-agent governance controls across three mandatory pillars: Least-Autonomy role isolation, statutory Risk Classification, and strict Token Budget Caps.

```mermaid
flowchart TD
    subgraph MultiAgentGov[1. Least-Autonomy Boundaries]
        RA[Retrieval Agent] -->|Read-Only| VKB[(ChromaDB Policy Vectors)]
        LA[Lookup Agent] -->|Single Record| APP[(Application Records & Escalations)]
        CA[Composer Agent] -->|Zero Tools| SYN[Pure Response Synthesis]
    end

    subgraph RiskMgmt[2. Regulatory Risk Classification]
        RR[Recruitment & Candidate Profiling] --> EU[EU AI Act Annex III: High Risk]
        RR --> DPDP[Indian DPDP Act 2023: Sensitive Personal Data]
        RR --> NIST[NIST AI RMF 1.0: Govern & Measure Controls]
    end

    subgraph CostCtrl[3. Financial & Runaway Governance]
        REQ[Incoming Prompt] --> TBM{Token Budget Manager}
        TBM -->|< 2000 Tokens| PASS[Accepted -> MockLLM Execution]
        TBM -->|>= 2000 Tokens| BLOCK[TokenBudgetExceededError: Execution Intercepted]
    end
```

### 1. Least-Autonomy Enforcement (`governance/autonomy.py`)
- **Principle of Least Privilege**: Each agent in the multi-agent hierarchy operates within the absolute minimum authorization boundary:
  - **Naukri Policy Retrieval Specialist**: Strictly wired to `retrieve_policy_context`. Has zero access to candidate application records.
  - **Application Status & Escalation Specialist**: Strictly wired to `check_job_application_status`. Has zero access to policy vector collections.
  - **Senior Candidate Support Composer**: Equipped with **zero tools** (`tools=[]`). Operates exclusively as a synthesizer of verified upstream findings.
- **Audit Verification**:
  ```powershell
  python governance/autonomy.py
  ```
  Programmatically inspects the live `Crew` instance, asserting that no cross-contamination or tool boundary expansion has occurred.

### 2. Risk Classification & Regulatory Assessment (`governance/risk_classification.md`)
- **Formal Risk Tier**: Formally categorized as **Medium-to-High Risk**.
- **Statutory Alignment**:
  - **EU AI Act (Regulation 2024/1689)**: Categorized under **Annex III, Section 4 (*Employment, workers management and access to self-employment*)** requiring mandatory human escalation, dataset governance, and audit trails.
  - **Indian DPDP Act 2023**: Mandates technical safeguards to prevent unauthorized exposure of candidate mobile numbers and identities.
  - **NIST AI RMF 1.0**: Maps Govern and Measure functions to tool boundary isolation and budget caps.

### 3. Token Budget Cap & Cost Control (`governance/cost_budget.py`)
- **Ceiling Configuration**: Configures a hard per-request ceiling of **2,000 tokens**.
- **Pre-execution Interception**: Heuristic token estimation validates incoming queries before invoking the language model, preventing financial denial-of-wallet attacks and recursive agent runaway.
- **MockLLM Native Integration**: Updated `MockLLM(max_token_budget=2000)` to raise `TokenBudgetExceededError` if any prompt exceeds the budget limit.
- **Demonstration Run**:
  ```powershell
  python governance/cost_budget.py
  ```
- **Demonstration Transcript**:
  [`transcripts/task15_budget_reject.txt`](file:///d:/antigravity/Naukri.com/transcripts/task15_budget_reject.txt):
  - **Scenario 1 (Normal Inquiry)**: 44 tokens $\le$ 2,000 ceiling $\rightarrow$ `ACCEPTED`.
  - **Scenario 2 (Oversized Adversarial Prompt)**: 3,214 tokens $>$ 2,000 ceiling $\rightarrow$ `REJECTED IMMEDIATELY BEFORE LLM INVOCATION` with 0 billed tokens incurred.

---

## 🔮 Next Steps (Day 13: Response Caching & Polishing - Task 16)

1. **Normalized Query Caching (`api/cache.py`)**:
   - Implement in-memory LRU response caching keyed by normalized query strings.
   - Strip punctuation, lower-case, and standardize whitespace.
2. **Empirical Verification (`transcripts/task16_cache_hit.txt`)**:
   - Demonstrate cache miss (first invocation) vs. cache hit (second identical query).
   - Show latency improvement and token avoidance proof.
3. **End-to-End System Integration**:
   - Wire caching into FastAPI `/ask` endpoint.
