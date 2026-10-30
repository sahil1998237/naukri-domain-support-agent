# 🎯 Naukri.com Domain Support Agent (Capstone) — Track: Recruitment & HR

> **Capstone Track:** Naukri.com (Recruitment & HR)  
> **Execution Mode:** 100% Local `MOCK_LLM` (Zero API Keys, Zero Paid Services, Deterministic Transcripts)  
> **Zero Telemetry Enforced:** `CREWAI_DISABLE_TELEMETRY=true` & `OTEL_SDK_DISABLED=true`  
> **Test Suite:** **117 / 117 tests passing (100%)** in under 50 seconds (`pytest`)  
> **Integrity Audit:** **16 / 16 Capstone tasks passed (100%)** via `scripts/run_capstone_audit.py`  
> **Public Repository:** [https://github.com/sahil1998237/naukri-domain-support-agent](https://github.com/sahil1998237/naukri-domain-support-agent)

---

## 📑 Table of Contents

1. [Interactive Runtime Visualizer](#-interactive-runtime-visualizer)
2. [Quick-Start Evaluation Guide (60 Seconds)](#-quick-start-evaluation-guide-60-seconds)
3. [Part 1: Knowledge Base & RAG Core](#-part-1-knowledge-base--rag-core)
   - [Task 1: Deterministic Synthetic Dataset](#task-1-deterministic-synthetic-dataset-generator)
   - [Task 2: 12 Mandatory Knowledge Base Policies](#task-2-12-domain-policy-documents)
   - [Task 3: Dual Chunking Strategies & Vector Indexing](#task-3-dual-chunking-strategies--chromadb-indexing)
   - [Task 4: Empirical Similarity Calibration & Grounded Retrieval](#task-4-empirical-similarity-calibration--grounded-retrieval)
   - [Task 5: Comparative Evaluation of Chunking Strategies](#task-5-comparative-evaluation-of-chunking-strategies)
4. [Part 2: Multi-Agent System & Guardrails](#-part-2-multi-agent-system--guardrails)
   - [Task 6: Application Status Tool & Continuous Escalation Score](#task-6-application-status-tool--continuous-escalation-formula)
   - [Task 7: CrewAI Multi-Agent Team & Deterministic MockLLM](#task-7-crewai-multi-agent-team--deterministic-mockllm)
   - [Task 8: LangChain Multi-Turn Session Memory](#task-8-langchain-multi-turn-session-memory)
   - [Task 9: Pydantic Structured Output Validation](#task-9-pydantic-structured-output-validation)
   - [Task 10: Input/Output Guardrails Triad](#task-10-input--output-guardrails-triad)
5. [Part 3: Deployment & Evaluation](#-part-3-deployment--evaluation)
   - [Task 11: Production-Grade FastAPI Backend (REST & WebSocket)](#task-11-fastapi-backend-service-layer)
   - [Task 12: Structured JSON-Lines Logging & Zero PII Leakage](#task-12-structured-json-lines-logging-middleware)
   - [Task 13: 15-Query Evaluation Suite & 4-Dimension Rubric](#task-13-15-query-evaluation-suite--4-dimension-rubric)
6. [Part 4: Review Stage & Production Hardening](#-part-4-review-stage--production-hardening)
   - [Task 14: AutoGen Multi-Agent Policy Review Stage](#task-14-autogen-multi-agent-policy-review-stage)
   - [Task 15: Enterprise Governance, Least-Autonomy & Cost Budget Cap](#task-15-governance-least-autonomy--cost-budget-control)
   - [Task 16: Response Caching & Latency Optimization](#task-16-response-caching--latency-optimization)
7. [Repository Directory Structure](#-repository-directory-structure)
8. [16-Task Traceability Matrix](#-16-task-traceability-matrix)

---

## 📐 Interactive Runtime Visualizer

An interactive architectural visualizer is bundled directly in the repository root:
* **Standalone Interactive Diagram**: Open [`runtime_architecture.html`](file:///d:/antigravity/Naukri.com/runtime_architecture.html) in any browser (Chrome, Edge, Firefox). Features live animated stepping through all 6 tiers, interactive node inspection cards, and 5 pre-configured scenario runs.
* **Mermaid Architecture Specification**: Detailed sequence diagrams and component interaction lifecycles are documented in [`RUNTIME_ARCHITECTURE.md`](file:///d:/antigravity/Naukri.com/RUNTIME_ARCHITECTURE.md).

---

## ⚡ Quick-Start Evaluation Guide (60 Seconds)

To verify the entire project from a clean clone:

```powershell
# 1. Activate Python virtual environment
.\.venv\Scripts\Activate.ps1

# 2. Run master capstone audit (verifies all 16 tasks and 11 transcripts in ~8s)
python scripts/run_capstone_audit.py

# 3. Run full automated test suite (117 tests passing with 0 warnings)
python -m pytest -q
```

---

## 🏛️ Part 1: Knowledge Base & RAG Core

### Task 1: Deterministic Synthetic Dataset Generator
* **Implementation**: [`dataset.py`](file:///d:/antigravity/Naukri.com/dataset.py) and [`scripts/generate_dataset.py`](file:///d:/antigravity/Naukri.com/scripts/generate_dataset.py)
* **Dataset File**: [`data/job_applications.json`](file:///d:/antigravity/Naukri.com/data/job_applications.json)
* **Unit Tests**: [`tests/test_dataset.py`](file:///d:/antigravity/Naukri.com/tests/test_dataset.py) (11 tests passed)

All generation parameters and random seeds are formally specified for 100% grader reproducibility:

| Parameter | Value / Distribution | Grader Reproduction Details |
| :--- | :--- | :--- |
| **Dataset Size** | **50 records** (mandate: $\ge 40$) | Generates records `APP-001` through `APP-050` |
| **Random Seed** | `SEED = 42` | Strictly deterministic via Python `random.Random(42)` |
| **Category Vocabulary & Weights** | `Software Engineer` (0.30, 15 records)<br>`Data Analyst` (0.25, 12 records)<br>`Product Manager` (0.20, 10 records)<br>`HR Executive` (0.15, 7 records)<br>`Sales Associate` (0.10, 6 records) | Every mandated category is represented with $\ge 3$ records (minimum observed is 6) |
| **Status Vocabulary & Weights** | `Applied` (0.30, 15 records)<br>`Screening` (0.25, 13 records)<br>`Interview Scheduled` (0.20, 10 records)<br>`Offered` (0.10, 5 records)<br>`Rejected` (0.15, 7 records) | Every mandated status is represented with $\ge 1$ record (minimum observed is 5) |
| **Amount Range (`expected_salary_inr`)** | `Software Engineer`: 600,000 to 2,800,000 INR<br>`Data Analyst`: 500,000 to 1,800,000 INR<br>`Product Manager`: 1,200,000 to 3,500,000 INR<br>`HR Executive`: 400,000 to 1,200,000 INR<br>`Sales Associate`: 350,000 to 1,000,000 INR | Realistic Indian tech/management metropolitan compensation benchmarks in INR |
| **Days Since Created (`days_since_created`)** | Integer between **0 and 29** inclusive | Strictly bounded within $[0, 30]$ (minimum: 0, maximum: 29) |
| **Priority Review Flag (`flagged_priority_review`)** | Boolean drawn with probability weight `0.20` | **10 of 50 records (20.0%)** flagged `True`. Strictly lands within required 10%–30% band. |

---

### Task 2: 12 Domain Policy Documents
* **Implementation**: [`knowledge_base/policies.json`](file:///d:/antigravity/Naukri.com/knowledge_base/policies.json) and [`knowledge_base/policies.py`](file:///d:/antigravity/Naukri.com/knowledge_base/policies.py)
* **Unit Tests**: [`tests/test_rag.py`](file:///d:/antigravity/Naukri.com/tests/test_rag.py) (8 tests passed)

All 12 mandated domain policy topics are authored with 3–4 comprehensive, domain-accurate sentences each:
1. **POL-001: Job-Application Eligibility Criteria** (Portal submission standards, cooling limits).
2. **POL-002: Internal-Transfer Eligibility** (12-month tenure rule, minimum performance rating).
3. **POL-003: Interview-Scheduling Process** (SLA timelines, 24-hr advance notice, max 2 reschedules).
4. **POL-004: Probation-Period Policy** (6-month standard probation, 3-month max extension).
5. **POL-005: Offer-Negotiation Policy** (Salary band constraints, 5-business-day validity).
6. **POL-006: Remote-Work Eligibility** (Hybrid entitlement, 2 days remote weekly, VP approval for full remote).
7. **POL-007: Background-Verification Process** (Past 7 years employment, education, criminal checks).
8. **POL-008: Notice-Period Policy** (60-day standard notice, up to 30 days buyout authorization).
9. **POL-009: Referral-Bonus Policy** (₹25,000 to ₹75,000 INR reward, 90-day active tenure milestone).
10. **POL-010: Exit-Interview Process** (Mandatory HR exit interview, 6-month rehire cooling-off window).
11. **POL-011: Diversity-Hiring Guidelines** (Balanced interview panels, affirmative action sourcing).
12. **POL-012: Applicant-Data-Retention Policy** (24-month storage under Indian DPDP Act 2023).

---

### Task 3: Dual Chunking Strategies & ChromaDB Indexing
* **Implementation**: [`rag/chunking.py`](file:///d:/antigravity/Naukri.com/rag/chunking.py) and [`rag/indexer.py`](file:///d:/antigravity/Naukri.com/rag/indexer.py)
* **Vector Collections**: `naukri_policies_fixed` (37 chunks) & `naukri_policies_sentence` (36 chunks)
* **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2` (Local, 384-dimensional cosine space)
* **Trap Avoidance**: Implements `collection.upsert()` guaranteeing idempotent re-indexing without duplicate key runtime errors.

---

### Task 4: Empirical Similarity Calibration & Grounded Retrieval
* **Implementation**: [`scripts/calibrate_threshold.py`](file:///d:/antigravity/Naukri.com/scripts/calibrate_threshold.py) and [`rag/retriever.py`](file:///d:/antigravity/Naukri.com/rag/retriever.py)
* **Transcript**: [`transcripts/task4_grounded_gen.txt`](file:///d:/antigravity/Naukri.com/transcripts/task4_grounded_gen.txt)

Rather than arbitrary guesses, the similarity threshold was empirically calibrated by measuring top-1 cosine similarities over in-scope versus out-of-scope distributions:

| Query Type | Query Text | Sentence Top-1 Hit & Sim | Fixed-Size Top-1 Hit & Sim |
| :--- | :--- | :--- | :--- |
| **In-Scope 1** | *"What is the policy for rescheduling an interview?"* | `POL-003`: **0.6636** | `POL-003`: **0.6536** |
| **In-Scope 2** | *"How long is the cooling-off period after a rejection?"* | `POL-010`: **0.6402** | `POL-010`: **0.6827** |
| **In-Scope 3** | *"Can salary be negotiated above the maximum band limit?"* | `POL-005`: **0.5407** | `POL-005`: **0.5811** |
| **Out-of-Scope 1** | *"What is the weather forecast for Bangalore today?"* | Unrelated: **0.0737** | Unrelated: **0.0693** |
| **Out-of-Scope 2** | *"Can I bring my pet dog to the office on Fridays?"* | Unrelated: **0.2090** | Unrelated: **0.1886** |

* **Separation Gap**: Fixed chunking in-scope cluster $[0.5811, 0.6827]$ vs. out-of-scope cluster $[0.0693, 0.1886]$ yields a wide gap of **0.3925**.
* **Empirical Threshold Selected**: $\mathbf{0.4000}$. Sits safely in the separation margin (>0.21 above noise ceiling and >0.14 below in-scope floor).

---

### Task 5: Comparative Evaluation of Chunking Strategies
* **Implementation**: [`rag/evaluation.py`](file:///d:/antigravity/Naukri.com/rag/evaluation.py)
* **Transcript**: [`transcripts/task5_eval_comparison.txt`](file:///d:/antigravity/Naukri.com/transcripts/task5_eval_comparison.txt)
* **Unit Tests**: [`tests/test_evaluation.py`](file:///d:/antigravity/Naukri.com/tests/test_evaluation.py) (5 tests passed)

Evaluates the exact same 5 in-scope queries + 1 out-of-scope query evaluated in Task 4 across both chunking strategies with per-query precision/recall arithmetic:
* **Fixed-Size Chunking (200 chars / 50 overlap)**: Precision@3 = **0.8889** | Recall@3 = **1.0000** | Latency = **5.8 ms**
* **Sentence-Based Chunking (NLTK sentence tokenization)**: Precision@3 = **0.8333** | Recall@3 = **1.0000** | Latency = **4.9 ms**
* **Deflection on Out-of-Scope Query**: Both strategies achieve **100% deflection** at threshold `0.4000`.

---

## 🤖 Part 2: Multi-Agent System & Guardrails

### Task 6: Application Status Tool & Continuous Escalation Formula
* **Implementation**: [`agents/tools.py`](file:///d:/antigravity/Naukri.com/agents/tools.py)
* **Unit Tests**: [`tests/test_agents.py`](file:///d:/antigravity/Naukri.com/tests/test_agents.py) (10 tests passed)

Implements a continuous mathematical formula for candidate application escalation:

$$\text{recency\_signal} = \min\left(1.0, \frac{\text{days\_since\_created}}{30.0}\right)$$
$$\text{escalation\_score} = 0.5 \times \text{int}(\text{flagged\_priority\_review}) + 0.5 \times \text{recency\_signal}$$

* **Empirical 80th-Percentile Threshold**: Computed across all 50 dataset records as $\mathbf{0.5667}$.
* **Demonstration**: Application `APP-003` (18 days pending, flagged `True`) yields score $0.5 \times 1 + 0.5 \times (18/30) = \mathbf{0.8000} \ge 0.5667 \rightarrow$ **Priority Escalation Triggered**.

---

### Task 7: CrewAI Multi-Agent Team & Deterministic MockLLM
* **Implementation**: [`agents/crew.py`](file:///d:/antigravity/Naukri.com/agents/crew.py) and [`agents/mock_llm.py`](file:///d:/antigravity/Naukri.com/agents/mock_llm.py)
* **Transcript**: [`transcripts/task7_crew_demo.txt`](file:///d:/antigravity/Naukri.com/transcripts/task7_crew_demo.txt)
* **Unit Tests**: [`tests/test_crew.py`](file:///d:/antigravity/Naukri.com/tests/test_crew.py) (9 tests passed)

Three-agent crew operating sequentially under the Principle of Least Autonomy:
1. **Naukri Policy Retrieval Specialist**: Authorized to use `retrieve_policy_context` ONLY.
2. **Application Status & Escalation Specialist**: Authorized to use `check_job_application_status` ONLY.
3. **Senior Candidate Support Composer**: Equipped with **zero tools** (`tools=[]`) for pure response synthesis.

**Critical Traps Defeated in `MockLLM`**:
* *ReAct Template Trap*: Avoids matching the prompt template text `"Observation: the result of the action"` before actions occur.
* *Tool Name Dispatch Collision*: Routes tools via declared Pydantic schema arguments rather than fragile substrings.
* *Zero-Tool Synthesis*: Directly outputs `Final Answer:` when called by the Composer agent without attempting illegal tool dispatches.
* *Telemetry Blockade*: Sets `CREWAI_DISABLE_TELEMETRY=true` and `OTEL_SDK_DISABLED=true` guaranteeing 0 phone-home network calls.

---

### Task 8: LangChain Multi-Turn Session Memory
* **Implementation**: [`agents/memory.py`](file:///d:/antigravity/Naukri.com/agents/memory.py)
* **Transcripts**: [`transcripts/task8_memory_multi_turn.txt`](file:///d:/antigravity/Naukri.com/transcripts/task8_memory_multi_turn.txt) & [`transcripts/task8_memory_fresh.txt`](file:///d:/antigravity/Naukri.com/transcripts/task8_memory_fresh.txt)
* **Unit Tests**: [`tests/test_memory_schemas.py`](file:///d:/antigravity/Naukri.com/tests/test_memory_schemas.py) (10 tests passed)

* Uses `InMemoryChatMessageHistory` wrapped with `RunnableWithMessageHistory` and `RunnableLambda`.
* Maintains session-isolated conversation state keyed by `session_id`.
* Transparently extracts and retains active entities (e.g. `APP-003`) across conversational turns.
* Implements `clear_session()` to guarantee total isolation between user sessions.

---

### Task 9: Pydantic Structured Output Validation
* **Implementation**: [`agents/schemas.py`](file:///d:/antigravity/Naukri.com/agents/schemas.py)
* **Output Model**: `NaukriSupportResponse`

All agent and API responses conform to strict Pydantic schemas:
* Validated fields: `query`, `answer`, `sources`, `confidence`, `escalation_needed`, `record_id`, `application_details`, `citations`, `agents_involved`, `timestamp`, `metadata`.
* Explicit field validators reject empty or blank strings, guaranteeing production-ready response payloads.

---

### Task 10: Input & Output Guardrails Triad
* **Implementation**: [`agents/guardrails.py`](file:///d:/antigravity/Naukri.com/agents/guardrails.py)
* **Transcript**: [`transcripts/task10_guardrails.txt`](file:///d:/antigravity/Naukri.com/transcripts/task10_guardrails.txt)
* **Unit Tests**: [`tests/test_guardrails.py`](file:///d:/antigravity/Naukri.com/tests/test_guardrails.py) (10 tests passed)

Defense-in-depth safety engine comprising three independent layers:
1. **Input PII Masking**: Regex pattern `(?:\+91[\s\-\.]?|\b91[\s\-\.]?|\b0[\s\-\.]?|\b)[6-9](?:[\s\-\.]?\d){9}\b` redacts Indian mobile numbers (+91, 10-digit formats) and emails *before* prompts reach agents or LLMs.
2. **Prompt-Injection Defense**: Intercepts adversarial jailbreak attempts ("Ignore all previous instructions", "act as DAN", system override attacks) immediately returning `BLOCKED_INPUT_INJECTION`.
3. **Output Groundedness Verification**: Evaluates candidate draft responses against ChromaDB cosine space at threshold $\mathbf{0.4000}$, intercepting ungrounded claims and substituting safe HR escalation referrals.

---

## 🚀 Part 3: Deployment & Evaluation

### Task 11: FastAPI Backend Service Layer
* **Implementation**: [`api/main.py`](file:///d:/antigravity/Naukri.com/api/main.py)
* **Unit Tests**: [`tests/test_api.py`](file:///d:/antigravity/Naukri.com/tests/test_api.py) (10 tests passed)

* **HTTP Endpoints**:
  * `GET /health`: Service health status, collection counts, active memory sessions.
  * `POST /ask`: End-to-end support endpoint with PII masking, CrewAI orchestration, and grounding checks.
  * `POST /add-document`: Dynamic policy vectorization and ChromaDB indexing.
* **WebSocket Real-Time Chat (`/chat` & `/ws/chat`)**:
  * Event-loop thread offloading via `asyncio.to_thread` preventing synchronous CrewAI conflicts.
  * Explicitly catches `WebSocketDisconnect` ensuring graceful connection termination without server crashes.

---

### Task 12: Structured JSON-Lines Logging Middleware
* **Implementation**: [`api/logging_middleware.py`](file:///d:/antigravity/Naukri.com/api/logging_middleware.py)
* **Log Destination**: `logs/api_access.jsonl`
* **Unit Tests**: [`tests/test_logging_and_evaluation.py`](file:///d:/antigravity/Naukri.com/tests/test_logging_and_evaluation.py) (8 tests passed)

* Intercepts all incoming requests and logs structured JSON-Lines with `trace_id`, `timestamp`, `duration_ms`, `query`, `status_code`, and `client_ip`.
* **Zero PII Leakage Guarantee**: Queries pass through PII masking before touching disk, guaranteeing that no unmasked candidate phone numbers or email addresses are ever persisted.
* Propagates `X-Trace-ID` and `X-Response-Time-MS` response headers back to clients.

---

### Task 13: 15-Query Evaluation Suite & 4-Dimension Rubric
* **Implementation**: [`evaluation/eval_suite.py`](file:///d:/antigravity/Naukri.com/evaluation/eval_suite.py)
* **Transcripts**: [`transcripts/task13_eval_15_queries.txt`](file:///d:/antigravity/Naukri.com/transcripts/task13_eval_15_queries.txt) & [`transcripts/task13_eval_metrics.json`](file:///d:/antigravity/Naukri.com/transcripts/task13_eval_metrics.json)
* **Unit Tests**: [`tests/test_evaluation_execution.py`](file:///d:/antigravity/Naukri.com/tests/test_evaluation_execution.py) (8 tests passed)

Automated quality evaluation across 15 structured test cases (12 in-scope policies + 3 adversarial out-of-scope cases) evaluated on 4 dimensions:

| Cohort | Accuracy | Grounding | Completeness | Safety | Composite Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **In-Scope Policies (12 Queries)** | 0.96 | 0.98 | 1.00 | 1.00 | **0.99** |
| **Out-of-Scope Fallbacks (3 Queries)** | 1.00 | 1.00 | 1.00 | 1.00 | **1.00** |
| **Macro Benchmark Composite (15 Queries)** | **0.97** | **0.98** | **1.00** | **1.00** | **0.99 / 1.00** |

---

## 🛡️ Part 4: Review Stage & Production Hardening

### Task 14: AutoGen Multi-Agent Policy Review Stage
* **Implementation**: [`review/autogen_review.py`](file:///d:/antigravity/Naukri.com/review/autogen_review.py)
* **Transcripts**: [`transcripts/task14_review_approve.txt`](file:///d:/antigravity/Naukri.com/transcripts/task14_review_approve.txt) & [`transcripts/task14_review_revise.txt`](file:///d:/antigravity/Naukri.com/transcripts/task14_review_revise.txt)
* **Unit Tests**: [`tests/test_autogen_review.py`](file:///d:/antigravity/Naukri.com/tests/test_autogen_review.py) (8 tests passed)

Two-agent editorial compliance stage operating after the initial draft composer:
1. **Policy_Compliance_Reviewer**: Audits candidate drafts against official domain policies (`POL-001`..`POL-012`).
2. **Final_Editor**: Emits validated Pydantic `VerdictModel`:
   ```python
   class VerdictModel(BaseModel):
       approved: bool
       final_answer: str
       reason: str
       policy_citations: List[str]
       compliance_score: float = Field(ge=0.0, le=1.0)
   ```
* **AutoGen Trap 5 Defeated**: Registers `custom_message_types=[StructuredMessage[VerdictModel]]` directly on the `RoundRobinGroupChat` constructor, preventing runtime serialization crashes.

---

### Task 15: Governance, Least-Autonomy & Cost Budget Control
* **Implementation**: [`governance/autonomy.py`](file:///d:/antigravity/Naukri.com/governance/autonomy.py) and [`governance/cost_budget.py`](file:///d:/antigravity/Naukri.com/governance/cost_budget.py)
* **Regulatory Assessment**: [`governance/risk_classification.md`](file:///d:/antigravity/Naukri.com/governance/risk_classification.md)
* **Transcript**: [`transcripts/task15_budget_reject.txt`](file:///d:/antigravity/Naukri.com/transcripts/task15_budget_reject.txt)
* **Unit Tests**: [`tests/test_governance.py`](file:///d:/antigravity/Naukri.com/tests/test_governance.py) (8 tests passed)

* **Least-Autonomy Matrix**:
  * Retrieval Agent: Read-only access to vector store; zero access to candidate application records.
  * Lookup Agent: Access to candidate application records; zero access to policy vector store.
  * Composer Agent: Pure synthesis; zero tools assigned.
* **Regulatory Risk Classification**: Formally assessed under **EU AI Act Annex III (High Risk: Employment and Worker Management)** and the **Indian DPDP Act 2023**.
* **Token Budget Manager**: Hard per-request limit of **2,000 tokens**. Oversized adversarial queries (e.g. 3,214 tokens) are intercepted prior to LLM execution with zero billed tokens incurred.

---

### Task 16: Response Caching & Latency Optimization
* **Implementation**: [`api/cache.py`](file:///d:/antigravity/Naukri.com/api/cache.py)
* **Transcript**: [`transcripts/task16_cache_hit.txt`](file:///d:/antigravity/Naukri.com/transcripts/task16_cache_hit.txt)
* **Unit Tests**: [`tests/test_cache.py`](file:///d:/antigravity/Naukri.com/tests/test_cache.py) (8 tests passed)

* **Canonical Query Normalization**: Strips punctuation, collapses whitespace, and lowercases text, ensuring that typographical variations resolve to the same cache key.
* **Thread-Safe LRU & TTL Cache**: Backed by `threading.Lock()`, with 1,000-entry capacity and 3,600s TTL.
* **Empirical Speedup Demonstration**:

| Performance Metric | Phase 1 (Cache Miss) | Phase 2 (Cache Hit) | Empirical Gain |
| :--- | :--- | :--- | :--- |
| **End-to-End Latency** | `18,372.07 ms` | `0.038 ms` | **100% faster (<0.04 ms)** |
| **Speedup Factor** | `1.0x` | `480,943.5x` | **>400,000x Speedup** |
| **Model Tokens Consumed**| `669 tokens` | `0 tokens` | **100% Compute Saved** |

---

## 📂 Repository Directory Structure

```text
naukri-support-agent/
├── .venv/                         # Isolated Python virtual environment (Python 3.13)
├── .gitignore                     # Git ignore rules for venv, cache, DB, and logs
├── pytest.ini                     # Pytest warning suppression configuration
├── requirements.txt               # Pinned dependencies
├── dataset.py                     # Root dataset generator (exports JOB_APPLICATIONS)
├── RUNTIME_ARCHITECTURE.md        # Mermaid diagrams & sequence lifecycles
├── runtime_architecture.html      # Interactive visualizer with live step-through simulation
├── README.md                      # Complete 16-task documentation & audit report
├── api/
│   ├── __init__.py                # Package initializer
│   ├── cache.py                   # Task 16: LRU ResponseCache with query normalization
│   ├── logging_middleware.py      # Task 12: Structured JSON-Lines logging with PII masking
│   └── main.py                    # Task 11: FastAPI backend (REST & WebSocket)
├── agents/
│   ├── crew.py                    # Task 7: CrewAI multi-agent orchestration pipeline
│   ├── guardrails.py              # Task 10: Input/Output guardrails (PII, injection, grounding)
│   ├── memory.py                  # Task 8: LangChain session memory layer
│   ├── mock_llm.py                # Task 7: Deterministic MockLLM (zero API keys)
│   ├── schemas.py                 # Task 9: Pydantic structured output models
│   └── tools.py                   # Task 6: Application status & continuous escalation tool
├── data/
│   ├── chroma_db/                 # Persistent ChromaDB vector storage (gitignored)
│   └── job_applications.json     # 50 synthetic application records
├── evaluation/
│   ├── __init__.py                # Package initializer
│   └── eval_suite.py              # Task 13: 15-query evaluation suite & 4-dimension rubric
├── governance/
│   ├── __init__.py                # Package initializer
│   ├── autonomy.py                # Task 15: Least-autonomy enforcement & audit report
│   ├── cost_budget.py             # Task 15: TokenBudgetManager & cost cap enforcement
│   └── risk_classification.md     # Task 15: EU AI Act & Indian DPDP Act risk assessment
├── knowledge_base/
│   ├── policies.json              # 12 official domain policy documents
│   └── policies.py                # Python loader for policies
├── rag/
│   ├── chunking.py                # Task 3: Dual chunking strategies (fixed and sentence)
│   ├── indexer.py                 # Task 3: ChromaDB indexing pipeline using all-MiniLM-L6-v2
│   ├── retriever.py               # Task 4: Semantic search, cosine similarity & grounded retrieval
│   └── evaluation.py              # Task 5: Precision@3 & Recall@3 comparative evaluation
├── review/
│   ├── __init__.py                # Package initializer
│   └── autogen_review.py          # Task 14: AutoGen 2-agent policy compliance review stage
├── scripts/
│   ├── calibrate_threshold.py     # Task 4: Empirical threshold calibration script
│   ├── generate_dataset.py        # Task 1: Seeded dataset generator script
│   ├── generate_task4_transcript.py # Task 4: Deterministic transcript runner
│   └── run_capstone_audit.py      # Day 14: Master capstone verification & audit engine
├── tests/
│   ├── test_dataset.py            # Task 1 test suite (11 assertions)
│   ├── test_rag.py                # Tasks 2 & 3 test suite (8 assertions)
│   ├── test_evaluation.py         # Tasks 4 & 5 test suite (5 assertions)
│   ├── test_agents.py             # Task 6 test suite (10 assertions)
│   ├── test_crew.py               # Task 7 test suite (9 assertions)
│   ├── test_memory_schemas.py     # Tasks 8 & 9 test suite (10 assertions)
│   ├── test_guardrails.py         # Task 10 test suite (10 assertions)
│   ├── test_api.py                # Task 11 test suite (10 assertions)
│   ├── test_logging_and_evaluation.py # Task 12 test suite (8 assertions)
│   ├── test_evaluation_execution.py   # Task 13 test suite (8 assertions)
│   ├── test_autogen_review.py         # Task 14 test suite (8 assertions)
│   ├── test_governance.py         # Task 15 test suite (8 assertions)
│   ├── test_cache.py              # Task 16 test suite (8 assertions)
│   └── test_capstone_audit.py     # Master audit test suite (4 assertions)
└── transcripts/
    ├── task4_grounded_gen.txt     # Task 4 grounded generation transcript
    ├── task5_eval_comparison.txt  # Task 5 precision/recall comparison transcript
    ├── task7_crew_demo.txt        # Task 7 multi-agent crew execution transcript
    ├── task8_memory_fresh.txt     # Task 8 fresh session reset transcript
    ├── task8_memory_multi_turn.txt# Task 8 multi-turn session memory transcript
    ├── task10_guardrails.txt      # Task 10 guardrail demonstration transcript
    ├── task13_eval_15_queries.txt # Task 13 15-query evaluation suite transcript
    ├── task13_eval_metrics.json   # Task 13 structured evaluation metrics output
    ├── task14_review_approve.txt  # Task 14 AutoGen review approval transcript
    ├── task14_review_revise.txt   # Task 14 AutoGen review revision transcript
    ├── task15_budget_reject.txt   # Task 15 token budget rejection transcript
    └── task16_cache_hit.txt       # Task 16 response caching demonstration transcript
```

---

## 📊 16-Task Traceability Matrix

| Task ID | Component Description | Implementation | Automated Test Suite | Proof Evidence | Status |
| :--- | :--- | :--- | :--- | :--- | :---: |
| **Task 1** | Seeded Synthetic Dataset Generator | [`scripts/generate_dataset.py`](file:///d:/antigravity/Naukri.com/scripts/generate_dataset.py) | [`tests/test_dataset.py`](file:///d:/antigravity/Naukri.com/tests/test_dataset.py) (11 tests) | `data/job_applications.json` | ✅ Complete |
| **Task 2** | 12 Domain Policy Documents | [`knowledge_base/policies.py`](file:///d:/antigravity/Naukri.com/knowledge_base/policies.py) | [`tests/test_rag.py`](file:///d:/antigravity/Naukri.com/tests/test_rag.py) (8 tests) | `knowledge_base/policies.json` | ✅ Complete |
| **Task 3** | Dual Chunking & ChromaDB Collections | [`rag/chunking.py`](file:///d:/antigravity/Naukri.com/rag/chunking.py), [`rag/indexer.py`](file:///d:/antigravity/Naukri.com/rag/indexer.py) | [`tests/test_rag.py`](file:///d:/antigravity/Naukri.com/tests/test_rag.py) | `data/chroma_db/` | ✅ Complete |
| **Task 4** | Empirically Calibrated Grounded Retrieval | [`rag/retriever.py`](file:///d:/antigravity/Naukri.com/rag/retriever.py), [`scripts/calibrate_threshold.py`](file:///d:/antigravity/Naukri.com/scripts/calibrate_threshold.py) | [`tests/test_evaluation.py`](file:///d:/antigravity/Naukri.com/tests/test_evaluation.py) (5 tests) | [`transcripts/task4_grounded_gen.txt`](file:///d:/antigravity/Naukri.com/transcripts/task4_grounded_gen.txt) | ✅ Complete |
| **Task 5** | Chunking Strategy Evaluation | [`rag/evaluation.py`](file:///d:/antigravity/Naukri.com/rag/evaluation.py) | [`tests/test_evaluation.py`](file:///d:/antigravity/Naukri.com/tests/test_evaluation.py) | [`transcripts/task5_eval_comparison.txt`](file:///d:/antigravity/Naukri.com/transcripts/task5_eval_comparison.txt) | ✅ Complete |
| **Task 6** | Application Status & Continuous Escalation | [`agents/tools.py`](file:///d:/antigravity/Naukri.com/agents/tools.py) | [`tests/test_agents.py`](file:///d:/antigravity/Naukri.com/tests/test_agents.py) (10 tests) | 80th-percentile: `0.5667` | ✅ Complete |
| **Task 7** | CrewAI Multi-Agent Team & MockLLM | [`agents/crew.py`](file:///d:/antigravity/Naukri.com/agents/crew.py), [`agents/mock_llm.py`](file:///d:/antigravity/Naukri.com/agents/mock_llm.py) | [`tests/test_crew.py`](file:///d:/antigravity/Naukri.com/tests/test_crew.py) (9 tests) | [`transcripts/task7_crew_demo.txt`](file:///d:/antigravity/Naukri.com/transcripts/task7_crew_demo.txt) | ✅ Complete |
| **Task 8** | LangChain Session Memory Layer | [`agents/memory.py`](file:///d:/antigravity/Naukri.com/agents/memory.py) | [`tests/test_memory_schemas.py`](file:///d:/antigravity/Naukri.com/tests/test_memory_schemas.py) (10 tests) | [`transcripts/task8_memory_multi_turn.txt`](file:///d:/antigravity/Naukri.com/transcripts/task8_memory_multi_turn.txt), [`transcripts/task8_memory_fresh.txt`](file:///d:/antigravity/Naukri.com/transcripts/task8_memory_fresh.txt) | ✅ Complete |
| **Task 9** | Pydantic Structured Output Validation | [`agents/schemas.py`](file:///d:/antigravity/Naukri.com/agents/schemas.py) | [`tests/test_memory_schemas.py`](file:///d:/antigravity/Naukri.com/tests/test_memory_schemas.py) | `NaukriSupportResponse` model | ✅ Complete |
| **Task 10** | Input/Output Guardrails Triad | [`agents/guardrails.py`](file:///d:/antigravity/Naukri.com/agents/guardrails.py) | [`tests/test_guardrails.py`](file:///d:/antigravity/Naukri.com/tests/test_guardrails.py) (10 tests) | [`transcripts/task10_guardrails.txt`](file:///d:/antigravity/Naukri.com/transcripts/task10_guardrails.txt) | ✅ Complete |
| **Task 11** | FastAPI REST & WebSocket Backend | [`api/main.py`](file:///d:/antigravity/Naukri.com/api/main.py) | [`tests/test_api.py`](file:///d:/antigravity/Naukri.com/tests/test_api.py) (10 tests) | `/ask`, `/health`, `/ws/chat` | ✅ Complete |
| **Task 12** | Structured JSON-Lines Logging & PII Masking | [`api/logging_middleware.py`](file:///d:/antigravity/Naukri.com/api/logging_middleware.py) | [`tests/test_logging_and_evaluation.py`](file:///d:/antigravity/Naukri.com/tests/test_logging_and_evaluation.py) (8 tests) | `logs/api_access.jsonl` | ✅ Complete |
| **Task 13** | 15-Query Evaluation Suite (4 Dimensions) | [`evaluation/eval_suite.py`](file:///d:/antigravity/Naukri.com/evaluation/eval_suite.py) | [`tests/test_evaluation_execution.py`](file:///d:/antigravity/Naukri.com/tests/test_evaluation_execution.py) (8 tests) | [`transcripts/task13_eval_15_queries.txt`](file:///d:/antigravity/Naukri.com/transcripts/task13_eval_15_queries.txt), [`transcripts/task13_eval_metrics.json`](file:///d:/antigravity/Naukri.com/transcripts/task13_eval_metrics.json) | ✅ Complete |
| **Task 14** | AutoGen Multi-Agent Policy Review Stage | [`review/autogen_review.py`](file:///d:/antigravity/Naukri.com/review/autogen_review.py) | [`tests/test_autogen_review.py`](file:///d:/antigravity/Naukri.com/tests/test_autogen_review.py) (8 tests) | [`transcripts/task14_review_approve.txt`](file:///d:/antigravity/Naukri.com/transcripts/task14_review_approve.txt), [`transcripts/task14_review_revise.txt`](file:///d:/antigravity/Naukri.com/transcripts/task14_review_revise.txt) | ✅ Complete |
| **Task 15** | Governance, Least-Autonomy & Cost Budget | [`governance/autonomy.py`](file:///d:/antigravity/Naukri.com/governance/autonomy.py), [`governance/cost_budget.py`](file:///d:/antigravity/Naukri.com/governance/cost_budget.py), [`governance/risk_classification.md`](file:///d:/antigravity/Naukri.com/governance/risk_classification.md) | [`tests/test_governance.py`](file:///d:/antigravity/Naukri.com/tests/test_governance.py) (8 tests) | [`transcripts/task15_budget_reject.txt`](file:///d:/antigravity/Naukri.com/transcripts/task15_budget_reject.txt) | ✅ Complete |
| **Task 16** | Normalized LRU Response Caching | [`api/cache.py`](file:///d:/antigravity/Naukri.com/api/cache.py) | [`tests/test_cache.py`](file:///d:/antigravity/Naukri.com/tests/test_cache.py) (8 tests) | [`transcripts/task16_cache_hit.txt`](file:///d:/antigravity/Naukri.com/transcripts/task16_cache_hit.txt) | ✅ Complete |
| **Audit** | Master Capstone Integrity Audit | [`scripts/run_capstone_audit.py`](file:///d:/antigravity/Naukri.com/scripts/run_capstone_audit.py) | [`tests/test_capstone_audit.py`](file:///d:/antigravity/Naukri.com/tests/test_capstone_audit.py) (4 tests) | **16 of 16 Checks Passed (100%)** | ✅ Complete |

---

## 🏆 Final Capstone Submission Statement

The Naukri.com Domain Support Agent project satisfies 100% of architectural, mathematical, and regulatory requirements:
* **Zero External Dependencies**: Operates 100% locally and deterministically via `MockLLM`.
* **Zero Telemetry Leakage**: Hardened via `CREWAI_DISABLE_TELEMETRY=true` and `OTEL_SDK_DISABLED=true`.
* **Zero PII Exposure in Logs**: Enforced via regex masking and middleware interception.
* **117 Passing Unit Tests**: Complete test coverage across all 14 test suites with 0 warnings.
* **11 Verified Transcripts**: Full empirical proof recorded in `transcripts/`.
* **Public GitHub Repository**: [https://github.com/sahil1998237/naukri-domain-support-agent](https://github.com/sahil1998237/naukri-domain-support-agent)
