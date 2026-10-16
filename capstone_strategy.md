# 🎯 Naukri.com Capstone — Deep Analysis & Strategy

## TL;DR — What This Project Actually Is

This is **not** a chatbot demo. It's a **full production-grade AI system** with 16 interlocking tasks across 4 layers. Think of it as building a real enterprise support agent — the kind that would survive a production governance audit. The good news: **everything runs locally with zero API keys** (MOCK_LLM mode). The bad news: every piece must genuinely work together, not just exist as isolated files.

---

## 🧩 Architecture at a Glance

```
┌─────────────────────────────────────────────────────────────────┐
│                     FastAPI Backend (Part 3)                     │
│  POST /ask  ·  POST /add-document  ·  WS /chat                 │
│  Structured JSON-Lines Logging (trace ID + timing)              │
├─────────────────────────────────────────────────────────────────┤
│                  Guardrails Layer (Part 2)                       │
│  Input: PII masking (phone) + prompt-injection detection        │
│  Output: groundedness check (similarity threshold)              │
├─────────────────────────────────────────────────────────────────┤
│              CrewAI Multi-Agent Crew (Part 2)                   │
│  ┌──────────────┐ ┌──────────────┐ ┌───────────────────┐       │
│  │  Retrieval   │ │   Lookup     │ │ Response Composer  │       │
│  │    Agent     │ │    Agent     │ │      Agent         │       │
│  │ (RAG tool)   │ │ (status tool)│ │ (combines outputs) │       │
│  └──────┬───────┘ └──────┬───────┘ └────────┬──────────┘       │
│         │                │                   │                   │
│  Session Memory (LangChain InMemoryChatMessageHistory)          │
│  Structured Output (Pydantic schema validation)                 │
│  Response Cache (normalized query key)                          │
├─────────────────────────────────────────────────────────────────┤
│           Autogen Review Stage (Part 4)                         │
│  ┌──────────────────────┐  ┌─────────────────────┐             │
│  │ Policy Compliance    │  │   Final Editor       │             │
│  │    Reviewer          │  │ (Pydantic verdict)   │             │
│  └──────────────────────┘  └─────────────────────┘             │
│  RoundRobinGroupChat · max_turns=2                              │
├─────────────────────────────────────────────────────────────────┤
│              RAG Core (Part 1)                                  │
│  Knowledge Base (≥12 docs) → Chunking (2 strategies)           │
│  → SentenceTransformers → ChromaDB (2 collections)             │
│  → Grounded Generation (similarity threshold)                   │
├─────────────────────────────────────────────────────────────────┤
│            Dataset (Part 1)                                     │
│  dataset.py → ≥40 seeded job applications                       │
│  5 categories · 5 statuses · flagged 10-30%                     │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📊 Difficulty Assessment — Where the Traps Are

### 🟢 Straightforward (but must be done right)
| Task | What | Why it's manageable |
|------|------|---------------------|
| T1 | Dataset generator | Pure Python, deterministic seed, simple validation |
| T2 | Knowledge base | 12 policy docs, 2-5 sentences each, your own words |
| T3 | Chunking + indexing | Two strategies into two ChromaDB collections |
| T9 | Pydantic schema | Define a BaseModel, validate responses |
| T12 | Structured logging | JSON-Lines with trace ID, straightforward |

### 🟡 Moderate (requires careful implementation)
| Task | What | Why it needs care |
|------|------|-------------------|
| T4 | Grounded generation | Must empirically calibrate threshold — no guessing |
| T5 | Eval both strategies | Per-query precision/recall with arithmetic shown |
| T6 | Escalation score | Must be a continuous [0,1] formula, not a boolean |
| T8 | Session memory | Two transcripts needed: multi-turn + fresh session |
| T10 | Guardrails | 3 guardrails, each must fire on a test case |
| T11 | FastAPI + WebSocket | WS must survive client disconnect gracefully |
| T16 | Response caching | Must show before/after evidence (counter or timing) |

### 🔴 Hard (this is where projects fail)
| Task | What | Why it's hard |
|------|------|---------------|
| T7 | CrewAI crew + MOCK_LLM | Extending `BaseLLM`, avoiding the two silent pitfalls, disabling telemetry |
| T13 | LLM-as-judge at scale | 15 queries × 4 scores, all under MOCK_LLM |
| T14 | Autogen review stage | Pydantic structured output with `custom_message_types`, approve vs. revise |
| T15 | Governance (4 layers) | Least-autonomy enforcement, risk classification, token budget cap |

---

## 🚨 Critical Traps Explicitly Called Out in the Spec

> [!CAUTION]
> These are not hypothetical — the spec literally warns about each one. Missing any = silent wrong output, not a crash.

1. **CrewAI ReAct parser trap**: The system prompt contains `"Observation: the result of the action"` — if your parser greps for `"Observation:"` in the full conversation, it matches the template before any tool runs. **Parse only the model's generated text, not the system prompt.**

2. **Tool dispatch by name trap**: If you match tool names by substring (`"lookup" in tool_name`), a tool named `rag_lookup` will match the wrong handler. **Dispatch by argument schema, not name.**

3. **CrewAI telemetry trap**: `crew.kickoff()` phones home by default. **Set `CREWAI_DISABLE_TELEMETRY=true` before running.** Confirm in README.

4. **Autogen `MaxMessageTermination` off-by-one**: The initiating message counts as message 1, so `MaxMessageTermination(3)` is needed for 2 agents to both speak (not `(2)`).

5. **Autogen `custom_message_types` crash**: If you use `output_content_type=YourModel` on the Final-Editor agent, you **must** also pass `custom_message_types=[StructuredMessage[YourVerdictModel]]` to the Team constructor, or it crashes with `ValueError`.

6. **PII in logs**: The logged request must have PII already masked. Apply the same masking to logs as to model input.

---

## 📅 14-Day Execution Plan

### Phase 1: Foundation (Days 1–3) — Parts 1

**Day 1: Dataset + Knowledge Base (Tasks 1-2)**
- Write `dataset.py` with a seeded generator
- Key decisions to make upfront:
  - Pick a seed (e.g., `42`) and salary range (e.g., ₹3,00,000 – ₹35,00,000 — reasoning: covers fresher to senior roles in Indian job market)
  - Weight categories so each has ≥3 records out of 40+
  - Weight `flagged_priority_review` to land 10-30% (e.g., 20% probability)
  - Run it, check thresholds, adjust seed/weights if needed
- Write 12+ knowledge-base documents (plain text or YAML/JSON)
  - Cover every required topic
  - Each 2-5 sentences, written in your own words
  - Think like a Naukri.com HR policy manual

**Day 2: Chunking + Embedding + Indexing (Task 3)**
- Implement two chunking strategies:
  - **Fixed-size with overlap**: e.g., 200 chars with 50 char overlap
  - **Sentence-based**: split on sentence boundaries
- Use a free SentenceTransformers model (e.g., `all-MiniLM-L6-v2`)
- Index each strategy into its own ChromaDB collection
- Verify retrieval works on a test query

**Day 3: Grounded Generation + Evaluation (Tasks 4-5)**
- Build the retrieval + generation pipeline
- **Critical**: empirically calibrate your similarity threshold
  - Run 3+ in-scope queries, record top-1 cosine similarity
  - Run 2+ out-of-scope queries, record top-1 cosine similarity
  - Set threshold between the two clusters
  - Document measured values in README
- Run ≥5 in-scope + 1 out-of-scope queries
- Compute precision/recall for BOTH collections on the same queries
- Write the 2-3 sentence recommendation

### Phase 2: CrewAI Orchestration (Days 4–7) — Part 2

**Day 4: MOCK_LLM + Escalation Tool (Tasks 6-7 setup)**
- **This is the hardest single day.** Get MOCK_LLM right first.
- Extend `crewai.llms.base_llm.BaseLLM` to create `MockLLM`
  - Must handle tool-calling format (return structured tool calls)
  - Must avoid both parser traps
  - Must produce deterministic, sensible outputs
- Implement `check_job_application_status(record_id)`:
  - Formula design: e.g., `escalation_score = 0.6 * flagged + 0.4 * (days_since_created / 30)`
  - State threshold and justify from your data's percentile distribution

**Day 5: CrewAI Crew Assembly (Task 7)**
- Build 3+ agents: Retrieval, Lookup, Response Composer
- Wire RAG tool to Retrieval Agent only
- Wire status tool to Lookup Agent only (this also satisfies Task 15's least-autonomy)
- Run `crew.kickoff()` with telemetry disabled
- Generate transcripts showing both tools invoked on different queries

**Day 6: Memory + Structured Output (Tasks 8-9)**
- Integrate LangChain's `InMemoryChatMessageHistory` + `RunnableWithMessageHistory`
- Generate two transcripts:
  - Multi-turn: show state carried across turns
  - Fresh session: show state correctly absent/reset
- Define Pydantic `AgentResponse` model (e.g., `answer: str`, `sources: list`, `confidence: float`, `escalation_needed: bool`)
- Validate every crew response against it

**Day 7: Guardrails (Task 10)**
- **Input-side**:
  - PII masking: regex for Indian phone numbers (10-digit, +91 prefix patterns)
  - Prompt-injection detection: pattern matching for common injection phrases
- **Output-side**:
  - Groundedness check using the similarity threshold from Task 4
- Generate test cases showing each guardrail firing

### Phase 3: Deployment + Evaluation (Days 8–10) — Part 3

**Day 8: FastAPI Backend (Task 11)**
- `POST /ask` — single query endpoint
- `POST /add-document` — add to knowledge base
- `WebSocket /chat` — multi-turn real-time chat
  - **Must catch `WebSocketDisconnect`** and continue serving other clients
- Use Pydantic models for all request/response schemas

**Day 9: Logging + Evaluation Setup (Tasks 12-13)**
- Structured logging middleware:
  - JSON-Lines format
  - Fields: `trace_id`, `timestamp`, `duration_ms`, `query` (masked), `response_summary`
  - Ensure PII is masked before logging
- Build evaluation test set: 15 queries
  - ≥1 per KB topic (12 queries minimum)
  - ≥2 out-of-scope/edge-case queries
  - Score each on 4 dimensions under MOCK_LLM

**Day 10: Evaluation Execution (Task 13 completion)**
- Run all 15 queries through the system
- Score Accuracy, Grounding, Completeness, Safety for each
- Compute averages
- Format results in a clear table

### Phase 4: Hardening + Governance (Days 11–13) — Part 4

**Day 11: Autogen Review Stage (Task 14)**
- Install `autogen-agentchat` and `autogen-ext`
- Build 2-agent RoundRobinGroupChat:
  - Policy-Compliance-Reviewer
  - Final-Editor with `output_content_type=VerdictModel`
- **Don't forget** `custom_message_types=[StructuredMessage[VerdictModel]]` on the Team
- Generate 2 transcripts:
  - One where review approves the draft
  - One where review revises the draft (deliberately inject an ungrounded claim)

**Day 12: Governance + Cost Control (Task 15)**
- Least-autonomy enforcement:
  - Show that only Lookup Agent has the status tool wired
  - Write the one-paragraph explanation
- Risk classification:
  - This system handles hiring data + salary info → **Medium-to-High risk**
  - Write one-paragraph justification
- Token/cost budget cap:
  - Implement a per-request token counter in MOCK_LLM
  - Set a budget (e.g., 2000 tokens)
  - Demonstrate rejection of an oversized request

**Day 13: Caching + Polish (Task 16)**
- Implement in-memory cache:
  - Normalize queries (lowercase, strip whitespace, etc.)
  - Key by normalized query text
  - Return cached result on hit
- Demonstrate with before/after:
  - Call counter showing 0 LLM calls on cache hit
  - Or timing comparison

### Phase 5: Documentation + Submission (Day 14)

**Day 14: README + Final Testing**
- Write comprehensive README.md:
  - Track declaration (Naukri.com Recruitment & HR)
  - Dataset design choices (seed, weights, salary range)
  - Similarity threshold values (measured + chosen)
  - Escalation formula + threshold justification
  - CREWAI_DISABLE_TELEMETRY confirmation
  - Instructions to run everything under MOCK_LLM
- End-to-end smoke test: run every task's transcript from scratch
- Push to GitHub

---

## 🧠 Key Design Decisions to Make Before Coding

### 1. MOCK_LLM Strategy (Most Critical Decision)

The entire project lives or dies on how well your MOCK_LLM works. It must:
- Return valid tool-calling responses when agents need to use tools
- Return coherent "answers" for grounded generation
- Produce structured output that validates against Pydantic schemas
- Work with CrewAI's ReAct loop without triggering the parser traps
- Work with Autogen's message protocol

**My recommendation**: Build a **rule-based MOCK_LLM** that:
- Detects when a tool call is expected (by inspecting the available tools in the prompt)
- Returns a deterministic tool call response based on the query content
- Returns template-based final answers that incorporate retrieved context
- Returns valid JSON for structured output requests

This is the single most complex piece. Invest time here early.

### 2. Project Structure

```
naukri-support-agent/
├── README.md
├── requirements.txt
├── dataset.py                    # Task 1: Seeded dataset generator
├── knowledge_base/
│   └── policies.py               # Task 2: 12+ policy documents
├── rag/
│   ├── chunking.py               # Task 3: Two chunking strategies
│   ├── indexer.py                 # Task 3: ChromaDB indexing
│   ├── retriever.py              # Task 4: Retrieval + grounded generation
│   └── evaluation.py             # Task 5: Precision/recall comparison
├── agents/
│   ├── mock_llm.py               # MOCK_LLM (BaseLLM extension)
│   ├── tools.py                  # Task 6: check_job_application_status
│   ├── crew.py                   # Task 7: CrewAI crew definition
│   ├── memory.py                 # Task 8: Session memory
│   ├── schemas.py                # Task 9: Pydantic output schemas
│   └── guardrails.py             # Task 10: Input/output guardrails
├── review/
│   └── autogen_review.py         # Task 14: Autogen review stage
├── governance/
│   ├── autonomy.py               # Task 15: Least-autonomy enforcement
│   ├── risk_classification.md    # Task 15: Risk assessment
│   └── cost_budget.py            # Task 15: Token budget cap
├── api/
│   ├── main.py                   # Task 11: FastAPI app
│   ├── logging_middleware.py     # Task 12: Structured logging
│   └── cache.py                  # Task 16: Response caching
├── evaluation/
│   └── eval_suite.py             # Task 13: 15-query evaluation
├── transcripts/                  # All demonstration transcripts
│   ├── task4_grounded_gen.txt
│   ├── task7_crew_demo.txt
│   ├── task8_memory_multi_turn.txt
│   ├── task8_memory_fresh.txt
│   ├── task10_guardrails.txt
│   ├── task14_review_approve.txt
│   ├── task14_review_revise.txt
│   ├── task15_budget_reject.txt
│   └── task16_cache_hit.txt
└── tests/                        # Optional but recommended
    └── ...
```

### 3. Escalation Score Formula (Task 6)

Proposed formula:
```
recency_score = days_since_created / 30    # normalized to [0, 1]
escalation_score = w1 * flagged_priority_review + w2 * (1 - recency_score)
```
Wait — re-read the spec: "combines flagged_priority_review with a normalized recency signal." So:
- `flagged_priority_review` is binary (0 or 1)
- `recency` = `days_since_created / 30` (0 = just created, 1 = 30 days old)
- Higher `days_since_created` → more urgent → higher escalation

```python
escalation_score = 0.5 * int(flagged) + 0.5 * (days_since_created / 30)
```

Threshold: compute the 80th percentile of your dataset's escalation scores, use that as the escalation recommendation threshold.

### 4. Similarity Threshold Calibration (Task 4)

**Do NOT guess.** The spec explicitly says common defaults (0.5/0.6/0.7) are unreliable.

Process:
1. Embed 3+ in-scope queries (e.g., "What is the notice period policy?")
2. Embed 2+ out-of-scope queries (e.g., "What is the weather today?")
3. Record top-1 cosine similarity for each
4. Find the gap between the two clusters
5. Set threshold in the middle of that gap
6. Document all measured values in README

### 5. Knowledge Base Content Strategy

Write these as if you're a Naukri.com HR policy writer. Each doc should be:
- **Specific enough** to be retrieved for the right query
- **Distinct enough** from other docs to avoid cross-contamination in retrieval
- **2-5 sentences** — not too short (poor embedding quality) or too long

---

## ⚠️ Risk Matrix — What Could Go Wrong

| Risk | Impact | Mitigation |
|------|--------|------------|
| MOCK_LLM doesn't produce valid tool calls | 🔴 Blocks Tasks 7-16 | Build and test MOCK_LLM on Day 4 before anything else |
| CrewAI parser traps cause silent wrong output | 🔴 Lose marks without knowing | Follow the two explicit warnings from the spec |
| Autogen version incompatibility | 🟡 Wastes debugging time | Pin exact versions in requirements.txt |
| Similarity threshold too high/low | 🟡 Wrong "I don't know" behavior | Empirical calibration, not guessing |
| Forgetting `custom_message_types` on Autogen Team | 🔴 Crashes at runtime | Spec explicitly warns about this |
| PII leaking into logs | 🟡 Fails Task 12 | Apply masking before logging, same function as Task 10 |
| `flagged_priority_review` percentage outside 10-30% | 🟡 Must regenerate | Check immediately after generation, adjust seed/weights |

---

## 💡 Key Insights From My Analysis

### 1. Everything feeds forward
The project is designed as a pipeline:
- Part 1 → feeds into Part 2 (RAG core becomes crew's tools)
- Part 2 → feeds into Part 3 (crew becomes FastAPI backend)
- Part 2 → feeds into Part 4 (crew's output goes through Autogen review)

**You cannot skip ahead.** Each part depends on the previous one being solid.

### 2. MOCK_LLM is the linchpin
Every graded transcript must run under MOCK_LLM. This means your mock must be smart enough to:
- Route queries to the right tool
- Produce answers that incorporate retrieved context
- Generate valid structured output
- Work with both CrewAI and Autogen

This is essentially writing a **rule-based agent** that pretends to be an LLM.

### 3. Transcripts are your proof
The spec says "demonstrate" for nearly every task. This means you need **runnable scripts that produce transcript output**. Not screenshots, not descriptions — actual text output that shows the system working.

### 4. The spec is unusually specific about failure modes
The spec calls out exact pitfalls (parser traps, off-by-one errors, telemetry, constructor parameters). This tells me the grading rubric will specifically check for these. Follow the spec's warnings to the letter.

### 5. "Production-minded" doesn't mean "production-deployed"
You don't need Kubernetes, Docker, or cloud deployment. You need:
- Structured logging ✓
- Guardrails ✓
- Governance policy ✓
- Cost control ✓
- Caching ✓
- Error handling (WebSocket disconnect) ✓

These are all local/in-process. The "production" part is the *discipline*, not the infrastructure.

---

## 🎯 Priority Order If Running Short on Time

If you're running behind, prioritize by marks:

| Priority | Parts | Marks | Why |
|----------|-------|-------|-----|
| 1st | Part 1 (Tasks 1-5) | 30 | Foundation — everything depends on this |
| 2nd | Part 2 (Tasks 6-10) | 30 | Core agent functionality |
| 3rd | Part 4 (Tasks 14-16) | 20 | Governance — easy marks if Part 2 works |
| 4th | Part 3 (Tasks 11-13) | 20 | Deployment — can be done quickly if crew works |

Within each part, the highest-risk tasks should be tackled first (MOCK_LLM, Autogen integration).

---

## 🔧 Technology Versions to Pin

```
crewai>=0.80.0        # Check for BaseLLM extension point
autogen-agentchat     # For RoundRobinGroupChat
autogen-ext           # For structured output support
chromadb              # Vector store
sentence-transformers # Embeddings (all-MiniLM-L6-v2)
fastapi               # API deployment
uvicorn               # ASGI server
pydantic>=2.0         # Structured output schemas
langchain-core        # For InMemoryChatMessageHistory
```

> [!IMPORTANT]
> Check the exact CrewAI and Autogen versions before starting. The APIs change between versions. Pin your versions in `requirements.txt` on Day 1.

---

## ✅ Summary — Your Attack Plan

1. **Days 1-3**: Build the data foundation (dataset, KB, RAG). This is straightforward but must be correct.
2. **Day 4**: Tackle the hardest piece first — MOCK_LLM. Get this right and everything downstream becomes manageable.
3. **Days 5-7**: Build the CrewAI crew with memory, schemas, and guardrails.
4. **Days 8-10**: Wrap it in FastAPI, add logging, run evaluation.
5. **Days 11-13**: Autogen review, governance, caching.
6. **Day 14**: README, final testing, submit.

**The project is difficult but very doable in 14 days if you follow the dependency chain and don't get stuck on MOCK_LLM.** That's where I'd allocate extra buffer time.
