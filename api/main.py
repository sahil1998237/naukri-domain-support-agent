"""
FastAPI Backend Service for Naukri.com Domain Support Agent (Task 11).

Endpoints:
- POST /ask: Single query support endpoint with full input/output guardrails and multi-agent execution.
- POST /add-document: Adds a new policy document into live ChromaDB vector collections.
- WebSocket /chat: Multi-turn real-time conversational streaming endpoint with graceful disconnect handling.
- GET /health: System health and component status check.

Guarantees:
- Full Pydantic validation across all requests and responses.
- Catches WebSocketDisconnect cleanly to ensure persistent service availability.
- Integrates seamlessly with SessionMemoryManager, Guardrails, and CrewAI orchestration.
"""

import asyncio
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional
import uuid

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from agents.guardrails import (
    detect_prompt_injection,
    mask_pii,
    verify_output_groundedness,
)
from agents.memory import _GLOBAL_MEMORY, process_support_query
from agents.schemas import ApplicationStatusInfo, PolicyCitation
from rag.chunking import chunk_policy_document
from rag.indexer import (
    COLLECTION_FIXED,
    COLLECTION_SENTENCE,
    get_chroma_client,
    get_embedding_function,
)
from api.cache import get_response_cache, normalize_query

logger = logging.getLogger("naukri_api")
logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="Naukri.com Domain Support Agent API",
    version="1.0.0",
    description="Production-grade AI Support Backend with Multi-Agent Orchestration, Guardrails, and RAG.",
)

from api.logging_middleware import StructuredLoggingMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(StructuredLoggingMiddleware)

POLICY_KEYWORDS = [
    "policy", "reschedule", "interview", "cooling", "salary",
    "offer", "bgv", "notice", "buyout", "screening", "grievance", "privacy"
]


# ==============================================================================
# PYDANTIC REQUEST & RESPONSE SCHEMAS
# ==============================================================================

class AskRequest(BaseModel):
    """Candidate inquiry request payload."""
    query: str = Field(..., description="Candidate question or support inquiry")
    session_id: Optional[str] = Field(default=None, description="Optional conversational session ID")
    bypass_cache: bool = Field(default=False, description="Optionally bypass response cache")

    @field_validator("query")
    @classmethod
    def query_not_blank(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Query string cannot be blank or whitespace-only.")
        return v.strip()


class AskResponse(BaseModel):
    """Guarded multi-agent response payload."""
    query: str
    clean_query: str
    answer: str
    sources: List[str] = Field(default_factory=list)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    escalation_needed: bool = False
    record_id: Optional[str] = None
    application_details: Optional[ApplicationStatusInfo] = None
    citations: List[PolicyCitation] = Field(default_factory=list)
    pii_masked: bool = False
    pii_details: List[str] = Field(default_factory=list)
    injection_detected: bool = False
    groundedness_verified: bool = True
    session_id: str
    execution_status: str = "COMPLETED_SAFE"
    is_cached: bool = False
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class AddDocumentRequest(BaseModel):
    """Payload to index a new policy document into live vector collections."""
    doc_id: str = Field(..., description="Unique policy identifier (e.g., POL-013)")
    title: str = Field(..., description="Policy document title")
    category: str = Field(..., description="Policy category")
    content: str = Field(..., description="Policy text (at least 2 sentences)")

    @field_validator("doc_id", "title", "category", "content")
    @classmethod
    def fields_not_blank(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Field cannot be empty or whitespace-only.")
        return v.strip()


class AddDocumentResponse(BaseModel):
    """Response returned upon indexing a document."""
    status: str = "success"
    message: str
    doc_id: str
    fixed_chunks_added: int
    sentence_chunks_added: int
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class HealthResponse(BaseModel):
    """Health check response schema."""
    status: str = "healthy"
    chromadb_connected: bool
    fixed_chunks_count: int
    sentence_chunks_count: int
    active_memory_sessions: int
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


# ==============================================================================
# CORE GUARDED EXECUTION PIPELINE
# ==============================================================================

def execute_guarded_ask(
    query: str,
    session_id: Optional[str] = None,
    bypass_cache: bool = False,
) -> AskResponse:
    """
    Executes the end-to-end support pipeline:
    1. Input PII masking (redacts Indian phone numbers and emails)
    2. Input Prompt-Injection detection (intercepts adversarial jailbreaks)
    3. Response caching lookup via normalized query key (Task 16)
    4. Multi-agent crew execution with session memory
    5. Output Groundedness verification (enforces calibrated 0.4000 threshold)
    """
    effective_session_id = session_id or f"session_{uuid.uuid4().hex[:8]}"

    # Step 1: Input PII Masking
    pii_audit = mask_pii(query)
    clean_query = pii_audit["masked_text"]

    # Step 2: Input Prompt Injection Defense
    injection_audit = detect_prompt_injection(clean_query)
    if injection_audit["is_injection"]:
        return AskResponse(
            query=query,
            clean_query=clean_query,
            answer=injection_audit["safe_response"],
            sources=[],
            confidence=0.0,
            escalation_needed=False,
            pii_masked=pii_audit["pii_detected"],
            pii_details=pii_audit["masked_entities"],
            injection_detected=True,
            groundedness_verified=False,
            session_id=effective_session_id,
            execution_status="BLOCKED_INPUT_INJECTION",
            is_cached=False,
        )

    # Step 2.5: Response Cache Lookup (Task 16)
    cache = get_response_cache()
    if not bypass_cache:
        cached_entry = cache.get(clean_query)
        if cached_entry:
            cached_data = dict(cached_entry)
            cached_data["session_id"] = effective_session_id
            cached_data["query"] = query
            cached_data["is_cached"] = True
            return AskResponse(**cached_data)

    # Step 3: Multi-Agent Crew & Session Memory Processing
    structured_res = process_support_query(clean_query, session_id=effective_session_id)

    # Step 4: Output Groundedness Verification (for all informational / non-application queries)
    final_answer = structured_res.answer
    is_grounded = True
    execution_status = "COMPLETED_SAFE"

    if "APP-" not in clean_query.upper():
        groundedness_audit = verify_output_groundedness(clean_query, structured_res.answer)
        is_grounded = groundedness_audit["is_grounded"]
        final_answer = groundedness_audit["verified_answer"]
        if not is_grounded:
            execution_status = groundedness_audit.get("status", "INTERCEPTED_LOW_SIMILARITY")

    response = AskResponse(
        query=query,
        clean_query=clean_query,
        answer=final_answer,
        sources=structured_res.sources,
        confidence=structured_res.confidence if is_grounded else 0.2,
        escalation_needed=structured_res.escalation_needed,
        record_id=structured_res.record_id,
        application_details=structured_res.application_details,
        citations=structured_res.citations if is_grounded else [],
        pii_masked=pii_audit["pii_detected"],
        pii_details=pii_audit["masked_entities"],
        injection_detected=False,
        groundedness_verified=is_grounded,
        session_id=effective_session_id,
        execution_status=execution_status,
        is_cached=False,
    )

    # Cache safe successful responses
    if not bypass_cache and execution_status == "COMPLETED_SAFE":
        cache.set(clean_query, response.model_dump())

    return response


# ==============================================================================
# FASTAPI HTTP ENDPOINTS
# ==============================================================================

@app.get("/health", response_model=HealthResponse)
@app.get("/api/v1/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    """Verifies service health, ChromaDB collection status, and session counts."""
    client = get_chroma_client()
    try:
        col_fixed = client.get_collection(COLLECTION_FIXED)
        col_sent = client.get_collection(COLLECTION_SENTENCE)
        fixed_count = col_fixed.count()
        sent_count = col_sent.count()
        chroma_ok = True
    except Exception:
        fixed_count = 0
        sent_count = 0
        chroma_ok = False

    active_sessions = len(_GLOBAL_MEMORY._sessions)

    return HealthResponse(
        status="healthy" if chroma_ok else "degraded",
        chromadb_connected=chroma_ok,
        fixed_chunks_count=fixed_count,
        sentence_chunks_count=sent_count,
        active_memory_sessions=active_sessions,
    )


@app.post("/ask", response_model=AskResponse)
@app.post("/api/v1/support/ask", response_model=AskResponse)
def ask_endpoint(payload: AskRequest) -> AskResponse:
    """
    Candidate inquiry endpoint:
    - Applies Input PII Masking & Prompt Injection Detection
    - Leverages Normalized Query Response Cache (Task 16)
    - Executes Multi-Agent Crew & LangChain Session Memory
    - Enforces Output Groundedness Verification (>= 0.4000 cutoff)
    """
    try:
        return execute_guarded_ask(
            query=payload.query,
            session_id=payload.session_id,
            bypass_cache=payload.bypass_cache,
        )
    except Exception as exc:
        logger.error(f"Error handling /ask request: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Support pipeline execution failed: {str(exc)}",
        )


@app.get("/cache/metrics")
@app.get("/api/v1/cache/metrics")
def get_cache_metrics():
    """Returns real-time response cache metrics (hits, misses, hit rate, tokens saved)."""
    return get_response_cache().get_metrics()


@app.post("/cache/clear")
@app.post("/api/v1/cache/clear")
def clear_cache():
    """Purges all entries from the response cache."""
    get_response_cache().clear()
    return {"status": "cleared", "message": "Response cache cleared successfully."}


@app.post("/add-document", response_model=AddDocumentResponse)
@app.post("/api/v1/knowledge/add-document", response_model=AddDocumentResponse)
def add_document_endpoint(payload: AddDocumentRequest) -> AddDocumentResponse:
    """
    Dynamically indexes a new policy document into ChromaDB:
    - Splits text into fixed-size and sentence-based chunks
    - Upserts chunks into both ChromaDB collections
    """
    try:
        client = get_chroma_client()
        ef = get_embedding_function()

        col_fixed = client.get_or_create_collection(
            name=COLLECTION_FIXED,
            embedding_function=ef,
            metadata={"hnsw:space": "cosine"},
        )
        col_sentence = client.get_or_create_collection(
            name=COLLECTION_SENTENCE,
            embedding_function=ef,
            metadata={"hnsw:space": "cosine"},
        )

        doc_dict = payload.model_dump()
        fixed_chunks = chunk_policy_document(doc_dict, strategy="fixed")
        sentence_chunks = chunk_policy_document(doc_dict, strategy="sentence")

        if fixed_chunks:
            col_fixed.upsert(
                ids=[c["chunk_id"] for c in fixed_chunks],
                documents=[c["text"] for c in fixed_chunks],
                metadatas=[
                    {
                        "doc_id": c["doc_id"],
                        "title": c["title"],
                        "category": c["category"],
                        "chunk_index": c["chunk_index"],
                        "strategy": "fixed",
                    }
                    for c in fixed_chunks
                ],
            )

        if sentence_chunks:
            col_sentence.upsert(
                ids=[c["chunk_id"] for c in sentence_chunks],
                documents=[c["text"] for c in sentence_chunks],
                metadatas=[
                    {
                        "doc_id": c["doc_id"],
                        "title": c["title"],
                        "category": c["category"],
                        "chunk_index": c["chunk_index"],
                        "strategy": "sentence",
                    }
                    for c in sentence_chunks
                ],
            )

        return AddDocumentResponse(
            status="success",
            message=f"Policy document '{payload.doc_id}' successfully indexed into vector collections.",
            doc_id=payload.doc_id,
            fixed_chunks_added=len(fixed_chunks),
            sentence_chunks_added=len(sentence_chunks),
        )
    except Exception as exc:
        logger.error(f"Error indexing document {payload.doc_id}: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to index policy document: {str(exc)}",
        )


# ==============================================================================
# WEBSOCKET REAL-TIME STREAMING ENDPOINT
# ==============================================================================

@app.websocket("/chat")
@app.websocket("/ws/chat")
async def websocket_chat_endpoint(websocket: WebSocket):
    """
    Multi-turn real-time WebSocket chat endpoint.
    Guarantees:
    - Maintains conversational session state across turns
    - Catches WebSocketDisconnect cleanly without crashing or blocking other clients
    """
    await websocket.accept()
    session_id = f"ws_{uuid.uuid4().hex[:8]}"
    logger.info(f"WebSocket client connected: session_id={session_id}")

    try:
        while True:
            raw_text = await websocket.receive_text()
            if not raw_text or not raw_text.strip():
                await websocket.send_json({"error": "Empty message received", "status": "error"})
                continue

            # Support either JSON payload or raw text prompt
            query_str = raw_text.strip()
            active_session = session_id
            try:
                data = json.loads(raw_text)
                if isinstance(data, dict):
                    query_str = data.get("query", query_str)
                    active_session = data.get("session_id", session_id)
            except Exception:
                pass

            response_obj = await asyncio.to_thread(
                execute_guarded_ask, query_str, active_session
            )
            await websocket.send_text(response_obj.model_dump_json())

    except WebSocketDisconnect:
        logger.info(f"WebSocket client disconnected cleanly: session_id={session_id}")
    except Exception as exc:
        logger.error(f"Unexpected WebSocket error: {exc}", exc_info=True)
        try:
            await websocket.send_json({"error": str(exc), "status": "error"})
            await websocket.close()
        except Exception:
            pass


if __name__ == "__main__":
    import uvicorn
    print("Starting Naukri.com Domain Support Agent API Server...")
    uvicorn.run("api.main:app", host="127.0.0.1", port=8000, reload=False)
