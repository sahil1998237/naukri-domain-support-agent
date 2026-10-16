"""
Retriever module for Naukri.com Policy Knowledge Base.

Provides high-level retrieval functions over both ChromaDB collections:
- Supports strategy selection ('fixed' or 'sentence')
- Computes cosine similarity (1.0 - cosine_distance)
- Returns structured hit dictionaries with document text and metadata
"""

from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag.indexer import (
    COLLECTION_FIXED,
    COLLECTION_SENTENCE,
    DEFAULT_DB_PATH,
    DEFAULT_MODEL_NAME,
    get_chroma_client,
    get_embedding_function,
)


def query_policies(
    query: str,
    strategy: str = "sentence",
    top_k: int = 3,
    db_path: Optional[Path | str] = None,
    model_name: str = DEFAULT_MODEL_NAME,
) -> List[Dict[str, Any]]:
    """
    Search the indexed policy knowledge base for relevant policy chunks.

    Args:
        query: Candidate or recruiter natural language query.
        strategy: 'sentence' (default) or 'fixed'.
        top_k: Number of most similar chunks to return.
        db_path: Optional custom path to persistent ChromaDB storage.
        model_name: SentenceTransformers model identifier.

    Returns:
        List of result dictionaries sorted by similarity descending.
    """
    collection_name = COLLECTION_SENTENCE if strategy == "sentence" else COLLECTION_FIXED
    client = get_chroma_client(db_path)
    ef = get_embedding_function(model_name)

    collection = client.get_collection(
        name=collection_name,
        embedding_function=ef,
    )

    results = collection.query(
        query_texts=[query],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    hits: List[Dict[str, Any]] = []
    if not results or not results["ids"] or not results["ids"][0]:
        return hits

    ids = results["ids"][0]
    docs = results["documents"][0] if results.get("documents") else []
    metas = results["metadatas"][0] if results.get("metadatas") else []
    distances = results["distances"][0] if results.get("distances") else []

    for i in range(len(ids)):
        dist = distances[i] if i < len(distances) else 1.0
        # For cosine space in ChromaDB, cosine_similarity = 1.0 - cosine_distance
        similarity = round(max(0.0, 1.0 - dist), 4)

        hits.append(
            {
                "chunk_id": ids[i],
                "doc_id": metas[i].get("doc_id", "") if i < len(metas) else "",
                "title": metas[i].get("title", "") if i < len(metas) else "",
                "category": metas[i].get("category", "") if i < len(metas) else "",
                "text": docs[i] if i < len(docs) else "",
                "strategy": strategy,
                "distance": round(dist, 4),
                "similarity": similarity,
            }
        )

    return hits


# Empirically calibrated on Day 3: in-scope min >= 0.58, out-of-scope max <= 0.20
CALIBRATED_SIMILARITY_THRESHOLD: float = 0.40


def generate_grounded_response(
    query: str,
    strategy: str = "sentence",
    threshold: float = CALIBRATED_SIMILARITY_THRESHOLD,
    db_path: Optional[Path | str] = None,
    model_name: str = DEFAULT_MODEL_NAME,
) -> Dict[str, Any]:
    """
    Generate a grounded response citing retrieved policy context if similarity >= threshold,
    or gracefully reject the query as out-of-scope if similarity < threshold.

    Args:
        query: User question.
        strategy: 'sentence' or 'fixed'.
        threshold: Minimum cosine similarity score required for grounding (default 0.40).
        db_path: Path to ChromaDB storage.
        model_name: SentenceTransformers model identifier.

    Returns:
        Structured response dictionary with grounding verdict, answer, citations, and scores.
    """
    hits = query_policies(
        query=query,
        strategy=strategy,
        top_k=1,
        db_path=db_path,
        model_name=model_name,
    )

    if not hits or hits[0]["similarity"] < threshold:
        top_similarity = hits[0]["similarity"] if hits else 0.0
        return {
            "query": query,
            "is_grounded": False,
            "answer": (
                "I do not have sufficient policy information in the knowledge base to answer this question. "
                "Please contact the Naukri Help Desk or an HR representative."
            ),
            "doc_id": None,
            "doc_title": None,
            "similarity": top_similarity,
            "threshold": threshold,
            "context_used": None,
            "strategy": strategy,
        }

    top_hit = hits[0]
    grounded_answer = (
        f"According to Naukri.com Policy [{top_hit['doc_id']}: {top_hit['title']}]: "
        f"{top_hit['text']}"
    )

    return {
        "query": query,
        "is_grounded": True,
        "answer": grounded_answer,
        "doc_id": top_hit["doc_id"],
        "doc_title": top_hit["title"],
        "similarity": top_hit["similarity"],
        "threshold": threshold,
        "context_used": top_hit["text"],
        "strategy": strategy,
    }


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    test_queries = [
        "What is the policy for rescheduling an interview?",
        "How long is the cooling-off period after a rejection?",
        "What is the weather forecast for Bangalore today?",
    ]

    print("=== Testing Grounded Response Generation (Task 4) ===")
    for q in test_queries:
        res = generate_grounded_response(q, strategy="sentence")
        print(f"\nQuery: '{q}'")
        print(f"Grounded: {res['is_grounded']} (Sim: {res['similarity']} vs Threshold: {res['threshold']})")
        print(f"Answer: {res['answer']}")
