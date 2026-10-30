"""
ChromaDB Indexer for Naukri.com Policy Knowledge Base.

Implements Task 3 requirements:
- Embeds policy chunks using SentenceTransformers (all-MiniLM-L6-v2)
- Creates two distinct persistent ChromaDB collections:
    1. 'naukri_policies_fixed' (Strategy 1: Fixed-size 200 chars with 50 overlap)
    2. 'naukri_policies_sentence' (Strategy 2: Sentence-based chunks)
- Configures cosine distance metric ('hnsw:space': 'cosine') for groundedness evaluation
"""

from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

# Add project root to path for direct script execution
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import chromadb
from chromadb.utils import embedding_functions

from knowledge_base.policies import load_policies
from rag.chunking import chunk_all_policies

# Default configuration
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "chroma_db"
DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"

COLLECTION_FIXED = "naukri_policies_fixed"
COLLECTION_SENTENCE = "naukri_policies_sentence"


def get_chroma_client(db_path: Optional[Path | str] = None) -> chromadb.PersistentClient:
    """Initialize a persistent ChromaDB client pointing to the target storage path."""
    target_path = Path(db_path) if db_path else DEFAULT_DB_PATH
    target_path.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(target_path))


def get_embedding_function(model_name: str = DEFAULT_MODEL_NAME) -> Any:
    """Create a SentenceTransformer embedding function for ChromaDB."""
    return embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model_name)


def index_collection(
    client: chromadb.PersistentClient,
    collection_name: str,
    chunks: List[Dict[str, Any]],
    embedding_function: Any,
    force_reindex: bool = False,
) -> chromadb.Collection:
    """
    Populate or update a ChromaDB collection with structured policy chunks.

    Args:
        client: Persistent ChromaDB client.
        collection_name: Unique collection identifier.
        chunks: List of chunk dictionaries from rag.chunking.
        embedding_function: SentenceTransformers embedding callable.
        force_reindex: If True, delete existing collection before indexing.

    Returns:
        The populated ChromaDB Collection object.
    """
    if force_reindex:
        try:
            client.delete_collection(name=collection_name)
        except Exception:
            pass

    # Note: 'hnsw:space': 'cosine' ensures distances reflect cosine distance (0.0 to 2.0),
    # where cosine_similarity = 1.0 - distance.
    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_function,
        metadata={"hnsw:space": "cosine"},
    )

    existing_count = collection.count()
    if existing_count > 0 and not force_reindex:
        return collection

    ids = [c["chunk_id"] for c in chunks]
    documents = [c["text"] for c in chunks]
    metadatas = [
        {
            "doc_id": c["doc_id"],
            "title": c["title"],
            "category": c["category"],
            "chunk_index": c["chunk_index"],
            "strategy": c["strategy"],
        }
        for c in chunks
    ]

    # collection.upsert() satisfies Task 3 specification
    collection.upsert(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
    )

    return collection


def build_indexes(
    db_path: Optional[Path | str] = None,
    model_name: str = DEFAULT_MODEL_NAME,
    force_reindex: bool = False,
) -> Dict[str, Any]:
    """
    Execute end-to-end indexing pipeline: load policies, chunk via both strategies,
    and persist into two ChromaDB collections.

    Returns:
        Summary dict containing index metrics and status.
    """
    # 1. Load 12 policy documents
    policies = load_policies()

    # 2. Chunk documents using both strategies
    fixed_chunks = chunk_all_policies(policies, strategy="fixed", chunk_size=200, overlap=50)
    sentence_chunks = chunk_all_policies(policies, strategy="sentence")

    # 3. Setup ChromaDB client and embedding model
    client = get_chroma_client(db_path)
    ef = get_embedding_function(model_name)

    # 4. Index Fixed-size collection
    col_fixed = index_collection(
        client,
        COLLECTION_FIXED,
        fixed_chunks,
        ef,
        force_reindex=force_reindex,
    )

    # 5. Index Sentence-based collection
    col_sentence = index_collection(
        client,
        COLLECTION_SENTENCE,
        sentence_chunks,
        ef,
        force_reindex=force_reindex,
    )

    summary = {
        "policies_loaded": len(policies),
        "collection_fixed": {
            "name": COLLECTION_FIXED,
            "count": col_fixed.count(),
            "strategy": "fixed_size (200 chars, 50 overlap)",
        },
        "collection_sentence": {
            "name": COLLECTION_SENTENCE,
            "count": col_sentence.count(),
            "strategy": "sentence_based",
        },
        "db_path": str(Path(db_path) if db_path else DEFAULT_DB_PATH),
        "model": model_name,
    }

    return summary


if __name__ == "__main__":
    print("Building ChromaDB policy indexes for both chunking strategies...")
    result = build_indexes(force_reindex=True)
    print("\n--- Indexing Completed Successfully ---")
    print(f"Policies Loaded: {result['policies_loaded']}")
    print(f"Fixed Collection: {result['collection_fixed']['count']} chunks ({result['collection_fixed']['name']})")
    print(f"Sentence Collection: {result['collection_sentence']['count']} chunks ({result['collection_sentence']['name']})")
    print(f"Storage Path: {result['db_path']}")
    print(f"Embedding Model: {result['model']}")
