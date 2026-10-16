"""
Chunking strategies for Naukri.com Knowledge Base policies.

Implements the two required chunking strategies for Task 3:
1. Fixed-size chunking with sliding-window character overlap (e.g., 200 chars, 50 overlap).
2. Sentence-based chunking splitting on linguistic sentence boundaries (. ? !).
"""

from pathlib import Path
import re
from typing import Any, Dict, List, Optional


def chunk_fixed_size_with_overlap(
    text: str,
    chunk_size: int = 200,
    overlap: int = 50,
) -> List[str]:
    """
    Split text into fixed-size character chunks with overlapping boundaries.

    Args:
        text: Input document text.
        chunk_size: Maximum character length per chunk (default 200).
        overlap: Character overlap between consecutive chunks (default 50).

    Returns:
        List of chunk strings.
    """
    cleaned_text = text.strip()
    if not cleaned_text:
        return []

    if len(cleaned_text) <= chunk_size:
        return [cleaned_text]

    step = max(1, chunk_size - overlap)
    chunks: List[str] = []
    start = 0

    while start < len(cleaned_text):
        end = min(start + chunk_size, len(cleaned_text))
        chunk = cleaned_text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(cleaned_text):
            break
        start += step

    return chunks


def chunk_sentence_based(text: str) -> List[str]:
    """
    Split text into discrete sentences based on punctuation boundaries (. ? !).

    Args:
        text: Input document text.

    Returns:
        List of individual sentence strings.
    """
    cleaned_text = text.strip()
    if not cleaned_text:
        return []

    # Split using lookbehind for terminal punctuation followed by whitespace
    sentences = re.split(r"(?<=[.?!])\s+", cleaned_text)
    return [s.strip() for s in sentences if s.strip()]


def chunk_policy_document(
    doc: Dict[str, Any],
    strategy: str = "fixed",
    chunk_size: int = 200,
    overlap: int = 50,
) -> List[Dict[str, Any]]:
    """
    Transform a single policy document into a list of structured chunk records.

    Args:
        doc: Policy dictionary containing doc_id, title, category, and content.
        strategy: Either 'fixed' or 'sentence'.
        chunk_size: For fixed strategy, character chunk length.
        overlap: For fixed strategy, character overlap length.

    Returns:
        List of chunk dictionaries with metadata and unique chunk IDs.
    """
    content = doc.get("content", "")
    if strategy == "fixed":
        raw_chunks = chunk_fixed_size_with_overlap(content, chunk_size=chunk_size, overlap=overlap)
    elif strategy == "sentence":
        raw_chunks = chunk_sentence_based(content)
    else:
        raise ValueError(f"Unknown chunking strategy: '{strategy}'. Choose 'fixed' or 'sentence'.")

    chunk_records: List[Dict[str, Any]] = []
    for idx, text in enumerate(raw_chunks):
        chunk_records.append(
            {
                "chunk_id": f"{doc['doc_id']}_{strategy}_{idx:02d}",
                "doc_id": doc["doc_id"],
                "title": doc.get("title", ""),
                "category": doc.get("category", ""),
                "chunk_index": idx,
                "strategy": strategy,
                "text": text,
            }
        )

    return chunk_records


def chunk_all_policies(
    policies: List[Dict[str, Any]],
    strategy: str = "fixed",
    chunk_size: int = 200,
    overlap: int = 50,
) -> List[Dict[str, Any]]:
    """
    Chunk an entire list of policy documents using the specified strategy.

    Args:
        policies: List of policy document dictionaries.
        strategy: 'fixed' or 'sentence'.

    Returns:
        Combined list of chunk records across all documents.
    """
    all_chunks: List[Dict[str, Any]] = []
    for doc in policies:
        chunks = chunk_policy_document(
            doc,
            strategy=strategy,
            chunk_size=chunk_size,
            overlap=overlap,
        )
        all_chunks.extend(chunks)
    return all_chunks


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from knowledge_base.policies import load_policies

    docs = load_policies()
    fixed_chunks = chunk_all_policies(docs, strategy="fixed", chunk_size=200, overlap=50)
    sentence_chunks = chunk_all_policies(docs, strategy="sentence")

    print(f"Loaded {len(docs)} documents.")
    print(f"Strategy 1 (Fixed 200/50): generated {len(fixed_chunks)} chunks.")
    print(f"Strategy 2 (Sentence-based): generated {len(sentence_chunks)} chunks.")
    print("\nSample Fixed Chunk:")
    print(fixed_chunks[0])
    print("\nSample Sentence Chunk:")
    print(sentence_chunks[0])
