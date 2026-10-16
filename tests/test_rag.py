"""
Automated Test Suite for Knowledge Base and RAG Pipeline (Day 2).

Validates:
1. Knowledge Base contains at least 12 policy documents.
2. Every policy document follows the required schema (doc_id, title, category, content, tags).
3. Every policy document contains between 2 and 5 sentences (as specified by capstone requirements).
4. Fixed-size chunking (200 chars, 50 overlap) produces valid chunks.
5. Sentence-based chunking properly splits on punctuation boundaries.
6. Chunk records retain all essential metadata.
7. ChromaDB collections can be created and populated.
8. Semantic search accurately retrieves relevant policy documents.
"""

from pathlib import Path
import re
import sys
import unittest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from knowledge_base.policies import load_policies
from rag.chunking import (
    chunk_all_policies,
    chunk_fixed_size_with_overlap,
    chunk_sentence_based,
)


class TestKnowledgeBaseAndRAG(unittest.TestCase):
    """Test suite verifying knowledge base policies, chunking strategies, and RAG retrieval."""

    @classmethod
    def setUpClass(cls):
        """Load the 12 policy documents once."""
        cls.policies = load_policies()

    def test_01_knowledge_base_count(self):
        """Check 1: Knowledge base has at least 12 policy documents."""
        self.assertGreaterEqual(
            len(self.policies),
            12,
            f"Expected at least 12 policy documents, found {len(self.policies)}",
        )

    def test_02_policy_schema(self):
        """Check 2: Every policy document contains all required schema fields."""
        required_keys = {"doc_id", "title", "category", "content", "tags"}
        for doc in self.policies:
            missing = required_keys - set(doc.keys())
            self.assertFalse(
                missing,
                f"Policy {doc.get('doc_id', 'unknown')} missing required keys: {missing}",
            )
            self.assertTrue(doc["content"].strip(), f"Policy {doc['doc_id']} content is empty")
            self.assertIsInstance(doc["tags"], list, f"Policy {doc['doc_id']} tags must be a list")

    def test_03_policy_sentence_length(self):
        """Check 3: Each policy document contains between 2 and 5 sentences."""
        for doc in self.policies:
            # Count sentences by splitting on punctuation
            sentences = [s.strip() for s in re.split(r"(?<=[.?!])\s+", doc["content"].strip()) if s.strip()]
            self.assertGreaterEqual(
                len(sentences),
                2,
                f"Policy {doc['doc_id']} has fewer than 2 sentences: {len(sentences)}",
            )
            self.assertLessEqual(
                len(sentences),
                5,
                f"Policy {doc['doc_id']} has more than 5 sentences: {len(sentences)}",
            )

    def test_04_fixed_chunking(self):
        """Check 4: Fixed-size chunking (200 chars, 50 overlap) produces valid chunks."""
        test_text = "A" * 350
        chunks = chunk_fixed_size_with_overlap(test_text, chunk_size=200, overlap=50)
        self.assertGreater(len(chunks), 1, "Expected multiple chunks for 350-char text")
        for chunk in chunks:
            self.assertLessEqual(len(chunk), 200, "Chunk exceeded maximum chunk_size")

    def test_05_sentence_chunking(self):
        """Check 5: Sentence-based chunking splits on punctuation boundaries."""
        sample = "First policy rule. Second policy rule? Third policy rule!"
        sentences = chunk_sentence_based(sample)
        self.assertEqual(len(sentences), 3, f"Expected 3 sentences, got {len(sentences)}")
        self.assertEqual(sentences[0], "First policy rule.")
        self.assertEqual(sentences[1], "Second policy rule?")
        self.assertEqual(sentences[2], "Third policy rule!")

    def test_06_chunk_metadata_retention(self):
        """Check 6: Chunk records preserve doc_id, title, category, and chunk_index."""
        fixed_chunks = chunk_all_policies(self.policies, strategy="fixed", chunk_size=200, overlap=50)
        sentence_chunks = chunk_all_policies(self.policies, strategy="sentence")

        self.assertGreater(len(fixed_chunks), 0)
        self.assertGreater(len(sentence_chunks), 0)

        for chunk in fixed_chunks + sentence_chunks:
            self.assertIn("chunk_id", chunk)
            self.assertIn("doc_id", chunk)
            self.assertIn("title", chunk)
            self.assertIn("category", chunk)
            self.assertIn("chunk_index", chunk)
            self.assertIn("strategy", chunk)
            self.assertIn("text", chunk)
            self.assertTrue(chunk["text"].strip())

    def test_07_chromadb_collections_exist(self):
        """Check 7: Both ChromaDB collections exist and are populated with 36 chunks each."""
        from rag.indexer import (
            COLLECTION_FIXED,
            COLLECTION_SENTENCE,
            get_chroma_client,
            get_embedding_function,
        )

        client = get_chroma_client()
        ef = get_embedding_function()

        col_fixed = client.get_collection(name=COLLECTION_FIXED, embedding_function=ef)
        col_sentence = client.get_collection(name=COLLECTION_SENTENCE, embedding_function=ef)

        self.assertGreaterEqual(col_fixed.count(), 36, f"Expected at least 36 fixed chunks, found {col_fixed.count()}")
        self.assertGreaterEqual(col_sentence.count(), 36, f"Expected at least 36 sentence chunks, found {col_sentence.count()}")

    def test_08_semantic_retrieval_relevance(self):
        """Check 8: Live semantic search retrieves the correct policy document as top-1."""
        from rag.retriever import query_policies

        # Test 1: Rescheduling interview should retrieve POL-003
        results = query_policies("rescheduling an interview", strategy="sentence", top_k=1)
        self.assertGreater(len(results), 0, "No results returned for interview reschedule query")
        self.assertEqual(
            results[0]["doc_id"],
            "POL-003",
            f"Expected POL-003 for reschedule query, got {results[0]['doc_id']}",
        )
        self.assertGreater(results[0]["similarity"], 0.45, "Similarity score unexpectedly low")

        # Test 2: Cooling-off period should retrieve POL-010
        cooling_results = query_policies("cooling-off period after rejection", strategy="sentence", top_k=1)
        self.assertGreater(len(cooling_results), 0)
        self.assertEqual(cooling_results[0]["doc_id"], "POL-010")


if __name__ == "__main__":
    unittest.main(verbosity=2)
