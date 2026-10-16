"""
Automated Test Suite for Day 13 / Task 16: Response Caching.

Validates:
1. Query normalization rules (casing, punctuation stripping, whitespace standardization).
2. ResponseCache basic insertion and retrieval (hits, misses).
3. Cache miss behavior for non-existent queries.
4. Least-Recently-Used (LRU) capacity eviction mechanics.
5. Time-to-Live (TTL) expiration and lazy eviction.
6. Cumulative cache performance metrics and token savings accounting.
7. FastAPI endpoint integration (POST /ask, GET /cache/metrics, POST /cache/clear).
8. Formal transcript verification for task16_cache_hit.txt.
"""

from pathlib import Path
import sys
import time
import unittest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient

from api.cache import (
    ResponseCache,
    compute_query_hash,
    get_response_cache,
    normalize_query,
)
from api.main import app


class TestDay13ResponseCaching(unittest.TestCase):
    """Test suite for Task 16: Response Caching with Normalized Query Keys."""

    def setUp(self):
        # Fresh local cache instance for isolated unit tests
        self.cache = ResponseCache(max_size=5, default_ttl_seconds=3600)
        # Clear global singleton cache between tests
        get_response_cache().clear()
        self.client = TestClient(app)

    def test_01_query_normalization_rules(self):
        """Check 1: Canonical normalization standardizes casing, punctuation, and whitespace."""
        q1 = "What is the notice period policy???"
        q2 = "  what IS the notice period policy!  "
        q3 = "What is the notice period policy"

        norm1 = normalize_query(q1)
        norm2 = normalize_query(q2)
        norm3 = normalize_query(q3)

        self.assertEqual(norm1, "what is the notice period policy")
        self.assertEqual(norm1, norm2)
        self.assertEqual(norm2, norm3)

        # Retain application ID hyphens
        app_q = "Status of APP-003, please?"
        self.assertEqual(normalize_query(app_q), "status of app-003 please")

    def test_02_cache_set_and_get(self):
        """Check 2: Basic insert and retrieval returns cached payload with metadata."""
        query = "What is the policy for cooling-off period?"
        payload = {"answer": "A candidate must wait 6 months before reapplying for the same profile.", "confidence": 1.0}

        self.cache.set(query, payload)
        self.assertEqual(self.cache.size(), 1)

        cached = self.cache.get("  what is the policy for COOLING-OFF period???  ")
        self.assertIsNotNone(cached)
        self.assertTrue(cached["is_cached"])
        self.assertEqual(cached["answer"], payload["answer"])
        self.assertEqual(self.cache.hits, 1)
        self.assertEqual(self.cache.misses, 0)

    def test_03_cache_miss(self):
        """Check 3: Non-existent query triggers cache miss and updates counters."""
        res = self.cache.get("Unknown non-cached inquiry")
        self.assertIsNone(res)
        self.assertEqual(self.cache.hits, 0)
        self.assertEqual(self.cache.misses, 1)

    def test_04_lru_capacity_eviction(self):
        """Check 4: Cache enforces max_size and evicts the least recently accessed item."""
        lru_cache = ResponseCache(max_size=3)

        lru_cache.set("query 1", {"ans": 1})
        lru_cache.set("query 2", {"ans": 2})
        lru_cache.set("query 3", {"ans": 3})
        self.assertEqual(lru_cache.size(), 3)

        # Access query 1 (moves it to MRU position; query 2 becomes oldest)
        _ = lru_cache.get("query 1")

        # Insert 4th item (must evict query 2)
        lru_cache.set("query 4", {"ans": 4})
        self.assertEqual(lru_cache.size(), 3)
        self.assertEqual(lru_cache.evictions, 1)

        # Query 2 should be gone
        self.assertIsNone(lru_cache.get("query 2"))
        # Query 1 and 4 should be present
        self.assertIsNotNone(lru_cache.get("query 1"))
        self.assertIsNotNone(lru_cache.get("query 4"))

    def test_05_ttl_expiration(self):
        """Check 5: Expired entries are discarded on lookup."""
        ttl_cache = ResponseCache(max_size=10, default_ttl_seconds=1)
        ttl_cache.set("quick_expire", {"msg": "hello"}, ttl_seconds=0.05)

        # Immediate lookup succeeds
        self.assertIsNotNone(ttl_cache.get("quick_expire"))

        # Wait for expiration
        time.sleep(0.08)

        # Subsequent lookup returns None
        expired_res = ttl_cache.get("quick_expire")
        self.assertIsNone(expired_res)
        self.assertGreaterEqual(ttl_cache.evictions, 1)

    def test_06_cache_metrics_and_token_savings(self):
        """Check 6: get_metrics() computes hit rate, capacity, and accumulated token savings."""
        self.cache.set("notice period", {"answer": "Notice period buyout is permitted subject to manager approval."})
        _ = self.cache.get("notice period")
        _ = self.cache.get("non-existent query")

        metrics = self.cache.get_metrics()
        self.assertEqual(metrics["capacity"], 5)
        self.assertEqual(metrics["current_size"], 1)
        self.assertEqual(metrics["hits"], 1)
        self.assertEqual(metrics["misses"], 1)
        self.assertEqual(metrics["total_lookups"], 2)
        self.assertEqual(metrics["hit_rate"], 0.5)
        self.assertEqual(metrics["hit_percentage"], "50.0%")
        self.assertGreater(metrics["estimated_tokens_saved"], 0)

    def test_07_fastapi_cache_integration(self):
        """Check 7: POST /ask populates cache on initial miss and serves from cache on repeat."""
        query = "What is the policy for executive salary bands?"
        payload = {"query": query}

        # 1. First invocation: Cache Miss
        resp1 = self.client.post("/ask", json=payload)
        self.assertEqual(resp1.status_code, 200)
        data1 = resp1.json()
        self.assertFalse(data1.get("is_cached", False))

        # 2. Second invocation with cosmetic spacing: Cache Hit
        payload_repeat = {"query": "  WHAT is the policy for executive salary bands???  "}
        resp2 = self.client.post("/ask", json=payload_repeat)
        self.assertEqual(resp2.status_code, 200)
        data2 = resp2.json()
        self.assertTrue(data2.get("is_cached", False))
        self.assertEqual(data2["answer"], data1["answer"])

        # 3. Invocation with bypass_cache=True forces fresh run
        payload_bypass = {"query": query, "bypass_cache": True}
        resp3 = self.client.post("/ask", json=payload_bypass)
        self.assertEqual(resp3.status_code, 200)
        data3 = resp3.json()
        self.assertFalse(data3.get("is_cached", False))

        # 4. Check cache metrics endpoint
        metrics_resp = self.client.get("/cache/metrics")
        self.assertEqual(metrics_resp.status_code, 200)
        metrics = metrics_resp.json()
        self.assertGreaterEqual(metrics["hits"], 1)

        # 5. Clear cache endpoint
        clear_resp = self.client.post("/cache/clear")
        self.assertEqual(clear_resp.status_code, 200)
        self.assertEqual(clear_resp.json()["status"], "cleared")

    def test_08_transcript_task16_cache_hit_verified(self):
        """Check 8: Official Task 16 transcript exists and contains required empirical before/after evidence."""
        transcript_path = PROJECT_ROOT / "transcripts" / "task16_cache_hit.txt"
        self.assertTrue(transcript_path.exists(), "task16_cache_hit.txt must exist")

        text = transcript_path.read_text(encoding="utf-8")
        self.assertIn("TASK 16 RESPONSE CACHING TRANSCRIPT", text)
        self.assertIn("QUERY NORMALIZATION VERIFICATION", text)
        self.assertIn("PHASE 1: INVOCATION 1 - INITIAL QUERY (CACHE MISS)", text)
        self.assertIn("PHASE 2: INVOCATION 2 - REPEATED QUERY (CACHE HIT)", text)
        self.assertIn("PHASE 3: BEFORE vs. AFTER EMPIRICAL EVIDENCE & PERFORMANCE GAINS", text)
        self.assertIn("VERDICT: Task 16 Response Caching Operational", text)


if __name__ == "__main__":
    unittest.main()
