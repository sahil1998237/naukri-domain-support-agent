"""
Structured JSON-Lines Logging Middleware for FastAPI (Task 12).

Guarantees:
1. Every HTTP request/response cycle generates a standardized JSON-Lines log entry.
2. Standardized audit fields: trace_id, timestamp, method, path, status_code, duration_ms, query, pii_masked.
3. Strict PII Protection (Trap 6): Raw user queries are masked before writing to log files
   so that no phone numbers or emails ever leak into storage or log aggregators.
4. Header Propagation: Injects X-Trace-ID and X-Response-Time-MS into HTTP response headers.
"""

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import sys
import time
from typing import Any, Callable, Dict, Optional, Tuple
import uuid

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from agents.guardrails import mask_pii

DEFAULT_LOG_DIR = PROJECT_ROOT / "logs"
DEFAULT_LOG_FILE = DEFAULT_LOG_DIR / "api_access.jsonl"

logger = logging.getLogger("structured_logger")


def append_jsonl_log(record: Dict[str, Any], log_file: Optional[Path | str] = None) -> None:
    """
    Append a structured dictionary as a single-line JSON string into the target JSONL file.
    Creates parent directories if not present.
    """
    target = Path(log_file) if log_file else DEFAULT_LOG_FILE
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


class StructuredLoggingMiddleware(BaseHTTPMiddleware):
    """
    ASGI middleware recording audit-compliant JSON-Lines logs for all API interactions.
    """

    def __init__(self, app: Any, log_file: Optional[Path | str] = None) -> None:
        super().__init__(app)
        self.log_file = Path(log_file) if log_file else DEFAULT_LOG_FILE

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        trace_id = str(uuid.uuid4().hex)
        start_time = time.perf_counter()

        masked_query: Optional[str] = None
        pii_masked: bool = False

        # If incoming request is POST to an inquiry endpoint, safely inspect query for logging
        if request.method == "POST" and any(p in request.url.path for p in ["/ask", "/chat"]):
            try:
                body = await request.body()
                # Reconstruct receive stream so downstream handler can read body untouched
                async def receive():
                    return {"type": "http.request", "body": body}
                request = Request(request.scope, receive=receive)

                if body:
                    data = json.loads(body.decode("utf-8"))
                    raw_q = data.get("query", "")
                    if raw_q:
                        audit = mask_pii(raw_q)
                        masked_query = audit["masked_text"]
                        pii_masked = audit["pii_detected"]
            except Exception:
                # If body is non-JSON or malformed, continue without crashing middleware
                masked_query = None
                pii_masked = False

        # Execute endpoint handler
        response = await call_next(request)

        duration_ms = round((time.perf_counter() - start_time) * 1000, 2)

        # Inject correlation trace headers
        response.headers["X-Trace-ID"] = trace_id
        response.headers["X-Response-Time-MS"] = str(duration_ms)

        # Build structured JSON-Lines record
        log_entry = {
            "trace_id": trace_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "duration_ms": duration_ms,
            "query": masked_query,
            "pii_masked": pii_masked,
            "client_ip": request.client.host if request.client else "127.0.0.1",
        }

        try:
            append_jsonl_log(log_entry, self.log_file)
        except Exception as exc:
            logger.error(f"Failed to write structured log: {exc}")

        return response
