"""Public mode: the app as a demonstration site anyone can reach, rather than the workbench
on someone's laptop. Set FRAUDVAE_PUBLIC=1 and the server

  * never mounts the remote-training API or shows the Compute tab (it would be a remote
    shell with the app's own key), and refuses every endpoint that trains, builds data or
    changes which model is the default;
  * keeps no visitor state: the customer store is not written, histories must come with the
    request, scheduled nothing;
  * hides the host name and file paths from /api/meta and the box details from run metadata;
  * rate-limits scoring per client address (a token bucket in memory, behind nginx's own
    limits) and caps a batch upload at FRAUDVAE_BATCH_ROWS rows with a small global
    concurrency, so one visitor cannot keep the CPU for everyone.

Everything else - the examples, the receipts, the model timeline, the batch upload - works
as on a laptop. A deployment also puts nginx in front (deploy/), which does TLS, the
first layer of rate limiting and the request size cap."""
from __future__ import annotations

import os
import threading
import time

from fastapi import HTTPException, Request

PUBLIC = os.environ.get("FRAUDVAE_PUBLIC", "").lower() in ("1", "true", "yes")
SCORE_PER_MINUTE = int(os.environ.get("FRAUDVAE_SCORE_PER_MINUTE", "60"))
SCORE_BURST = int(os.environ.get("FRAUDVAE_SCORE_BURST", "20"))
BATCH_PER_MINUTE = int(os.environ.get("FRAUDVAE_BATCH_PER_MINUTE", "4"))
BATCH_ROWS = int(os.environ.get("FRAUDVAE_BATCH_ROWS", "5000"))
BATCH_CONCURRENCY = int(os.environ.get("FRAUDVAE_BATCH_CONCURRENCY", "2"))
COMPARE_MAX = int(os.environ.get("FRAUDVAE_COMPARE_MAX", "6"))

_buckets: dict[tuple[str, str], list[float]] = {}
_lock = threading.Lock()
batch_slots = threading.BoundedSemaphore(BATCH_CONCURRENCY)


def client_ip(request: Request) -> str:
    """the visitor's address: nginx passes it in X-Forwarded-For, otherwise the socket's"""
    fwd = request.headers.get("x-forwarded-for")
    if fwd:
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def take(kind: str, ip: str, per_minute: int, burst: int) -> float | None:
    """token bucket per (kind, ip): None when a token was taken, otherwise seconds to wait"""
    now = time.time()
    rate = per_minute / 60.0
    with _lock:
        tokens, last = _buckets.get((kind, ip), (float(burst), now))
        tokens = min(float(burst), tokens + (now - last) * rate)
        if tokens >= 1.0:
            _buckets[(kind, ip)] = [tokens - 1.0, now]
            # keep the table from growing with every address that ever visited
            if len(_buckets) > 50_000:
                stale = [k for k, (t, l) in _buckets.items() if now - l > 3600]
                for k in stale:
                    _buckets.pop(k, None)
            return None
        _buckets[(kind, ip)] = [tokens, now]
        return (1.0 - tokens) / rate


def limit(kind: str, request: Request):
    """raise 429 for a visitor past their budget (public mode only)"""
    if not PUBLIC:
        return
    per_minute, burst = (SCORE_PER_MINUTE, SCORE_BURST) if kind == "score" else (BATCH_PER_MINUTE, max(1, BATCH_PER_MINUTE))
    wait = take(kind, client_ip(request), per_minute, burst)
    if wait is not None:
        raise HTTPException(429, f"rate limit: try again in {wait:.0f}s", headers={"Retry-After": str(int(wait) + 1)})


def forbid(what: str):
    """the endpoints a public site does not offer"""
    if PUBLIC:
        raise HTTPException(403, f"{what} is not available on the public site; run the app on your own machine for that "
                                 f"(see the repository's README)")


def limits() -> dict:
    return {"public": PUBLIC, "score_per_minute": SCORE_PER_MINUTE, "batch_rows": BATCH_ROWS,
            "batch_per_minute": BATCH_PER_MINUTE, "compare_max": COMPARE_MAX}
