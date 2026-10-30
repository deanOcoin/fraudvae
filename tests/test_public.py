"""the public-mode limiter on its own: the bucket's refill arithmetic, its table bound, and
whose forwarded headers are believed"""
from types import SimpleNamespace

from fraudvae.server import public


def _req(peer, **headers):
    return SimpleNamespace(headers={k.lower().replace("_", "-"): v for k, v in headers.items()},
                           client=SimpleNamespace(host=peer) if peer else None)


def test_bucket_refill(monkeypatch):
    now = [1000.0]
    monkeypatch.setattr(public.time, "time", lambda: now[0])
    monkeypatch.setattr(public, "_buckets", {})
    # burst 3 at 60 a minute: three straight away, the fourth waits a second
    assert [public.take("score", "a", 60, 3) for _ in range(3)] == [None, None, None]
    wait = public.take("score", "a", 60, 3)
    assert abs(wait - 1.0) < 1e-9
    now[0] += 0.5
    assert public.take("score", "a", 60, 3) is not None     # half a token is not a token
    now[0] += 0.5
    assert public.take("score", "a", 60, 3) is None
    assert public.take("score", "b", 60, 3) is None          # another address, another bucket
    now[0] += 3600
    assert [public.take("score", "a", 60, 3) for _ in range(3)] == [None, None, None]   # refilled, capped at the burst


def test_bucket_table_is_bounded(monkeypatch):
    now = [1000.0]
    monkeypatch.setattr(public.time, "time", lambda: now[0])
    monkeypatch.setattr(public, "_buckets", {})
    for i in range(50_001):
        public.take("score", f"ip{i}", 60, 3)
    assert 40_000 <= len(public._buckets) <= 40_001             # crossing 50k trims to 40k
    now[0] += 3601
    for i in range(10_001):                                      # crossing it again with every old bucket idle an hour
        public.take("score", f"new{i}", 60, 3)
    assert len(public._buckets) == 10_001 and all(k[1].startswith("new") for k in public._buckets)
    # a flood of fresh addresses within the hour: the least recently seen go, down to 40k
    monkeypatch.setattr(public, "_buckets", {})
    for i in range(50_001):
        now[0] += 0.001
        public.take("score", f"ip{i}", 60, 3)
    assert 40_000 <= len(public._buckets) <= 40_001 and ("score", "ip0") not in public._buckets


def test_client_ip_trusts_only_a_proxy(monkeypatch):
    monkeypatch.setattr(public, "TRUSTED_PROXIES", {"127.0.0.1", "::1", "10.0.0.2"})
    # from loopback (nginx on the box): the proxy's headers, X-Real-IP first, else the last forwarded hop
    assert public.client_ip(_req("127.0.0.1", X_REAL_IP="203.0.113.9", X_FORWARDED_FOR="1.1.1.1, 203.0.113.9")) == "203.0.113.9"
    assert public.client_ip(_req("127.0.0.1", X_FORWARDED_FOR="9.9.9.9, 203.0.113.9")) == "203.0.113.9"
    assert public.client_ip(_req("10.0.0.2", X_REAL_IP=" 203.0.113.9 ")) == "203.0.113.9"
    assert public.client_ip(_req("127.0.0.1")) == "127.0.0.1"
    # from anywhere else the headers are a visitor's claim, and the socket is the visitor
    assert public.client_ip(_req("198.51.100.4", X_REAL_IP="203.0.113.9", X_FORWARDED_FOR="203.0.113.9")) == "198.51.100.4"
    assert public.client_ip(_req(None, X_REAL_IP="203.0.113.9")) == "unknown"
