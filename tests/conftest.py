"""Every test runs against a throwaway data/runs/app-home tree. The env vars have
to be set before fraudvae.paths is imported anywhere, hence the top of conftest."""
import atexit
import os
import shutil
import tempfile
from pathlib import Path

_tmp = Path(tempfile.mkdtemp(prefix="fraudvae-test-"))
atexit.register(shutil.rmtree, _tmp, True)   # half a gigabyte per run otherwise, left in /tmp
os.environ["FRAUDVAE_DATA_DIR"] = str(_tmp / "data")
os.environ["FRAUDVAE_RUNS_DIR"] = str(_tmp / "runs")
os.environ["FRAUDVAE_HOME"] = str(_tmp / "home")
os.environ["FRAUDVAE_RAW_DIR"] = str(_tmp / "data" / "raw")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from fraudvae.data import schema  # noqa: E402
from fraudvae.paths import PROCESSED_DIR, ensure_dirs  # noqa: E402


def synthetic_source(key: str, n: int = 6000, fraud_rate: float = 0.03, seed: int = 0, wide: bool = False,
                     device: bool = False) -> pd.DataFrame:
    """a fake unified-schema source with a planted fraud pattern (big amounts at night, out of
    scale for the customer's own history). v5: 300 customers with real sequences, a merchant
    per row, the velocity + entity-graph blocks computed for real, optionally a small wide
    block (3 raw columns, one of them categorical) and optionally the device block (os,
    browser, screen, session length, with fraud leaning on one os / short sessions)."""
    from fraudvae.data import entities, velocity

    rng = np.random.default_rng(seed)
    df = schema.empty_frame(n)
    y = (rng.random(n) < fraud_rate).astype(np.int8)
    cust = rng.integers(0, 300, n)
    # v5: five of the fifty counterparties are "compromised": a quarter of their traffic is
    # fraud, which only the entity graph / the graph block can know
    cpty = rng.integers(0, 50, n)
    y = (y | ((cpty < 5) & (rng.random(n) < 0.25))).astype(np.int8)
    scale = rng.lognormal(3.5, 0.5, 300)[cust]          # each customer has their own typical amount
    amount = (scale * rng.lognormal(0, 0.4, n)).astype(np.float32)
    amount[y == 1] *= 6
    hour = rng.integers(6, 23, n).astype(np.float32)
    hour[y == 1] = rng.integers(0, 5, (y == 1).sum())
    df["txn_id"] = [f"{key}-{i}" for i in range(n)]
    df["ts"] = np.sort(rng.uniform(1.5e9, 1.5e9 + 90 * 86400, n))
    df["amount"] = amount
    df["hour"] = hour
    df["dow"] = rng.integers(0, 7, n).astype(np.float32)
    df["is_fraud"] = y
    df["channel"] = rng.integers(1, 4, n).astype(np.int8)
    df["txn_type"] = 1
    df["merchant_category"] = rng.integers(1, 8, n).astype(np.int8)
    df["customer"] = cust.astype(np.int64)
    df["counterparty"] = cpty.astype(np.int64)
    if device:
        # device block on ~80% of the rows; fraud is mostly "linux" (code 5) on short sessions
        has = rng.random(n) < 0.8
        os_code = rng.integers(1, 7, n)
        os_code[y == 1] = np.where(rng.random((y == 1).sum()) < 0.7, 5, os_code[y == 1])
        df["device_os"] = np.where(has, os_code, 0).astype(np.int8)
        df["browser"] = np.where(has, rng.integers(1, 9, n), 0).astype(np.int8)
        sess = rng.lognormal(2.0, 0.6, n).astype(np.float32)
        sess[y == 1] *= 0.3
        df["session_length_min"] = np.where(has, sess, np.nan).astype(np.float32)
        df["screen_w"] = np.where(has, rng.choice([1366, 1920, 375, 414], n), np.nan).astype(np.float32)
        df["screen_h"] = np.where(has, rng.choice([768, 1080, 812, 896], n), np.nan).astype(np.float32)
        df["device"] = np.where(has, rng.integers(0, 400, n), -1).astype(np.int64)
    df = velocity.attach(df, "customer", merchant_col="counterparty")
    df = entities.attach(df)
    if wide:
        df["wide__raw_a"] = (amount * 2 + rng.normal(0, 1, n)).astype(np.float32)
        df["wide__raw_b"] = np.where(rng.random(n) < 0.3, np.nan, rng.normal(0, 1, n)).astype(np.float32)
        df["wide__raw_cat"] = rng.integers(1, 5, n).astype(np.float32)
        df.attrs["wide_columns"] = [{"name": "raw_a", "kind": "num"}, {"name": "raw_b", "kind": "num"},
                                    {"name": "raw_cat", "kind": "cat", "values": ["p", "q", "r", "s"]}]
    df = schema.finalize(df, key)
    return schema.assign_time_splits(df)


def synthetic_text_sets(raw_dir: Path, n: int = 600, seed: int = 0):
    """tiny malurl / fakejob files with a planted vocabulary difference, so the text heads
    have something to learn in the smoke pipeline"""
    rng = np.random.default_rng(seed)
    bad_words = ["verify", "login", "secure", "update", "account", "free", "prize", "wire", "urgent", "fee"]
    good_words = ["news", "recipe", "weather", "about", "blog", "team", "docs", "music", "sports", "shop"]
    tlds = ["com", "org", "net", "io", "ru", "cn"]

    def words(k, bad):
        pool = bad_words if bad else good_words
        return [str(rng.choice(pool)) if rng.random() < 0.7 else f"w{rng.integers(0, 500)}" for _ in range(k)]

    d = raw_dir / "malurl"
    d.mkdir(parents=True, exist_ok=True)
    rows = []
    for _ in range(n):
        bad = rng.random() < 0.35
        rows.append({"url": "-".join(words(3, bad)) + "." + str(rng.choice(tlds)) + "/" + "/".join(words(2, bad)),
                     "type": str(rng.choice(["phishing", "malware", "defacement"])) if bad else "benign"})
    pd.DataFrame(rows).to_csv(d / "malicious_phish.csv", index=False)
    d = raw_dir / "fakejob"
    d.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(n):
        bad = rng.random() < 0.15
        rows.append({"job_id": i, "title": " ".join(words(2, bad)), "company_profile": " ".join(words(6, bad)),
                     "description": " ".join(words(25, bad)), "requirements": " ".join(words(8, bad)),
                     "benefits": " ".join(words(4, bad)), "fraudulent": int(bad)})
    pd.DataFrame(rows).to_csv(d / "fake_job_postings.csv", index=False)


def synthetic_mouse_session(rng, speed: float, jitter: float, n_strokes: int = 40) -> pd.DataFrame:
    """balabit-shaped csv rows: strokes of a few points, each user with their own speed / wobble"""
    rows = []
    t, x, y = 0.0, 500.0, 400.0
    for _ in range(n_strokes):
        tx, ty = rng.uniform(0, 1900), rng.uniform(0, 1000)
        k = int(rng.integers(6, 14))
        for i in range(1, k + 1):
            f = i / k
            t += 0.016 * 60.0 / speed
            rows.append({"record timestamp": t, "client timestamp": t, "button": "NoButton", "state": "Move",
                         "x": int(x + (tx - x) * f + rng.normal(0, jitter)), "y": int(y + (ty - y) * f + rng.normal(0, jitter))})
        x, y = tx, ty
        t += 0.05
        rows.append({"record timestamp": t, "client timestamp": t, "button": "Left", "state": "Pressed", "x": int(x), "y": int(y)})
        t += rng.uniform(0.05, 0.15) * (60.0 / speed)
        rows.append({"record timestamp": t, "client timestamp": t, "button": "Left", "state": "Released", "x": int(x), "y": int(y)})
        t += rng.uniform(0.6, 1.2)
    return pd.DataFrame(rows)


def synthetic_balabit(raw_dir: Path, seed: int = 0):
    """4 users with distinct mouse habits, 3 training sessions each, 6 test sessions each of
    which half come from another user (the challenge's intruder labels)"""
    rng = np.random.default_rng(seed)
    root = raw_dir / "balabit"
    users = {"user1": (60.0, 1.0), "user2": (120.0, 4.0), "user3": (40.0, 8.0), "user4": (90.0, 2.0)}
    labels = []
    for u, (speed, jitter) in users.items():
        (root / "training_files" / u).mkdir(parents=True, exist_ok=True)
        (root / "test_files" / u).mkdir(parents=True, exist_ok=True)
        for k in range(3):
            synthetic_mouse_session(rng, speed, jitter).to_csv(root / "training_files" / u / f"session_{k}", index=False)
        others = [v for v in users if v != u]
        for k in range(6):
            illegal = k % 2 == 1
            sp, jt = users[str(rng.choice(others))] if illegal else (speed, jitter)
            name = f"session_{u}_{k}"
            synthetic_mouse_session(rng, sp, jt, n_strokes=25).to_csv(root / "test_files" / u / name, index=False)
            labels.append({"filename": name, "is_illegal": int(illegal)})
    pd.DataFrame(labels).to_csv(root / "public_labels.csv", index=False)


@pytest.fixture(scope="session")
def processed_corpus():
    """two synthetic sources written as processed parquet + minimal stats.json, plus the tiny
    text sets under the raw folder"""
    ensure_dirs()
    import json

    from fraudvae.paths import RAW_DIR

    synthetic_text_sets(RAW_DIR)
    synthetic_balabit(RAW_DIR)

    from fraudvae.data.build import STATS, SAMPLES, write_source, summarize, sample_rows

    stats = {"sources": {}, "schema_version": schema.SCHEMA_VERSION}
    samples = []
    for i, key in enumerate(("fdh", "capone")):
        df = synthetic_source(key, seed=i, wide=(key == "capone"), device=(key == "fdh"))
        out = PROCESSED_DIR / f"{key}.parquet"
        write_source(key, df, out)
        s = summarize(out, key, 0.1)
        s["parquet"] = out.name
        stats["sources"][key] = s
        samples.append(sample_rows(out, 30, 30))
    STATS.write_text(json.dumps(stats))
    pd.concat(samples).to_parquet(SAMPLES, index=False)
    return PROCESSED_DIR


@pytest.fixture(scope="session")
def prepared_set(processed_corpus):
    from fraudvae.data.prepare import prepare

    prepare(name="t", max_rows_per_source=3000)
    return "t"


@pytest.fixture(scope="session")
def trained_run(prepared_set):
    from fraudvae.config import load_config
    from fraudvae.train import train

    cfg = load_config("smoke")
    cfg.train.epochs = 2
    cfg.train.batch_size = 512
    cfg.train.score_samples = 2
    run_dir = train(cfg, prepared=prepared_set, name="test-run")
    return run_dir
