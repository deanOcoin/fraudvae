"""Run every adapter whose raw data is present and write the unified parquet
files + a stats.json the UI reads.

    python -m fraudvae.data.build                 # all available sources
    python -m fraudvae.data.build --source fdh    # one
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ..paths import PROCESSED_DIR, RAW_DIR, ensure_dirs
from . import schema
from .adapters import get_adapter
from .adapters.base import MissingData, available
from .registry import ORDER, SOURCES

STATS = PROCESSED_DIR / "stats.json"
VELOCITY_KEY = "card_key"
WINDOWS = {"1h": 3_600, "24h": 86_400, "7d": 604_800}


# ---- causal velocity features --------------------------------------------

def _window(ts: np.ndarray, amt: np.ndarray, w: float):
    """count and amount sum over [ts - w, ts), the current row excluded"""
    lo = np.searchsorted(ts, ts - w, side="left")
    idx = np.arange(len(ts))
    csum = np.concatenate([[0.0], np.cumsum(amt)])
    return idx - lo, csum[idx] - csum[lo]


def add_velocity(df: pd.DataFrame) -> pd.DataFrame:
    """per card counts / sums over trailing windows. strictly causal: a row only sees
    earlier rows of the same card, so nothing leaks from val/test into train"""
    if VELOCITY_KEY not in df or df["ts"].isna().all():
        return df
    df = df.reset_index(drop=True).sort_values([VELOCITY_KEY, "ts"], kind="stable")
    keys = df[VELOCITY_KEY].to_numpy()
    ts = df["ts"].to_numpy(dtype=np.float64)
    amt = np.nan_to_num(df["amount"].to_numpy(dtype=np.float64))
    starts = np.flatnonzero(np.r_[True, keys[1:] != keys[:-1]])
    ends = np.r_[starts[1:], len(keys)]
    cols = {f"vel_{k}_{n}": np.full(len(df), np.nan, np.float32) for n in WINDOWS for k in ("count", "amount")}
    gap = np.full(len(df), np.nan, np.float32)
    for a, b in zip(starts, ends):
        t, m = ts[a:b], amt[a:b]
        for name, w in WINDOWS.items():
            c, s = _window(t, m, w)
            cols[f"vel_count_{name}"][a:b] = c
            cols[f"vel_amount_{name}"][a:b] = s
        gap[a + 1:b] = np.diff(t)
    for c, v in cols.items():
        df[c] = v
    df["secs_since_last"] = gap
    return df.sort_index()


# ---- stats ---------------------------------------------------------------

def _rate_by(df: pd.DataFrame, col: str, labels=None, max_levels=30) -> dict:
    g = df.groupby(col, observed=True)["is_fraud"].agg(["mean", "size"])
    g = g.sort_index().head(max_levels)
    out = {}
    for k, row in g.iterrows():
        name = labels[int(k)] if labels is not None else (int(k) if isinstance(k, (int, np.integer, float)) else str(k))
        out[str(name)] = {"fraud_rate": float(row["mean"]), "count": int(row["size"])}
    return out


def summarize(df: pd.DataFrame, seconds: float) -> dict:
    y = df["is_fraud"].to_numpy()
    legit = df[y == 0]
    fraud = df[y == 1]
    out = {
        "rows": int(len(df)),
        "fraud": int(y.sum()),
        "fraud_rate": float(y.mean()),
        "build_seconds": round(seconds, 1),
        "splits": {s: {"rows": int((df["split"] == s).sum()), "fraud": int(df.loc[df["split"] == s, "is_fraud"].sum())}
                   for s in ("train", "val", "test")},
        "ts_min": float(np.nanmin(df["ts"])) if df["ts"].notna().any() else None,
        "ts_max": float(np.nanmax(df["ts"])) if df["ts"].notna().any() else None,
        "coverage": {f.name: float(df[f.name].notna().mean()) for f in schema.NUMERIC},
        "amount": {
            "legit_quantiles": {q: float(legit["amount"].quantile(q)) for q in (0.01, 0.1, 0.5, 0.9, 0.99)},
            "fraud_quantiles": {q: float(fraud["amount"].quantile(q)) for q in (0.01, 0.1, 0.5, 0.9, 0.99)} if len(fraud) else {},
        },
    }
    if df["hour"].notna().any():
        out["by_hour"] = _rate_by(df.dropna(subset=["hour"]).assign(hour=lambda d: d["hour"].astype(int)), "hour")
    if (df["merchant_category"] != 0).any():
        out["by_category"] = _rate_by(df, "merchant_category", schema.CAT_VOCAB["merchant_category"])
    return out


def build_source(key: str, raw_dir: Path) -> dict:
    t0 = time.time()
    adapter = get_adapter(key)
    print(f"[{key}] loading...", flush=True)
    df = adapter.load(raw_dir)
    if "split" not in df or df["split"].isna().all():
        df = schema.assign_time_splits(df)
    print(f"[{key}] velocity features...", flush=True)
    df = add_velocity(df)
    df["source"] = pd.Categorical(df["source"], categories=schema.SOURCES)
    df["split"] = pd.Categorical(df["split"], categories=["train", "val", "test"])
    out = PROCESSED_DIR / f"{key}.parquet"
    df[schema.ALL_COLUMNS].to_parquet(out, index=False, compression="zstd")
    stats = summarize(df, time.time() - t0)
    stats["parquet"] = out.name
    print(f"[{key}] {stats['rows']:,} rows, {stats['fraud']:,} fraud ({stats['fraud_rate']:.3%}) in {stats['build_seconds']}s")
    return stats


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", action="append")
    ap.add_argument("--raw-dir", default=str(RAW_DIR))
    args = ap.parse_args(argv)
    ensure_dirs()
    raw_dir = Path(args.raw_dir)
    keys = args.source or ORDER

    stats = json.loads(STATS.read_text()) if STATS.exists() else {"sources": {}}
    built = []
    for key in keys:
        if not available(raw_dir, key):
            print(f"[{key}] raw data not found, skipping ({SOURCES[key].url})")
            continue
        try:
            stats["sources"][key] = build_source(key, raw_dir)
        except MissingData as e:
            print(f"[{key}] {e}")
            continue
        built.append(key)
    stats["generated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    STATS.write_text(json.dumps(stats, indent=1))

    total = sum(s["rows"] for s in stats["sources"].values())
    fraud = sum(s["fraud"] for s in stats["sources"].values())
    print(f"\nbuilt {built}; corpus now {total:,} rows / {fraud:,} fraud across {len(stats['sources'])} sources")
    return 0


if __name__ == "__main__":
    sys.exit(main())
