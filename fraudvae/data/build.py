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


def summarize(df: pd.DataFrame, seconds: float) -> dict:
    y = df["is_fraud"].to_numpy()
    legit = df.loc[y == 0, "amount"]
    fraud = df.loc[y == 1, "amount"]
    return {
        "rows": int(len(df)),
        "fraud": int(y.sum()),
        "fraud_rate": float(y.mean()),
        "build_seconds": round(seconds, 1),
        "splits": {s: {"rows": int((df["split"] == s).sum()), "fraud": int(df.loc[df["split"] == s, "is_fraud"].sum())}
                   for s in ("train", "val", "test")},
        "ts_min": float(np.nanmin(df["ts"])) if df["ts"].notna().any() else None,
        "ts_max": float(np.nanmax(df["ts"])) if df["ts"].notna().any() else None,
        "coverage": {f.name: float(df[f.name].notna().mean()) for f in schema.NUMERIC},
        "amount_median": {"legit": float(legit.median()), "fraud": float(fraud.median()) if len(fraud) else None},
    }


def build_source(key: str, raw_dir: Path) -> dict:
    t0 = time.time()
    adapter = get_adapter(key)
    print(f"[{key}] loading...", flush=True)
    df = adapter.load(raw_dir)
    if "split" not in df or df["split"].isna().all():
        df = schema.assign_time_splits(df)
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
