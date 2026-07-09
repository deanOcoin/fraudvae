"""Run every adapter whose raw data is present and write the unified parquet
files + a stats.json the UI reads.

    python -m fraudvae.data.build                 # all available sources
    python -m fraudvae.data.build --source fdh    # one

Adapters return either a DataFrame or a generator of DataFrames (tabformer);
both end up as one zstd parquet per source. Stats are then computed from the
parquet with a light column subset, so even the 24M row file is fine on a
laptop.
"""
from __future__ import annotations

import argparse
import gc
import inspect
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ..paths import PROCESSED_DIR, RAW_DIR, ensure_dirs
from . import schema
from .adapters import get_adapter
from .adapters.base import MissingData, available
from .registry import ORDER, SOURCES

STATS = PROCESSED_DIR / "stats.json"
LIGHT_COLS = ["ts", "is_fraud", "split", "amount", "hour", "merchant_category"]


def _arrow_schema():
    fields = []
    for c, dt in schema.META_COLUMNS.items():
        t = {"category": pa.dictionary(pa.int8(), pa.string()), "float64": pa.float64(), "int8": pa.int8()}[dt]
        fields.append(pa.field(c, t))
    for f in schema.NUMERIC:
        fields.append(pa.field(f.name, pa.float32()))
    for f in schema.CATEGORICAL:
        fields.append(pa.field(f.name, pa.int8()))
    return pa.schema(fields)


def _to_table(df: pd.DataFrame) -> pa.Table:
    df = df.copy(deep=False)
    df["source"] = pd.Categorical(df["source"], categories=schema.SOURCES)
    df["split"] = pd.Categorical(df["split"], categories=["train", "val", "test"])
    tbl = pa.Table.from_pandas(df[schema.ALL_COLUMNS], preserve_index=False)
    return tbl.cast(_arrow_schema())


def write_source(key: str, result, out: Path) -> int:
    """result = DataFrame or iterator of DataFrames. returns rows written"""
    tmp = out.with_suffix(".parquet.tmp")
    rows = 0
    writer = None
    try:
        chunks = [result] if isinstance(result, pd.DataFrame) else result
        for i, df in enumerate(chunks):
            if "split" not in df or df["split"].isna().all():
                if not isinstance(result, pd.DataFrame):
                    raise ValueError(f"{key}: chunked adapters must assign 'split' themselves")
                df = schema.assign_time_splits(df)
            tbl = _to_table(df)
            if writer is None:
                writer = pq.ParquetWriter(tmp, tbl.schema, compression="zstd")
            writer.write_table(tbl, row_group_size=500_000)
            rows += len(df)
            del tbl, df
            gc.collect()
            if not isinstance(result, pd.DataFrame):
                print(f"[{key}]   wrote chunk {i + 1} ({rows:,} rows so far)", flush=True)
    finally:
        if writer is not None:
            writer.close()
    tmp.replace(out)   # atomic-ish, a half written parquet would poison prepare
    return rows


# ---- stats from the written parquet --------------------------------------

def _coverage(path: Path) -> dict:
    md = pq.ParquetFile(path).metadata
    n = md.num_rows
    nulls = {}
    for rg in range(md.num_row_groups):
        g = md.row_group(rg)
        for j in range(g.num_columns):
            col = g.column(j)
            st = col.statistics
            if st is not None and st.has_null_count:
                nulls[col.path_in_schema] = nulls.get(col.path_in_schema, 0) + st.null_count
    return {f.name: 1.0 - nulls.get(f.name, 0) / max(n, 1) for f in schema.NUMERIC}


def _rate_by(df: pd.DataFrame, col: str, labels=None, max_levels=30) -> dict:
    g = df.groupby(col, observed=True)["is_fraud"].agg(["mean", "size"])
    g = g.sort_index().head(max_levels)
    out = {}
    for k, row in g.iterrows():
        name = labels[int(k)] if labels is not None else (int(k) if isinstance(k, (int, np.integer, float)) else str(k))
        out[str(name)] = {"fraud_rate": float(row["mean"]), "count": int(row["size"])}
    return out


def _log_hist(x: np.ndarray, bins=40):
    x = x[np.isfinite(x) & (x >= 0)]
    if len(x) == 0:
        return {"edges": [], "counts": []}
    lx = np.log10(x + 1)
    counts, edges = np.histogram(lx, bins=bins)
    return {"edges": [round(float(e), 4) for e in edges], "counts": [int(c) for c in counts]}


def summarize(path: Path, key: str, seconds: float) -> dict:
    df = pq.read_table(path, columns=LIGHT_COLS).to_pandas()
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
        "coverage": _coverage(path),
        "amount": {
            "legit_quantiles": {q: float(legit["amount"].quantile(q)) for q in (0.01, 0.1, 0.5, 0.9, 0.99)},
            "fraud_quantiles": {q: float(fraud["amount"].quantile(q)) for q in (0.01, 0.1, 0.5, 0.9, 0.99)} if len(fraud) else {},
            "legit_hist_log10": _log_hist(legit["amount"].to_numpy()),
            "fraud_hist_log10": _log_hist(fraud["amount"].to_numpy()),
        },
    }
    if df["hour"].notna().any():
        out["by_hour"] = _rate_by(df.dropna(subset=["hour"]).assign(hour=lambda d: d["hour"].astype(int)), "hour")
    if (df["merchant_category"] != 0).any():
        out["by_category"] = _rate_by(df, "merchant_category", schema.CAT_VOCAB["merchant_category"])
    return out


def build_source(key: str, raw_dir: Path, max_rows=None) -> dict:
    t0 = time.time()
    adapter = get_adapter(key)
    print(f"[{key}] loading...", flush=True)
    if "max_rows" in inspect.signature(adapter.load).parameters:
        result = adapter.load(raw_dir, max_rows=max_rows)
    else:
        result = adapter.load(raw_dir)
        if max_rows and isinstance(result, pd.DataFrame) and len(result) > max_rows:
            result = result.iloc[:max_rows].reset_index(drop=True)
    out = PROCESSED_DIR / f"{key}.parquet"
    rows = write_source(key, result, out)
    del result
    stats = summarize(out, key, time.time() - t0)
    stats["parquet"] = out.name
    print(f"[{key}] {rows:,} rows, {stats['fraud']:,} fraud ({stats['fraud_rate']:.3%}) in {stats['build_seconds']}s")
    return stats


def load_stats() -> dict:
    if STATS.exists():
        return json.loads(STATS.read_text())
    return {"sources": {}}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", action="append")
    ap.add_argument("--raw-dir", default=str(RAW_DIR))
    ap.add_argument("--max-rows", type=int, default=None, help="debug: cap rows per source")
    args = ap.parse_args(argv)
    ensure_dirs()
    raw_dir = Path(args.raw_dir)
    keys = args.source or ORDER

    stats = load_stats()
    built = []
    for key in keys:
        if not available(raw_dir, key):
            print(f"[{key}] raw data not found, skipping ({SOURCES[key].url})")
            continue
        try:
            stats["sources"][key] = build_source(key, raw_dir, args.max_rows)
        except MissingData as e:
            print(f"[{key}] {e}")
            continue
        built.append(key)
        # write after every source so a crash on tabformer doesn't lose the rest
        stats["generated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        STATS.write_text(json.dumps(stats, indent=1))

    total = sum(s["rows"] for s in stats["sources"].values())
    fraud = sum(s["fraud"] for s in stats["sources"].values())
    print(f"\nbuilt {built}; corpus now {total:,} rows / {fraud:,} fraud across {len(stats['sources'])} sources")
    return 0


if __name__ == "__main__":
    sys.exit(main())
