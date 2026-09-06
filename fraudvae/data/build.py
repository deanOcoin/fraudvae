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
SAMPLES = PROCESSED_DIR / "samples.parquet"
LIGHT_COLS = ["ts", "is_fraud", "split", "amount", "hour", "merchant_category", "channel", "txn_type", "fraud_scenario"]


def _arrow_schema():
    fields = []
    for c, dt in schema.META_COLUMNS.items():
        t = {"category": pa.dictionary(pa.int8(), pa.string()), "string": pa.string(), "float64": pa.float64(),
             "int8": pa.int8(), "int64": pa.int64()}[dt]
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


def _wide_table(df: pd.DataFrame, cols: list[str]) -> pa.Table:
    """the source's own raw columns, float32, same row order as the unified table"""
    arrays = [pa.array(df[c].to_numpy(dtype=np.float32), type=pa.float32()) for c in cols]
    return pa.table(arrays, names=[c[len(schema.WIDE_PREFIX):] for c in cols])


def wide_path(out: Path) -> Path:
    return out.with_name(out.stem + ".wide.parquet")


def write_source(key: str, result, out: Path) -> int:
    """result = DataFrame or iterator of DataFrames. returns rows written. a frame with
    wide__ columns also produces <key>.wide.parquet (+ .wide.json describing the columns)"""
    tmp = out.with_suffix(".parquet.tmp")
    wtmp = wide_path(out).with_suffix(".parquet.tmp")
    rows = 0
    writer = wwriter = None
    wide_cols, wide_desc = None, None
    try:
        if isinstance(result, pd.DataFrame):
            chunks = [result]
        else:
            chunks = result
        for i, df in enumerate(chunks):
            if "split" not in df or df["split"].isna().all():
                if not isinstance(result, pd.DataFrame):
                    raise ValueError(f"{key}: chunked adapters must assign 'split' themselves")
                df = schema.assign_time_splits(df)
            if wide_cols is None:
                wide_cols = [c for c in df.columns if c.startswith(schema.WIDE_PREFIX)]
                # the adapter's descriptions where it gave them (raw categoricals carry their
                # values), a numeric default for the rest (the entity block's wide__*_fast columns)
                given = {d["name"]: d for d in (df.attrs.get("wide_columns") or [])}
                wide_desc = [given.get(c[len(schema.WIDE_PREFIX):], {"name": c[len(schema.WIDE_PREFIX):], "kind": "num"}) for c in wide_cols]
            tbl = _to_table(df)
            if writer is None:
                writer = pq.ParquetWriter(tmp, tbl.schema, compression="zstd")
            writer.write_table(tbl, row_group_size=500_000)
            if wide_cols:
                wt = _wide_table(df, wide_cols)
                if wwriter is None:
                    wwriter = pq.ParquetWriter(wtmp, wt.schema, compression="zstd")
                wwriter.write_table(wt, row_group_size=500_000)
                del wt
            rows += len(df)
            del tbl, df
            gc.collect()   # the 24M row sources run close to the memory limit
            if not isinstance(result, pd.DataFrame):
                print(f"[{key}]   wrote chunk {i + 1} ({rows:,} rows so far)", flush=True)
    finally:
        if writer is not None:
            writer.close()
        if wwriter is not None:
            wwriter.close()
    tmp.replace(out)   # atomic-ish, a half written parquet would poison prepare
    wp = wide_path(out)
    if wide_cols:
        wtmp.replace(wp)
        wp.with_suffix(".json").write_text(json.dumps({"source": key, "columns": wide_desc}, indent=1))
    else:
        # a rebuild of a source that used to have a wide block: don't leave a stale one behind
        wp.unlink(missing_ok=True)
        wp.with_suffix(".json").unlink(missing_ok=True)
    return rows


# ---- stats from the written parquet --------------------------------------

def _coverage(path: Path) -> dict:
    pf = pq.ParquetFile(path)
    md = pf.metadata
    n = md.num_rows
    names = pf.schema_arrow.names
    nulls = {name: 0 for name in names}
    for rg in range(md.num_row_groups):
        g = md.row_group(rg)
        for j in range(g.num_columns):
            st = g.column(j).statistics
            if st is not None and st.has_null_count:
                nulls[names[j]] += st.null_count
    cov = {f.name: 1.0 - nulls.get(f.name, n) / max(n, 1) for f in schema.NUMERIC}
    present = [c for c in schema.CATEGORICAL_NAMES if c in names]
    cats = pf.read(columns=present)
    for f in schema.CATEGORICAL:
        if f.name not in present:
            cov[f.name] = 0.0
            continue
        col = cats.column(f.name).to_numpy()
        cov[f.name] = float((col != 0).mean())
    return cov


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
        "absolute_time": bool(df["ts"].dropna().gt(1e9).all()) if df["ts"].notna().any() else False,
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
    if (df["channel"] != 0).any():
        out["by_channel"] = _rate_by(df, "channel", schema.CAT_VOCAB["channel"])
    if (df["txn_type"] != 0).any():
        out["by_txn_type"] = _rate_by(df, "txn_type", schema.CAT_VOCAB["txn_type"])
    if (df["fraud_scenario"] != 0).any():
        out["by_scenario"] = {int(k): int(v) for k, v in df["fraud_scenario"].value_counts().items()}
    return out


def sample_rows(path: Path, n_legit=150, n_fraud=150, seed=0) -> pd.DataFrame:
    """a few hundred rows for the ui, fraud oversampled. row-group aware so it's
    cheap on the big files"""
    pf = pq.ParquetFile(path)
    tbl = pf.read(columns=["is_fraud", "split"])
    y = tbl.column("is_fraud").to_numpy()
    held = tbl.column("split").to_numpy().astype(str) == "test"   # examples come from data the model never trained on
    rng = np.random.default_rng(seed)
    legit = np.flatnonzero((y == 0) & held)
    fraud = np.flatnonzero((y == 1) & held)
    if len(fraud) < n_fraud:
        fraud = np.flatnonzero(y == 1)
    pick = np.sort(np.concatenate([
        rng.choice(legit, min(n_legit, len(legit)), replace=False),
        rng.choice(fraud, min(n_fraud, len(fraud)), replace=False),
    ]))
    frames = []
    wp = wide_path(path)
    pw = pq.ParquetFile(wp) if wp.exists() else None
    start = 0
    for rg in range(pf.metadata.num_row_groups):
        n = pf.metadata.row_group(rg).num_rows
        lo, hi = np.searchsorted(pick, [start, start + n])
        if hi > lo:
            tbl = pf.read_row_group(rg)
            sel = pa.array(pick[lo:hi] - start)
            frame = tbl.take(sel).to_pandas()
            if pw is not None:
                # the same rows of the wide block, as one json dict per row (NaN dropped) so
                # the samples file keeps its flat schema and the ui can hand it back to /api/score
                w = pw.read_row_group(rg).take(sel).to_pandas()
                frame["wide"] = [json.dumps({k: float(v) for k, v in r.items() if v == v}) for r in w.to_dict("records")]
            frames.append(frame)
        start += n
    out = pd.concat(frames, ignore_index=True)
    if "wide" not in out:
        out["wide"] = None
    return out


def build_source(key: str, raw_dir: Path, max_rows=None):
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
    stats["schema_version"] = schema.SCHEMA_VERSION
    sample = sample_rows(out)
    print(f"[{key}] {rows:,} rows, {stats['fraud']:,} fraud ({stats['fraud_rate']:.3%}) in {stats['build_seconds']}s")
    return stats, sample


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
    samples = []
    if SAMPLES.exists():
        old = pd.read_parquet(SAMPLES)
        samples.append(old[~old["source"].isin(keys)])
    built = []
    for key in keys:
        if not available(raw_dir, key):
            print(f"[{key}] raw data not found, skipping ({SOURCES[key].url})")
            continue
        try:
            s, sample = build_source(key, raw_dir, args.max_rows)
        except MissingData as e:
            print(f"[{key}] {e}")
            continue
        stats["sources"][key] = s
        samples.append(sample)
        built.append(key)
        # write after every source so a crash on tabformer doesn't lose the rest
        stats["generated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        stats["schema_version"] = schema.SCHEMA_VERSION
        STATS.write_text(json.dumps(stats, indent=1))
        if samples:
            pd.concat(samples, ignore_index=True).to_parquet(SAMPLES, index=False)

    total = sum(s["rows"] for s in stats["sources"].values())
    fraud = sum(s["fraud"] for s in stats["sources"].values())
    print(f"\nbuilt {built}; corpus now {total:,} rows / {fraud:,} fraud across {len(stats['sources'])} sources")
    return 0


if __name__ == "__main__":
    sys.exit(main())
