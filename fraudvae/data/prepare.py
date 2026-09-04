"""processed parquet -> model ready arrays.

Writes one directory per "prepared set" (default data/prepared/default):
    meta.json                 feature list, transforms, standardisation stats, vocab
    {split}_num.npy  float32  [N, D] standardised numerics, 0 where unobserved
    {split}_mask.npy uint8    [N, D] 1 = observed
    {split}_cat.npy  int16    [N, C] categorical codes, 0 = unknown
    {split}_src.npy  int8     [N]    source index (schema.SOURCE_INDEX)
    {split}_y.npy    int8     [N]    fraud label
    {split}_row.npy  int64    [N]    row offset into the source parquet (for lookups)

Standardisation uses train legit rows only. hour / dow become categoricals here
(they're stored as numbers upstream because that's what they are, but the
model does better treating them as 24 / 7 classes).

Memory: streamed in row-group batches via pyarrow so the 24M row tabformer
file doesn't need to fit in RAM.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from ..paths import PREPARED_DIR, PROCESSED_DIR, ensure_dirs
from . import schema

DERIVED_CATS = {"hour": 24, "dow": 7}   # numeric column -> number of classes
SPLITS = ("train", "val", "test")
BATCH = 1_000_000


def model_numeric_features():
    return [f for f in schema.NUMERIC if f.name not in DERIVED_CATS]


def model_categorical_features():
    cats = [{"name": f.name, "size": len(f.values), "values": list(f.values), "group": f.group, "label": f.label}
            for f in schema.CATEGORICAL]
    for name, k in DERIVED_CATS.items():
        cats.append({"name": name, "size": k + 1, "values": ["unknown"] + [str(i) for i in range(k)],
                     "group": "core", "label": {"hour": "Hour of day", "dow": "Day of week"}[name]})
    return cats


def transform_numeric(x: np.ndarray, kind: str) -> np.ndarray:
    x = x.astype(np.float32, copy=True)
    if kind == "log1p":
        return np.log1p(np.clip(x, 0, None))
    if kind == "signed_log1p":
        return np.sign(x) * np.log1p(np.abs(x))
    return x


def _available_sources():
    return [k for k in schema.SOURCES if (PROCESSED_DIR / f"{k}.parquet").exists()]


def _plan(sources, max_rows_per_source, seed):
    """which rows of each source go where. returns {source: {split: np.ndarray of row ids}}"""
    rng = np.random.default_rng(seed)
    plan, counts = {}, {s: 0 for s in SPLITS}
    for key in sources:
        pf = pq.ParquetFile(PROCESSED_DIR / f"{key}.parquet")
        tbl = pf.read(columns=["split", "is_fraud"])
        split = tbl.column("split").to_numpy().astype(str)
        y = tbl.column("is_fraud").to_numpy()
        plan[key] = {}
        for s in SPLITS:
            idx = np.flatnonzero(split == s)
            if s == "train" and max_rows_per_source and len(idx) > max_rows_per_source:
                legit = idx[y[idx] == 0]
                fraud = idx[y[idx] == 1]
                keep = rng.choice(legit, size=max(max_rows_per_source - len(fraud), 1), replace=False)
                idx = np.sort(np.concatenate([keep, fraud]))
            plan[key][s] = idx
            counts[s] += len(idx)
    return plan, counts


def _stats_from_run(run) -> dict:
    """{feature name: (mean, std)} from a trained run's checkpoint, so a set prepared here is
    standardised exactly like the one that run was trained on (calibrating / evaluating a
    gpu-trained model on a laptop-sized set, say)"""
    import torch

    from ..paths import RUNS_DIR

    p = Path(run) if Path(run).exists() else RUNS_DIR / run
    p = p / "model.pt" if p.is_dir() else p
    ck = torch.load(p, map_location="cpu", weights_only=False)
    return {f["name"]: (float(f["mean"]), float(f["std"])) for f in ck["prepared_meta"]["numeric"]}


def prepare(name="default", sources=None, max_rows_per_source=None, seed=1337, out_root=PREPARED_DIR, stats_from_run=None):
    t0 = time.time()
    fixed_stats = _stats_from_run(stats_from_run) if stats_from_run else None
    sources = sources or _available_sources()
    if not sources:
        raise SystemExit("nothing in data/processed, run build first")
    out = Path(out_root) / name
    out.mkdir(parents=True, exist_ok=True)

    num_feats = model_numeric_features()
    cat_feats = model_categorical_features()
    num_names = [f.name for f in num_feats]
    cat_names = [c["name"] for c in cat_feats]
    D, C = len(num_names), len(cat_names)

    plan, counts = _plan(sources, max_rows_per_source, seed)
    print("rows per split:", counts, flush=True)

    arrays = {}
    for s in SPLITS:
        n = counts[s]
        arrays[s] = {
            "num": np.lib.format.open_memmap(out / f"{s}_num.npy", "w+", np.float32, (n, D)),
            "mask": np.lib.format.open_memmap(out / f"{s}_mask.npy", "w+", np.uint8, (n, D)),
            "cat": np.lib.format.open_memmap(out / f"{s}_cat.npy", "w+", np.int16, (n, C)),
            "src": np.lib.format.open_memmap(out / f"{s}_src.npy", "w+", np.int8, (n,)),
            "y": np.lib.format.open_memmap(out / f"{s}_y.npy", "w+", np.int8, (n,)),
            "row": np.lib.format.open_memmap(out / f"{s}_row.npy", "w+", np.int64, (n,)),
            "pos": 0,
        }

    cols = num_names + [c for c in schema.CATEGORICAL_NAMES] + list(DERIVED_CATS) + ["is_fraud"]
    for key in sources:
        pf = pq.ParquetFile(PROCESSED_DIR / f"{key}.parquet")
        src_idx = schema.SOURCE_INDEX[key]
        offset = 0
        for batch in pf.iter_batches(batch_size=BATCH, columns=cols):
            b = batch.to_pandas()
            n_b = len(b)
            rows = np.arange(offset, offset + n_b)
            for s in SPLITS:
                want = plan[key][s]
                # rows in this batch that belong to split s (plan ids are sorted)
                lo, hi = np.searchsorted(want, [offset, offset + n_b])
                sel = want[lo:hi] - offset
                if len(sel) == 0:
                    continue
                sub = b.iloc[sel]
                a = arrays[s]
                p, q = a["pos"], a["pos"] + len(sel)
                num = np.empty((len(sel), D), dtype=np.float32)
                for j, f in enumerate(num_feats):
                    num[:, j] = transform_numeric(sub[f.name].to_numpy(), f.transform)
                mask = np.isfinite(num)
                num[~mask] = 0.0
                a["num"][p:q] = num
                a["mask"][p:q] = mask.astype(np.uint8)
                cat = np.zeros((len(sel), C), dtype=np.int16)
                for j, c in enumerate(cat_feats):
                    if c["name"] in DERIVED_CATS:
                        v = sub[c["name"]].to_numpy()
                        ok = np.isfinite(v)
                        cat[:, j] = np.where(ok, np.nan_to_num(v).astype(np.int16) + 1, 0)
                    else:
                        cat[:, j] = sub[c["name"]].to_numpy().astype(np.int16)
                a["cat"][p:q] = cat
                a["src"][p:q] = src_idx
                a["y"][p:q] = sub["is_fraud"].to_numpy().astype(np.int8)
                a["row"][p:q] = rows[sel]
                a["pos"] = q
            offset += n_b
        print(f"  {key}: streamed {offset:,} rows ({time.time() - t0:.0f}s)", flush=True)

    for s in SPLITS:
        assert arrays[s]["pos"] == counts[s], (s, arrays[s]["pos"], counts[s])

    # standardisation from train legit, observed values only
    # FIXME this is a column-at-a-time loop over a memmap, slow-ish for the full set (minutes). fine for now
    tr = arrays["train"]
    legit = tr["y"][:] == 0
    mean = np.zeros(D, dtype=np.float64)
    std = np.ones(D, dtype=np.float64)
    for j in range(D):
        if fixed_stats is not None:
            mean[j], std[j] = fixed_stats.get(num_names[j], (0.0, 1.0))
            continue
        m = (tr["mask"][:, j] == 1) & legit
        if m.sum() > 1:
            v = tr["num"][:, j][m].astype(np.float64)
            mean[j] = v.mean()
            std[j] = max(v.std(), 1e-3)
    for s in SPLITS:
        a = arrays[s]
        for start in range(0, counts[s], BATCH):
            sl = slice(start, min(start + BATCH, counts[s]))
            chunk = a["num"][sl]
            chunk = (chunk - mean.astype(np.float32)) / std.astype(np.float32)
            chunk[a["mask"][sl] == 0] = 0.0
            a["num"][sl] = chunk
        for k in ("num", "mask", "cat", "src", "y", "row"):
            a[k].flush()

    meta = {
        "name": name,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "schema_version": schema.SCHEMA_VERSION,
        "sources": sources,
        "source_index": {k: schema.SOURCE_INDEX[k] for k in sources},
        "max_rows_per_source": max_rows_per_source,
        "stats_from_run": str(stats_from_run) if stats_from_run else None,
        "seed": seed,
        "counts": {s: int(counts[s]) for s in SPLITS},
        "fraud_counts": {s: int(arrays[s]["y"][:].sum()) for s in SPLITS},
        "numeric": [{"name": f.name, "transform": f.transform, "group": f.group, "label": f.label, "unit": f.unit,
                     "mean": float(mean[j]), "std": float(std[j])} for j, f in enumerate(num_feats)],
        "categorical": cat_feats,
        "seconds": round(time.time() - t0, 1),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"prepared set '{name}' at {out} in {meta['seconds']}s: " + ", ".join(f"{s}={counts[s]:,}" for s in SPLITS))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", default="default")
    ap.add_argument("--source", action="append")
    ap.add_argument("--max-rows-per-source", type=int, default=None,
                    help="cap TRAIN rows per source (legit subsampled, fraud kept). val/test are never capped")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--stats-from-run", default=None,
                    help="standardise with the mean/std a trained run used, so its model can be calibrated / evaluated on this set")
    args = ap.parse_args(argv)
    ensure_dirs()
    prepare(args.name, args.source, args.max_rows_per_source, args.seed, stats_from_run=args.stats_from_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
