"""processed parquet -> model ready arrays.

Writes one directory per "prepared set" (default data/prepared/default):
    meta.json                 feature list, transforms, standardisation stats, vocab
    {split}_num.npy  float32  [N, D] standardised numerics, 0 where unobserved
    {split}_mask.npy uint8    [N, D] 1 = observed
    {split}_cat.npy  int16    [N, C] categorical codes, 0 = unknown
    {split}_src.npy  int8     [N]    source index (schema.SOURCE_INDEX)
    {split}_y.npy    int8     [N]    fraud label

Standardisation uses train legit rows only. hour / dow become categoricals here
(they're stored as numbers upstream because that's what they are, but the
model does better treating them as 24 / 7 classes).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ..paths import PREPARED_DIR, PROCESSED_DIR, ensure_dirs
from . import schema

DERIVED_CATS = {"hour": 24, "dow": 7}   # numeric column -> number of classes
SPLITS = ("train", "val", "test")


def model_numeric_features():
    return [f for f in schema.NUMERIC if f.name not in DERIVED_CATS]


def model_categorical_features():
    cats = [{"name": f.name, "size": len(f.values), "values": list(f.values)} for f in schema.CATEGORICAL]
    for name, k in DERIVED_CATS.items():
        cats.append({"name": name, "size": k + 1, "values": ["unknown"] + [str(i) for i in range(k)]})
    return cats


def transform_numeric(x: np.ndarray, kind: str) -> np.ndarray:
    x = x.astype(np.float32, copy=True)
    if kind == "log1p":
        return np.log1p(np.clip(x, 0, None))
    if kind == "signed_log1p":
        return np.sign(x) * np.log1p(np.abs(x))
    return x


def _encode(df, num_feats, cat_feats):
    num = np.empty((len(df), len(num_feats)), dtype=np.float32)
    for j, f in enumerate(num_feats):
        num[:, j] = transform_numeric(df[f.name].to_numpy(), f.transform)
    mask = np.isfinite(num)
    num[~mask] = 0.0
    cat = np.zeros((len(df), len(cat_feats)), dtype=np.int16)
    for j, c in enumerate(cat_feats):
        v = df[c["name"]].to_numpy()
        if c["name"] in DERIVED_CATS:
            ok = np.isfinite(v)
            cat[:, j] = np.where(ok, np.nan_to_num(v).astype(np.int16) + 1, 0)
        else:
            cat[:, j] = v.astype(np.int16)
    return num, mask.astype(np.uint8), cat


def prepare(name="default", sources=None, out_root=PREPARED_DIR):
    t0 = time.time()
    sources = sources or [k for k in schema.SOURCES if (PROCESSED_DIR / f"{k}.parquet").exists()]
    if not sources:
        raise SystemExit("nothing in data/processed, run build first")
    out = Path(out_root) / name
    out.mkdir(parents=True, exist_ok=True)

    num_feats = model_numeric_features()
    cat_feats = model_categorical_features()
    num_names = [f.name for f in num_feats]
    D = len(num_names)

    cols = num_names + list(schema.CATEGORICAL_NAMES) + list(DERIVED_CATS) + ["is_fraud", "split"]
    parts = {s: [] for s in SPLITS}
    for key in sources:
        df = pd.read_parquet(PROCESSED_DIR / f"{key}.parquet", columns=cols)
        src_idx = schema.SOURCE_INDEX[key]
        for s in SPLITS:
            sub = df[df["split"] == s]
            num, mask, cat = _encode(sub, num_feats, cat_feats)
            parts[s].append((num, mask, cat, np.full(len(sub), src_idx, dtype=np.int8),
                             sub["is_fraud"].to_numpy().astype(np.int8)))
        print(f"  {key}: {len(df):,} rows ({time.time() - t0:.0f}s)", flush=True)
        del df

    keys = ("num", "mask", "cat", "src", "y")
    arrays = {s: {k: np.concatenate([p[i] for p in parts[s]]) for i, k in enumerate(keys)} for s in SPLITS}
    del parts
    counts = {s: len(arrays[s]["y"]) for s in SPLITS}

    # standardisation from train legit, observed values only
    tr = arrays["train"]
    legit = tr["y"] == 0
    mean = np.zeros(D, dtype=np.float64)
    std = np.ones(D, dtype=np.float64)
    for j in range(D):
        m = (tr["mask"][:, j] == 1) & legit
        if m.sum() > 1:
            v = tr["num"][:, j][m].astype(np.float64)
            mean[j] = v.mean()
            std[j] = max(v.std(), 1e-3)
    for s in SPLITS:
        a = arrays[s]
        a["num"] = (a["num"] - mean.astype(np.float32)) / std.astype(np.float32)
        a["num"][a["mask"] == 0] = 0.0
        for k in keys:
            np.save(out / f"{s}_{k}.npy", a[k])

    meta = {
        "name": name,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "schema_version": schema.SCHEMA_VERSION,
        "sources": sources,
        "source_index": {k: schema.SOURCE_INDEX[k] for k in sources},
        "counts": {s: int(counts[s]) for s in SPLITS},
        "fraud_counts": {s: int(arrays[s]["y"].sum()) for s in SPLITS},
        "numeric": [{"name": f.name, "transform": f.transform,
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
    args = ap.parse_args(argv)
    ensure_dirs()
    prepare(args.name, args.source)
    return 0


if __name__ == "__main__":
    sys.exit(main())
