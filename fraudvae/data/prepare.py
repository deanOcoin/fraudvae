"""processed parquet -> model ready arrays.

Writes one directory per "prepared set" (default data/prepared/default):
    meta.json                 feature list, transforms, standardisation stats, vocab
    {split}_num.npy  float32  [N, D] standardised numerics, 0 where unobserved
    {split}_mask.npy uint8    [N, D] 1 = observed
    {split}_cat.npy  int16    [N, C] categorical codes, 0 = unknown
    {split}_src.npy  int8     [N]    source index (schema.SOURCE_INDEX)
    {split}_y.npy    int8     [N]    fraud label
    {split}_row.npy  int64    [N]    row offset into the source parquet (for lookups)
    {split}_cust.npy int64    [N]    v5 global customer key (source << 40 | code), -1 unknown
    {split}_ts.npy   float64  [N]    v5 timestamp, for the sequence index and history windows
    {split}_ent.npy  int32    [N, 6] v5 entity codes within the source (schema.ENTITY_COLUMNS order), -1 none;
                              the graph module looks embeddings up by them
    {split}_wide_<source>.npy float32 [n_source_rows, W_source]  v5 wide block, NaN = missing;
                              rows of one source are contiguous in a split (meta.wide.offsets)
    seq_order.npy  int64     [N_all] v5 sequence index: global row ids (train, then val, then
                              test; meta.split_offsets) sorted by (customer, time)
    seq_pos.npy    int32     [N_all] position of each global row in seq_order
    seq_start.npy  int32     [N_all] position where that row's customer block starts (== seq_pos
                              for rows without a customer, i.e. no history)

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
BATCH_WIDE = 200_000   # a 431-column wide block at 1M rows is 1.7 GB per copy; keep the lockstep batches small


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
            "cust": np.lib.format.open_memmap(out / f"{s}_cust.npy", "w+", np.int64, (n,)),
            "ts": np.lib.format.open_memmap(out / f"{s}_ts.npy", "w+", np.float64, (n,)),
            "ent": np.lib.format.open_memmap(out / f"{s}_ent.npy", "w+", np.int32, (n, len(schema.ENTITY_COLUMNS))),
            "pos": 0,
        }

    cols = num_names + [c for c in schema.CATEGORICAL_NAMES] + list(DERIVED_CATS) + ["is_fraud", "ts"] + list(schema.ENTITY_COLUMNS)
    cols = list(dict.fromkeys(cols))
    wide_meta = {}
    for key in sources:
        pf = pq.ParquetFile(PROCESSED_DIR / f"{key}.parquet")
        src_idx = schema.SOURCE_INDEX[key]
        have = set(pf.schema_arrow.names)
        missing = [c for c in cols if c not in have]
        if missing:
            # a field added to the schema after this source was built: unobserved for it
            print(f"  {key}: parquet predates {len(missing)} schema field(s) ({', '.join(missing[:4])}...), treated as unobserved")
        # v5 wide block: the source's own raw columns, read in lockstep with the unified file
        wp = PROCESSED_DIR / f"{key}.wide.parquet"
        wide_iter, wide_arrays = None, None
        if wp.exists():
            wdesc = json.loads(wp.with_suffix(".json").read_text())["columns"]
            wcols = [c["name"] for c in wdesc]
            wide_iter = pq.ParquetFile(wp).iter_batches(batch_size=BATCH_WIDE)
            wide_arrays = {s: np.lib.format.open_memmap(out / f"{s}_wide_{key}.npy", "w+", np.float32, (len(plan[key][s]), len(wcols)))
                           for s in SPLITS}
            wide_meta[key] = {"columns": wcols, "kinds": [c["kind"] for c in wdesc],
                              "sizes": [len(c.get("values", [])) + 2 if c["kind"] == "cat" else 0 for c in wdesc],
                              "offsets": {s: [int(arrays[s]["pos"]), int(arrays[s]["pos"]) + len(plan[key][s])] for s in SPLITS}}
        offset = 0
        for batch in pf.iter_batches(batch_size=BATCH_WIDE if wide_iter is not None else BATCH, columns=[c for c in cols if c in have]):
            b = batch.to_pandas()
            for c in missing:
                b[c] = np.int8(0) if c in schema.CATEGORICAL_NAMES else (np.int64(-1) if c in schema.ENTITY_COLUMNS else np.float32(np.nan))
            n_b = len(b)
            wb = None
            if wide_iter is not None:
                wb = next(wide_iter)
                assert len(wb) == n_b, (key, len(wb), n_b)
                wb = np.column_stack([wb.column(i).to_numpy(zero_copy_only=False) for i in range(wb.num_columns)])
                if wb.dtype != np.float32:
                    wb = wb.astype(np.float32)
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
                if wb is not None:
                    w0 = p - wide_meta[key]["offsets"][s][0]
                    wide_arrays[s][w0 : w0 + len(sel)] = wb[sel]
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
                cust = sub["customer"].to_numpy().astype(np.int64)
                a["cust"][p:q] = np.where(cust >= 0, (np.int64(src_idx) << 40) | cust, -1)
                a["ts"][p:q] = sub["ts"].to_numpy().astype(np.float64)
                a["ent"][p:q] = np.column_stack([sub[c].to_numpy().astype(np.int64) for c in schema.ENTITY_COLUMNS]).astype(np.int32)
                a["pos"] = q
            offset += n_b
        if wide_arrays is not None:
            for s in SPLITS:
                wide_arrays[s].flush()
            del wide_arrays
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
        for k in ("num", "mask", "cat", "src", "y", "row", "cust", "ts"):
            a[k].flush()

    # v5 sequence index: every row of every split in (customer, time) order, so a row's history
    # window is the slice of seq_order just before its own position, whatever split those
    # earlier rows fell into (they're older, so train rows for a val row, train+val for test)
    offsets = {}
    acc = 0
    for s in SPLITS:
        offsets[s] = acc
        acc += counts[s]
    cust_all = np.concatenate([arrays[s]["cust"][:] for s in SPLITS])
    ts_all = np.concatenate([arrays[s]["ts"][:] for s in SPLITS])
    order = np.lexsort((ts_all, cust_all)).astype(np.int64)
    del ts_all
    c_sorted = cust_all[order]
    del cust_all
    n_all = len(order)
    same = np.empty(n_all, dtype=bool)
    same[0] = False
    np.equal(c_sorted[1:], c_sorted[:-1], out=same[1:])
    no_cust = c_sorted < 0
    del c_sorted
    starts = np.flatnonzero(~same)
    block_id = np.cumsum(~same, dtype=np.int32) - 1
    start_pos = starts[block_id].astype(np.int32)
    del same, starts, block_id
    pos_sorted = np.arange(n_all, dtype=np.int32)
    start_pos[no_cust] = pos_sorted[no_cust]   # a row with no customer has no history
    del no_cust
    seq_pos = np.empty(n_all, dtype=np.int32)
    seq_pos[order] = pos_sorted
    seq_start = np.empty(n_all, dtype=np.int32)
    seq_start[order] = start_pos
    del pos_sorted, start_pos
    np.save(out / "seq_order.npy", order)
    np.save(out / "seq_pos.npy", seq_pos)
    np.save(out / "seq_start.npy", seq_start)
    n_hist = int((seq_pos - seq_start > 0).sum())
    del order, seq_pos, seq_start
    print(f"  sequence index: {n_all:,} rows, {n_hist:,} with at least one earlier row of the same customer", flush=True)

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
        "split_offsets": offsets,
        "fraud_counts": {s: int(arrays[s]["y"][:].sum()) for s in SPLITS},
        "numeric": [{"name": f.name, "transform": f.transform, "group": f.group, "label": f.label, "unit": f.unit,
                     "mean": float(mean[j]), "std": float(std[j])} for j, f in enumerate(num_feats)],
        "categorical": cat_feats,
        "wide": wide_meta,
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
