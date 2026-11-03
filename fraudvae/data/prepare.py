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
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from ..paths import PREPARED_DIR, PROCESSED_DIR, ensure_dirs
from . import schema
from ..model.loading import load_checkpoint

DERIVED_CATS = {"hour": 24, "dow": 7}   # numeric column -> number of classes
SPLITS = ("train", "val", "test")
BATCH = 1_000_000
BATCH_WIDE = 200_000   # a 431-column wide block at 1M rows is 1.7 GB per copy; keep the lockstep batches small
ROWS_PER_TASK = 5_000_000   # a task streams about this many rows of one source (whole row groups)
ARRAY_DTYPES = {"num": np.float32, "mask": np.uint8, "cat": np.int16, "src": np.int8, "y": np.int8, "row": np.int64,
                "cust": np.int64, "ts": np.float64, "ent": np.int32}


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


def _available_sources(include_pretrain_only=False):
    """the built sources, in schema order; the pretraining-only ones (a generator we ran ourselves,
    registry.pretrain_only) only when asked, so the evaluation frame never holds a simulator a
    model was tuned on"""
    from .registry import SOURCES as _REG

    return [k for k in schema.SOURCES if (PROCESSED_DIR / f"{k}.parquet").exists()
            and (include_pretrain_only or not getattr(_REG.get(k), "pretrain_only", False))]


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
    from ..paths import RUNS_DIR

    p = Path(run) if Path(run).exists() else RUNS_DIR / run
    p = p / "model.pt" if p.is_dir() else p
    ck = load_checkpoint(p, "cpu")
    return {f["name"]: (float(f["mean"]), float(f["std"])) for f in ck["prepared_meta"]["numeric"]}


def _default_workers() -> int:
    return max(1, min(os.cpu_count() or 1, 64))


def _source_base(sources, plan) -> dict:
    """where each source's rows start within each split: sources are laid out one after the
    other in registry order, each one's rows in the order of its parquet"""
    base, pos = {}, {s: 0 for s in SPLITS}
    for key in sources:
        base[key] = dict(pos)
        for s in SPLITS:
            pos[s] += len(plan[key][s])
    return base


def _row_group_ranges(pf, rows_per_task) -> list[tuple[int, int, int]]:
    """(rg_lo, rg_hi, row_offset) ranges of whole row groups, about rows_per_task rows each"""
    n = pf.metadata.num_row_groups
    sizes = [pf.metadata.row_group(i).num_rows for i in range(n)]
    ranges, lo, acc, off = [], 0, 0, 0
    for i, sz in enumerate(sizes):
        acc += sz
        if acc >= rows_per_task or i == n - 1:
            ranges.append((lo, i + 1, off))
            off += acc
            lo, acc = i + 1, 0
    return ranges


def _stream_task(task):
    """one task of prepare: the rows of row groups [rg_lo, rg_hi) of one source go to their
    places in the split memmaps (opened read-write; tasks never share a row). returns
    (key, rows written per split, rows read, seconds)"""
    key, rg_lo, rg_hi, row_offset, out, base = task
    out = Path(out)
    t0 = time.time()
    num_feats = model_numeric_features()
    cat_feats = model_categorical_features()
    num_names = [f.name for f in num_feats]
    D, C = len(num_names), len(cat_feats)
    cols = num_names + [c for c in schema.CATEGORICAL_NAMES] + list(DERIVED_CATS) + ["is_fraud", "ts"] + list(schema.ENTITY_COLUMNS)
    cols = list(dict.fromkeys(cols))
    pf = pq.ParquetFile(PROCESSED_DIR / f"{key}.parquet")
    src_idx = schema.SOURCE_INDEX[key]
    have = set(pf.schema_arrow.names)
    missing = [c for c in cols if c not in have]
    plan = {s: np.load(out / f"_plan_{key}_{s}.npy", mmap_mode="r") for s in SPLITS}
    arrays = {s: {k: np.load(out / f"{s}_{k}.npy", mmap_mode="r+") for k in ARRAY_DTYPES} for s in SPLITS}
    wp = PROCESSED_DIR / f"{key}.wide.parquet"
    wide_iter, wide_arrays = None, None
    row_groups = list(range(rg_lo, rg_hi))
    if wp.exists():
        wide_iter = pq.ParquetFile(wp).iter_batches(batch_size=BATCH_WIDE, row_groups=row_groups)
        wide_arrays = {s: np.load(out / f"{s}_wide_{key}.npy", mmap_mode="r+") for s in SPLITS}
    written = {s: 0 for s in SPLITS}
    offset = row_offset
    for batch in pf.iter_batches(batch_size=BATCH_WIDE if wide_iter is not None else BATCH, row_groups=row_groups,
                                 columns=[c for c in cols if c in have]):
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
            want = plan[s]
            # rows in this batch that belong to split s (plan ids are sorted)
            lo, hi = np.searchsorted(want, [offset, offset + n_b])
            sel = want[lo:hi] - offset
            if len(sel) == 0:
                continue
            sub = b.iloc[sel]
            a = arrays[s]
            p, q = base[s] + lo, base[s] + hi
            if wb is not None:
                wide_arrays[s][lo:hi] = wb[sel]
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
            written[s] += len(sel)
        offset += n_b
    for s in SPLITS:
        for a in arrays[s].values():
            a.flush()
        if wide_arrays is not None:
            wide_arrays[s].flush()
    return key, written, offset - row_offset, time.time() - t0


def prepare(name="default", sources=None, max_rows_per_source=None, seed=1337, out_root=PREPARED_DIR, stats_from_run=None,
            workers=None, rows_per_task=ROWS_PER_TASK, include_pretrain_only=False):
    t0 = time.time()
    fixed_stats = _stats_from_run(stats_from_run) if stats_from_run else None
    sources = sources or _available_sources(include_pretrain_only)
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

    base = _source_base(sources, plan)
    for s in SPLITS:
        n = counts[s]
        shapes = {"num": (n, D), "mask": (n, D), "cat": (n, C), "src": (n,), "y": (n,), "row": (n,), "cust": (n,), "ts": (n,),
                  "ent": (n, len(schema.ENTITY_COLUMNS))}
        for k, shape in shapes.items():
            np.lib.format.open_memmap(out / f"{s}_{k}.npy", "w+", ARRAY_DTYPES[k], shape).flush()
        for key in sources:
            np.save(out / f"_plan_{key}_{s}.npy", plan[key][s].astype(np.int64))

    cols = num_names + [c for c in schema.CATEGORICAL_NAMES] + list(DERIVED_CATS) + ["is_fraud", "ts"] + list(schema.ENTITY_COLUMNS)
    cols = list(dict.fromkeys(cols))
    wide_meta = {}
    tasks = []
    for key in sources:
        pf = pq.ParquetFile(PROCESSED_DIR / f"{key}.parquet")
        have = set(pf.schema_arrow.names)
        missing = [c for c in cols if c not in have]
        if missing:
            # a field added to the schema after this source was built: unobserved for it
            print(f"  {key}: parquet predates {len(missing)} schema field(s) ({', '.join(missing[:4])}...), treated as unobserved")
        # v5 wide block: the source's own raw columns, read in lockstep with the unified file
        wp = PROCESSED_DIR / f"{key}.wide.parquet"
        if wp.exists():
            wdesc = json.loads(wp.with_suffix(".json").read_text())["columns"]
            wcols = [c["name"] for c in wdesc]
            for s in SPLITS:
                np.lib.format.open_memmap(out / f"{s}_wide_{key}.npy", "w+", np.float32, (len(plan[key][s]), len(wcols))).flush()
            wide_meta[key] = {"columns": wcols, "kinds": [c["kind"] for c in wdesc],
                              "sizes": [len(c.get("values", [])) + 2 if c["kind"] == "cat" else 0 for c in wdesc],
                              "offsets": {s: [int(base[key][s]), int(base[key][s]) + len(plan[key][s])] for s in SPLITS}}
        for rg_lo, rg_hi, row_offset in _row_group_ranges(pf, rows_per_task):
            tasks.append((key, rg_lo, rg_hi, row_offset, str(out), base[key]))

    # every task writes its own rows of the shared memmaps; the tasks of one source are its row
    # groups in order, so each one knows its destination from the sorted plan alone
    workers = max(1, int(workers or _default_workers()))
    streamed = {key: {s: 0 for s in SPLITS} for key in sources}
    done_t = time.time()
    if workers == 1 or len(tasks) == 1:
        results = [_stream_task(t) for t in tasks]
    else:
        results = []
        with ProcessPoolExecutor(max_workers=min(workers, len(tasks))) as pool:
            futs = {pool.submit(_stream_task, t): t for t in tasks}
            for f in as_completed(futs):
                results.append(f.result())
    read = {key: 0 for key in sources}
    for key, rows, n_rows, secs in results:
        read[key] += n_rows
        for s in SPLITS:
            streamed[key][s] += rows[s]
    for key in sources:
        print(f"  {key}: streamed {read[key]:,} rows ({time.time() - t0:.0f}s)", flush=True)
        for s in SPLITS:
            assert streamed[key][s] == len(plan[key][s]), (key, s, streamed[key][s], len(plan[key][s]))
        for s in SPLITS:
            (out / f"_plan_{key}_{s}.npy").unlink()
    print(f"  streamed {sum(counts.values()):,} rows in {len(tasks)} task(s) on {min(workers, len(tasks))} worker(s) "
          f"({time.time() - done_t:.0f}s)", flush=True)

    arrays = {s: {k: np.load(out / f"{s}_{k}.npy", mmap_mode="r+") for k in ARRAY_DTYPES} for s in SPLITS}

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
        for k in ARRAY_DTYPES:
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
    ap.add_argument("--workers", type=int, default=None, help=f"processes streaming the parquet files (default: cpus, at most {_default_workers()})")
    ap.add_argument("--rows-per-task", type=int, default=ROWS_PER_TASK, help="rows of one source per streaming task (whole row groups)")
    ap.add_argument("--include-pretrain-only", action="store_true",
                    help="v7: also take the pretraining-only sources (the generators run locally); for a set the encoder pretrains on, "
                         "never for one a model is evaluated on")
    args = ap.parse_args(argv)
    ensure_dirs()
    prepare(args.name, args.source, args.max_rows_per_source, args.seed, stats_from_run=args.stats_from_run,
            workers=args.workers, rows_per_task=args.rows_per_task, include_pretrain_only=args.include_pretrain_only)
    return 0


if __name__ == "__main__":
    sys.exit(main())
