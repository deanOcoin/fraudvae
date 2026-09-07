"""Loads a prepared set (see data/prepare.py) and hands out batches as tensors."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch

from ..paths import PREPARED_DIR


def _ram_bytes():
    try:
        total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError):
        total = 8 << 30
    # containers lie about RAM, the cgroup limit is what actually kills you
    candidates = ["/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"]
    try:
        for line in open("/proc/self/cgroup"):
            parts = line.strip().split(":")
            if len(parts) == 3 and parts[1] in ("memory", ""):
                candidates.append(f"/sys/fs/cgroup/memory{parts[2]}/memory.limit_in_bytes")
                candidates.append(f"/sys/fs/cgroup{parts[2]}/memory.max")
    except OSError:
        pass
    for p in candidates:
        try:
            v = int(open(p).read().strip())
            if 0 < v < total:
                total = v
        except (OSError, ValueError):
            pass
    return total


class PreparedSplit:
    """One split. Arrays are memmapped unless they comfortably fit in RAM."""

    def __init__(self, root: Path, split: str, in_memory: bool | None = None, wide_meta: dict | None = None):
        self.root = Path(root)
        self.split = split
        files = {k: self.root / f"{split}_{k}.npy" for k in ("num", "mask", "cat", "src", "y", "row")}
        total = sum(f.stat().st_size for f in files.values())
        if in_memory is None:
            in_memory = total < 0.25 * _ram_bytes()
        mode = None if in_memory else "r"
        self.num = np.load(files["num"], mmap_mode=mode)
        self.mask = np.load(files["mask"], mmap_mode=mode)
        self.cat = np.load(files["cat"], mmap_mode=mode)
        self.src = np.load(files["src"], mmap_mode=mode)
        self.y = np.load(files["y"], mmap_mode=mode)
        self.row = np.load(files["row"], mmap_mode=mode)
        self.in_memory = in_memory
        # v5 extras: customer keys + timestamps (sequence index), and the per-source wide blocks
        self.cust = np.load(self.root / f"{split}_cust.npy", mmap_mode="r") if (self.root / f"{split}_cust.npy").exists() else None
        self.ts = np.load(self.root / f"{split}_ts.npy", mmap_mode="r") if (self.root / f"{split}_ts.npy").exists() else None
        # v5: entity codes per row [N, 6] (schema.ENTITY_COLUMNS order, -1 none), for the graph module
        self.ent = np.load(self.root / f"{split}_ent.npy", mmap_mode="r") if (self.root / f"{split}_ent.npy").exists() else None
        self.wide_meta = wide_meta or {}
        self.wide_blocks = {}
        for key, wm in self.wide_meta.items():
            f = self.root / f"{split}_wide_{key}.npy"
            if f.exists():
                self.wide_blocks[key] = (np.load(f, mmap_mode="r"), wm["offsets"][split][0])
        # column layout of the union wide matrix: source blocks one after another
        self.wide_names, self.wide_cat_cols, self.wide_slices = [], [], {}
        for key, wm in self.wide_meta.items():
            start = len(self.wide_names)
            self.wide_names += [f"{key}.{c}" for c in wm["columns"]]
            self.wide_cat_cols += [start + j for j, k in enumerate(wm["kinds"]) if k == "cat"]
            self.wide_slices[key] = (start, start + len(wm["columns"]))

    @property
    def n_wide(self):
        return len(self.wide_names)

    def wide(self, idx) -> np.ndarray:
        """[len(idx), n_wide] float32, NaN everywhere a row's source has no wide block. idx
        in any order."""
        idx = np.asarray(idx)
        out = np.full((len(idx), self.n_wide), np.nan, dtype=np.float32)
        if not self.wide_blocks:
            return out
        for key, (arr, start) in self.wide_blocks.items():
            lo, hi = start, start + len(arr)
            pos = np.flatnonzero((idx >= lo) & (idx < hi))
            if len(pos) == 0:
                continue
            local = idx[pos] - lo
            order = np.argsort(local, kind="stable")          # sorted read, memmap friendly
            a, b = self.wide_slices[key]
            out[pos[order], a:b] = arr[local[order]]
        return out

    def __len__(self):
        return len(self.y)

    def legit_indices(self):
        return np.flatnonzero(self.y == 0)

    def batch(self, idx, device="cpu"):
        """rows come back in the order of idx. the read itself is done sorted (way kinder to a
        memmap) and un-permuted after; a caller scoring a shuffled index must get matching rows"""
        idx = np.asarray(idx)
        order = np.argsort(idx, kind="stable")
        sidx = idx[order]
        inv = np.empty_like(order)
        inv[order] = np.arange(len(order))
        return (
            torch.from_numpy(np.ascontiguousarray(self.num[sidx][inv])).to(device, non_blocking=True),
            torch.from_numpy(np.ascontiguousarray(self.mask[sidx][inv]).astype(np.float32)).to(device, non_blocking=True),
            torch.from_numpy(np.ascontiguousarray(self.cat[sidx][inv]).astype(np.int64)).to(device, non_blocking=True),
            torch.from_numpy(np.ascontiguousarray(self.src[sidx][inv]).astype(np.int64)).to(device, non_blocking=True),
        )

    def iter_batches(self, batch_size, idx=None, shuffle=False, seed=0, device="cpu"):
        idx = np.arange(len(self)) if idx is None else np.asarray(idx)
        if shuffle:
            idx = np.random.default_rng(seed).permutation(idx)
        for s in range(0, len(idx), batch_size):
            b = idx[s : s + batch_size]
            yield b, self.batch(b, device)


class PreparedSet:
    def __init__(self, name="default", root=PREPARED_DIR, splits=("train", "val", "test"), in_memory=None):
        self.path = Path(root) / name
        if not (self.path / "meta.json").exists():
            raise FileNotFoundError(f"no prepared set at {self.path}; run `fraudvae prepare`")
        self.meta = json.loads((self.path / "meta.json").read_text())
        self.splits = {s: PreparedSplit(self.path, s, in_memory, self.meta.get("wide") or {})
                       for s in splits if (self.path / f"{s}_y.npy").exists()}
        # v5: the history windows need every split a row's past can fall into, so the index
        # opens the earlier splits read-only even when the caller asked for one
        self.seq = None
        if (self.path / "seq_order.npy").exists() and "split_offsets" in self.meta:
            all_splits = dict(self.splits)
            for s in self.meta["split_offsets"]:
                if s not in all_splits and (self.path / f"{s}_y.npy").exists():
                    all_splits[s] = PreparedSplit(self.path, s, in_memory=False, wide_meta=self.meta.get("wide") or {})
            self.seq = SequenceIndex(self.path, self.meta, all_splits)

    def __getitem__(self, split):
        return self.splits[split]

    @property
    def numeric_names(self):
        return [f["name"] for f in self.meta["numeric"]]

    @property
    def categorical_names(self):
        return [c["name"] for c in self.meta["categorical"]]

    @property
    def source_names(self):
        inv = {v: k for k, v in self.meta["source_index"].items()}
        return inv


# ---- v5: customer history windows -------------------------------------------------------

class SequenceIndex:
    """Every row of every split in (customer, time) order (data/prepare.py writes it). A row's
    history is the window of seq_order just before its own position, cut at the start of its
    customer's block. The earlier rows can live in any split - they are older, that is all
    that matters - so the gather goes through global ids: train rows first, then val, then
    test (meta.split_offsets)."""

    def __init__(self, root: Path, meta: dict, splits: dict):
        self.root = Path(root)
        self.order = np.load(self.root / "seq_order.npy", mmap_mode="r")
        self.pos = np.load(self.root / "seq_pos.npy", mmap_mode="r")
        self.start = np.load(self.root / "seq_start.npy", mmap_mode="r")
        self.offsets = {k: int(v) for k, v in meta["split_offsets"].items()}
        self.counts = {k: int(v) for k, v in meta["counts"].items()}
        self.splits = splits
        self.ranges = sorted((off, off + self.counts[name], name) for name, off in self.offsets.items())

    def global_ids(self, split: str, local_idx) -> np.ndarray:
        return self.offsets[split] + np.asarray(local_idx, dtype=np.int64)

    def positions(self, split: str, local_idx) -> np.ndarray:
        """where the given rows sit in seq_order (sort rows by this before batching, so the
        windows of neighbouring rows overlap and the gathers stay local)"""
        return self.pos[self.global_ids(split, local_idx)]

    def gather_rows(self, gids: np.ndarray):
        """global ids (-1 = padding) -> num [n, D] f32, mask [n, D] f32, cat [n, C] i64, ts [n] f64.
        padding rows are zeros. sorted reads per split, which is what a memmap wants."""
        gids = np.asarray(gids, dtype=np.int64)
        n = len(gids)
        any_split = next(iter(self.splits.values()))
        D, C = any_split.num.shape[1], any_split.cat.shape[1]
        num = np.zeros((n, D), dtype=np.float32)
        mask = np.zeros((n, D), dtype=np.float32)
        cat = np.zeros((n, C), dtype=np.int64)
        ts = np.zeros(n, dtype=np.float64)
        for lo, hi, name in self.ranges:
            sp = self.splits.get(name)
            if sp is None:
                continue
            sel = np.flatnonzero((gids >= lo) & (gids < hi))
            if len(sel) == 0:
                continue
            local = gids[sel] - lo
            uniq, inv = np.unique(local, return_inverse=True)   # sorted + deduplicated read
            num[sel] = sp.num[uniq][inv]
            mask[sel] = sp.mask[uniq][inv].astype(np.float32)
            cat[sel] = sp.cat[uniq][inv].astype(np.int64)
            if sp.ts is not None:
                ts[sel] = sp.ts[uniq][inv]
        return num, mask, cat, ts

    def history(self, split: str, local_idx, window: int, device="cpu"):
        """the `window` rows before each given row, oldest first:
        (h_num [n, L, D], h_mask [n, L, D], h_cat [n, L, C], h_dt [n, L], h_valid [n, L]) as tensors.
        h_dt is log1p(seconds between that past row and the current one) / 10; h_valid is 0 where
        the customer's block ran out (or the row has no customer at all)."""
        local_idx = np.asarray(local_idx)
        g = self.global_ids(split, local_idx)
        p = self.pos[g].astype(np.int64)
        st = self.start[g].astype(np.int64)
        wpos = p[:, None] - window + np.arange(window, dtype=np.int64)[None, :]   # oldest .. newest
        valid = wpos >= st[:, None]
        wg = np.where(valid, self.order[np.clip(wpos, 0, len(self.order) - 1)], -1)
        num, mask, cat, ts = self.gather_rows(wg.ravel())
        n, L = wpos.shape
        cur_ts = self.splits[split].ts[local_idx] if self.splits[split].ts is not None else np.zeros(n)
        dt = np.where(valid, cur_ts[:, None] - ts.reshape(n, L), 0.0)
        dt = np.log1p(np.clip(dt, 0, None)) / 10.0
        return (torch.from_numpy(num.reshape(n, L, -1)).to(device, non_blocking=True),
                torch.from_numpy(mask.reshape(n, L, -1)).to(device, non_blocking=True),
                torch.from_numpy(cat.reshape(n, L, -1)).to(device, non_blocking=True),
                torch.from_numpy(dt.astype(np.float32)).to(device, non_blocking=True),
                torch.from_numpy(valid).to(device, non_blocking=True))

    def iter_batches(self, split: str, batch_size: int, window: int, idx=None, shuffle=False, seed=0, device="cpu"):
        """like PreparedSplit.iter_batches but with history windows: rows are batched in
        sequence order (neighbours share most of their window, so the gather is cheap) and
        it is the order of the batches that gets shuffled. yields (idx, (num, mask, cat, src), hist)."""
        sp = self.splits[split]
        idx = np.arange(len(sp)) if idx is None else np.asarray(idx)
        idx = idx[np.argsort(self.positions(split, idx), kind="stable")]
        starts = np.arange(0, len(idx), batch_size)
        if shuffle:
            starts = np.random.default_rng(seed).permutation(starts)

        def make(s):
            b = idx[s : s + batch_size]
            return b, sp.batch(b, device), self.history(split, b, window, device)

        if len(starts) < 4:
            for s in starts:
                yield make(s)
            return
        # the window gather is numpy work on the cpu; a worker thread keeps it one or two
        # batches ahead of the model, which on a gpu is most of the epoch time otherwise
        import queue
        import threading

        q: queue.Queue = queue.Queue(maxsize=3)
        stop = object()

        def worker():
            try:
                for s in starts:
                    q.put(make(s))
            except Exception as e:  # noqa: BLE001 - surface in the consumer
                q.put(e)
            q.put(stop)

        t = threading.Thread(target=worker, daemon=True)
        t.start()
        while True:
            item = q.get()
            if item is stop:
                break
            if isinstance(item, Exception):
                raise item
            yield item
