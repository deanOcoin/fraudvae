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


def _to_tensor(a, device, dtype=None):
    a = np.ascontiguousarray(a)
    if dtype is not None:
        a = a.astype(dtype)
    return torch.from_numpy(a).to(device)


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
        self.wide_meta = wide_meta or {}
        self.wide_blocks = self._open_wide()
        # column layout of the union wide matrix: source blocks one after another
        self.wide_names, self.wide_cat_cols, self.wide_slices = [], [], {}
        for key, wm in self.wide_meta.items():
            start = len(self.wide_names)
            self.wide_names += [f"{key}.{c}" for c in wm["columns"]]
            self.wide_cat_cols += [start + j for j, k in enumerate(wm["kinds"]) if k == "cat"]
            self.wide_slices[key] = (start, start + len(wm["columns"]))

    def _open_wide(self):
        """per-source wide blocks of raw columns, always memmapped (they can be huge)"""
        blocks = {}
        for key, wm in self.wide_meta.items():
            f = self.root / f"{self.split}_wide_{key}.npy"
            if f.exists():
                blocks[key] = (np.load(f, mmap_mode="r"), wm["offsets"][self.split][0])
        return blocks

    @property
    def n_wide(self):
        return len(self.wide_names)

    def wide(self, idx) -> np.ndarray:
        """[len(idx), n_wide] float32, NaN everywhere a row's source has no wide block."""
        idx = np.asarray(idx)
        out = np.full((len(idx), self.n_wide), np.nan, dtype=np.float32)
        for key, (arr, start) in self.wide_blocks.items():
            lo, hi = start, start + len(arr)
            pos = np.flatnonzero((idx >= lo) & (idx < hi))
            if len(pos) == 0:
                continue
            a, b = self.wide_slices[key]
            out[pos, a:b] = arr[idx[pos] - lo]
        return out

    def __len__(self):
        return len(self.y)

    def legit_indices(self):
        return np.flatnonzero(self.y == 0)

    def batch(self, idx, device="cpu"):
        idx = np.asarray(idx)
        return (
            _to_tensor(self.num[idx], device),
            _to_tensor(self.mask[idx], device, np.float32),
            _to_tensor(self.cat[idx], device, np.int64),
            _to_tensor(self.src[idx], device, np.int64),
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
