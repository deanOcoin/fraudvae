"""Loads a prepared set (see data/prepare.py) and hands out batches as tensors."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from ..paths import PREPARED_DIR

ARRAYS = ("num", "mask", "cat", "src", "y", "row")


def _ram_bytes():
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return 8 << 30


def _to_tensor(a, device, dtype=None):
    a = np.ascontiguousarray(a)
    if dtype is not None:
        a = a.astype(dtype)
    return torch.from_numpy(a).to(device)


class PreparedSplit:
    """One split. Arrays are memmapped unless they comfortably fit in RAM."""

    def __init__(self, root, split, in_memory=None):
        self.root = Path(root)
        self.split = split
        files = {k: self.root / f"{split}_{k}.npy" for k in ARRAYS}
        total = sum(f.stat().st_size for f in files.values())
        if in_memory is None:
            in_memory = total < 0.5 * _ram_bytes()
        mode = None if in_memory else "r"
        for k, f in files.items():
            setattr(self, k, np.load(f, mmap_mode=mode))
        self.in_memory = in_memory

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
        self.splits = {s: PreparedSplit(self.path, s, in_memory) for s in splits if (self.path / f"{s}_y.npy").exists()}

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
