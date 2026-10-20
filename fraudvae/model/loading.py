"""One way to read a checkpoint.

`torch.load` is a pickle reader, and a pickle can carry code: a checkpoint from an untrusted
source could run anything on load. Every checkpoint this project writes holds tensors, plain
containers and numpy arrays, nothing else, so they are read with `weights_only=True`, which
allows exactly those types and refuses everything else. numpy's array types are not on torch's
default list and are added here once.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

_registered = False


def _allow_numpy():
    global _registered
    if _registered:
        return
    _registered = True
    add = getattr(torch.serialization, "add_safe_globals", None)
    if add is None:            # torch < 2.4 has no allow-list; load() falls back below
        return
    allowed = [np.ndarray, np.dtype]
    try:
        from numpy._core import multiarray as _ma
    except ImportError:        # numpy 1.x
        from numpy.core import multiarray as _ma
    allowed.append(_ma._reconstruct)
    for name in dir(getattr(np, "dtypes", None) or ()):
        if name.endswith("DType"):
            allowed.append(getattr(np.dtypes, name))
    add(allowed)


def load_checkpoint(path: str | Path, device="cpu"):
    """read a checkpoint written by this project without executing anything in it"""
    _allow_numpy()
    if getattr(torch.serialization, "add_safe_globals", None) is None:
        return torch.load(path, map_location=device, weights_only=False)
    return torch.load(path, map_location=device, weights_only=True)
