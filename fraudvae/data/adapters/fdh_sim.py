"""v7: the Fraud Detection Handbook simulator, run locally. Pretraining only.

The published fdh set is this generator's own output (183 days, 5k customers); run at the volume
the encoder's pretraining wants it gives per-customer card sequences with fraud episodes over
days - the one shape of data the v6 encoder measurably helped on. Save the generator's frame
as one pickle per day (as the handbook does) anywhere under data/raw/fdh_sim/. The same columns
as fdh, through the same frame-building; never in the evaluation frame (registry: pretrain_only).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .base import MissingData, source_dir
from .fdh import _read_pickle, frame

KEY = "fdh_sim"


def load(raw_dir: Path, max_rows=None) -> pd.DataFrame:
    root = source_dir(raw_dir, KEY)
    files = sorted(root.rglob("*.pkl")) if root.exists() else []
    if not files:
        raise MissingData(f"{KEY}: no *.pkl under {root} (run the handbook's generate_dataset and save one pickle per day there)")
    frames, n = [], 0
    for p in files:
        f = _read_pickle(p)
        frames.append(f)
        n += len(f)
        if max_rows and n >= max_rows:
            break
    raw = pd.concat(frames, ignore_index=True)
    del frames
    if max_rows and len(raw) > max_rows:
        raw = raw.iloc[:max_rows].reset_index(drop=True)
    return frame(raw, KEY)
