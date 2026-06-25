"""ULB creditcard.csv. Time is seconds since the first transaction of the two day
window, we assume the window started at midnight (that's what everyone does, the
paper doesn't say). V1..V28 are PCA components, kept in their own block."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..schema import cat_code, empty_frame, finalize
from .base import require

KEY = "ulb"


def load(raw_dir: Path) -> pd.DataFrame:
    path = require(raw_dir, KEY, "creditcard.csv")
    raw = pd.read_csv(path, dtype=np.float64)
    n = len(raw)
    df = empty_frame(n)
    df["txn_id"] = "ulb-" + pd.Series(np.arange(n)).astype(str)
    t = raw["Time"].to_numpy()
    df["ts"] = t
    df["hour"] = ((t // 3600) % 24).astype(np.float32)
    df["amount"] = raw["Amount"].astype(np.float32)
    df["is_fraud"] = raw["Class"].astype(np.int8)
    df["txn_type"] = cat_code("txn_type", "purchase")
    for i in range(1, 29):
        df[f"ulb_v{i}"] = raw[f"V{i}"].astype(np.float32)
    # no customer ids -> no velocity features for this one
    return finalize(df, KEY)
