"""BankSim. Values come single-quoted ('C1093826151'), step is a day."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .. import entities, velocity
from ..schema import cat_code, empty_frame, entity_codes, finalize
from ..taxonomy import BANKSIM_CATEGORY, encode_series
from .base import require

KEY = "banksim"

# bucket -> rough midpoint in years
_AGE = {"0": 16.0, "1": 22.0, "2": 30.0, "3": 40.0, "4": 50.0, "5": 60.0, "6": 72.0}


def _unquote(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip("'")


def load(raw_dir: Path) -> pd.DataFrame:
    path = require(raw_dir, KEY, "bs140513_032310.csv")
    raw = pd.read_csv(path)
    for c in ("customer", "age", "gender", "zipcodeOri", "merchant", "zipMerchant", "category"):
        raw[c] = _unquote(raw[c])

    n = len(raw)
    df = empty_frame(n)
    df["txn_id"] = "banksim-" + pd.Series(np.arange(n)).astype(str)
    df["ts"] = raw["step"].to_numpy().astype(np.float64) * 86400.0
    df["amount"] = raw["amount"].astype(np.float32)
    df["is_fraud"] = raw["fraud"].astype(np.int8)
    df["merchant_category"] = encode_series("merchant_category", raw["category"], BANKSIM_CATEGORY)
    df["txn_type"] = cat_code("txn_type", "purchase")
    df["cust_age"] = raw["age"].map(_AGE).astype(np.float32)
    df["cust_gender"] = encode_series("cust_gender", raw["gender"])   # 'E' and 'U' -> unknown

    df["customer"] = entity_codes(raw["customer"].to_numpy())
    df["counterparty"] = entity_codes(raw["merchant"].to_numpy())
    df = velocity.attach(df, "customer", merchant_col="counterparty")
    df = entities.attach(df)
    return finalize(df, KEY)
