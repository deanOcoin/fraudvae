"""Fraud Detection Handbook simulated data: one pandas pickle per day."""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from .. import entities, velocity
from ..schema import cat_code, empty_frame, entity_codes, finalize
from .base import epoch_seconds, hour_dow, require

KEY = "fdh"


class _CompatUnpickler(pickle.Unpickler):
    # the pickles were written with pandas 1.x which still had Int64Index
    def find_class(self, module, name):
        if name in ("Int64Index", "Float64Index", "UInt64Index"):
            return pd.Index
        return super().find_class(module, name)


def _read_pickle(path: Path) -> pd.DataFrame:
    try:
        return pd.read_pickle(path)
    except Exception:
        with open(path, "rb") as f:
            return _CompatUnpickler(f).load()


def load(raw_dir: Path) -> pd.DataFrame:
    first = require(raw_dir, KEY, "simulated-data-raw/data/2018-04-01.pkl")
    files = sorted(first.parent.glob("*.pkl"))
    frames = [_read_pickle(p) for p in files]
    raw = pd.concat(frames, ignore_index=True)
    del frames

    n = len(raw)
    df = empty_frame(n)
    df["txn_id"] = "fdh-" + raw["TRANSACTION_ID"].astype(str)
    df["ts"] = epoch_seconds(raw["TX_DATETIME"])
    df["amount"] = raw["TX_AMOUNT"].astype(np.float32)
    df["hour"], df["dow"] = hour_dow(raw["TX_DATETIME"])
    df["is_fraud"] = raw["TX_FRAUD"].astype(np.int8)
    df["fraud_scenario"] = raw["TX_FRAUD_SCENARIO"].astype(np.int8)
    df["channel"] = cat_code("channel", "card_present")   # terminals = POS
    df["txn_type"] = cat_code("txn_type", "purchase")

    df["customer"] = entity_codes(raw["CUSTOMER_ID"].to_numpy())
    df["counterparty"] = entity_codes(raw["TERMINAL_ID"].to_numpy())   # the terminal is the closest thing to a merchant here
    df = velocity.attach(df, "customer", merchant_col="counterparty")
    df = entities.attach(df)
    return finalize(df, KEY)
