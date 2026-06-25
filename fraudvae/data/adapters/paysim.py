"""PaySim. step = 1 hour of simulated time, 744 steps. 6.4M rows, so read
with pyarrow and keep the account ids as dictionary codes, never python
strings (that is what blew the memory limit the first time)."""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv

from .. import entities, velocity
from ..schema import cat_code, time_split_codes
from .base import chunked_frames, release_memory, require

KEY = "paysim"
pa.set_memory_pool(pa.system_memory_pool())   # see tabformer.py

_TYPES = {"CASH_IN": "cash_in", "CASH_OUT": "cash_out", "DEBIT": "debit", "PAYMENT": "payment", "TRANSFER": "transfer"}


def load(raw_dir: Path) -> Iterator[pd.DataFrame]:
    path = require(raw_dir, KEY, "PS_20174392719_1491204439457_log.csv")
    co = pacsv.ConvertOptions(
        column_types={"step": pa.int32(), "type": pa.string(), "amount": pa.float32(),
                      "oldbalanceOrg": pa.float32(), "newbalanceOrig": pa.float32(),
                      "oldbalanceDest": pa.float32(), "newbalanceDest": pa.float32(),
                      "isFraud": pa.int8(), "isFlaggedFraud": pa.int8(),
                      "nameOrig": pa.string(), "nameDest": pa.string()},
        auto_dict_encode=True,
    )
    tbl = pacsv.read_csv(path, convert_options=co, read_options=pacsv.ReadOptions(block_size=64 << 20))
    n = tbl.num_rows
    step = tbl["step"].to_numpy().astype(np.float64)
    ts = step * 3600.0
    amount = tbl["amount"].to_numpy(zero_copy_only=False)

    ttype = tbl["type"].combine_chunks()
    if pa.types.is_dictionary(ttype.type):
        ttype = ttype.dictionary_decode()
    type_lut = {t: cat_code("txn_type", _TYPES.get(t, "unknown")) for t in pc.unique(ttype).to_pylist()}
    txn_type = pd.Series(ttype.to_numpy(zero_copy_only=False)).map(type_lut).fillna(0).astype(np.int8).to_numpy()

    # customer key as integer codes, origin accounts are nearly unique anyway
    orig = tbl["nameOrig"].combine_chunks()
    orig_codes = (orig.indices if pa.types.is_dictionary(orig.type) else pc.dictionary_encode(orig).indices).to_numpy(zero_copy_only=False)
    dest = tbl["nameDest"].combine_chunks()
    if pa.types.is_dictionary(dest.type):
        dest = dest.dictionary_decode()
    dest_is_m = pc.starts_with(dest, "M").to_numpy(zero_copy_only=False)
    dest_codes = pc.dictionary_encode(dest).indices.to_numpy(zero_copy_only=False).astype(np.int32)
    cols = {
        "ts": ts,
        "hour": (step % 24).astype(np.float32),
        "amount": amount,
        "is_fraud": tbl["isFraud"].to_numpy(zero_copy_only=False),
        "channel": np.int8(cat_code("channel", "mobile_money")),
        "txn_type": txn_type,
        "balance_before": tbl["oldbalanceOrg"].to_numpy(zero_copy_only=False),
        "balance_after": tbl["newbalanceOrig"].to_numpy(zero_copy_only=False),
        "dest_balance_before": tbl["oldbalanceDest"].to_numpy(zero_copy_only=False),
        "dest_balance_after": tbl["newbalanceDest"].to_numpy(zero_copy_only=False),
        "dest_is_merchant": np.where(dest_is_m, 2, 1).astype(np.int8),
    }
    # isFlaggedFraud deliberately dropped, it's the simulator's own rule firing
    del tbl, orig, dest, ttype
    release_memory()
    cols.update(velocity.rolling_customer_features(orig_codes, ts, amount, merchant=dest_codes))
    cols.update(entities.entity_features({"cpty": dest_codes}, orig_codes, ts, cols["is_fraud"]))
    cols.update(entities.fast_label_features({"cpty": dest_codes}, orig_codes, ts, cols["is_fraud"]))
    cols["customer"] = orig_codes.astype(np.int64)
    cols["counterparty"] = dest_codes.astype(np.int64)
    del orig_codes, dest_codes
    cols["split"] = time_split_codes(ts)
    yield from chunked_frames(KEY, n, cols)
