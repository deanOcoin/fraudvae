"""IBM AMLworld (kaggle ealtman2019/ibm-transactions-for-anti-money-laundering-aml), HI-Small.

Account-to-account transfers between banks, 5M rows over 18 days in sept 2022.
The label is "is laundering", which is a different animal from card fraud: the
flagged rows are the legs of fan-in / fan-out / cycle patterns and mostly look
ordinary on their own. It's in the corpus for diversity (another kind of bank,
another kind of "normal") rather than for its per-row signal.

The customer is the paying account, the "merchant" is the receiving account.
Amounts are in 13 currencies; they're converted to USD with fixed rough 2022
rates so one amount column makes sense across rows. Nothing here is a real
exchange rate and nothing depends on it being one.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv

from .. import entities, velocity
from ..schema import cat_code, time_split_codes
from .base import chunked_frames, release_memory, require, source_dir

KEY = "amlworld"
CHUNK = 1_000_000

_COLS = ["Timestamp", "From Bank", "From Account", "To Bank", "To Account", "Amount Received",
         "Receiving Currency", "Amount Paid", "Payment Currency", "Payment Format", "Is Laundering"]

# rough sept-2022 rates to USD. synthetic data, the point is a common scale not accuracy
_USD = {
    "US Dollar": 1.0, "Euro": 1.0, "UK Pound": 1.15, "Swiss Franc": 1.03, "Canadian Dollar": 0.76,
    "Australian Dollar": 0.67, "Yen": 0.007, "Yuan": 0.14, "Rupee": 0.0125, "Ruble": 0.017,
    "Mexican Peso": 0.05, "Brazil Real": 0.19, "Saudi Riyal": 0.27, "Shekel": 0.29, "Bitcoin": 20000.0,
}
_FORMAT = {"ACH": "ach", "Wire": "wire", "Cheque": "cheque", "Cash": "cash", "Credit Card": "credit_card",
           "Reinvestment": "reinvestment", "Bitcoin": "crypto"}
_TXN_TYPE = {"ACH": "transfer", "Wire": "transfer", "Cheque": "payment", "Cash": "cash_out", "Credit Card": "payment",
             "Reinvestment": "transfer", "Bitcoin": "transfer"}
PATTERNS = ["FAN-OUT", "FAN-IN", "GATHER-SCATTER", "SCATTER-GATHER", "CYCLE", "BIPARTITE", "STACK", "RANDOM"]


def _read(path: Path) -> pa.Table:
    # the header has two columns called "Account"; name them ourselves
    ro = pacsv.ReadOptions(column_names=_COLS, skip_rows=1, block_size=64 << 20)
    co = pacsv.ConvertOptions(
        column_types={"Timestamp": pa.string(), "From Bank": pa.int32(), "From Account": pa.string(),
                      "To Bank": pa.int32(), "To Account": pa.string(), "Amount Received": pa.float64(),
                      "Receiving Currency": pa.string(), "Amount Paid": pa.float64(), "Payment Currency": pa.string(),
                      "Payment Format": pa.string(), "Is Laundering": pa.int8()},
        auto_dict_encode=True,
    )
    return pacsv.read_csv(path, read_options=ro, convert_options=co)


def _lut(col: pa.ChunkedArray, mapping: dict, default, dtype):
    """dictionary column -> numpy via a lookup on the dictionary values (no python per row)"""
    col = col.combine_chunks()
    if not pa.types.is_dictionary(col.type):
        col = pc.dictionary_encode(col)
    vals = [mapping.get(v, default) for v in col.dictionary.to_pylist()]
    lut = np.asarray(vals + [default], dtype=dtype)
    idx = col.indices.to_numpy(zero_copy_only=False)
    idx = np.where(col.is_null().to_numpy(zero_copy_only=False), len(vals), idx) if col.null_count else idx
    return lut[idx]


def _scenarios(raw_dir: Path, keys: np.ndarray) -> np.ndarray:
    """laundering pattern per row from HI-Small_Patterns.txt, 0 where none / file missing.
    keys = the csv line minus the label, which is how the patterns file identifies rows"""
    root = source_dir(raw_dir, KEY)
    p = next(iter(root.rglob("HI-Small_Patterns.txt")), None)
    out = np.zeros(len(keys), dtype=np.int8)
    if p is None:
        return out
    lines, codes, cur = [], [], 0
    with open(p) as f:
        for line in f:
            line = line.strip()
            if line.startswith("BEGIN LAUNDERING ATTEMPT"):
                m = re.match(r"BEGIN LAUNDERING ATTEMPT\s*-\s*([A-Z\- ]+?)\s*:", line)
                name = m.group(1).strip() if m else ""
                cur = PATTERNS.index(name) + 1 if name in PATTERNS else 8
            elif line.startswith("END LAUNDERING ATTEMPT") or not line:
                cur = 0
            elif cur:
                lines.append(line.rsplit(",", 1)[0])
                codes.append(cur)
    if not lines:
        return out
    m = pd.Series(codes, index=pd.Index(lines)).groupby(level=0).first()
    hit = pd.Series(keys).map(m)
    return hit.fillna(0).to_numpy().astype(np.int8)


def load(raw_dir: Path, max_rows: int | None = None) -> Iterator[pd.DataFrame]:
    path = require(raw_dir, KEY, "HI-Small_Trans.csv|HI-Small_Trans.csv.gz")
    tbl = _read(path)
    if max_rows:
        tbl = tbl.slice(0, max_rows)
    n = tbl.num_rows

    when = pc.strptime(tbl["Timestamp"], format="%Y/%m/%d %H:%M", unit="s")
    ts = when.cast(pa.int64()).to_numpy(zero_copy_only=False).astype(np.float64)
    hour = pc.hour(when).to_numpy(zero_copy_only=False).astype(np.float32)
    dow = pc.day_of_week(when).to_numpy(zero_copy_only=False).astype(np.float32)   # monday = 0, like pandas
    del when

    rate = _lut(tbl["Payment Currency"], _USD, np.nan, np.float32)
    amount = (tbl["Amount Paid"].to_numpy(zero_copy_only=False).astype(np.float32) * rate).astype(np.float32)
    del rate
    is_fraud = tbl["Is Laundering"].to_numpy(zero_copy_only=False).astype(np.int8)
    payment_format = _lut(tbl["Payment Format"], {k: cat_code("payment_format", v) for k, v in _FORMAT.items()}, 0, np.int8)
    txn_type = _lut(tbl["Payment Format"], {k: cat_code("txn_type", v) for k, v in _TXN_TYPE.items()}, 0, np.int8)
    same_ccy = pc.equal(tbl["Payment Currency"].cast(pa.string()), tbl["Receiving Currency"].cast(pa.string()))
    is_foreign = np.where(same_ccy.to_numpy(zero_copy_only=False), cat_code("is_foreign", "no"), cat_code("is_foreign", "yes")).astype(np.int8)
    del same_ccy

    # payer / payee keys: bank + account, as dense codes
    payer = pc.binary_join_element_wise(pc.cast(tbl["From Bank"], pa.string()), tbl["From Account"].cast(pa.string()), "-")
    payee = pc.binary_join_element_wise(pc.cast(tbl["To Bank"], pa.string()), tbl["To Account"].cast(pa.string()), "-")
    payer_codes = pc.dictionary_encode(payer).combine_chunks().indices.to_numpy(zero_copy_only=False).astype(np.int32)
    payee_codes = pc.dictionary_encode(payee).combine_chunks().indices.to_numpy(zero_copy_only=False).astype(np.int32)
    del payer, payee

    # the patterns file quotes csv lines verbatim (minus the label). re-read the flagged
    # lines as plain text rather than trying to re-format floats the way the file did
    scen = np.zeros(n, dtype=np.int8)
    flagged = np.flatnonzero(is_fraud == 1)
    if len(flagged):
        lines = pacsv.read_csv(path, read_options=pacsv.ReadOptions(column_names=["line"], skip_rows=1, block_size=64 << 20),
                               parse_options=pacsv.ParseOptions(delimiter="\x01", quote_char=False))
        if max_rows:
            lines = lines.slice(0, max_rows)
        flagged_lines = lines["line"].take(pa.array(flagged)).to_pylist()
        del lines
        keys = np.array([ln.rsplit(",", 1)[0] for ln in flagged_lines], dtype=object)
        scen[flagged] = _scenarios(raw_dir, keys)
        del flagged_lines, keys
    del tbl
    release_memory()

    vel = velocity.rolling_customer_features(payer_codes, ts, amount, merchant=payee_codes, foreign=is_foreign)
    graph = entities.entity_features({"cpty": payee_codes}, payer_codes, ts, is_fraud)
    graph.update(entities.fast_label_features({"cpty": payee_codes}, payer_codes, ts, is_fraud))
    split = time_split_codes(ts)
    cols = {"ts": ts, "amount": amount, "hour": hour, "dow": dow, "is_fraud": is_fraud, "fraud_scenario": scen,
            "txn_type": txn_type, "payment_format": payment_format, "is_foreign": is_foreign, "split": split,
            "customer": payer_codes.astype(np.int64), "counterparty": payee_codes.astype(np.int64)}
    del payer_codes, payee_codes
    cols.update(vel)
    cols.update(graph)
    yield from chunked_frames(KEY, n, cols, chunk=CHUNK)
