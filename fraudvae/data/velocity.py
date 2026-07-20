"""Causal per-customer rolling features, vectorised.

The trick: sort by (customer, ts) and build a monotone key
    k = customer_rank * BIG + ts
so one searchsorted over the whole array finds each row's window start without
leaving the customer's block. Beats groupby().rolling() by a couple orders of
magnitude on the 24M row tabformer file.

Everything here is strictly "past only": a row never sees itself or anything
after it, otherwise the fraud label leaks through the amount features.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DAY = 86400.0

CUSTOMER_FEATURES = ("secs_since_prev_txn", "cust_txn_count_24h", "cust_amount_sum_24h", "cust_txn_count_7d",
                     "cust_amount_mean_30d", "amount_over_cust_mean")


def _codes(values) -> np.ndarray:
    """anything -> dense int32 codes, -1 for missing. factorize handles strings, ints, floats"""
    codes, _ = pd.factorize(pd.Series(np.asarray(values)), sort=False)
    return codes.astype(np.int32)


def _sorted_key(codes: np.ndarray, ts: np.ndarray, max_window: float):
    """(order, inverse, key) for a (group, time) sort. key is monotone within each group block
    and jumps by more than any window between blocks, so searchsorted never crosses a group."""
    order = np.lexsort((ts, codes)).astype(np.int32)
    t = ts[order]
    tmin = float(np.nanmin(t))
    span = float(np.nanmax(t) - tmin)
    big = span + max_window + 10.0
    key = codes[order].astype(np.float64)
    key *= big
    key += t
    key -= tmin
    inv = np.empty(len(ts), dtype=np.int32)
    inv[order] = np.arange(len(ts), dtype=np.int32)
    return order, inv, key


def rolling_customer_features(
    customer: np.ndarray,
    ts: np.ndarray,
    amount: np.ndarray,
    windows=(DAY, 7 * DAY, 30 * DAY),
) -> dict[str, np.ndarray]:
    """customer: any id per row (None / NaN = unknown, those rows get NaN features)"""
    n = len(ts)
    out = {k: np.full(n, np.nan, dtype=np.float32) for k in CUSTOMER_FEATURES}
    if n == 0:
        return out

    cust_codes = _codes(customer)
    rows = np.flatnonzero(cust_codes >= 0)
    if len(rows) == 0:
        return out
    cust = cust_codes[rows]
    ts = np.asarray(ts, dtype=np.float64)[rows]
    amt32 = np.nan_to_num(np.asarray(amount, dtype=np.float32))[rows]
    m = len(rows)
    w24, w7, w30 = windows

    order, inv, key = _sorted_key(cust, ts, max(windows))
    c = cust[order]
    t = ts[order]
    same = np.empty(m, dtype=bool)
    same[0] = False
    np.equal(c[1:], c[:-1], out=same[1:])
    dt = np.full(m, np.nan)
    dt[1:] = np.where(same[1:], t[1:] - t[:-1], np.nan)
    out["secs_since_prev_txn"][rows] = dt[inv]
    del dt, same, c, t

    a_sorted = amt32[order]
    csum = np.concatenate([[0.0], np.cumsum(a_sorted, dtype=np.float64)])
    idx = np.arange(m, dtype=np.int32)
    # window start: first index with key >= key_i - w. because big > w the
    # search can't cross into another customer's block
    for w, cnt_name, sum_name in ((w24, "cust_txn_count_24h", "cust_amount_sum_24h"),
                                  (w7, "cust_txn_count_7d", None)):
        s = np.searchsorted(key, key - w, side="left").astype(np.int32)
        out[cnt_name][rows] = (idx - s)[inv]
        if sum_name:
            out[sum_name][rows] = (csum[idx] - csum[s])[inv]
    s = np.searchsorted(key, key - w30, side="left").astype(np.int32)
    cnt30 = idx - s
    sum30 = csum[idx] - csum[s]
    with np.errstate(invalid="ignore", divide="ignore"):
        mean30 = sum30 / np.maximum(cnt30, 1)
        mean30[cnt30 == 0] = np.nan
        ratio = a_sorted / np.where(mean30 > 0, mean30, np.nan)
    out["cust_amount_mean_30d"][rows] = mean30[inv]
    out["amount_over_cust_mean"][rows] = ratio[inv]
    return out


def attach(df: pd.DataFrame, customer_col: str, ts_col: str = "ts", amount_col: str = "amount") -> pd.DataFrame:
    """the DataFrame flavour"""
    feats = rolling_customer_features(
        df[customer_col].to_numpy(), df[ts_col].to_numpy(), df[amount_col].to_numpy(),
    )
    for k, v in feats.items():
        df[k] = v
    return df
