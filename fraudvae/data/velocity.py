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


def rolling_customer_features(customer, ts, amount, windows=(DAY, 7 * DAY, 30 * DAY)) -> dict[str, np.ndarray]:
    n = len(ts)
    codes, _ = pd.factorize(pd.Series(customer), sort=False)
    ts = np.asarray(ts, dtype=np.float64)
    amt = np.nan_to_num(np.asarray(amount, dtype=np.float64))
    order = np.lexsort((ts, codes))
    c = codes[order]
    t = ts[order]
    a = amt[order]
    big = (t.max() - t.min()) + max(windows) + 10.0
    key = c * big + (t - t.min())
    inv = np.empty(n, dtype=np.int64)
    inv[order] = np.arange(n)

    out = {}
    same = np.zeros(n, dtype=bool)
    same[1:] = c[1:] == c[:-1]
    dt = np.full(n, np.nan)
    dt[1:] = np.where(same[1:], t[1:] - t[:-1], np.nan)
    out["secs_since_prev_txn"] = dt[inv].astype(np.float32)

    csum = np.concatenate([[0.0], np.cumsum(a)])
    idx = np.arange(n)
    w24, w7, w30 = windows
    # window start: first index with key >= key_i - w. because big > w the
    # search can't cross into another customer's block
    s24 = np.searchsorted(key, key - w24, side="left")
    out["cust_txn_count_24h"] = (idx - s24)[inv].astype(np.float32)
    out["cust_amount_sum_24h"] = (csum[idx] - csum[s24])[inv].astype(np.float32)
    s7 = np.searchsorted(key, key - w7, side="left")
    out["cust_txn_count_7d"] = (idx - s7)[inv].astype(np.float32)
    s30 = np.searchsorted(key, key - w30, side="left")
    cnt30 = idx - s30
    with np.errstate(invalid="ignore", divide="ignore"):
        mean30 = (csum[idx] - csum[s30]) / cnt30
        ratio = a / np.where(mean30 > 0, mean30, np.nan)
    out["cust_amount_mean_30d"] = mean30[inv].astype(np.float32)
    out["amount_over_cust_mean"] = ratio[inv].astype(np.float32)
    return out
