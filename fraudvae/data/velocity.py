"""Causal per-customer (and per-merchant) rolling features, vectorised.

The trick: sort by (customer, ts) and build a monotone key
    k = customer_rank * BIG + ts
so one searchsorted over the whole array finds each row's window start without
leaving the customer's block. Beats groupby().rolling() by a couple orders of
magnitude on the 24M row tabformer file.

Everything here is strictly "past only": a row never sees itself or anything
after it, otherwise the fraud label leaks through the amount features.

v4 added the 1h burst window, the customer's own amount z-score, tenure
(prior count, days since first seen), first-time-seen flags for merchant /
category / cross-border, and two merchant-side features. Same machinery,
sorted a second time by merchant.

Memory: every feature only needs the rows of its own customer (or merchant),
so the work is done in groups of customers of ~6M rows each and scattered
into the output. The 24M row file then peaks around 3 GB instead of 6.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

HOUR = 3600.0
DAY = 86400.0
MAX_CHUNK = 3_000_000

# first-seen flags are categoricals in the schema: 0 unknown, 1 no, 2 yes
_NO, _YES = np.int8(1), np.int8(2)

CUSTOMER_FEATURES = ("secs_since_prev_txn", "cust_txn_count_24h", "cust_amount_sum_24h", "cust_txn_count_7d",
                     "cust_amount_mean_30d", "amount_over_cust_mean", "cust_txn_count_1h", "cust_amount_sum_1h",
                     "cust_amount_z_30d", "cust_txn_count_prior", "cust_days_since_first")
FLAG_FEATURES = ("cust_new_merchant", "cust_new_category", "cust_first_foreign")
MERCHANT_FEATURES = ("merch_txn_count_24h", "merch_amount_ratio")


def _codes(values) -> np.ndarray:
    """anything -> dense int32 codes, -1 for missing. factorize handles strings, ints, floats;
    an integer input is a v5 entity key column where -1 already means missing and must stay so"""
    v = np.asarray(values)
    codes, _ = pd.factorize(pd.Series(v), sort=False)
    codes = codes.astype(np.int32)
    if v.dtype.kind in "iu":
        codes[v < 0] = -1
    return codes


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
    del t
    inv = np.empty(len(ts), dtype=np.int32)
    inv[order] = np.arange(len(ts), dtype=np.int32)
    return order, inv, key


def _first_in_group(group_codes: np.ndarray, ts: np.ndarray) -> np.ndarray:
    """bool per row: is this the earliest row of its group (ties by file order)"""
    order = np.lexsort((ts, group_codes)).astype(np.int32)
    g = group_codes[order]
    first_sorted = np.empty(len(g), dtype=bool)
    first_sorted[0] = True
    np.not_equal(g[1:], g[:-1], out=first_sorted[1:])
    del g
    first = np.empty(len(order), dtype=bool)
    first[order] = first_sorted
    return first


def _block_prior(same: np.ndarray):
    """for a sorted array where same[i] says row i shares a group with i-1:
    (starts, block_id int32, prior int32) - prior = rows before i in its block"""
    starts = np.flatnonzero(~same)
    block_id = np.cumsum(~same, dtype=np.int32)
    block_id -= 1
    prior = np.arange(len(same), dtype=np.int32)
    prior -= starts[block_id].astype(np.int32)
    return starts, block_id, prior


def _customer_block(cust: np.ndarray, ts: np.ndarray, amt32: np.ndarray, merchant, category, foreign, windows) -> dict:
    """all customer-side features for one group of customers (every row of each customer present)"""
    n = len(ts)
    w24, w7, w30 = windows
    out = {}
    order, inv, key = _sorted_key(cust, ts, max(windows))
    c = cust[order]
    t = ts[order]

    same = np.empty(n, dtype=bool)
    same[0] = False
    np.equal(c[1:], c[:-1], out=same[1:])
    del c
    dt = np.full(n, np.nan)
    dt[1:] = np.where(same[1:], t[1:] - t[:-1], np.nan)
    out["secs_since_prev_txn"] = dt[inv].astype(np.float32)
    del dt
    starts, block_id, prior = _block_prior(same)
    out["cust_txn_count_prior"] = prior[inv].astype(np.float32)
    del prior
    first_t = t[starts][block_id]
    first_t -= t
    first_t /= -DAY
    out["cust_days_since_first"] = first_t[inv].astype(np.float32)
    del first_t, block_id, starts, same, t

    a_sorted = amt32[order]
    del order
    csum = np.concatenate([[0.0], np.cumsum(a_sorted, dtype=np.float64)])
    idx = np.arange(n, dtype=np.int32)
    # window start: first index with key >= key_i - w. because big > w the
    # search can't cross into another customer's block
    for w, cnt_name, sum_name in ((HOUR, "cust_txn_count_1h", "cust_amount_sum_1h"),
                                  (w24, "cust_txn_count_24h", "cust_amount_sum_24h"),
                                  (w7, "cust_txn_count_7d", None)):
        s = np.searchsorted(key, key - w, side="left").astype(np.int32)
        out[cnt_name] = (idx - s)[inv].astype(np.float32)
        if sum_name:
            out[sum_name] = (csum[idx] - csum[s])[inv].astype(np.float32)
        del s
    s = np.searchsorted(key, key - w30, side="left").astype(np.int32)
    del key
    cnt30 = idx - s
    sum30 = csum[idx] - csum[s]
    del csum
    csq = np.concatenate([[0.0], np.cumsum(a_sorted.astype(np.float64) ** 2)])
    sq30 = csq[idx] - csq[s]
    del csq, s, idx
    with np.errstate(invalid="ignore", divide="ignore"):
        denom = np.maximum(cnt30, 1)
        sum30 /= denom                      # -> mean30
        sum30[cnt30 == 0] = np.nan
        mean30 = sum30
        ratio = a_sorted / np.where(mean30 > 0, mean30, np.nan)
        out["amount_over_cust_mean"] = ratio[inv].astype(np.float32)
        del ratio
        sq30 /= denom                       # E[x^2]
        sq30 -= mean30 * mean30             # variance
        np.clip(sq30, 0, None, out=sq30)
        np.sqrt(sq30, out=sq30)             # std
        floor = np.abs(mean30)
        floor *= 0.05
        floor += 0.01                       # so a customer who always spends 9.99 doesn't get z=inf on 10.49
        np.maximum(sq30, floor, out=sq30)
        del floor, denom
        z = a_sorted - mean30
        z /= sq30
        z[cnt30 < 2] = np.nan
        np.clip(z, -20, 50, out=z)
        out["cust_amount_z_30d"] = z[inv].astype(np.float32)
        del z, sq30, cnt30
    out["cust_amount_mean_30d"] = mean30[inv].astype(np.float32)
    del mean30, a_sorted, inv

    # first-time-seen flags: first row of each (customer, thing) pair in time order
    if merchant is not None:
        m_ok = merchant >= 0
        pair = cust.astype(np.int64)
        pair *= int(merchant.max()) + 2
        pair += merchant
        pair[~m_ok] = -1
        first = _first_in_group(pair, ts)
        del pair
        out["cust_new_merchant"] = np.where(m_ok, np.where(first, _YES, _NO), 0).astype(np.int8)
        del first, m_ok
    if category is not None:
        cat = category.astype(np.int64)
        c_ok = cat > 0
        pair = cust.astype(np.int64)
        pair *= 64
        pair += cat
        pair[~c_ok] = -1
        first = _first_in_group(pair, ts)
        del pair, cat
        out["cust_new_category"] = np.where(c_ok, np.where(first, _YES, _NO), 0).astype(np.int8)
        del first, c_ok
    if foreign is not None:
        f_ok = foreign > 0
        is_f = foreign == 2
        # "first cross-border": this row is foreign and none of the customer's earlier rows were
        pair = np.where(f_ok & is_f, cust.astype(np.int64), -1)
        first = _first_in_group(pair, ts)
        del pair
        out["cust_first_foreign"] = np.where(f_ok, np.where(is_f & first, _YES, _NO), 0).astype(np.int8)
        del first, is_f, f_ok
    return out


def _merchant_block(merch: np.ndarray, ts: np.ndarray, amt32: np.ndarray, w24: float) -> dict:
    """merchant-side features for one group of merchants (every row of each merchant present)"""
    n = len(ts)
    order, inv, key = _sorted_key(merch, ts, w24)
    g = merch[order]
    same = np.empty(n, dtype=bool)
    same[0] = False
    np.equal(g[1:], g[:-1], out=same[1:])
    del g
    starts, block_id, prior = _block_prior(same)
    del same
    a_sorted = amt32[order]
    del order
    csum = np.concatenate([[0.0], np.cumsum(a_sorted, dtype=np.float64)])
    idx = np.arange(n, dtype=np.int32)
    s = np.searchsorted(key, key - w24, side="left").astype(np.int32)
    del key
    cnt24 = (idx - s).astype(np.float32)
    del s
    # expanding mean of everything the merchant saw before this row
    prior_sum = csum[idx] - csum[starts[block_id]]
    del csum, idx, starts, block_id
    with np.errstate(invalid="ignore", divide="ignore"):
        prior_sum /= np.maximum(prior, 1)
        prior_sum[prior == 0] = np.nan
        ratio = a_sorted / np.where(prior_sum > 0, prior_sum, np.nan)
    del prior, prior_sum, a_sorted
    return {"merch_txn_count_24h": cnt24[inv], "merch_amount_ratio": ratio[inv].astype(np.float32)}


def rolling_customer_features(
    customer: np.ndarray,
    ts: np.ndarray,
    amount: np.ndarray,
    merchant: np.ndarray | None = None,
    category: np.ndarray | None = None,
    foreign: np.ndarray | None = None,
    windows=(DAY, 7 * DAY, 30 * DAY),
    max_chunk: int = MAX_CHUNK,
    scratch_dir=None,
) -> dict[str, np.ndarray]:
    """merchant: any id per row (None / NaN / -1 = unknown). category: schema codes (0 = unknown).
    foreign: schema codes of is_foreign (0 unknown, 1 no, 2 yes).
    scratch_dir: put the 16 output arrays in .npy memmaps there instead of RAM (the 24M row
    source: 1.3 GB of outputs that the kernel can then page out while the parquet is written)."""
    n = len(ts)

    def alloc(name, dtype, fill):
        if scratch_dir is not None:
            a = np.lib.format.open_memmap(Path(scratch_dir) / f"{name}.npy", "w+", dtype, (n,))
        else:
            a = np.empty(n, dtype=dtype)
        a[:] = fill
        return a

    out = {}
    for k in CUSTOMER_FEATURES:
        nan_default = k in ("secs_since_prev_txn", "cust_amount_mean_30d", "amount_over_cust_mean", "cust_amount_z_30d")
        out[k] = alloc(k, np.float32, np.nan if nan_default else 0.0)
    for k in FLAG_FEATURES:
        out[k] = alloc(k, np.int8, 0)
    for k in MERCHANT_FEATURES:
        out[k] = alloc(k, np.float32, np.nan)
    if n == 0:
        return out

    cust_codes = _codes(customer)
    valid = cust_codes >= 0
    ts = np.asarray(ts, dtype=np.float64)
    amt32 = np.nan_to_num(np.asarray(amount, dtype=np.float32))
    m_codes = _codes(merchant) if merchant is not None else None
    cat8 = np.asarray(category).astype(np.int8) if category is not None else None
    fo8 = np.asarray(foreign).astype(np.int8) if foreign is not None else None
    if not valid.all():
        for k in CUSTOMER_FEATURES:
            out[k][~valid] = np.nan

    # customers in groups: a customer's rows all land in the same group, so the result is exact
    n_groups = max(1, math.ceil(n / max_chunk))
    for g in range(n_groups):
        rows = np.flatnonzero(valid) if n_groups == 1 else np.flatnonzero(valid & (cust_codes % n_groups == g))
        if len(rows) == 0:
            continue
        res = _customer_block(cust_codes[rows], ts[rows], amt32[rows],
                              m_codes[rows] if m_codes is not None else None,
                              cat8[rows] if cat8 is not None else None,
                              fo8[rows] if fo8 is not None else None, windows)
        for k, v in res.items():
            out[k][rows] = v
        del res, rows
    del cust_codes

    if m_codes is not None and (m_codes >= 0).any():
        m_ok = m_codes >= 0
        for g in range(n_groups):
            rows = np.flatnonzero(m_ok) if n_groups == 1 else np.flatnonzero(m_ok & (m_codes % n_groups == g))
            if len(rows) == 0:
                continue
            res = _merchant_block(m_codes[rows], ts[rows], amt32[rows], windows[0])
            for k, v in res.items():
                out[k][rows] = v
            del res, rows
    return out


def attach(df: pd.DataFrame, customer_col: str, ts_col: str = "ts", amount_col: str = "amount",
           merchant_col: str | None = None) -> pd.DataFrame:
    """the DataFrame flavour: category / foreign are read from the frame's own schema columns"""
    feats = rolling_customer_features(
        df[customer_col].to_numpy(), df[ts_col].to_numpy(), df[amount_col].to_numpy(),
        merchant=df[merchant_col].to_numpy() if merchant_col else None,
        category=df["merchant_category"].to_numpy() if "merchant_category" in df else None,
        foreign=df["is_foreign"].to_numpy() if "is_foreign" in df else None,
    )
    for k, v in feats.items():
        df[k] = v
    return df
