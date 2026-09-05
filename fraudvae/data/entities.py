"""v5 entity-graph features: what the things a transaction touches (counterparty, device,
email, address, card, and the customer's partners) have been up to, computed causally.

Same machinery as velocity.py - sort by (entity, time), one monotone key, searchsorted -
but with two things v4 deliberately did not have:

  * cross-entity structure: how many new customers this device / address / email was
    seen with in the last 30 days, how many new counterparties or devices this customer
    used. Shared devices and addresses are where fraud rings show up in e-commerce data.
    Windowed on purpose: the lifetime versions ("distinct customers ever", "days since
    first seen") grow with calendar time, and on a 30-year file the booster read them as
    the era, which does not transfer to the newest months (tabformer's test AUROC went
    0.98 -> 0.93 with them and back with them gone). Ages are capped at a year for the
    same reason.
  * label-derived history with a delay: confirmed fraud among the entity's rows that are
    older than ENTITY_LABEL_DELAY_DAYS. A bank does know, at scoring time, that a
    terminal was compromised last month; what it never knows is yesterday's label. The
    delay keeps that honest. Rows with no delayed history get NaN (unobserved), not 0,
    so "no information" and "clean so far" stay different things.

  * the same label history at a second, shorter delay (ENTITY_LABEL_DELAY_FAST_DAYS, a
    week: the lag the Fraud Detection Handbook assumes), for the booster only. Written as
    wide__<entity>_fraud_{count,share}_fast so the VAE never sees it; the schema columns
    keep the conservative delay. A compromised terminal in the handbook simulator stays
    compromised for 28 days, which a 30-day-old label can never catch and a 7-day-old one
    catches for three weeks; whether a week is honest depends on how fast a deployment's
    labels arrive, which is why both are there and the receipt says which one fired.

All of it is strictly past-only: a row never sees itself or anything after it.
"""
from __future__ import annotations

import math

import numpy as np

from .schema import ENTITY_LABEL_DELAY_DAYS, ENTITY_LABEL_DELAY_FAST_DAYS, WIDE_PREFIX
from .velocity import DAY, MAX_CHUNK, _block_prior, _first_in_group, _sorted_key

ENTITY_KINDS = {"cpty": "counterparty", "device": "device", "email": "email", "addr": "address", "card": "card"}
ENTITY_FEATURES = ("txn_count_30d", "new_cust_30d", "age_days", "fraud_count_delayed", "fraud_share_delayed")
CUSTOMER_FEATURES = ("cust_new_cpty_30d", "cust_new_device_30d",
                     "cust_fraud_count_delayed", "cust_fraud_share_delayed")
AGE_CAP_DAYS = 365.0


def _pair_first(group: np.ndarray, partner: np.ndarray, ts: np.ndarray) -> np.ndarray:
    """bool per row: first time this (group, partner) pair appears, in time order.
    partner < 0 means unknown and never counts as a first"""
    ok = partner >= 0
    pair = group.astype(np.int64)
    pair *= int(partner.max()) + 2
    pair += partner
    pair[~ok] = -1
    first = _first_in_group(pair, ts)
    first &= ok
    return first


def _group_block(codes: np.ndarray, ts: np.ndarray, y: np.ndarray, partner, delay: float, w30: float,
                 label_windows: tuple = ()) -> dict:
    """every feature for one group of entities (all rows of each entity present).
    partner: codes of the thing to count distinct values of (customers for an entity,
    counterparties / devices for a customer), or None. label_windows: extra window lengths
    (seconds) over the delayed labels: fraud count / share among the entity's rows in the
    window that ends at the label horizon, named fraud_count_w<days> / fraud_share_w<days>."""
    n = len(ts)
    out = {}
    order, inv, key = _sorted_key(codes, ts, max([delay, w30] + [delay + w for w in label_windows]))
    g = codes[order]
    same = np.empty(n, dtype=bool)
    same[0] = False
    np.equal(g[1:], g[:-1], out=same[1:])
    del g
    starts, block_id, prior = _block_prior(same)
    del same
    t = ts[order]
    first_t = t[starts][block_id]
    first_t = np.minimum((t - first_t) / DAY, AGE_CAP_DAYS)
    out["age_days"] = first_t[inv].astype(np.float32)
    del first_t, t
    idx = np.arange(n, dtype=np.int32)
    block_start = starts[block_id].astype(np.int32)
    del starts

    s = np.searchsorted(key, key - w30, side="left").astype(np.int32)
    out["txn_count_30d"] = (idx - s)[inv].astype(np.float32)

    # delayed labels: rows of the same entity with ts <= t - delay
    y_sorted = y[order].astype(np.int32)
    fcum = np.concatenate([[0], np.cumsum(y_sorted, dtype=np.int64)])
    del y_sorted
    k = np.searchsorted(key, key - delay, side="right").astype(np.int32)
    cnt = (k - block_start).astype(np.float64)
    fr = (fcum[k] - fcum[block_start]).astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        share = fr / cnt
    share[cnt <= 0] = np.nan
    out["fraud_count_delayed"] = fr[inv].astype(np.float32)
    out["fraud_share_delayed"] = share[inv].astype(np.float32)
    del fr, share, cnt
    for w in label_windows:
        # rows of the entity inside (horizon - w, horizon]: the recent labelled past, so a
        # terminal compromised this month looks different from one compromised years ago
        kw = np.searchsorted(key, key - delay - w, side="right").astype(np.int32)
        kw = np.maximum(kw, block_start)
        cnt = (k - kw).astype(np.float64)
        fr = (fcum[k] - fcum[kw]).astype(np.float64)
        with np.errstate(invalid="ignore", divide="ignore"):
            share = fr / cnt
        share[cnt <= 0] = np.nan
        days = int(round(w / DAY))
        out[f"fraud_count_w{days}"] = fr[inv].astype(np.float32)
        out[f"fraud_share_w{days}"] = share[inv].astype(np.float32)
        del kw, cnt, fr, share
    del key, fcum, k

    if partner is not None:
        first = _pair_first(codes, partner, ts)
        f_sorted = first[order].astype(np.int32)
        del first
        ccum = np.cumsum(f_sorted, dtype=np.int32)
        ccum -= f_sorted                       # exclusive: pairs first seen strictly before this row
        del f_sorted
        # pairs first seen in the last 30 days (rows s .. i-1 are the entity's rows in the window)
        new30 = ccum - ccum[s]
        del ccum
        d = new30[inv].astype(np.float32)
        if (partner < 0).all():   # nothing to count distinct values of
            d[:] = np.nan
        out["new_cust_30d"] = d
    del order, inv, block_id, prior, block_start, idx, s
    return out


def _grouped(codes: np.ndarray, ts: np.ndarray, y: np.ndarray, partner, delay: float, w30: float,
             max_chunk: int, names: dict, label_windows: tuple = ()) -> dict:
    """run _group_block over chunks of entities and scatter into full-length arrays"""
    n = len(ts)
    out = {k: np.full(n, np.nan, dtype=np.float32) for k in names.values()}
    valid = codes >= 0
    if not valid.any():
        return out
    n_groups = max(1, math.ceil(int(valid.sum()) / max_chunk))
    for gi in range(n_groups):
        rows = np.flatnonzero(valid) if n_groups == 1 else np.flatnonzero(valid & (codes % n_groups == gi))
        if len(rows) == 0:
            continue
        res = _group_block(codes[rows], ts[rows], y[rows], None if partner is None else partner[rows], delay, w30,
                           label_windows=label_windows)
        for k, name in names.items():
            if k in res:
                out[name][rows] = res[k]
        del res, rows
    return out


def entity_features(entity: dict[str, np.ndarray], customer: np.ndarray, ts: np.ndarray, is_fraud: np.ndarray,
                    delay_days: float = ENTITY_LABEL_DELAY_DAYS, max_chunk: int = MAX_CHUNK) -> dict[str, np.ndarray]:
    """entity: {"cpty": codes, "device": codes, ...} (dense int codes, -1 unknown; keys from
    ENTITY_KINDS, any subset). customer: codes (-1 unknown). returns {schema feature name: array}
    for every graph-block feature, NaN where the source has no such entity."""
    n = len(ts)
    ts = np.asarray(ts, dtype=np.float64)
    y = np.asarray(is_fraud).astype(np.int8)
    cust = np.asarray(customer).astype(np.int64)
    delay = float(delay_days) * DAY
    w30 = 30 * DAY
    out = {}
    for short in ENTITY_KINDS:
        names = {f: f"{short}_{f}" for f in ENTITY_FEATURES}
        codes = entity.get(short)
        if codes is None:
            for name in names.values():
                out[name] = np.full(n, np.nan, dtype=np.float32)
            continue
        codes = np.asarray(codes).astype(np.int64)
        out.update(_grouped(codes, ts, y, cust, delay, w30, max_chunk, names))
    # the customer's side: its own delayed labels and how many partners it has had
    cust_names = {"fraud_count_delayed": "cust_fraud_count_delayed", "fraud_share_delayed": "cust_fraud_share_delayed",
                  "new_cust_30d": "cust_new_cpty_30d"}
    cp = entity.get("cpty")
    res = _grouped(cust, ts, y, None if cp is None else np.asarray(cp).astype(np.int64), delay, w30, max_chunk, cust_names)
    out.update(res)
    if cp is None:
        out["cust_new_cpty_30d"] = np.full(n, np.nan, dtype=np.float32)
    dv = entity.get("device")
    if dv is not None:
        res = _grouped(cust, ts, y, np.asarray(dv).astype(np.int64), delay, w30, max_chunk,
                       {"new_cust_30d": "cust_new_device_30d"})
        out["cust_new_device_30d"] = res["cust_new_device_30d"]
    else:
        out["cust_new_device_30d"] = np.full(n, np.nan, dtype=np.float32)
    return out


FAST_WINDOW_DAYS = (7, 30)
FAST_FEATURES = ("fraud_count_delayed", "fraud_share_delayed") + tuple(
    f"fraud_{what}_w{d}" for d in FAST_WINDOW_DAYS for what in ("count", "share"))


def fast_feature_name(short: str, feature: str) -> str:
    """wide__<kind>_fraud_count_fast, wide__<kind>_fraud_share_fast_w7, ..."""
    if feature.endswith("_delayed"):
        return f"{WIDE_PREFIX}{short}_{feature[: -len('_delayed')]}_fast"
    what, win = feature.rsplit("_w", 1)
    return f"{WIDE_PREFIX}{short}_{what}_fast_w{win}"


def fast_label_features(entity: dict[str, np.ndarray], customer: np.ndarray, ts: np.ndarray, is_fraud: np.ndarray,
                        delay_days: float = ENTITY_LABEL_DELAY_FAST_DAYS, max_chunk: int = MAX_CHUNK) -> dict[str, np.ndarray]:
    """the delayed-label pair at the short delay, lifetime and over the last 7 / 30 days before
    the label horizon, for every entity kind present and for the customer, named for the
    wide block (fast_feature_name)"""
    ts = np.asarray(ts, dtype=np.float64)
    y = np.asarray(is_fraud).astype(np.int8)
    delay = float(delay_days) * DAY
    w30 = 30 * DAY
    out = {}
    groups = {short: np.asarray(codes).astype(np.int64) for short, codes in entity.items() if codes is not None}
    groups["cust"] = np.asarray(customer).astype(np.int64)
    for short, codes in groups.items():
        names = {f: fast_feature_name(short, f) for f in FAST_FEATURES}
        out.update(_grouped(codes, ts, y, None, delay, w30, max_chunk, names, label_windows=tuple(d * DAY for d in FAST_WINDOW_DAYS)))
    return out


def attach(df, delay_days: float = ENTITY_LABEL_DELAY_DAYS, fast_delay_days: float | None = ENTITY_LABEL_DELAY_FAST_DAYS):
    """the DataFrame flavour: reads the v5 entity key columns off the frame (-1 = none),
    writes the graph block onto it, and (fast_delay_days) the booster-only short-delay
    label block as wide__ columns"""
    ent = {}
    for short, col in ENTITY_KINDS.items():
        if col in df and (df[col].to_numpy() >= 0).any():
            ent[short] = df[col].to_numpy()
    feats = entity_features(ent, df["customer"].to_numpy(), df["ts"].to_numpy(), df["is_fraud"].to_numpy(), delay_days)
    for k, v in feats.items():
        df[k] = v
    if fast_delay_days and (ent or (df["customer"].to_numpy() >= 0).any()):
        fast = fast_label_features(ent, df["customer"].to_numpy(), df["ts"].to_numpy(), df["is_fraud"].to_numpy(), fast_delay_days)
        for k, v in fast.items():
            df[k] = v
    return df
