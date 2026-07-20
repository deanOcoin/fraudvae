import numpy as np

from fraudvae.data.velocity import rolling_customer_features


def brute(cust, ts, amt, i, merch=None, cat=None, foreign=None):
    past = (cust == cust[i]) & (ts < ts[i])
    w1 = past & (ts >= ts[i] - 3600)
    w24 = past & (ts >= ts[i] - 86400)
    w7 = past & (ts >= ts[i] - 7 * 86400)
    w30 = past & (ts >= ts[i] - 30 * 86400)
    out = {
        "cust_txn_count_24h": w24.sum(), "cust_amount_sum_24h": amt[w24].sum(), "cust_txn_count_7d": w7.sum(),
        "cust_amount_mean_30d": amt[w30].mean() if w30.any() else np.nan,
        "amount_over_cust_mean": amt[i] / amt[w30].mean() if w30.any() else np.nan,
        "secs_since_prev_txn": ts[i] - ts[past].max() if past.any() else np.nan,
        "cust_txn_count_1h": w1.sum(), "cust_amount_sum_1h": amt[w1].sum(),
        "cust_txn_count_prior": past.sum(),
        "cust_days_since_first": (ts[i] - ts[past].min()) / 86400 if past.any() else 0.0,
    }
    if w30.sum() >= 2:
        m, sd = amt[w30].mean(), amt[w30].std()
        out["cust_amount_z_30d"] = np.clip((amt[i] - m) / max(sd, 0.05 * abs(m) + 0.01), -20, 50)
    else:
        out["cust_amount_z_30d"] = np.nan
    if merch is not None:
        out["cust_new_merchant"] = 2 if not (past & (merch == merch[i])).any() else 1
        mpast = (merch == merch[i]) & (ts < ts[i])
        out["merch_txn_count_24h"] = (mpast & (ts >= ts[i] - 86400)).sum()
        out["merch_amount_ratio"] = amt[i] / amt[mpast].mean() if mpast.any() else np.nan
    if cat is not None:
        out["cust_new_category"] = 2 if not (past & (cat == cat[i])).any() else 1
    if foreign is not None:
        if foreign[i] == 2:
            out["cust_first_foreign"] = 2 if not (past & (foreign == 2)).any() else 1
        else:
            out["cust_first_foreign"] = 1
    return out


def _check(f, exp, i):
    for k, v in exp.items():
        got = f[k][i]
        if isinstance(v, float) and np.isnan(v):
            assert np.isnan(got), (k, i)
        else:
            assert np.isclose(got, v, rtol=1e-4, atol=1e-2), (k, i, got, v)


def test_matches_brute_force():
    rng = np.random.default_rng(1)
    n = 2500
    cust = rng.integers(0, 30, n)
    ts = rng.uniform(0, 60 * 86400, n)
    amt = rng.exponential(50, n)
    f = rolling_customer_features(cust, ts, amt)
    for i in rng.choice(n, 300, replace=False):
        _check(f, brute(cust, ts, amt, i), i)


def test_merchant_category_foreign_flags_match_brute_force():
    rng = np.random.default_rng(2)
    n = 3000
    cust = rng.integers(0, 25, n)
    ts = np.round(rng.uniform(0, 40 * 86400, n))
    amt = rng.exponential(30, n)
    merch = rng.integers(0, 40, n)
    cat = rng.integers(1, 6, n)
    foreign = rng.choice([1, 1, 1, 2], n)
    f = rolling_customer_features(cust, ts, amt, merchant=merch, category=cat, foreign=foreign)
    for i in rng.choice(n, 300, replace=False):
        _check(f, brute(cust, ts, amt, i, merch, cat, foreign), i)
    # unknown merchant / category / foreign -> flag 0 and merchant features NaN
    merch2 = merch.astype(object)
    merch2[:50] = None
    cat2 = cat.copy(); cat2[:50] = 0
    fo2 = foreign.copy(); fo2[:50] = 0
    g = rolling_customer_features(cust, ts, amt, merchant=merch2, category=cat2, foreign=fo2)
    assert (g["cust_new_merchant"][:50] == 0).all() and (g["cust_new_category"][:50] == 0).all()
    assert (g["cust_first_foreign"][:50] == 0).all()
    assert np.isnan(g["merch_txn_count_24h"][:50]).all()
    assert (g["cust_new_merchant"][50:] > 0).all()


def test_no_leakage_from_future():
    # a customer with one huge transaction at the end must not affect earlier rows
    cust = np.zeros(5, dtype=int)
    ts = np.array([0, 100, 200, 300, 400], dtype=float)
    amt = np.array([10, 10, 10, 10, 1e6])
    f = rolling_customer_features(cust, ts, amt)
    assert f["cust_amount_sum_24h"][3] == 30
    assert np.isclose(f["cust_amount_mean_30d"][3], 10)
    assert f["cust_txn_count_24h"][0] == 0 and np.isnan(f["secs_since_prev_txn"][0])
    assert f["cust_txn_count_prior"].tolist() == [0, 1, 2, 3, 4]
    assert np.isnan(f["cust_amount_z_30d"][1]) and np.isclose(f["cust_amount_z_30d"][2], 0.0)


def test_customers_do_not_mix():
    cust = np.array([1, 2, 1, 2])
    ts = np.array([0, 1, 2, 3], dtype=float)
    amt = np.array([5.0, 7.0, 5.0, 7.0])
    f = rolling_customer_features(cust, ts, amt)
    assert f["cust_txn_count_24h"].tolist() == [0, 0, 1, 1]
    assert f["cust_amount_sum_24h"].tolist() == [0, 0, 5, 7]


def test_string_customer_ids_and_empty():
    f = rolling_customer_features(np.array(["a", "b", "a"]), np.array([0.0, 1.0, 2.0]), np.array([1.0, 2.0, 3.0]))
    assert f["cust_txn_count_7d"].tolist() == [0, 0, 1]
    e = rolling_customer_features(np.array([]), np.array([]), np.array([]))
    assert len(e["cust_txn_count_24h"]) == 0
