"""the entity-graph features against a brute-force definition: every number is computed from
rows strictly before the row's own time (or before its label horizon), and a label that
arrives later than the delay never reaches an earlier row"""
import numpy as np

from fraudvae.data.entities import AGE_CAP_DAYS, entity_features, fast_feature_name, fast_label_features
from fraudvae.data.velocity import DAY


def _data(n=600, seed=3):
    rng = np.random.default_rng(seed)
    ts = np.sort(rng.uniform(0, 400 * DAY, n))
    ts += np.arange(n) * 1e-3                       # distinct times, so "before" is unambiguous
    device = rng.integers(0, 6, n).astype(np.int64)
    device[rng.random(n) < 0.1] = -1                # some rows with no device
    cust = rng.integers(0, 25, n).astype(np.int64)
    y = (rng.random(n) < 0.08).astype(np.int8)
    return ts, device, cust, y


def _brute_entity(codes, ts, y, partner, i, delay, w30):
    e = codes[i]
    if e < 0:
        return None
    same = np.flatnonzero(codes == e)
    before = same[ts[same] < ts[i]]
    labelled = same[ts[same] <= ts[i] - delay]
    recent = before[ts[before] >= ts[i] - w30]
    first_ts = ts[same].min()
    out = {"age_days": min((ts[i] - first_ts) / DAY, AGE_CAP_DAYS), "txn_count_30d": len(recent),
           "fraud_count_delayed": float(y[labelled].sum()),
           "fraud_share_delayed": (y[labelled].sum() / len(labelled)) if len(labelled) else np.nan}
    # pairs (entity, partner) whose first row falls in the last 30 days before this row
    seen = set()
    firsts = []
    for j in same[np.argsort(ts[same])]:
        key = partner[j]
        if key >= 0 and key not in seen:
            seen.add(key)
            firsts.append(j)
    out["new_cust_30d"] = sum(1 for j in firsts if ts[i] - w30 <= ts[j] < ts[i])
    return out


def test_entity_features_match_brute_force():
    ts, device, cust, y = _data()
    delay_days = 30
    f = entity_features({"device": device}, cust, ts, y, delay_days=delay_days)
    delay, w30 = delay_days * DAY, 30 * DAY
    for i in range(len(ts)):
        ref = _brute_entity(device, ts, y, cust, i, delay, w30)
        if ref is None:
            assert all(np.isnan(f[f"device_{k}"][i]) for k in ("age_days", "txn_count_30d", "fraud_count_delayed", "new_cust_30d"))
            continue
        for k, v in ref.items():
            got = f[f"device_{k}"][i]
            assert (np.isnan(v) and np.isnan(got)) or np.isclose(got, v, atol=1e-5), (i, k, got, v)


def test_fast_label_windows_match_brute_force():
    ts, device, cust, y = _data(seed=5)
    delay = 7 * DAY
    f = fast_label_features({"device": device}, cust, ts, y, delay_days=7)
    for i in range(len(ts)):
        e = device[i]
        if e < 0:
            continue
        same = np.flatnonzero(device == e)
        horizon = ts[i] - delay
        labelled = same[ts[same] <= horizon]
        assert np.isclose(f[fast_feature_name("device", "fraud_count_delayed")][i], y[labelled].sum())
        for days in (7, 30):
            win = same[(ts[same] <= horizon) & (ts[same] > horizon - days * DAY)]
            got = f[fast_feature_name("device", f"fraud_count_w{days}")][i]
            share = f[fast_feature_name("device", f"fraud_share_w{days}")][i]
            assert np.isclose(got, y[win].sum()), (i, days, got, y[win].sum())
            assert (np.isnan(share) and len(win) == 0) or np.isclose(share, y[win].sum() / len(win))


def test_a_late_label_never_reaches_an_earlier_row():
    ts, device, cust, y = _data(seed=7)
    base = entity_features({"device": device}, cust, ts, y, delay_days=30)
    y2 = y.copy()
    last = np.argmax(ts)
    y2[last] = 1 - y2[last]                          # the newest row's label flips
    changed = entity_features({"device": device}, cust, ts, y2, delay_days=30)
    for k in base:
        a, b = base[k][:last], changed[k][:last]
        assert np.array_equal(np.nan_to_num(a, nan=-9), np.nan_to_num(b, nan=-9)), k
    # a row's own label is never in its own features: flipping it leaves its row unchanged
    for name in ("device_fraud_count_delayed", "cust_fraud_count_delayed"):
        assert np.nan_to_num(base[name][last], nan=-9) == np.nan_to_num(changed[name][last], nan=-9)


def test_rows_with_no_entity_are_unobserved():
    ts, device, cust, y = _data(n=50)
    f = entity_features({}, cust, ts, y)
    assert all(np.isnan(f[k]).all() for k in f if k.startswith("device_") or k.startswith("cpty_"))
    assert np.isnan(f["cust_new_cpty_30d"]).all() and np.isnan(f["cust_new_device_30d"]).all()
    assert not np.isnan(f["cust_fraud_count_delayed"]).all()
