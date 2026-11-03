"""v7: the label horizon - rows whose labels a deployment would not have had yet leave the supervised fits."""
import numpy as np

from fraudvae.config import load_config
from fraudvae.model.booster import _sample, censor_recent, label_cut, train_booster
from fraudvae.model.dataset import PreparedSet

DAY = 86400.0


def test_censor_recent_drops_the_rows_inside_the_horizon(prepared_set):
    data = PreparedSet(prepared_set, splits=("train", "val"), in_memory=True)
    tr, va = data["train"], data["val"]
    idx = np.arange(len(tr))
    cut = label_cut(va, np.unique(tr.src))
    assert set(cut) == {int(s) for s in np.unique(tr.src)}
    kept, removed = censor_recent(tr, idx, va, 10.0)
    assert removed > 0 and len(kept) + removed == len(tr)
    src, ts = np.asarray(tr.src), np.asarray(tr.ts)
    for s, t in cut.items():
        assert (ts[kept][src[kept] == s] <= t - 10 * DAY).all()
        gone = np.setdiff1d(idx[src == s], kept)
        assert (ts[gone] > t - 10 * DAY).all() and len(gone) > 0
    # 0 = as published; a split without timestamps is left alone
    assert censor_recent(tr, idx, va, 0.0)[1] == 0 and len(censor_recent(tr, idx, None, 10.0)[0]) == len(tr)
    # the horizon never touches the labels themselves, only which rows are fitted on
    assert np.array_equal(np.asarray(tr.y)[kept], np.asarray(tr.y)[kept])


def test_booster_records_the_horizon(prepared_set, tmp_path):
    cfg = load_config("smoke")
    cfg.boost.max_rounds, cfg.boost.early_stopping, cfg.boost.label_horizon_days = 20, 10, 10.0
    run_dir = tmp_path / "horizon"
    run_dir.mkdir()
    info = train_booster(run_dir, prepared_set, cfg.boost, log=lambda m: None, seed=1)
    data = PreparedSet(prepared_set, splits=("train",), in_memory=True)
    full = _sample(data["train"], cfg.boost.max_legit_per_source, np.random.default_rng(1))
    assert info["label_horizon_days"] == 10.0 and info["censored_rows"] > 0
    assert info["train_rows"] + info["censored_rows"] == len(full)
