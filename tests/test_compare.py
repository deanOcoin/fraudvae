"""`fraudvae compare`: the paired bootstrap over two runs' per-row scores."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from fraudvae.compare import Ranked, compare_runs, compare_scores, format_table, main, metric_keys
from fraudvae.evaluate import read_scores, scores_name, write_scores


def _scores(n=20000, fraud=0.03, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < fraud).astype(np.int8)
    return y, rng.normal(0, 1, n) + 1.2 * y


def test_weighted_metrics_match_the_plain_ones():
    from sklearn.metrics import roc_auc_score

    y, s = _scores()
    s[::97] = s[1::97]                         # ties, so the grouping by value is exercised
    r = Ranked(s, y)
    m = r.metrics(np.ones(len(y)), (0.01, 0.005))
    assert abs(m["auroc"] - roc_auc_score(y, s)) < 1e-12
    for f in (0.01, 0.005):
        thr = np.quantile(s[y == 0], 1 - f, method="inverted_cdf")
        assert abs(m[f"recall@{f:g}"] - (s[y == 1] > thr).mean()) < 1e-12
    thr = np.quantile(s[y == 0], 0.99, method="inverted_cdf")
    flagged = s > thr
    assert abs(m["precision@0.01"] - (y[flagged] == 1).mean()) < 1e-12
    # a replicate's weights change the numbers a little, never the keys; no fraud = no number
    w = np.random.default_rng(1).poisson(1.0, len(y)).astype(float)
    mw = r.metrics(w, (0.01, 0.005))
    assert set(mw) == set(metric_keys((0.01, 0.005))) and abs(mw["auroc"] - m["auroc"]) < 0.05
    assert all(v != v for v in Ranked(s[y == 0], y[y == 0]).metrics(np.ones(int((y == 0).sum()))).values())


def _write_pair(tmp_path, planted=0.8, boot_seed=0):
    """two runs on the same rows of two sources: B carries a planted gain on fdh and noise on capone"""
    rng = np.random.default_rng(boot_seed)
    ya, sa = _scores(seed=10)
    yb, sb = _scores(seed=11)
    y = np.concatenate([ya, yb])
    src = np.concatenate([np.ones(len(ya), dtype=np.int8), np.full(len(yb), 2, dtype=np.int8)])
    row = np.arange(len(y), dtype=np.int64)
    a = np.concatenate([sa, sb]).astype(np.float32)
    b = a.copy()
    b[: len(ya)] += planted * ya                               # the gain where the label is 1, fdh only
    b[len(ya):] += rng.normal(0, 0.02, len(yb)).astype(np.float32)   # noise on capone
    te = SimpleNamespace(row=row, src=src, y=y)
    names = {1: "fdh", 2: "capone"}
    runs = {}
    for name, s in (("a", a), ("b", b)):
        d = tmp_path / name
        d.mkdir()
        write_scores(d / scores_name(None), {"norm": s, "blend": 1 / (1 + np.exp(-s)), "booster": s, "local": np.full(len(s), np.nan)},
                     te, names, "norm", None, "t")
        runs[name] = d
    return runs, te


def test_compare_finds_the_planted_gain_and_not_the_noise(tmp_path):
    runs, te = _write_pair(tmp_path)
    a, b = read_scores(runs["a"] / "scores_test.npz"), read_scores(runs["b"] / "scores_test.npz")
    assert a["meta"]["signals"] == ["vae", "booster", "blend", "local"] and a["meta"]["n"] == len(te.y)
    res = compare_scores(a, b, signal="blend", fprs=(0.01, 0.005), boot=200, seed=0, log=lambda m: None)
    fdh, cap = res["sources"]["fdh"], res["sources"]["capone"]
    assert fdh["verdict"] == "better" and fdh["metrics"]["auroc"]["lo"] > 0 and fdh["metrics"]["recall@0.01"]["lo"] > 0
    assert cap["verdict"] == "undecided" and cap["metrics"]["auroc"]["lo"] < 0 < cap["metrics"]["auroc"]["hi"]
    assert fdh["n"] == 20000 and fdh["fraud"] == int(te.y[:20000].sum())
    # the mean over sources pairs replicate by replicate: half the planted gain, interval above zero
    mean = res["mean"]["metrics"]["auroc"]
    assert mean["n_sources"] if "n_sources" in mean else res["mean"]["n_sources"] == 2
    assert abs(mean["delta"] - fdh["metrics"]["auroc"]["delta"] / 2 - cap["metrics"]["auroc"]["delta"] / 2) < 1e-9 and mean["lo"] > 0
    table = format_table(res, "a", "b")
    assert "| fdh |" in table and "| mean of 2 |" in table and "better" in table and "recall@1%" in table
    # the same rows are a precondition: a run on other rows is refused, and so is another signal
    c = {k: v for k, v in b.items()}
    c["row"] = c["row"] + 1
    with pytest.raises(SystemExit):
        compare_scores(a, c, boot=10, log=lambda m: None)
    with pytest.raises(SystemExit):
        compare_scores(a, b, signal="seq", boot=10, log=lambda m: None)
    # the local signal is NaN everywhere: no fraud above the threshold is a number, not a crash
    res_local = compare_scores(a, b, signal="local", boot=10, log=lambda m: None)
    assert res_local["sources"]["fdh"]["verdict"] == "undecided"


def test_compare_command_over_run_dirs(tmp_path, capsys):
    runs, _ = _write_pair(tmp_path)
    out = tmp_path / "cmp.json"
    assert main([str(runs["a"]), str(runs["b"]), "--boot", "50", "--json", str(out), "--quiet"]) == 0
    printed = capsys.readouterr().out
    assert "### blend, labels as published" in printed and "| fdh |" in printed
    res = json.loads(out.read_text())
    assert len(res) == 1 and res[0]["a"]["run"] == "a" and res[0]["b"]["run"] == "b" and res[0]["boot"] == 50
    # a delayed-label pass is compared only when both runs have it
    assert compare_runs(str(runs["a"]), str(runs["b"]), boot=10, log=lambda m: None)[0]["label_delay_days"] is None
    with pytest.raises(SystemExit):
        compare_runs(str(runs["a"]), str(runs["b"]), delay=30, boot=10, log=lambda m: None)
