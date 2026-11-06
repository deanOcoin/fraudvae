"""the calibration arithmetic on its own: thresholds at a false-positive budget, the logistic
blender with its standardisation folded back into the weights, and the per-bank shrinkage that
keeps a bank with a handful of fraud rows from being learnt by heart"""
import json

import numpy as np

from fraudvae.calibrate import FPR_TARGETS, SHRINK_REF, _fit_logistic, _interaction_shrink, _sigmoid, _thresholds


def test_thresholds_hit_the_budget_and_report_what_they_flag():
    rng = np.random.default_rng(0)
    legit = rng.normal(0.0, 1.0, 200_000)
    fraud = rng.normal(2.5, 1.0, 2_000)
    s = np.concatenate([legit, fraud])
    y = np.concatenate([np.zeros(len(legit), np.int8), np.ones(len(fraud), np.int8)])
    out = _thresholds(y, s)
    for fpr in FPR_TARGETS:
        op = out[f"fpr_{fpr}"]
        assert op["target_fpr"] == fpr
        assert abs((legit > op["threshold"]).mean() - fpr) < fpr * 0.15 + 1e-4   # the budget is met on the legit rows
        assert op["recall"] == ((fraud > op["threshold"]).sum() / len(fraud))
        flagged = s > op["threshold"]
        assert op["precision"] == (flagged & (y == 1)).sum() / flagged.sum() and op["flagged_share"] == flagged.mean()
    recalls = [out[f"fpr_{fpr}"]["recall"] for fpr in FPR_TARGETS]
    assert recalls == sorted(recalls)                            # a bigger budget never catches less
    assert 0.0 < out["best_f1"]["f1"] <= 1.0 and out["best_f1"]["threshold"] > 0.0


def test_thresholds_without_fraud_or_without_legit():
    s = np.linspace(0, 1, 1000)
    out = _thresholds(np.zeros(1000, np.int8), s)
    assert out["fpr_0.01"]["recall"] is None and "best_f1" not in out
    out = _thresholds(np.ones(1000, np.int8), s)
    assert np.isnan(out["fpr_0.01"]["threshold"]) and "best_f1" not in out


def test_fit_logistic_folds_the_standardisation_back():
    rng = np.random.default_rng(1)
    n = 20_000
    X = np.column_stack([rng.normal(5, 2, n), rng.normal(-3, 0.5, n), rng.normal(0, 10, n)])
    logit = 1.2 * (X[:, 0] - 5) / 2 - 0.8 * (X[:, 1] + 3) / 0.5 + 0.0 * X[:, 2]
    y = (rng.random(n) < _sigmoid(logit)).astype(int)
    fit = _fit_logistic(X, y, C=10.0)
    coef = np.array(fit["coef"])
    assert fit["n_features"] == 3 and len(coef) == 3
    # the weights apply to the raw columns: one dot product gives the probabilities sklearn gives
    # on the standardised columns it was actually fit on
    from sklearn.linear_model import LogisticRegression

    mean, std = X.mean(0), np.maximum(X.std(0), 1e-3)
    clf = LogisticRegression(C=10.0, class_weight="balanced", max_iter=2000, solver="lbfgs").fit((X - mean) / std, y)
    p = _sigmoid(X @ coef + fit["intercept"])
    assert np.allclose(p, clf.predict_proba((X - mean) / std)[:, 1], atol=1e-6)
    assert coef[0] > 0 and coef[1] < 0 and abs(coef[2]) < 0.01       # signs recovered, the noise column near zero
    # a scaled column fits the same decision boundary with a correspondingly larger raw weight
    shrunk = _fit_logistic(X, y, C=10.0, col_scale=[1.0, 1.0, 1.0])
    assert np.allclose(shrunk["coef"], fit["coef"], atol=1e-6)


def test_interaction_shrink_is_sqrt_of_fraud_over_reference():
    names = ["booster", "vae", "booster*src1", "vae*src1", "booster*src2", "booster*src3"]
    src = np.array([1] * 1000 + [2] * 20_000 + [3] * 50)
    y = np.zeros(len(src), np.int8)
    y[:21] = 1                      # bank 1: 21 fraud rows, the ULB case
    y[1000:1000 + SHRINK_REF] = 1   # bank 2: exactly the reference, trusted in full
    f = _interaction_shrink(names, src, y)
    assert f[0] == 1.0 and f[1] == 1.0                       # shared weights are never shrunk
    assert np.isclose(f[2], np.sqrt(21 / SHRINK_REF)) and f[2] == f[3]
    assert f[4] == 1.0
    assert f[5] == 0.0                                       # a bank with no fraud in the fit half gets no own weights


def test_blend_without_vae_drops_the_novelty_columns():
    """v7: boost.blend_without_vae fits the blender on the booster logit, the one-hot and booster x
    one-hot (plus the extra signals) alone: the feature count drops by the VAE's 2 + 2 (S + 1)
    columns, the names agree, and blend_prob is the logistic on exactly those columns"""
    from fraudvae.model.scoring import Scorer

    n_src, n = 3, 40
    rng = np.random.default_rng(0)
    base = {"generic": {"feat_mean": [0.0] * 4, "feat_std": [1.0] * 4, "kl_mean": 0.0, "kl_std": 1.0}}
    sc = Scorer(base, n_features=4, n_sources=n_src)
    r = {"raw": rng.normal(size=n).astype(np.float32), "kl": rng.normal(size=n).astype(np.float32),
         "seq": np.where(rng.random(n) < 0.7, rng.normal(size=n), np.nan).astype(np.float32)}
    src = rng.integers(0, n_src + 1, size=n)
    b = rng.normal(size=n).astype(np.float32)
    mean, std = np.zeros(n_src + 1, dtype=np.float32), np.ones(n_src + 1, dtype=np.float32)
    full = sc.blend_features(r, src, b, "raw", mean, std, signals=["seq"])
    slim = sc.blend_features(r, src, b, "raw", mean, std, signals=["seq"], without_vae=True)
    assert full.shape[1] - slim.shape[1] == 2 + 2 * (n_src + 1)
    assert full.shape[1] == len(Scorer.blend_feature_names(n_src, ["seq"]))
    assert slim.shape[1] == len(Scorer.blend_feature_names(n_src, ["seq"], without_vae=True))
    assert Scorer.blend_feature_names(n_src, ["seq"], without_vae=True)[:2] == ["booster", "src0"]
    # by hand: the booster logit, the one-hot, booster x one-hot, then the signal's three parts
    onehot = np.zeros((n, n_src + 1), dtype=np.float32)
    onehot[np.arange(n), src] = 1
    bc = np.clip(b, -15, 15)[:, None]
    present = np.isfinite(r["seq"]).astype(np.float32)[:, None]
    val = np.clip(np.nan_to_num(r["seq"]), -15, 15)[:, None] * present
    hand = np.concatenate([bc, onehot, bc * onehot, val, present, val * onehot], axis=1)
    assert np.allclose(slim, hand)
    coef = rng.normal(size=slim.shape[1]).round(3)
    sc.set_blend({"coef": coef.tolist(), "intercept": 0.25, "score_type": "raw", "score_mean": mean.tolist(),
                  "score_std": std.tolist(), "signals": ["seq"], "without_vae": True})
    assert sc.blend_without_vae is True
    assert np.allclose(sc.blend_prob(r, src, b), 1 / (1 + np.exp(-(hand @ coef + 0.25))), atol=1e-5)
    sc.set_blend({"coef": [0.0] * full.shape[1], "intercept": 0.0, "score_type": "raw", "score_mean": mean.tolist(),
                  "score_std": std.tolist(), "signals": ["seq"]})
    assert sc.blend_without_vae is False and sc.blend_features(r, src, b).shape == full.shape


def test_calibrate_run_without_vae_records_the_layout(trained_run, prepared_set, tmp_path):
    """the whole calibration with the flag: calibration.json says without_vae, its feature count and
    names are the slim layout's, and a Scorer built from it (as the server and evaluate do) carries
    the choice"""
    import shutil

    from fraudvae.calibrate import calibrate_run
    from fraudvae.model.scoring import Scorer

    run = tmp_path / "novae"
    shutil.copytree(trained_run, run)
    calibrate_run(run, prepared=prepared_set, device="cpu", n_samples=2, log=lambda m: None, blend_without_vae=True)
    cal = json.loads((run / "calibration.json").read_text())
    blend = cal["blend"]
    n_src = len(blend["score_mean"]) - 1
    assert blend["without_vae"] is True and "vae_z" not in blend["feature_names"]
    assert blend["feature_names"] == Scorer.blend_feature_names(n_src, blend["signals"], without_vae=True)
    assert blend["n_features"] == len(blend["feature_names"]) == len(blend["coef"])
    sc = Scorer(cal["baselines"], len(cal["features"]["numeric"]) + len(cal["features"]["categorical"]), blend=blend)
    assert sc.blend_without_vae is True and len(sc.blend_coef) == blend["n_features"]
    before = json.loads((trained_run / "calibration.json").read_text())["blend"]
    assert before.get("without_vae", False) is False and before["n_features"] > blend["n_features"]
