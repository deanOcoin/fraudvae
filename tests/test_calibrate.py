"""the calibration arithmetic on its own: thresholds at a false-positive budget, the logistic
blender with its standardisation folded back into the weights, and the per-bank shrinkage that
keeps a bank with a handful of fraud rows from being learnt by heart"""
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
