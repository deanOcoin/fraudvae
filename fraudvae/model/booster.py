"""The supervised half of the hybrid: a LightGBM classifier on the observable
fields + customer context + source, trained on the labelled train split.

The VAE answers "is this unusual for this bank"; the booster answers "does
this look like the fraud we have labels for". They are combined per bank in
calibrate.py (the blender). Nothing here depends on the VAE, which keeps the
stacking honest: the booster never sees features computed by a model that
trained on the same rows.

Inputs come straight from a prepared set. The standardised numerics are fine
for trees; the only thing we undo is the "0 where unobserved" fill, which
becomes NaN so LightGBM treats it as missing instead of as a value.

Artifacts in runs/<name>/: booster.txt (the model) and booster.json (features,
params, early-stopping curve, per-source val AUROC, gain importance).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

from .dataset import PreparedSet

BATCH = 500_000


class BoosterUnavailable(RuntimeError):
    """lightgbm (or its native library) can't be loaded here. The VAE half still works,
    so callers that can live without the booster catch this instead of failing."""


def _lgb():
    try:
        import lightgbm as lgb
    except ImportError as e:   # pragma: no cover
        raise BoosterUnavailable("lightgbm is not installed (pip install lightgbm)") from e
    except OSError as e:   # pragma: no cover - the wheel is there but its dylib won't load
        hint = " (on macOS: brew install libomp)" if sys.platform == "darwin" else ""
        raise BoosterUnavailable(f"lightgbm's native library failed to load{hint}: {e}") from e
    if sys.platform == "darwin":
        _check_single_openmp()
    return lgb


def _loaded_openmp_runtimes() -> list[str]:
    """real paths of every libomp dyld has mapped into this process (macOS only)"""
    import ctypes
    import os

    libc = ctypes.CDLL(None)
    libc._dyld_get_image_name.restype = ctypes.c_char_p
    found = set()
    for i in range(libc._dyld_image_count()):
        name = libc._dyld_get_image_name(i)
        if name and name.decode(errors="replace").endswith("libomp.dylib"):
            found.add(os.path.realpath(name.decode()))
    return sorted(found)


def _check_single_openmp():
    """The PyTorch wheel carries its own libomp and the LightGBM wheel links the homebrew
    one. With both mapped, the first LightGBM call segfaults the process (seen on
    macOS 26 / torch 2.14 / lightgbm 4.7, even with KMP_DUPLICATE_LIB_OK set). lightgbm's
    import is still safe, so this runs right after it and refuses before the crash.

    Only torch's *bundled* copy is the problem: scikit-learn ships one too and that
    coexists with the homebrew runtime fine, and once scripts/macos_openmp.sh has
    turned torch/lib/libomp.dylib into a symlink to the homebrew file there is
    nothing to object to."""
    import os

    torch_mod = sys.modules.get("torch")
    if torch_mod is None:   # no torch in this process, no fight to referee
        return
    bundled = os.path.join(os.path.dirname(torch_mod.__file__), "lib", "libomp.dylib")
    if not os.path.isfile(bundled) or os.path.islink(bundled):
        return
    try:
        paths = _loaded_openmp_runtimes()
    except (OSError, AttributeError):   # pragma: no cover - not a dyld platform after all
        return
    others = [p for p in paths if p != os.path.realpath(bundled)]
    if os.path.realpath(bundled) in paths and others:
        raise BoosterUnavailable(
            "torch's bundled OpenMP runtime and another one (" + ", ".join(others) + ") are both loaded; "
            "the first LightGBM prediction would crash the process. Point torch at the homebrew libomp "
            "with `make fix-omp` (scripts/macos_openmp.sh) and restart.")


def _default_threads(cap=32) -> int:
    """num_threads=0 means "all cores" to lightgbm, which on a rented 256-thread box means a
    thread per hardware thread fighting over 2M rows: 50 rounds took 87s there against 17s
    on a 2-core sandbox. so: the cores we are actually allowed (affinity), capped where the
    scaling stops anyway."""
    import os

    try:
        n = len(os.sched_getaffinity(0))
    except AttributeError:   # macOS has no affinity api
        n = os.cpu_count() or 4
    return max(1, min(cap, n))


def feature_names(num_names, cat_names, wide_names=(), hist_dim=0):
    """[common numerics | common categoricals | source | wide block | history read-out]"""
    return list(num_names) + list(cat_names) + ["source"] + list(wide_names) + [f"hist_{i}" for i in range(hist_dim)]


def source_column(n_num: int, n_cat: int) -> int:
    return n_num + n_cat


def build_features(num, mask, cat, src, wide=None, hist=None) -> np.ndarray:
    """[N, D + C + 1 (+ W) (+ H)] float32. numerics NaN where unobserved, categorical codes as
    is, source after them, then the v5 wide block (NaN for sources without one), then the
    history encoder's read-out when the booster was trained with it (hist: [N, H])."""
    num = np.asarray(num, dtype=np.float32)
    n_wide = 0 if wide is None else wide.shape[1]
    n_hist = 0 if hist is None else hist.shape[1]
    x = np.empty((num.shape[0], num.shape[1] + cat.shape[1] + 1 + n_wide + n_hist), dtype=np.float32)
    x[:, : num.shape[1]] = np.where(np.asarray(mask) > 0, num, np.nan)
    x[:, num.shape[1] : num.shape[1] + cat.shape[1]] = np.asarray(cat, dtype=np.float32)
    sc = num.shape[1] + cat.shape[1]
    x[:, sc] = np.asarray(src, dtype=np.float32)
    if n_wide:
        x[:, sc + 1 : sc + 1 + n_wide] = wide
    if n_hist:
        x[:, sc + 1 + n_wide :] = np.asarray(hist, dtype=np.float32)
    return x


def history_readout(model, split, idx, seq, split_name, device, src_override=None, batch_size=16384) -> np.ndarray:
    """[len(idx), H] the frozen history encoder's summary of each row's customer past (its
    context vector), for the booster to read as features. unsupervised, so no label leaks"""
    from .scoring import score_split

    r = score_split(model, split, batch_size=batch_size, n_samples=1, device=device, idx=np.asarray(idx), keep_features=False,
                    keep_mu=False, seq=seq, split_name=split_name, keep_h=True, src_override=src_override)
    return np.asarray(r["h"], dtype=np.float32)


FAST_MARK = "_fast"   # wide columns of the one-week-delay label block end with _fast or _fast_w<days>


def fast_columns(names) -> list[int]:
    """indices of the one-week-delay label columns (wide__*_fast, *_fast_w7, *_fast_w30)"""
    out = []
    for j, n in enumerate(names):
        tail = n.rsplit("_fast", 1)
        if len(tail) == 2 and (tail[1] == "" or (tail[1].startswith("_w") and tail[1][2:].isdigit())):
            out.append(j)
    return out


def blank_fast(X: np.ndarray, names, rng, fraction: float, log=None, what="booster") -> np.ndarray:
    """label-latency dropout: on a random `fraction` of rows, set every fast-delay label column
    to NaN. the model then learns both regimes - labels that arrive in a week and labels that
    take a month - instead of meeting the second one for the first time in production"""
    cols = fast_columns(names)
    if not cols or fraction <= 0:
        return X
    rows = rng.random(len(X)) < fraction
    X[np.ix_(rows, cols)] = np.nan
    if log:
        log(f"  {what}: one-week label block blanked on {int(rows.sum()):,} of {len(X):,} rows ({len(cols)} columns)")
    return X


def _sample(split, per_source_legit: int, rng, keep_fraud=True):
    src, y = split.src[:], split.y[:]
    keep = []
    for s in np.unique(src):
        idx = np.flatnonzero(src == s)
        lg = idx[y[idx] == 0]
        if per_source_legit and len(lg) > per_source_legit:
            lg = rng.choice(lg, per_source_legit, replace=False)
        keep.append(lg)
        if keep_fraud:
            keep.append(idx[y[idx] == 1])
    return np.sort(np.concatenate(keep))


def _legit_weights(split, idx, cap=3.0):
    """legit rows were subsampled per source; weight them back up towards the base rate, but
    no further than `cap`. The natural weights on the full corpus are 85 for tabformer (17M
    legit train rows behind 200k sampled) against 1 for ulb, and with those one bank's
    traffic becomes the whole definition of "normal" in the source-blanked generic mode:
    the same booster config scored 0.936 generic-mode AUROC on the full set and 0.955 on a
    500k-rows-per-source set, on identical test rows, and it was the weights, not the rows
    or the number of trees. The booster is a ranker, not a base-rate estimator (the blender
    and the per-bank thresholds take care of calibration), so a bounded weight keeps every
    bank's say in what normal looks like."""
    src, y = split.src[:], split.y[:]
    w = np.ones(len(idx), dtype=np.float32)
    si, yi = src[idx], y[idx]
    for s in np.unique(si):
        total = int(((src == s) & (y == 0)).sum())
        got = int(((si == s) & (yi == 0)).sum())
        if got:
            w[(si == s) & (yi == 0)] = min(total / got, cap) if cap else total / got
    return w


def _gather(split, idx, batch=BATCH):
    """features for sorted idx, written straight into one preallocated array in batches
    (a list + concatenate would briefly need twice the memory)"""
    n_feat = split.num.shape[1] + split.cat.shape[1] + 1 + split.n_wide
    out = np.empty((len(idx), n_feat), dtype=np.float32)
    for s in range(0, len(idx), batch):
        b = idx[s : s + batch]
        out[s : s + len(b)] = build_features(split.num[b], split.mask[b], split.cat[b], split.src[b],
                                             split.wide(b) if split.n_wide else None)
    return out


class Booster:
    def __init__(self, model, info: dict):
        self.model = model
        self.info = info
        self.features = info["features"]
        self.n_num = info["n_numeric"]
        self.wide_names = info.get("wide_names", [])
        self.wide_index = {n: i for i, n in enumerate(self.wide_names)}

    def wide_vector(self, wide: dict | None) -> np.ndarray:
        """{'ieee.V1': 3.0, ...} -> [1, n_wide] float32 with NaN for everything not given"""
        v = np.full((1, len(self.wide_names)), np.nan, dtype=np.float32)
        for k, val in (wide or {}).items():
            j = self.wide_index.get(k)
            if j is not None and val is not None and val == val:
                v[0, j] = float(val)
        return v

    @classmethod
    def load(cls, run_dir: Path):
        """None when the run has no booster; BoosterUnavailable when it has one we can't load"""
        run_dir = Path(run_dir)
        p = run_dir / "booster.txt"
        if not p.exists():
            return None
        lgb = _lgb()
        info = json.loads((run_dir / "booster.json").read_text())
        return cls(lgb.Booster(model_file=str(p)), info)

    @property
    def hist_dim(self) -> int:
        return int(self.info.get("hist_dim", 0) or 0)

    def _x(self, num, mask, cat, src, wide, hist=None):
        n = np.asarray(num).shape[0]
        if self.wide_names and wide is None:
            wide = np.full((n, len(self.wide_names)), np.nan, dtype=np.float32)
        if self.hist_dim and hist is None:
            hist = np.full((n, self.hist_dim), np.nan, dtype=np.float32)   # a caller without the encoder: "no summary"
        return build_features(num, mask, cat, src, wide if self.wide_names else None, hist if self.hist_dim else None)

    def logit(self, num, mask, cat, src, threads=0, wide=None, hist=None) -> np.ndarray:
        """threads=1 for single rows (spinning up the pool costs more than the prediction)"""
        x = self._x(num, mask, cat, src, wide, hist)
        return np.asarray(self.model.predict(x, raw_score=True, num_threads=threads or _default_threads()), dtype=np.float32)

    def prob(self, num, mask, cat, src, threads=0, wide=None, hist=None) -> np.ndarray:
        z = self.logit(num, mask, cat, src, threads=threads, wide=wide, hist=hist)
        return (1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))).astype(np.float32)

    def contributions(self, num, mask, cat, src, wide=None, hist=None) -> np.ndarray:
        """SHAP-style per-feature contributions in log-odds, [N, F + 1]; last column is the bias"""
        x = self._x(num, mask, cat, src, wide, hist)
        return np.asarray(self.model.predict(x, pred_contrib=True, num_threads=1))

    def logit_split(self, split, idx=None, batch=BATCH, src_override=None, progress=None):
        """straight off the prepared arrays; a booster trained with the history read-out gets
        NaN there (score_split is the path that has the encoder)"""
        idx = np.arange(len(split)) if idx is None else np.asarray(idx)
        out = np.empty(len(idx), dtype=np.float32)
        for s in range(0, len(idx), batch):
            b = idx[s : s + batch]
            src = split.src[b] if src_override is None else np.full(len(b), src_override, dtype=np.int8)
            x = self._x(split.num[b], split.mask[b], split.cat[b], src, split.wide(b) if (self.wide_names and split.n_wide) else None)
            out[s : s + len(b)] = self.model.predict(x, raw_score=True, num_threads=_default_threads())
            if progress:
                progress(s + len(b))
        return out


def _hist_dim(model, cfg) -> int:
    from .scoring import history_window

    if model is None or not getattr(cfg, "history_features", False) or not history_window(model):
        return 0
    return int(model.hdim)


def train_booster(run_dir: Path, prepared: str, cfg, log=print, seed=1337, model=None, device="cpu") -> dict:
    """cfg is a BoostConfig. returns the booster.json dict. model: the trained VAE, so that with
    cfg.history_features its history encoder's read-out becomes booster input (hist_* columns)"""
    lgb = _lgb()
    run_dir = Path(run_dir)
    t0 = time.time()
    rng = np.random.default_rng(seed)
    data = PreparedSet(prepared, splits=("train", "val"))
    tr, va = data["train"], data["val"]
    vae = model   # `model` becomes the lightgbm model below
    hist_dim = _hist_dim(vae, cfg)
    names = feature_names(data.numeric_names, data.categorical_names, tr.wide_names, hist_dim)
    n_num = len(data.numeric_names)
    n_common = n_num + len(data.categorical_names) + 1
    src_col = source_column(n_num, len(data.categorical_names))
    # categoricals: the common ones + source, then the wide block's own categorical columns
    cat_idx = list(range(n_num, n_common)) + [n_common + j for j in tr.wide_cat_cols]
    source_names = data.source_names

    tr_idx = _sample(tr, cfg.max_legit_per_source, rng)
    va_idx = _sample(va, cfg.val_legit_per_source, rng)
    X = _gather(tr, tr_idx)
    y = tr.y[tr_idx].astype(np.float32)
    w = _legit_weights(tr, tr_idx, cap=cfg.legit_weight_cap)
    Xv = _gather(va, va_idx)
    yv = va.y[va_idx].astype(np.float32)
    if hist_dim:
        # the frozen history encoder's summary of each row's customer past, as features
        t1 = time.time()
        X = np.concatenate([X, history_readout(vae, tr, tr_idx, data.seq, "train", device)], axis=1)
        Xv = np.concatenate([Xv, history_readout(vae, va, va_idx, data.seq, "val", device)], axis=1)
        log(f"  booster: history read-out ({hist_dim} dims) for {len(tr_idx) + len(va_idx):,} rows ({time.time() - t1:.0f}s)")
    log(f"  booster: {len(tr_idx):,} train rows ({int(y.sum()):,} fraud), {len(va_idx):,} val rows, {X.shape[1]} features")
    X = blank_fast(X, names, rng, getattr(cfg, "fast_dropout", 0.0), log=log)

    # a slice of rows with the source blanked, so the model has a "generic" mode like the vae does
    if cfg.generic_fraction > 0:
        g = np.sort(rng.choice(len(X), int(len(X) * cfg.generic_fraction), replace=False))
        Xg_ = X[g].copy()
        Xg_[:, src_col] = 0
        if hist_dim:
            # the encoder's summary with the bank blanked, as it would be computed in generic mode
            Xg_[:, -hist_dim:] = history_readout(vae, tr, tr_idx[g], data.seq, "train", device, src_override=0)
        X = np.concatenate([X, Xg_])
        del Xg_
        y = np.concatenate([y, y[g]])
        w = np.concatenate([w, w[g] * 0.5])

    params = {
        "objective": "binary", "metric": "auc", "learning_rate": cfg.learning_rate, "num_leaves": cfg.num_leaves,
        "min_data_in_leaf": cfg.min_data_in_leaf, "feature_fraction": cfg.feature_fraction,
        "bagging_fraction": cfg.bagging_fraction, "bagging_freq": 1, "lambda_l2": cfg.lambda_l2,
        "max_bin": 255, "num_threads": cfg.num_threads or _default_threads(), "seed": seed, "verbose": -1,
        "cat_l2": 10.0, "cat_smooth": 10.0, "max_cat_to_onehot": 8,
    }
    dtrain = lgb.Dataset(X, label=y, weight=w, feature_name=names, categorical_feature=cat_idx, free_raw_data=True)
    # two validation sets: the named-bank rows as they are, and the same rows with the source
    # blanked. early stopping watches the mean of the two AUCs, so the generic mode (a small
    # minority of the training rows) cannot be quietly traded away for named-bank fit. the
    # first full run did exactly that: pooled val auc still creeping up at 1000 rounds while
    # the generic test auroc had fallen 0.02 below a 574-round fit
    Xv_g = Xv.copy()
    Xv_g[:, src_col] = 0
    if hist_dim:
        Xv_g[:, -hist_dim:] = history_readout(vae, va, va_idx, data.seq, "val", device, src_override=0)
    dval = lgb.Dataset(Xv, label=yv, reference=dtrain, feature_name=names, categorical_feature=cat_idx)
    dval_g = lgb.Dataset(Xv_g, label=yv, reference=dtrain, feature_name=names, categorical_feature=cat_idx)
    curve = {}
    stopper = _BalancedStop(cfg.early_stopping)
    model = lgb.train(params, dtrain, num_boost_round=cfg.max_rounds, valid_sets=[dval, dval_g], valid_names=["val", "generic"],
                      callbacks=[stopper, lgb.record_evaluation(curve), _LogEvery(log, 50)])
    Xv_g_hist = Xv_g[:, -hist_dim:].copy() if hist_dim else None
    del dtrain, dval, dval_g, Xv_g, X, y, w
    best = stopper.best_iter + 1
    log(f"  booster: stopped at {model.num_trees()} rounds, best {best} (val auc {curve['val']['auc'][best - 1]:.4f}, "
        f"generic {curve['generic']['auc'][best - 1]:.4f}), {time.time() - t0:.0f}s")

    # per-source val auroc on the plain (not source-blanked) val rows + a generic read
    from .scoring import auprc, auroc

    pv = model.predict(Xv, num_iteration=best, raw_score=True)
    sv = va.src[va_idx]
    per_source = {}
    for s in np.unique(sv):
        m = sv == s
        per_source[source_names.get(int(s), str(int(s)))] = {"auroc": auroc(yv[m], pv[m]), "auprc": auprc(yv[m], pv[m]),
                                                             "n": int(m.sum()), "fraud": int(yv[m].sum())}
    Xg = Xv.copy()
    Xg[:, src_col] = 0
    if hist_dim:
        Xg[:, -hist_dim:] = Xv_g_hist
    pg = model.predict(Xg, num_iteration=best, raw_score=True)
    generic = {"auroc": auroc(yv, pg), "auprc": auprc(yv, pg), "n": int(len(yv))}
    del Xg, Xv
    imp = model.feature_importance(importance_type="gain")
    order = np.argsort(-imp)
    info = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "prepared": prepared, "features": names, "n_numeric": n_num,
        "wide_names": list(tr.wide_names), "hist_dim": hist_dim,
        "categorical_index": cat_idx, "params": params, "best_iteration": best, "num_trees": int(model.num_trees()),
        "train_rows": int(len(tr_idx)), "train_fraud": int(tr.y[tr_idx].sum()), "val_rows": int(len(va_idx)),
        "generic_fraction": cfg.generic_fraction, "val_auc_curve": [float(v) for v in curve["val"]["auc"]],
        "generic_auc_curve": [float(v) for v in curve["generic"]["auc"]],
        "val_auroc": float(np.mean([v["auroc"] for v in per_source.values() if v["auroc"] == v["auroc"]])),
        "sources": per_source, "generic": generic,
        "importance_gain": [{"feature": names[i], "gain": float(imp[i])} for i in order[:40]],
        "seconds": round(time.time() - t0, 1),
    }
    model.save_model(str(run_dir / "booster.txt"), num_iteration=best)
    (run_dir / "booster.json").write_text(json.dumps(info, indent=1))
    log("  booster val auroc: " + " ".join(f"{k}={v['auroc']:.3f}" for k, v in per_source.items())
        + f" | generic {generic['auroc']:.3f}")
    return info


class _BalancedStop:
    """early stopping on the mean AUC over the validation sets (named banks, generic),
    instead of lightgbm's own "stop when any one of them stalls". best_iter is 0-based."""

    def __init__(self, rounds):
        self.rounds = rounds
        self.best, self.best_iter = -1.0, 0
        self.order = 20

    def __call__(self, env):
        score = float(np.mean([r[2] for r in env.evaluation_result_list]))
        if score > self.best:
            self.best, self.best_iter = score, env.iteration
        elif env.iteration - self.best_iter >= self.rounds:
            raise _lgb().callback.EarlyStopException(self.best_iter, env.evaluation_result_list)


class _LogEvery:
    """lightgbm callback that writes to our logger instead of stdout"""

    def __init__(self, log, every):
        self.log, self.every = log, every
        self.order = 10

    def __call__(self, env):
        if env.evaluation_result_list and (env.iteration + 1) % self.every == 0:
            parts = " | ".join(f"{name} {metric} {val:.4f}" for name, metric, val, _ in env.evaluation_result_list)
            self.log(f"  booster round {env.iteration + 1}: {parts}")


def main(argv=None):
    import argparse

    from ..config import load_config
    from ..paths import RUNS_DIR

    ap = argparse.ArgumentParser(description="train the supervised booster for an existing run")
    ap.add_argument("run")
    ap.add_argument("--prepared", default=None)
    ap.add_argument("--config", default=None, help="defaults to the config the run was trained with")
    args = ap.parse_args(argv)
    run_dir = Path(args.run) if Path(args.run).exists() else RUNS_DIR / args.run
    run = json.loads((run_dir / "run.json").read_text())
    prepared = args.prepared or run.get("prepared", "default")
    if args.config:
        cfg = load_config(args.config)
    else:
        from .. import config as _c

        cfg = _c.Config()
        _c._merge(cfg, run.get("config", {}))
    train_booster(run_dir, prepared, cfg.boost)
    return 0


if __name__ == "__main__":
    sys.exit(main())


# ---- v5: per-source specialists -------------------------------------------------------------------

MIN_SPECIALIST_FRAUD = 300


def train_specialists(run_dir: Path, prepared: str, cfg, log=print, seed=1337, min_fraud=MIN_SPECIALIST_FRAUD, model=None,
                      device="cpu") -> dict:
    """one booster per source with enough labelled fraud, trained on that source's rows alone
    (its own wide block included, no generic slice, early stopped on its own validation auc).
    the shared booster splits its capacity across eleven banks and learns what they have in
    common; a specialist spends all of it on one bank's quirks, which is what a bank deploying
    this would want for itself. its logit is the "local" signal: present on that bank's rows,
    absent everywhere else, blended with its own weight like every other signal.
    writes runs/<run>/specialist/<source>/booster.{txt,json} + specialists.json"""
    lgb = _lgb()
    run_dir = Path(run_dir)
    out_root = run_dir / "specialist"
    out_root.mkdir(parents=True, exist_ok=True)
    t_all = time.time()
    rng = np.random.default_rng(seed)
    data = PreparedSet(prepared, splits=("train", "val"))
    tr, va = data["train"], data["val"]
    vae = model
    hist_dim = _hist_dim(vae, cfg)
    names = feature_names(data.numeric_names, data.categorical_names, tr.wide_names, hist_dim)
    n_num = len(data.numeric_names)
    n_common = n_num + len(data.categorical_names) + 1
    cat_idx = list(range(n_num, n_common)) + [n_common + j for j in tr.wide_cat_cols]
    src_tr, y_tr = tr.src[:], tr.y[:]
    src_va, y_va = va.src[:], va.y[:]
    summary = {"created": time.strftime("%Y-%m-%dT%H:%M:%S"), "prepared": prepared, "sources": {}, "hist_dim": hist_dim}
    for s in np.unique(src_tr):
        key = data.source_names.get(int(s), str(int(s)))
        rows = np.flatnonzero(src_tr == s)
        fraud = rows[y_tr[rows] == 1]
        if len(fraud) < min_fraud:
            log(f"  specialist {key}: {len(fraud)} labelled fraud rows, fewer than {min_fraud}, skipping")
            continue
        t0 = time.time()
        legit = rows[y_tr[rows] == 0]
        cap = int(cfg.max_legit_per_source * 2)   # one bank gets twice the shared booster's share of its legit rows
        if len(legit) > cap:
            legit = rng.choice(legit, cap, replace=False)
        tr_idx = np.sort(np.concatenate([legit, fraud]))
        vrows = np.flatnonzero(src_va == s)
        vl = vrows[y_va[vrows] == 0]
        if len(vl) > cfg.val_legit_per_source:
            vl = rng.choice(vl, cfg.val_legit_per_source, replace=False)
        va_idx = np.sort(np.concatenate([vl, vrows[y_va[vrows] == 1]]))
        if y_va[va_idx].sum() < 20:
            log(f"  specialist {key}: too few validation fraud rows to stop on, skipping")
            continue
        X, y = _gather(tr, tr_idx), y_tr[tr_idx].astype(np.float32)
        Xv, yv = _gather(va, va_idx), y_va[va_idx].astype(np.float32)
        if hist_dim:
            X = np.concatenate([X, history_readout(vae, tr, tr_idx, data.seq, "train", device)], axis=1)
            Xv = np.concatenate([Xv, history_readout(vae, va, va_idx, data.seq, "val", device)], axis=1)
        X = blank_fast(X, names, rng, getattr(cfg, "fast_dropout", 0.0))
        total_legit = int(((src_tr == s) & (y_tr == 0)).sum())
        w = np.where(y == 0, min(total_legit / max(len(legit), 1), cfg.legit_weight_cap or 1e9), 1.0).astype(np.float32)
        # the source column is constant here: lightgbm just never splits on it
        params = {
            "objective": "binary", "metric": "auc", "learning_rate": cfg.learning_rate, "num_leaves": cfg.num_leaves,
            "min_data_in_leaf": cfg.min_data_in_leaf, "feature_fraction": cfg.feature_fraction,
            "bagging_fraction": cfg.bagging_fraction, "bagging_freq": 1, "lambda_l2": cfg.lambda_l2,
            "max_bin": 255, "num_threads": cfg.num_threads or _default_threads(), "seed": seed, "verbose": -1,
            "cat_l2": 10.0, "cat_smooth": 10.0, "max_cat_to_onehot": 8,
        }
        dtrain = lgb.Dataset(X, label=y, weight=w, feature_name=names, categorical_feature=cat_idx, free_raw_data=True)
        dval = lgb.Dataset(Xv, label=yv, reference=dtrain, feature_name=names, categorical_feature=cat_idx)
        curve = {}
        model = lgb.train(params, dtrain, num_boost_round=cfg.max_rounds, valid_sets=[dval], valid_names=["val"],
                          callbacks=[lgb.early_stopping(cfg.early_stopping, verbose=False), lgb.record_evaluation(curve)])
        best = int(model.best_iteration or model.num_trees())
        from .scoring import auprc, auroc

        pv = model.predict(Xv, num_iteration=best, raw_score=True)
        imp = model.feature_importance(importance_type="gain")
        order = np.argsort(-imp)
        info = {
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "prepared": prepared, "source": key, "features": names, "n_numeric": n_num,
            "wide_names": list(tr.wide_names), "hist_dim": hist_dim, "categorical_index": cat_idx, "params": params, "best_iteration": best,
            "num_trees": int(model.num_trees()), "train_rows": int(len(tr_idx)), "train_fraud": int(len(fraud)),
            "val_rows": int(len(va_idx)), "val_auroc": auroc(yv, pv), "val_auprc": auprc(yv, pv),
            "val_auc_curve": [float(v) for v in curve["val"]["auc"]],
            "importance_gain": [{"feature": names[i], "gain": float(imp[i])} for i in order[:30]],
            "seconds": round(time.time() - t0, 1),
        }
        d = out_root / key
        d.mkdir(parents=True, exist_ok=True)
        model.save_model(str(d / "booster.txt"), num_iteration=best)
        (d / "booster.json").write_text(json.dumps(info, indent=1))
        summary["sources"][key] = {k: info[k] for k in ("best_iteration", "train_rows", "train_fraud", "val_rows", "val_auroc", "val_auprc", "seconds")}
        log(f"  specialist {key}: {len(tr_idx):,} rows ({len(fraud):,} fraud), {best} rounds, val auroc {info['val_auroc']:.4f} ({time.time() - t0:.0f}s)")
        del X, Xv, dtrain, dval, model
    summary["seconds"] = round(time.time() - t_all, 1)
    (run_dir / "specialists.json").write_text(json.dumps(summary, indent=1))
    return summary


class Specialists:
    """the per-source boosters of a run: {source idx: Booster}. logits NaN for rows of a
    source without one, which the blender reads as absent."""

    def __init__(self, boosters: dict[int, Booster]):
        self.boosters = boosters

    def __bool__(self):
        return bool(self.boosters)

    @classmethod
    def load(cls, run_dir: Path):
        from ..data import schema

        root = Path(run_dir) / "specialist"
        if not root.exists():
            return None
        out = {}
        for d in sorted(root.iterdir()):
            if (d / "booster.txt").exists() and d.name in schema.SOURCE_INDEX:
                out[schema.SOURCE_INDEX[d.name]] = Booster.load(d)
        return cls(out) if out else None

    def logit(self, num, mask, cat, src, wide=None, threads=0, hist=None) -> np.ndarray:
        src = np.asarray(src)
        out = np.full(len(src), np.nan, dtype=np.float32)
        for s, bst in self.boosters.items():
            m = src == s
            if m.any():
                out[m] = bst.logit(np.asarray(num)[m], np.asarray(mask)[m], np.asarray(cat)[m], src[m], threads=threads,
                                   wide=None if wide is None else np.asarray(wide)[m],
                                   hist=None if hist is None else np.asarray(hist)[m])
        return out

    def logit_split(self, split, idx=None, batch=BATCH) -> np.ndarray:
        idx = np.arange(len(split)) if idx is None else np.asarray(idx)
        out = np.full(len(idx), np.nan, dtype=np.float32)
        src = split.src[idx] if len(idx) < len(split) else split.src[:]
        for s, bst in self.boosters.items():
            where = np.flatnonzero(src == s)
            if len(where) == 0:
                continue
            rows = idx[where]
            for k in range(0, len(rows), batch):
                b = rows[k : k + batch]
                x = bst._x(split.num[b], split.mask[b], split.cat[b], split.src[b], split.wide(b) if (bst.wide_names and split.n_wide) else None)
                out[where[k : k + len(b)]] = bst.model.predict(x, raw_score=True, num_threads=_default_threads())
        return out

    def describe(self) -> dict:
        from ..data import schema

        inv = {v: k for k, v in schema.SOURCE_INDEX.items()}
        return {inv.get(s, str(s)): {"best_iteration": b.info.get("best_iteration"), "val_auroc": b.info.get("val_auroc"),
                                    "train_rows": b.info.get("train_rows")} for s, b in self.boosters.items()}
