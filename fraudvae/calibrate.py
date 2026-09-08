"""Post-training calibration on the validation split.

Pass 1: per-source NLL baselines from a legit subsample (and a "generic" set
        scored with the source token zeroed).
Pass 2: on a labelled subsample, pick the VAE score type (raw vs normalised)
        and fit the blender: a logistic regression on [booster logit, VAE
        score z, KL z, source one-hot and the three interactions], inputs
        clipped. That is the model's final p(fraud).
Pass 3: score the whole split once (VAE, booster, blend) and derive operating
        thresholds, legit quantiles and the PCA latent map.

Writes runs/<name>/calibration.json and latent_map.npz.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from .model.booster import Booster, Specialists
from .graph.scorer import GraphScorer
from .model.heads import load_heads
from .model.seqhead import load_seqhead
from .model.dataset import PreparedSet
from .model.scoring import Scorer, auprc, auroc, extra_signals, score_split
from .model.vae import load_checkpoint
from .paths import RUNS_DIR

FPR_TARGETS = (0.001, 0.005, 0.01, 0.02, 0.05)
QS = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.995, 0.999, 0.9999]
BASELINE_ROWS = 100_000      # legit rows per source for pass 1
GENERIC_ROWS = 60_000        # legit rows per source in the generic passes
FIT_ROWS = 100_000           # legit rows per source for fitting the blender (all fraud rows are used)


def _thresholds(y, s):
    out = {}
    legit, fraud = s[y == 0], s[y == 1]
    for fpr in FPR_TARGETS:
        thr = float(np.quantile(legit, 1 - fpr)) if len(legit) else float("nan")
        flagged = s > thr
        tp = int((flagged & (y == 1)).sum())
        fp = int((flagged & (y == 0)).sum())
        out[f"fpr_{fpr}"] = {
            "target_fpr": fpr, "threshold": thr,
            "recall": float(tp / max(len(fraud), 1)) if len(fraud) else None,
            "precision": float(tp / max(tp + fp, 1)) if (tp + fp) else None,
            "flagged_share": float(flagged.mean()),
        }
    if len(fraud) and len(legit):
        cand = np.quantile(s, np.linspace(0.5, 0.9999, 400))
        best = None
        for thr in cand:
            flagged = s > thr
            tp = (flagged & (y == 1)).sum()
            fp = (flagged & (y == 0)).sum()
            fn = (~flagged & (y == 1)).sum()
            f1 = 2 * tp / max(2 * tp + fp + fn, 1)
            if best is None or f1 > best["f1"]:
                best = {"threshold": float(thr), "f1": float(f1), "precision": float(tp / max(tp + fp, 1)),
                        "recall": float(tp / max(tp + fn, 1))}
        out["best_f1"] = best
    return out


def _dist_block(y, s, extra=None):
    legit = s[y == 0]
    block = {
        "n": int(len(s)), "fraud": int(y.sum()),
        "auroc": auroc(y, s), "auprc": auprc(y, s),
        "legit_mean": float(legit.mean()), "legit_std": float(max(legit.std(), 1e-6)),
        "legit_quantiles": {str(q): float(np.quantile(legit, q)) for q in QS},
        "fraud_median": float(np.median(s[y == 1])) if y.sum() else None,
        "thresholds": _thresholds(y, s),
    }
    if extra:
        block.update(extra)
    return block


def _signal_block(y, logit, extra=None, min_rows=200):
    """_dist_block on sigmoid(logit) over the rows where the signal is present (finite); the
    field heads leave NaN where their block is unobserved. None when too few rows carry it."""
    ok = np.isfinite(logit)
    if ok.sum() < min_rows or y[ok].sum() < 5 or (y[ok] == 0).sum() < 50:
        return None
    block = _dist_block(y[ok], _sigmoid(logit[ok]), extra=extra)
    block["coverage"] = float(ok.mean())
    return block


def _fit_logistic(X, y, C=0.5, col_scale=None):
    """L2 logistic regression on standardised columns. col_scale (one factor per column, default
    1) multiplies a standardised column before the fit: a column scaled by f needs a 1/f larger
    weight for the same effect and pays 1/f^2 more penalty, which is how the per-bank
    interaction terms are shrunk toward the shared weights (see _interaction_shrink). the
    scaling and the standardisation are folded back into the weights, so inference stays one
    dot product on the raw features."""
    from sklearn.linear_model import LogisticRegression

    mean = X.mean(0)
    std = np.maximum(X.std(0), 1e-3)
    f = np.ones(X.shape[1]) if col_scale is None else np.asarray(col_scale, dtype=np.float64)
    clf = LogisticRegression(C=C, class_weight="balanced", max_iter=2000, solver="lbfgs")
    clf.fit((X - mean) / std * f, y)
    coef = clf.coef_[0] * f / std
    intercept = float(clf.intercept_[0] - (coef * mean).sum())
    return {"coef": coef.round(6).tolist(), "intercept": intercept, "n_features": int(X.shape[1]), "C": C,
            "class_weight": "balanced"}


# a bank's own blender weights (the "signal x one-hot" interaction columns) are trusted in full
# once its fit half holds this many fraud rows; with fewer they are shrunk toward the shared
# weights by sqrt(fraud / SHRINK_REF). the v5 run had 21 ULB fraud rows in the fit half and the
# unshrunk interactions learnt them by heart: held-out ULB AUROC 0.915 against the booster's
# 0.953 on the same rows. shrunk, 0.953, with every other bank unchanged.
SHRINK_REF = 10_000


def _interaction_shrink(feature_names: list[str], src_fit: np.ndarray, y_fit: np.ndarray) -> np.ndarray:
    pos = {int(s): int(y_fit[src_fit == s].sum()) for s in np.unique(src_fit)}
    f = np.ones(len(feature_names))
    for j, name in enumerate(feature_names):
        if "*src" in name:
            f[j] = np.sqrt(min(1.0, pos.get(int(name.split("*src")[1]), 0) / SHRINK_REF))
    return f


def _subsample(src, y, rng, per_source, with_fraud=True):
    keep = []
    for s in np.unique(src):
        idx = np.flatnonzero(src == s)
        lg = idx[y[idx] == 0]
        keep.append(rng.choice(lg, min(per_source, len(lg)), replace=False))
        if with_fraud:
            keep.append(idx[y[idx] == 1])
    return np.sort(np.concatenate(keep))


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def calibrate_run(run_dir: Path, prepared="default", device="cpu", n_samples=8, log=print, batch_size=16384):
    run_dir = Path(run_dir)
    model, ck = load_checkpoint(run_dir / "model.pt", device=device)
    booster = Booster.load(run_dir)
    seqhead = load_seqhead(run_dir, device=device)   # v5: None without a history encoder / before the post stage
    heads = load_heads(run_dir, device=device)        # v5: the field heads (device ...), {} before the post stage
    graph = GraphScorer.load(run_dir, device=device)  # v5: the entity-graph models, None without any
    specialists = Specialists.load(run_dir)            # v5: per-source boosters, None without any
    data = PreparedSet(prepared, splits=("val",))
    va = data["val"]
    names = data.source_names
    y, src = va.y[:], va.src[:]
    rng = np.random.default_rng(0)
    n_feat = len(data.numeric_names) + len(data.categorical_names)
    t0 = time.time()
    if booster is None:
        log("  no booster.txt in this run: the blender will be fit on the VAE features alone")

    # ---- pass 1: baselines --------------------------------------------------------
    base_idx = _subsample(src, y, rng, BASELINE_ROWS, with_fraud=False)
    r1 = score_split(model, va, batch_size=batch_size, n_samples=n_samples, device=device, idx=base_idx, keep_mu=False, seq=data.seq, split_name="val")
    nll1 = np.concatenate([r1["nll_num"], r1["nll_cat"]], axis=1)
    baselines = {"by_source": {}}
    for s in np.unique(src[base_idx]):
        m = src[base_idx] == s
        baselines["by_source"][str(int(s))] = Scorer.fit_baselines(nll1[m], r1["kl"][m])
    gen_idx = _subsample(src, y, rng, GENERIC_ROWS, with_fraud=True)
    g_legit = gen_idx[y[gen_idx] == 0]
    rg = score_split(model, va, batch_size=batch_size, n_samples=n_samples, device=device, idx=g_legit, src_override=0,
                     keep_mu=False, seq=data.seq, split_name="val")
    baselines["generic"] = Scorer.fit_baselines(np.concatenate([rg["nll_num"], rg["nll_cat"]], axis=1), rg["kl"])
    scorer = Scorer(baselines, n_feat)
    n_src = scorer.n_sources
    log(f"  pass 1: baselines from {len(base_idx):,} legit rows ({time.time() - t0:.0f}s)")
    del r1, nll1, rg

    # ---- pass 2: score type + blender on a labelled subsample --------------------------------
    fit_idx = _subsample(src, y, rng, FIT_ROWS, with_fraud=True)
    rf = score_split(model, va, batch_size=batch_size, n_samples=n_samples, device=device, idx=fit_idx, scorer=scorer,
                     booster=booster, keep_features=False, keep_mu=False, seq=data.seq, split_name="val", seqhead=seqhead, heads=heads, graph=graph, specialists=specialists)
    yf, sf = y[fit_idx], src[fit_idx]
    cands = {}
    for kind in ("raw", "norm"):
        aucs = [auroc(yf[sf == s], rf[kind][sf == s]) for s in np.unique(sf)]
        aucs = [a for a in aucs if a == a]
        cands[kind] = float(np.mean(aucs)) if aucs else float("nan")
    score_type = "norm" if cands["norm"] >= cands["raw"] - 1e-9 else "raw"
    log(f"  pass 2: mean val auroc raw={cands['raw']:.4f} norm={cands['norm']:.4f} -> using '{score_type}'")
    # legit mean/std of the chosen score per source idx, for the blender's "score z" column
    score_mean = np.zeros(n_src + 1, dtype=np.float32)
    score_std = np.ones(n_src + 1, dtype=np.float32)
    for s in np.unique(sf):
        m = (sf == s) & (yf == 0)
        score_mean[s] = rf[score_type][m].mean()
        score_std[s] = max(float(rf[score_type][m].std()), 1e-6)
    gen_fit = np.sort(rng.choice(gen_idx, min(len(gen_idx), 200_000), replace=False))
    rfg = score_split(model, va, batch_size=batch_size, n_samples=n_samples, device=device, idx=gen_fit, scorer=scorer,
                      booster=booster, src_override=0, keep_features=False, keep_mu=False, seq=data.seq, split_name="val", seqhead=seqhead, heads=heads, graph=graph, specialists=specialists)
    yg = y[gen_fit]
    score_mean[0] = rfg[score_type][yg == 0].mean()
    score_std[0] = max(float(rfg[score_type][yg == 0].std()), 1e-6)

    signals = extra_signals(rf)
    X = scorer.blend_features(rf, sf, rf.get("booster"), score_type, score_mean, score_std, signals=signals)
    Xg = scorer.blend_features(rfg, np.zeros(len(gen_fit), dtype=int), rfg.get("booster"), score_type, score_mean, score_std,
                               signals=signals)
    X_all = np.concatenate([X, Xg])
    y_all = np.concatenate([yf, yg])
    s_all = np.concatenate([sf, np.zeros(len(gen_fit), dtype=sf.dtype)])
    fit_mask = (np.arange(len(X_all)) % 2) == 0
    feature_names = Scorer.blend_feature_names(n_src, signals)
    shrink = _interaction_shrink(feature_names, s_all[fit_mask], y_all[fit_mask])
    blend = _fit_logistic(X_all[fit_mask], y_all[fit_mask], col_scale=shrink)
    blend.update({"score_type": score_type, "score_mean": score_mean.round(6).tolist(), "score_std": score_std.round(6).tolist(),
                  "signals": signals,
                  "features": "booster logit, VAE score z, KL z (each clipped), source one-hot, each of the three x "
                              "one-hot" + (", then per extra signal: value, present, value x one-hot" if signals else ""),
                  "feature_names": feature_names,
                  "interaction_shrink": {"ref_fraud_rows": SHRINK_REF,
                                         "per_source": {names.get(int(s), "generic") if s else "generic": round(float(shrink[feature_names.index(f"booster*src{int(s)}")]), 4)
                                                        for s in np.unique(s_all[fit_mask])}},
                  "fit_rows": int(fit_mask.sum()), "holdout_rows": int((~fit_mask).sum()), "with_booster": booster is not None,
                  "with_seq": "seq" in signals})
    scorer.set_blend(blend)
    p_hold = _sigmoid(X_all[~fit_mask] @ scorer.blend_coef + scorer.blend_intercept)
    hold = {"auroc": auroc(y_all[~fit_mask], p_hold), "sources": {}}
    for s_ in np.unique(s_all[~fit_mask]):
        m = s_all[~fit_mask] == s_
        hold["sources"][names.get(int(s_), "generic") if s_ else "generic"] = auroc(y_all[~fit_mask][m], p_hold[m])
    blend["holdout"] = hold
    shrunk = {k: v for k, v in blend["interaction_shrink"]["per_source"].items() if v < 0.999}
    log(f"  blender fit on {blend['fit_rows']:,} rows; held-out half auroc {hold['auroc']:.4f}; per-bank weights shrunk "
        f"toward the shared ones for {len(shrunk)} banks (smallest: "
        + ", ".join(f"{k} {v:.2f}" for k, v in sorted(shrunk.items(), key=lambda kv: kv[1])[:3]) + f") ({time.time() - t0:.0f}s)")
    del X, Xg, X_all, rf, rfg

    # ---- pass 3: everything, once ---------------------------------------------------------------
    r = score_split(model, va, batch_size=batch_size, n_samples=n_samples, device=device, keep_features=False,
                    scorer=scorer, booster=booster, progress=lambda n: log(f"  calibrate: scored {n:,}/{len(va):,}"), seq=data.seq, split_name="val", seqhead=seqhead, heads=heads, graph=graph, specialists=specialists)
    log(f"  pass 3: scored {len(va):,} val rows ({time.time() - t0:.0f}s)")
    rgen = score_split(model, va, batch_size=batch_size, n_samples=n_samples, device=device, idx=gen_idx,
                       keep_features=False, scorer=scorer, booster=booster, src_override=0, keep_mu=False, seq=data.seq, split_name="val", seqhead=seqhead, heads=heads, graph=graph, specialists=specialists)
    gy = y[gen_idx]
    score = r[score_type]

    cal = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "n_samples": n_samples, "prepared": prepared,
        "schema_version": data.meta.get("schema_version"),
        "features": {"numeric": data.numeric_names, "categorical": data.categorical_names},
        "score_type": score_type, "score_type_candidates": cands,
        "baselines": baselines, "fpr_targets": list(FPR_TARGETS),
        "sources": {}, "booster": {"sources": {}} if booster is not None else None,
        "blend": {**blend, "sources": {}},
    }
    for sig in extra_signals(r):
        cal[sig] = {"sources": {}}
    cal.setdefault("seq", None)
    std_score = np.zeros_like(score)
    for s in np.unique(src):
        m = src == s
        name = names.get(int(s), str(int(s)))
        block = _dist_block(y[m], score[m], extra={
            "source_index": int(s),
            "auroc_raw": auroc(y[m], r["raw"][m]), "auroc_norm": auroc(y[m], r["norm"][m]),
            "feature_observed": np.concatenate([va.mask[:][m].mean(0), (va.cat[:][m] != 0).mean(0)]).round(4).tolist(),
        })
        std_score[m] = (score[m] - block["legit_mean"]) / block["legit_std"]
        cal["sources"][name] = block
        if booster is not None:
            cal["booster"]["sources"][name] = _dist_block(y[m], _sigmoid(r["booster"][m]))
        extra_txt = ""
        for sig in extra_signals(r):
            sb = _signal_block(y[m], r[sig][m])
            cal[sig]["sources"][name] = sb
            extra_txt += f" | {sig} {sb['auroc']:.4f}" if sb else f" | {sig} -"
        cal["blend"]["sources"][name] = _dist_block(y[m], r["blend"][m])
        log(f"  {name}: vae {block['auroc']:.4f} (raw {block['auroc_raw']:.4f} / norm {block['auroc_norm']:.4f})"
            + (f" | booster {cal['booster']['sources'][name]['auroc']:.4f}" if booster is not None else "")
            + extra_txt
            + f" | blend {cal['blend']['sources'][name]['auroc']:.4f} recall@1%fpr "
            f"{cal['blend']['sources'][name]['thresholds']['fpr_0.01']['recall']:.3f}")

    note = "scored with the source token set to generic (0); legit rows subsampled per source"
    cal["generic"] = _dist_block(gy, rgen[score_type], extra={
        "auroc_raw": auroc(gy, rgen["raw"]), "auroc_norm": auroc(gy, rgen["norm"]), "note": note})
    if booster is not None:
        cal["booster"]["generic"] = _dist_block(gy, _sigmoid(rgen["booster"]), extra={"note": note})
    for sig in extra_signals(rgen):
        cal[sig]["generic"] = _signal_block(gy, rgen[sig], extra={"note": note})
    cal["blend"]["generic"] = _dist_block(gy, rgen["blend"], extra={"note": note})
    log(f"  generic context: vae {cal['generic']['auroc']:.4f}"
        + (f" | booster {cal['booster']['generic']['auroc']:.4f}" if booster is not None else "")
        + f" | blend {cal['blend']['generic']['auroc']:.4f} on {len(gen_idx):,} rows")
    cal["pooled"] = {"auroc": auroc(y, std_score), "auprc": auprc(y, std_score),
                     "blend_auroc": auroc(y, r["blend"]), "blend_auprc": auprc(y, r["blend"]),
                     "note": "all sources together: VAE on per-source standardised scores, blend on p(fraud)"}
    cal["mean_val_auroc"] = {
        "vae": float(np.mean([b["auroc"] for b in cal["sources"].values()])),
        "booster": float(np.mean([b["auroc"] for b in cal["booster"]["sources"].values()])) if booster is not None else None,
        "blend": float(np.mean([b["auroc"] for b in cal["blend"]["sources"].values()])),
    }
    for sig in extra_signals(r):
        vals = [b["auroc"] for b in cal[sig]["sources"].values() if b]
        cal["mean_val_auroc"][sig] = float(np.mean(vals)) if vals else None
    cal["mean_val_auroc"].setdefault("seq", None)

    # ---- latent map ------------------------------------------------------------------------------
    legit_idx = np.flatnonzero(y == 0)
    fit_idx = rng.choice(legit_idx, min(200_000, len(legit_idx)), replace=False)
    mu_all = r["mu"]
    center = mu_all[fit_idx].mean(0)
    _, sv, vt = np.linalg.svd(mu_all[fit_idx] - center, full_matrices=False)
    comps = vt[:2]
    explained = (sv[:2] ** 2 / (sv ** 2).sum()).tolist()
    cal["latent_pca"] = {"mean": center.tolist(), "components": comps.tolist(), "explained_variance_ratio": explained}
    keep = []
    for s in np.unique(src):
        idx = np.flatnonzero(src == s)
        fr, lg = idx[y[idx] == 1], idx[y[idx] == 0]
        keep.append(rng.choice(lg, min(2500, len(lg)), replace=False))
        keep.append(rng.choice(fr, min(1200, len(fr)), replace=False) if len(fr) else fr)
    keep = np.concatenate(keep)
    xy = (mu_all[keep] - center) @ comps.T
    np.savez_compressed(run_dir / "latent_map.npz", xy=xy.astype(np.float32), y=y[keep].astype(np.int8),
                        src=src[keep].astype(np.int8), score=std_score[keep].astype(np.float32),
                        blend=r["blend"][keep].astype(np.float32))

    (run_dir / "calibration.json").write_text(json.dumps(cal, indent=1))
    log(f"  calibration written ({len(keep):,} latent points, pca explains {sum(explained):.0%}, {time.time() - t0:.0f}s total)")
    return cal


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run", help="run name under runs/ or a path")
    ap.add_argument("--prepared", default=None, help="defaults to the set the run was trained on")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--samples", type=int, default=8)
    args = ap.parse_args(argv)
    run_dir = Path(args.run) if Path(args.run).exists() else RUNS_DIR / args.run
    prepared = args.prepared or json.loads((run_dir / "run.json").read_text()).get("prepared", "default")
    calibrate_run(run_dir, prepared=prepared, device=torch.device(args.device), n_samples=args.samples)
    return 0


if __name__ == "__main__":
    sys.exit(main())
