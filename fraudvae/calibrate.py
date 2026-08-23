"""Post-training calibration on the validation split.

Pass 1: per-source NLL baselines from a legit subsample (and a "generic" set
        scored with the source token zeroed).
Pass 2: score the whole split once with those baselines, pick the VAE score
        type (raw vs normalised) on mean per-source AUROC and derive operating
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

from .model.dataset import PreparedSet
from .model.scoring import Scorer, auprc, auroc, score_split
from .model.vae import load_checkpoint
from .paths import RUNS_DIR

FPR_TARGETS = (0.001, 0.005, 0.01, 0.02, 0.05)
QS = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.995, 0.999, 0.9999]
BASELINE_ROWS = 100_000      # legit rows per source for pass 1
GENERIC_ROWS = 60_000        # legit rows per source in the generic passes


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


def _subsample(src, y, rng, per_source, with_fraud=True):
    keep = []
    for s in np.unique(src):
        idx = np.flatnonzero(src == s)
        lg = idx[y[idx] == 0]
        keep.append(rng.choice(lg, min(per_source, len(lg)), replace=False))
        if with_fraud:
            keep.append(idx[y[idx] == 1])
    return np.sort(np.concatenate(keep))


def calibrate_run(run_dir: Path, prepared="default", device="cpu", n_samples=8, log=print, batch_size=16384):
    run_dir = Path(run_dir)
    model, ck = load_checkpoint(run_dir / "model.pt", device=device)
    data = PreparedSet(prepared, splits=("val",))
    va = data["val"]
    names = data.source_names
    y, src = va.y[:], va.src[:]
    rng = np.random.default_rng(0)
    n_feat = len(data.numeric_names) + len(data.categorical_names)
    t0 = time.time()

    # ---- pass 1: baselines --------------------------------------------------------
    base_idx = _subsample(src, y, rng, BASELINE_ROWS, with_fraud=False)
    r1 = score_split(model, va, batch_size=batch_size, n_samples=n_samples, device=device, idx=base_idx, keep_mu=False)
    nll1 = np.concatenate([r1["nll_num"], r1["nll_cat"]], axis=1)
    baselines = {"by_source": {}}
    for s in np.unique(src[base_idx]):
        m = src[base_idx] == s
        baselines["by_source"][str(int(s))] = Scorer.fit_baselines(nll1[m], r1["kl"][m])
    gen_idx = _subsample(src, y, rng, GENERIC_ROWS, with_fraud=True)
    g_legit = gen_idx[y[gen_idx] == 0]
    rg = score_split(model, va, batch_size=batch_size, n_samples=n_samples, device=device, idx=g_legit, src_override=0,
                     keep_mu=False)
    baselines["generic"] = Scorer.fit_baselines(np.concatenate([rg["nll_num"], rg["nll_cat"]], axis=1), rg["kl"])
    scorer = Scorer(baselines, n_feat)
    log(f"  pass 1: baselines from {len(base_idx):,} legit rows ({time.time() - t0:.0f}s)")
    del r1, nll1, rg

    # ---- pass 2: everything, once ---------------------------------------------------------------
    r = score_split(model, va, batch_size=batch_size, n_samples=n_samples, device=device, keep_features=False,
                    scorer=scorer, progress=lambda n: log(f"  calibrate: scored {n:,}/{len(va):,}"))
    log(f"  pass 2: scored {len(va):,} val rows ({time.time() - t0:.0f}s)")
    cands = {}
    for kind in ("raw", "norm"):
        aucs = [auroc(y[src == s], r[kind][src == s]) for s in np.unique(src)]
        aucs = [a for a in aucs if a == a]
        cands[kind] = float(np.mean(aucs)) if aucs else float("nan")
    score_type = "norm" if cands["norm"] >= cands["raw"] - 1e-9 else "raw"
    log(f"  mean val auroc raw={cands['raw']:.4f} norm={cands['norm']:.4f} -> using '{score_type}'")
    rgen = score_split(model, va, batch_size=batch_size, n_samples=n_samples, device=device, idx=gen_idx,
                       keep_features=False, scorer=scorer, src_override=0, keep_mu=False)
    gy = y[gen_idx]
    score = r[score_type]

    cal = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "n_samples": n_samples, "prepared": prepared,
        "schema_version": data.meta.get("schema_version"),
        "features": {"numeric": data.numeric_names, "categorical": data.categorical_names},
        "score_type": score_type, "score_type_candidates": cands,
        "baselines": baselines, "fpr_targets": list(FPR_TARGETS),
        "sources": {},
    }
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
        log(f"  {name}: vae {block['auroc']:.4f} (raw {block['auroc_raw']:.4f} / norm {block['auroc_norm']:.4f}) recall@1%fpr "
            f"{block['thresholds']['fpr_0.01']['recall']:.3f}")

    note = "scored with the source token set to generic (0); legit rows subsampled per source"
    cal["generic"] = _dist_block(gy, rgen[score_type], extra={
        "auroc_raw": auroc(gy, rgen["raw"]), "auroc_norm": auroc(gy, rgen["norm"]), "note": note})
    log(f"  generic context: vae {cal['generic']['auroc']:.4f} on {len(gen_idx):,} rows")
    cal["pooled"] = {"auroc": auroc(y, std_score), "auprc": auprc(y, std_score),
                     "note": "all sources together on per-source standardised scores"}
    cal["mean_val_auroc"] = float(np.mean([b["auroc"] for b in cal["sources"].values()]))

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
                        src=src[keep].astype(np.int8), score=std_score[keep].astype(np.float32))

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
