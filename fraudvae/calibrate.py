"""Post-training calibration on the validation split.

Scores the validation split once and derives per-source operating thresholds
and legit score quantiles.

Writes runs/<name>/calibration.json.
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
from .model.scoring import auprc, auroc, score_split
from .model.vae import load_checkpoint
from .paths import RUNS_DIR

FPR_TARGETS = (0.001, 0.005, 0.01, 0.02, 0.05)
QS = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.999]


def _thresholds(y, s):
    out = {}
    legit, fraud = s[y == 0], s[y == 1]
    for fpr in FPR_TARGETS:
        thr = float(np.quantile(legit, 1 - fpr)) if len(legit) else float("nan")
        flagged = s > thr
        tp = int((flagged & (y == 1)).sum())
        out[f"fpr_{fpr}"] = {
            "target_fpr": fpr, "threshold": thr,
            "recall": float(tp / max(len(fraud), 1)) if len(fraud) else None,
            "flagged_share": float(flagged.mean()),
        }
    return out


def calibrate_run(run_dir: Path, prepared="default", device="cpu", n_samples=8, log=print, batch_size=16384):
    run_dir = Path(run_dir)
    model, ck = load_checkpoint(run_dir / "model.pt", device=device)
    data = PreparedSet(prepared, splits=("val",))
    va = data["val"]
    names = data.source_names
    y, src = va.y[:], va.src[:]
    t0 = time.time()

    r = score_split(model, va, batch_size=batch_size, n_samples=n_samples, device=device, keep_features=False,
                    keep_mu=False, progress=lambda n: log(f"  calibrate: scored {n:,}/{len(va):,}"))
    log(f"  scored {len(va):,} val rows ({time.time() - t0:.0f}s)")
    score = r["raw"]

    cal = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "n_samples": n_samples, "prepared": prepared,
        "schema_version": data.meta.get("schema_version"),
        "features": {"numeric": data.numeric_names, "categorical": data.categorical_names},
        "score_type": "raw", "fpr_targets": list(FPR_TARGETS),
        "sources": {},
    }
    std_score = np.zeros_like(score)
    for s in np.unique(src):
        m = src == s
        name = names.get(int(s), str(int(s)))
        ys, ss = y[m], score[m]
        legit = ss[ys == 0]
        block = {
            "source_index": int(s), "n": int(m.sum()), "fraud": int(ys.sum()),
            "auroc": auroc(ys, ss), "auprc": auprc(ys, ss),
            "legit_mean": float(legit.mean()), "legit_std": float(max(legit.std(), 1e-6)),
            "legit_quantiles": {str(q): float(np.quantile(legit, q)) for q in QS},
            "thresholds": _thresholds(ys, ss),
        }
        std_score[m] = (ss - block["legit_mean"]) / block["legit_std"]
        cal["sources"][name] = block
        log(f"  {name}: auroc {block['auroc']:.4f} recall@1%fpr {block['thresholds']['fpr_0.01']['recall']:.3f}")

    cal["pooled"] = {"auroc": auroc(y, std_score), "auprc": auprc(y, std_score),
                     "note": "all sources together on per-source standardised scores"}

    (run_dir / "calibration.json").write_text(json.dumps(cal, indent=1))
    log(f"  calibration written ({time.time() - t0:.0f}s total)")
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
