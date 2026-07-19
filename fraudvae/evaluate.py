"""Held-out test metrics for a run -> runs/<name>/eval.json

Three scores per row (VAE novelty, booster, blend), each judged per source
with the thresholds that calibration froze on val, so the realised FPR /
recall / precision here are honest "what would have happened" numbers. The
headline mean_auroc is the blend's, because that is what the app flags on.

A generic pass scores a test subsample with the source token blanked, which
is the number for "a bank the model has never seen".
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from .calibrate import _subsample
from .model.booster import Booster, Specialists
from .graph.scorer import GraphScorer
from .model.heads import load_heads
from .model.seqhead import load_seqhead
from .model.dataset import PreparedSet
from .model.scoring import Scorer, auprc, auroc, extra_signals, recall_at_fpr, score_split
from .model.vae import load_checkpoint
from .paths import RUNS_DIR

FPRS = (0.001, 0.005, 0.01, 0.02, 0.05)
GENERIC_ROWS = 60_000


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def _metrics(ys, ss, thresholds: dict | None = None) -> dict:
    rec = {"auroc": auroc(ys, ss), "auprc": auprc(ys, ss), "recall_at_fpr": {}}
    for fpr in FPRS:
        _, rr = recall_at_fpr(ys, ss, fpr)
        rec["recall_at_fpr"][str(fpr)] = rr
    if thresholds:
        rec["at_val_thresholds"] = {}
        for k, t in thresholds.items():
            if not t:
                continue
            flagged = ss > t["threshold"]
            tp = int((flagged & (ys == 1)).sum())
            fp = int((flagged & (ys == 0)).sum())
            rec["at_val_thresholds"][k] = {
                "threshold": t["threshold"],
                "fpr": float(fp / max((ys == 0).sum(), 1)),
                "recall": float(tp / max(ys.sum(), 1)) if ys.sum() else None,
                "precision": float(tp / max(tp + fp, 1)) if tp + fp else None,
                "flagged": int(tp + fp),
            }
    return rec


def _signal_metrics(ys, logit, thresholds=None, min_rows=200):
    """_metrics over the rows where a head's signal is present; None when too few carry it"""
    ok = np.isfinite(logit)
    if ok.sum() < min_rows or ys[ok].sum() < 5 or (ys[ok] == 0).sum() < 50:
        return None
    rec = _metrics(ys[ok], _sigmoid(logit[ok]), thresholds)
    rec["coverage"] = float(ok.mean())
    rec["n"] = int(ok.sum())
    return rec


def evaluate_run(run_dir: Path, prepared="default", device="cpu", n_samples=8, log=print, batch_size=16384,
                 label_delay_days=None):
    """label_delay_days: score as a deployment with that label latency would (the one-week
    label block blanked past 7 days); the result then goes to eval_delay<days>.json next to
    eval.json, thresholds unchanged (they were frozen on validation as trained)"""
    run_dir = Path(run_dir)
    tag = f" ({label_delay_days:g}-day labels)" if label_delay_days else ""
    model, ck = load_checkpoint(run_dir / "model.pt", device=device)
    cal = json.loads((run_dir / "calibration.json").read_text())
    booster = Booster.load(run_dir)
    seqhead = load_seqhead(run_dir, device=device)
    heads = load_heads(run_dir, device=device)
    graph = GraphScorer.load(run_dir, device=device)
    specialists = Specialists.load(run_dir)
    data = PreparedSet(prepared, splits=("test",))
    te = data["test"]
    names = data.source_names
    n_feat = len(data.numeric_names) + len(data.categorical_names)
    scorer = Scorer(cal["baselines"], n_feat, blend=cal.get("blend"))
    score_type = cal["score_type"]
    t0 = time.time()
    r = score_split(model, te, batch_size=batch_size, n_samples=n_samples, device=device, keep_features=False,
                    keep_mu=False, scorer=scorer, booster=booster,
                    progress=lambda n: log(f"  evaluate{tag}: scored {n:,}/{len(te):,}"), seq=data.seq, split_name="test",
                    seqhead=seqhead, heads=heads, graph=graph, specialists=specialists, label_delay_days=label_delay_days)
    y, src = te.y[:], te.src[:]
    s = r[score_type]
    log(f"  scored {len(te):,} test rows in {time.time() - t0:.0f}s (vae score type: {score_type})")

    out = {"created": time.strftime("%Y-%m-%dT%H:%M:%S"), "n_samples": n_samples, "score_type": score_type,
           "schema_version": cal.get("schema_version"), "with_booster": booster is not None,
           "label_delay_days": label_delay_days, "sources": {}, "n": int(len(te))}
    std_score = np.zeros_like(s)
    for code in np.unique(src):
        m = src == code
        name = names.get(int(code), str(int(code)))
        ys = y[m]
        c = cal["sources"].get(name)
        rec = {"n": int(m.sum()), "fraud": int(ys.sum())}
        rec["vae"] = _metrics(ys, s[m], c["thresholds"] if c else None)
        rec["vae"]["auroc_raw"] = auroc(ys, r["raw"][m])
        rec["vae"]["auroc_norm"] = auroc(ys, r["norm"][m])
        if c:
            std_score[m] = (s[m] - c["legit_mean"]) / c["legit_std"]
        if booster is not None:
            cb = (cal.get("booster") or {}).get("sources", {}).get(name)
            rec["booster"] = _metrics(ys, _sigmoid(r["booster"][m]), cb["thresholds"] if cb else None)
        for sig in extra_signals(r):
            cs = (cal.get(sig) or {}).get("sources", {}).get(name)
            rec[sig] = _signal_metrics(ys, r[sig][m], cs["thresholds"] if cs else None)
        cbl = cal["blend"]["sources"].get(name)
        rec["blend"] = _metrics(ys, r["blend"][m], cbl["thresholds"] if cbl else None)
        # the receipt's headline columns, kept flat for the ui
        rec["auroc"] = rec["blend"]["auroc"]
        rec["auprc"] = rec["blend"]["auprc"]
        out["sources"][name] = rec
        log(f"  {name}{tag}: vae {rec['vae']['auroc']:.4f}"
            + (f" | booster {rec['booster']['auroc']:.4f}" if "booster" in rec else "")
            + "".join(f" | {sig} {rec[sig]['auroc']:.4f}" if rec.get(sig) else f" | {sig} -" for sig in extra_signals(r))
            + f" | blend {rec['blend']['auroc']:.4f} auprc {rec['blend']['auprc']:.4f} "
            f"recall@1%fpr {rec['blend']['recall_at_fpr']['0.01']:.3f}")

    # generic: a test subsample with the source token blanked
    rng = np.random.default_rng(1)
    gen_idx = _subsample(src, y, rng, GENERIC_ROWS, with_fraud=True)
    rg = score_split(model, te, batch_size=batch_size, n_samples=n_samples, device=device, idx=gen_idx, keep_features=False,
                     keep_mu=False, scorer=scorer, booster=booster, src_override=0, seq=data.seq, split_name="test",
                     seqhead=seqhead, heads=heads, graph=graph, specialists=specialists, label_delay_days=label_delay_days)
    gy = y[gen_idx]
    gen = {"n": int(len(gen_idx)), "fraud": int(gy.sum()),
           "vae": _metrics(gy, rg[score_type], cal["generic"]["thresholds"]),
           "blend": _metrics(gy, rg["blend"], cal["blend"]["generic"]["thresholds"]),
           "note": "test rows scored with the source token set to generic (0); legit rows subsampled per source"}
    if booster is not None:
        gen["booster"] = _metrics(gy, _sigmoid(rg["booster"]), cal["booster"]["generic"]["thresholds"])
    for sig in extra_signals(rg):
        gen[sig] = _signal_metrics(gy, rg[sig], ((cal.get(sig) or {}).get("generic") or {}).get("thresholds"))
    gen["auroc"] = gen["blend"]["auroc"]
    out["generic"] = gen
    log(f"  generic: vae {gen['vae']['auroc']:.4f}"
        + (f" | booster {gen['booster']['auroc']:.4f}" if "booster" in gen else "")
        + f" | blend {gen['blend']['auroc']:.4f} recall@1%fpr {gen['blend']['recall_at_fpr']['0.01']:.3f}")

    out["pooled"] = {"auroc": auroc(y, std_score), "auprc": auprc(y, std_score),
                     "blend_auroc": auroc(y, r["blend"]), "blend_auprc": auprc(y, r["blend"])}

    def _mean(key):
        v = [x[key]["auroc"] for x in out["sources"].values() if x.get(key) and x[key]["auroc"] == x[key]["auroc"]]
        return float(np.mean(v)) if v else None

    out["mean"] = {"vae": _mean("vae"), "booster": _mean("booster"), "seq": _mean("seq"), "blend": _mean("blend")}
    for sig in extra_signals(r):
        out["mean"][sig] = _mean(sig)
    out["mean_auroc"] = out["mean"]["blend"]
    out["mean_vae_auroc"] = out["mean"]["vae"]
    (run_dir / (f"eval_delay{label_delay_days:g}.json" if label_delay_days else "eval.json")).write_text(json.dumps(out, indent=1))
    log(f"  mean test auroc{tag}: vae {out['mean']['vae']:.4f}"
        + (f" | booster {out['mean']['booster']:.4f}" if out["mean"]["booster"] else "")
        + "".join(f" | {sig} {out['mean'][sig]:.4f}" for sig in extra_signals(r) if out["mean"].get(sig))
        + f" | blend {out['mean']['blend']:.4f}")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run")
    ap.add_argument("--prepared", default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--label-delay", type=float, default=None,
                    help="score as a deployment whose fraud labels take this many days (blanks the one-week block past 7); "
                         "writes eval_delay<N>.json")
    args = ap.parse_args(argv)
    run_dir = Path(args.run) if Path(args.run).exists() else RUNS_DIR / args.run
    prepared = args.prepared or json.loads((run_dir / "run.json").read_text()).get("prepared", "default")
    evaluate_run(run_dir, prepared=prepared, device=torch.device(args.device), n_samples=args.samples,
                 label_delay_days=args.label_delay)
    return 0


if __name__ == "__main__":
    sys.exit(main())
