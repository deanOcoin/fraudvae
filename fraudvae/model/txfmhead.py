"""v7: the supervised head on the transaction encoder's block - the "txsup" signal.

The encoder is pretrained with no label in sight and the booster reads its block through a
few hundred tree splits; the fraud labels have never reached the encoder's state directly.
This head does that, the cheapest way: per source with enough labelled fraud, a small MLP on
the standardised block (state, per-field surprise, mean surprise), trained on the same
labelled subsample the booster uses and early stopped on its own validation AUROC. Its logit
is a blend signal beside the booster, the specialist and the sequence head - present on the
rows of a source that has a head, NaN everywhere else, like the specialists' "local". It reads
the block through the scorer, so the cache `fraudvae txfm-cache` wrote serves it and no encoder
pass is needed; a row whose block is NaN throughout (gated, routed away) gets no logit.

The encoder stays frozen here. Whether the labels should reach deeper (the top layers unfrozen,
`docs/TRAINING_NOTES.md` v7) is decided by what this head earns first.

    fraudvae txfm-head RUN --prepared SET      # train the head for a run that holds txfm.pt, then
                                               # calibrate and evaluate again so the blend reads it
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .booster import censor_recent
from .dataset import PreparedSet
from .loading import load_checkpoint
from .scoring import auroc

SIGNAL = "txsup"
FILE = "txfmhead.pt"


def _mlp(in_dim: int, hidden: int, dropout: float) -> nn.Module:
    return nn.Sequential(nn.Linear(in_dim, hidden), nn.SiLU(), nn.Dropout(dropout),
                         nn.Linear(hidden, hidden // 2), nn.SiLU(), nn.Linear(hidden // 2, 1))


class TxfmHead(nn.Module):
    """one MLP per source over the standardised block; logit() answers NaN for rows of a source
    without a head and for rows whose block is NaN throughout"""

    def __init__(self, in_dim: int, sources: list[int], hidden: int = 256, dropout: float = 0.2):
        super().__init__()
        self.in_dim, self.hidden, self.dropout_p = int(in_dim), int(hidden), float(dropout)
        self.sources = [int(s) for s in sources]
        self.nets = nn.ModuleDict({str(s): _mlp(self.in_dim, self.hidden, self.dropout_p) for s in self.sources})
        self.register_buffer("mean", torch.zeros(len(self.sources), self.in_dim))
        self.register_buffer("std", torch.ones(len(self.sources), self.in_dim))

    def standardise(self, x: torch.Tensor, k: int) -> torch.Tensor:
        z = (x - self.mean[k]) / self.std[k]
        return torch.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)   # an unobserved field sits at the mean

    def forward(self, x: torch.Tensor, k: int) -> torch.Tensor:
        return self.nets[str(self.sources[k])](self.standardise(x, k)).squeeze(1)

    @torch.no_grad()
    def logit(self, block: np.ndarray, src) -> np.ndarray:
        src = np.asarray(src.cpu().numpy() if hasattr(src, "cpu") else src).astype(np.int64).reshape(-1)
        block = np.asarray(block, dtype=np.float32)
        out = np.full(len(src), np.nan, dtype=np.float32)
        device = self.mean.device
        for k, s in enumerate(self.sources):
            m = (src == s) & ~np.isnan(block).all(1)
            if m.any():
                out[m] = self.forward(torch.from_numpy(block[m]).to(device), k).float().cpu().numpy()
        return out

    def config(self) -> dict:
        return {"in_dim": self.in_dim, "sources": self.sources, "hidden": self.hidden, "dropout": self.dropout_p}


def load_txfmhead(run_dir: Path, device="cpu") -> TxfmHead | None:
    p = Path(run_dir) / FILE
    if not p.exists():
        return None
    ck = load_checkpoint(p, device)
    head = TxfmHead(ck["in_dim"], ck["sources"], ck["hidden"], ck["dropout"])
    head.load_state_dict(ck["state_dict"])
    head.to(device).eval()
    return head


def _rows(split, s: int, per_source_legit: int, rng) -> np.ndarray:
    idx = np.flatnonzero(split.src[:] == s)
    y = split.y[:][idx]
    legit, fraud = idx[y == 0], idx[y == 1]
    if per_source_legit and len(legit) > per_source_legit:
        legit = rng.choice(legit, per_source_legit, replace=False)
    return np.sort(np.concatenate([legit, fraud]))


def train_txfmhead(run_dir: Path, prepared: str, txfm, device="cpu", log=print, max_legit_per_source=200_000, val_legit_per_source=100_000,
                   min_fraud=300, epochs=20, patience=3, batch_size=4096, lr=1e-3, hidden=256, dropout=0.2, seed=1337,
                   label_horizon_days=0.0) -> dict | None:
    """fit one head per source with at least min_fraud labelled fraud rows in train (the
    specialists' rule), on the encoder's block of the booster's labelled subsample, early
    stopped on that source's validation AUROC; write txfmhead.pt + txfmhead.json. None when no
    source qualifies. txfm: the run's TxfmScorer (its cache, gate and routing included)."""
    run_dir = Path(run_dir)
    t0 = time.time()
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    device = torch.device(device)
    data = PreparedSet(prepared, splits=("train", "val"))
    if data.seq is None:
        log("  encoder head: the prepared set has no sequence index, skipping")
        return None
    tr, va = data["train"], data["val"]
    names = data.source_names
    src_tr, y_tr = tr.src[:], tr.y[:]
    trained, per_source = [], {}
    for s in np.unique(src_tr):
        key = names.get(int(s), str(int(s)))
        n_fraud = int(((src_tr == s) & (y_tr == 1)).sum())
        if n_fraud < min_fraud:
            log(f"  encoder head {key}: {n_fraud} labelled fraud rows, fewer than {min_fraud}, skipping")
            continue
        ts = time.time()
        tr_idx, va_idx = _rows(tr, int(s), max_legit_per_source, rng), _rows(va, int(s), val_legit_per_source, rng)
        tr_idx, censored = censor_recent(tr, tr_idx, va, label_horizon_days, log=log, what=f"encoder head {key}")
        Xb = txfm.features_split(tr, tr_idx, data.seq, "train", device)
        Xv = txfm.features_split(va, va_idx, data.seq, "val", device)
        keep = ~np.isnan(Xb).all(1)                       # gated or routed-away rows carry no block
        if keep.sum() < 50 or y_tr[tr_idx][keep].sum() < 20:
            log(f"  encoder head {key}: the block is blank on too many rows ({int(keep.sum())} with one), skipping")
            continue
        Xb, yb = Xb[keep], y_tr[tr_idx][keep].astype(np.float32)
        yv, keep_v = va.y[:][va_idx].astype(np.float32), ~np.isnan(Xv).all(1)
        mean = np.nanmean(Xb, axis=0)
        std = np.nanstd(Xb, axis=0)
        mean = np.where(np.isfinite(mean), mean, 0.0).astype(np.float32)
        std = np.where(np.isfinite(std) & (std > 1e-6), std, 1.0).astype(np.float32)
        head = TxfmHead(Xb.shape[1], [int(s)], hidden, dropout).to(device)
        head.mean[0], head.std[0] = torch.from_numpy(mean).to(device), torch.from_numpy(std).to(device)
        pos_weight = min(float((yb == 0).sum() / max(1.0, yb.sum())), 50.0)
        opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-5)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))
        Xb_t, yb_t = torch.from_numpy(Xb).to(device), torch.from_numpy(yb).to(device)
        Xv_t = torch.from_numpy(Xv[keep_v]).to(device)
        best, best_state, best_ep, bad = -1.0, None, 0, 0
        for ep in range(epochs):
            head.train()
            order = torch.from_numpy(rng.permutation(len(Xb_t))).to(device)
            tot = 0.0
            for i in range(0, len(order), batch_size):
                b = order[i:i + batch_size]
                loss = loss_fn(head(Xb_t[b], 0), yb_t[b])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                tot += float(loss) * len(b)
            head.eval()
            with torch.no_grad():
                lv = np.full(len(va_idx), np.nan, dtype=np.float32)
                lv[keep_v] = torch.cat([head(Xv_t[i:i + 8192], 0) for i in range(0, len(Xv_t), 8192)]).float().cpu().numpy() if len(Xv_t) else lv[keep_v]
            a = auroc(yv[keep_v], lv[keep_v]) if keep_v.any() else float("nan")
            if a == a and a > best + 1e-4:
                best, best_ep, bad = a, ep + 1, 0
                best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
            else:
                bad += 1
            if bad >= patience:
                break
        if best_state is None:
            log(f"  encoder head {key}: no validation number to stop on, skipping")
            continue
        head.load_state_dict(best_state)
        trained.append((int(s), head))
        per_source[key] = {"auroc": best, "best_epoch": best_ep, "epochs_run": ep + 1, "train_rows": int(len(Xb)), "train_fraud": int(yb.sum()),
                           "censored_rows": censored,
                           "val_rows": int(keep_v.sum()), "val_fraud": int(yv[keep_v].sum()), "pos_weight": pos_weight, "seconds": round(time.time() - ts, 1)}
        log(f"  encoder head {key}: {len(Xb):,} rows ({int(yb.sum()):,} fraud), best val auroc {best:.4f} at epoch {best_ep} ({time.time() - ts:.0f}s)")
        del Xb_t, yb_t, Xv_t
    if not trained:
        log("  encoder head: no source qualified, nothing written")
        return None
    merged = TxfmHead(trained[0][1].in_dim, [s for s, _ in trained], hidden, dropout)
    for k, (s, head) in enumerate(trained):
        merged.nets[str(s)].load_state_dict(head.nets[str(s)].state_dict())
        merged.mean[k], merged.std[k] = head.mean[0].cpu(), head.std[0].cpu()
    merged.eval()
    torch.save({"state_dict": merged.state_dict(), **merged.config()}, run_dir / FILE)
    info = {"created": time.strftime("%Y-%m-%dT%H:%M:%S"), "prepared": prepared, "signal": SIGNAL, **merged.config(),
            "source_names": [names.get(s, str(s)) for s, _ in trained], "sources": per_source,
            "val_auroc": float(np.mean([v["auroc"] for v in per_source.values()])), "min_fraud": min_fraud, "epochs": epochs,
            "label_horizon_days": float(label_horizon_days or 0.0),
            "patience": patience, "lr": lr, "seconds": round(time.time() - t0, 1)}
    (run_dir / "txfmhead.json").write_text(json.dumps(info, indent=1))
    log("  encoder head val auroc: " + " ".join(f"{k}={v['auroc']:.3f}" for k, v in per_source.items()) + f" ({time.time() - t0:.0f}s)")
    return info


def main(argv=None):
    from ..calibrate import calibrate_run
    from ..evaluate import evaluate_run
    from ..manifest import clear_evaluations, write_manifest
    from ..paths import RUNS_DIR
    from .booster import fast_columns
    from .txfm import TxfmScorer

    ap = argparse.ArgumentParser(prog="fraudvae txfm-head", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", help="a run holding txfm.pt and a booster (a v6 post-stage run)")
    ap.add_argument("--prepared", default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--min-fraud", type=int, default=300)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--samples", type=int, default=8, help="posterior samples for the calibration and evaluation that follow")
    args = ap.parse_args(argv)
    run_dir = Path(args.run) if Path(args.run).exists() else RUNS_DIR / args.run
    run = json.loads((run_dir / "run.json").read_text())
    prepared = args.prepared or run.get("prepared", "default")
    txfm = TxfmScorer.load(run_dir, device=args.device)
    if txfm is None:
        raise SystemExit(f"{run_dir} has no txfm.pt: the head reads the encoder's block")
    txfm.with_cache(prepared)
    log = print
    info = train_txfmhead(run_dir, prepared, txfm, device=args.device, log=log, min_fraud=args.min_fraud, epochs=args.epochs)
    if info is None:
        return 1
    # the blend has to learn the new signal, and the receipt has to be remade with it
    clear_evaluations(run_dir, log=log)
    calibrate_run(run_dir, prepared=prepared, device=args.device, n_samples=args.samples, log=log)
    evaluate_run(run_dir, prepared=prepared, device=args.device, n_samples=args.samples, log=log)
    bj = run_dir / "booster.json"
    if bj.exists() and fast_columns(json.loads(bj.read_text()).get("features", [])):
        from ..data.schema import ENTITY_LABEL_DELAY_DAYS

        evaluate_run(run_dir, prepared=prepared, device=args.device, n_samples=args.samples, log=log, label_delay_days=ENTITY_LABEL_DELAY_DAYS)
    write_manifest(run_dir, "txfm-head", None, prepared, log=log)
    return 0


if __name__ == "__main__":
    sys.exit(main())
