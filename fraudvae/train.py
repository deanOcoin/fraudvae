"""Train the VAE on legitimate transactions.

    python -m fraudvae.train --config configs/cpu_small.yaml

Writes runs/<name>/{run.json, metrics.jsonl, status.json, train.log, model.pt,
calibration.json, latent_map.npz}
"""
from __future__ import annotations

import argparse
import json
import math
import os
import socket
import sys
import time
from pathlib import Path

import numpy as np
import torch

from . import __version__
from .config import Config, load_config, to_dict
from .model.dataset import PreparedSet
from .model.scoring import auroc, batches, score_split
from .model.vae import build_model, save_checkpoint
from .paths import RUNS_DIR, ensure_dirs


def pick_device(name="auto"):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class Logger:
    def __init__(self, run_dir: Path):
        self.f = open(run_dir / "train.log", "a")

    def __call__(self, msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        self.f.write(line + "\n")
        self.f.flush()


def write_status(run_dir: Path, **kw):
    p = run_dir / "status.json"
    cur = json.loads(p.read_text()) if p.exists() else {}
    cur.update(kw)
    cur["updated"] = time.time()
    p.write_text(json.dumps(cur, indent=1))


def _eval_indices(split, per_source=300_000, seed=0):
    """subsample legit rows per source for the per-epoch eval, keep every fraud row"""
    rng = np.random.default_rng(seed)
    src = split.src[:]
    y = split.y[:]
    keep = []
    for s in np.unique(src):
        idx = np.flatnonzero(src == s)
        fraud = idx[y[idx] == 1]
        legit = idx[y[idx] == 0]
        if per_source and len(legit) > per_source:
            legit = rng.choice(legit, per_source, replace=False)
        keep.append(np.concatenate([legit, fraud]))
    return np.sort(np.concatenate(keep))


def evaluate_split(model, split, idx, source_names, device, batch_size=8192, n_samples=1):
    """per-epoch check on a val subsample. raw -elbo score only; the normalised
    score needs baselines which calibrate() builds at the end"""
    r = score_split(model, split, batch_size=batch_size, n_samples=n_samples, device=device, idx=idx, keep_features=False)
    y = split.y[idx]
    src = split.src[idx]
    out = {"n": int(len(idx)), "per_source": {}}
    aucs = []
    for s in np.unique(src):
        m = src == s
        name = source_names.get(int(s), str(s))
        a = auroc(y[m], r["raw"][m])
        out["per_source"][name] = {
            "auroc": a,
            "legit_score_mean": float(r["raw"][m & (y == 0)].mean()) if (m & (y == 0)).any() else None,
            "fraud_score_mean": float(r["raw"][m & (y == 1)].mean()) if (m & (y == 1)).any() else None,
            "n": int(m.sum()), "fraud": int(y[m].sum()),
        }
        if not math.isnan(a):
            aucs.append(a)
    out["mean_auroc"] = float(np.mean(aucs)) if aucs else float("nan")
    legit = y == 0
    out["legit_score_mean"] = float(r["raw"][legit].mean())
    out["legit_kl_mean"] = float(r["kl"][legit].mean())
    return out


def train(cfg: Config, prepared="default", run_dir: Path | None = None, name: str | None = None):
    ensure_dirs()
    torch.manual_seed(cfg.train.seed)
    np.random.seed(cfg.train.seed)
    device = pick_device(cfg.device)
    name = name or f"{cfg.name}-{time.strftime('%Y%m%d-%H%M')}"
    run_dir = Path(run_dir) if run_dir else RUNS_DIR / name
    run_dir.mkdir(parents=True, exist_ok=True)
    log = Logger(run_dir)
    log(f"fraudvae {__version__} | run {run_dir.name} | device {device} | torch {torch.__version__}")

    data = PreparedSet(prepared)
    meta = data.meta
    tr, va = data["train"], data["val"]
    legit_idx = tr.legit_indices()
    log(f"prepared set '{prepared}': train {len(tr):,} rows ({len(legit_idx):,} legit), val {len(va):,}")

    model = build_model(meta, cfg.model).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"model: {n_params / 1e6:.2f}M params, latent {cfg.model.latent_dim}, hidden {cfg.model.hidden}")

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    steps_per_epoch = math.ceil(len(legit_idx) / cfg.train.batch_size)
    total_steps = steps_per_epoch * cfg.train.epochs
    warmup = min(500, steps_per_epoch)

    def lr_at(step):
        if step < warmup:
            return cfg.train.lr * (step + 1) / warmup
        p = (step - warmup) / max(1, total_steps - warmup)
        return cfg.train.lr * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * p)))

    run_info = {
        "name": run_dir.name, "started": time.strftime("%Y-%m-%dT%H:%M:%S"), "config": to_dict(cfg),
        "prepared": prepared, "device": str(device), "host": socket.gethostname(), "torch": torch.__version__,
        "params": n_params, "pid": os.getpid(), "fraudvae": __version__,
    }
    (run_dir / "run.json").write_text(json.dumps(run_info, indent=1))
    write_status(run_dir, state="running", epoch=0, epochs=cfg.train.epochs)

    source_names = data.source_names
    eval_idx = _eval_indices(va, per_source=300_000, seed=cfg.train.seed)
    use_amp = cfg.train.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    metrics_f = open(run_dir / "metrics.jsonl", "a")
    step = 0
    best = -float("inf")
    bad_epochs = 0
    t_start = time.time()

    for epoch in range(cfg.train.epochs):
        model.train()
        beta = cfg.train.beta * min(1.0, (epoch + 1) / max(1, cfg.train.beta_warmup_epochs))
        t0 = time.time()
        sums = {"loss": 0.0, "nll": 0.0, "kl": 0.0}
        seen = 0
        skipped = 0
        for b, (num, mask, cat, src) in batches(tr, cfg.train.batch_size, idx=legit_idx, shuffle=True,
                                                seed=cfg.train.seed + epoch, device=device):
            if cfg.train.source_dropout > 0:
                drop = torch.rand(src.shape[0], device=device) < cfg.train.source_dropout
                src = torch.where(drop, torch.zeros_like(src), src)
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
                out = model(num, mask, cat, src)
                nll_num, nll_cat = model.nll_terms(out, num, mask, cat)
                nll = nll_num.sum(1) + nll_cat.sum(1)
                kl_dims = model.kl_terms(out["mu"].float(), out["logvar"].float())
                if cfg.train.free_bits > 0:
                    kl_dims = torch.clamp(kl_dims, min=cfg.train.free_bits)
                kl = kl_dims.sum(1)
                loss = (nll + beta * kl).mean()
            if not torch.isfinite(loss):
                skipped += 1
                log(f"non-finite loss at step {step}, step skipped")
                opt.zero_grad(set_to_none=True)
                continue
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            scaler.step(opt)
            scaler.update()
            step += 1
            n = num.shape[0]
            seen += n
            sums["loss"] += loss.item() * n
            sums["nll"] += nll.mean().item() * n
            sums["kl"] += kl.mean().item() * n
            if step % 200 == 0:
                rate = seen / (time.time() - t0)
                eta = (len(legit_idx) - seen) / max(rate, 1)
                log(f"ep {epoch + 1}/{cfg.train.epochs} step {step} loss {sums['loss'] / seen:.3f} "
                    f"nll {sums['nll'] / seen:.3f} kl {sums['kl'] / seen:.3f} beta {beta:.2f} lr {lr_at(step):.2e} "
                    f"{rate:,.0f} rows/s, {eta / 60:.1f} min left in epoch")
                write_status(run_dir, state="running", epoch=epoch, epochs=cfg.train.epochs, step=step,
                             progress=seen / len(legit_idx), rows_per_sec=rate, loss=sums["loss"] / seen)

        train_time = time.time() - t0
        rec = {"epoch": epoch + 1, "beta": beta, "lr": lr_at(step), "train_loss": sums["loss"] / max(seen, 1),
               "train_nll": sums["nll"] / max(seen, 1), "train_kl": sums["kl"] / max(seen, 1),
               "rows_per_sec": seen / max(train_time, 1e-6), "epoch_seconds": train_time, "skipped_steps": skipped}
        if (epoch + 1) % cfg.train.eval_every == 0 or epoch + 1 == cfg.train.epochs:
            t1 = time.time()
            ev = evaluate_split(model, va, eval_idx, source_names, device, batch_size=cfg.train.batch_size * 4)
            rec.update({"val_mean_auroc": ev["mean_auroc"], "val_legit_score": ev["legit_score_mean"],
                        "val_legit_kl": ev["legit_kl_mean"], "val_per_source": ev["per_source"],
                        "eval_seconds": time.time() - t1})
            if cfg.train.select_by == "auroc" and not math.isnan(ev["mean_auroc"]):
                crit = ev["mean_auroc"]
            else:
                crit = -ev["legit_score_mean"]   # higher is better: lower held-out -ELBO
            per_src = " ".join(f"{k}={v['auroc']:.3f}" for k, v in ev["per_source"].items() if v["auroc"] == v["auroc"])
            log(f"epoch {epoch + 1} done in {train_time / 60:.1f} min | train loss {rec['train_loss']:.3f} | "
                f"val legit -elbo {ev['legit_score_mean']:.3f} | val auroc mean {ev['mean_auroc']:.4f} ({per_src})")
            if crit > best:
                best = crit
                bad_epochs = 0
                save_checkpoint(run_dir / "model.pt", model, meta, to_dict(cfg),
                                extra={"epoch": epoch, "best": best, "val": ev})
                log(f"  new best ({cfg.train.select_by}), checkpoint saved")
            else:
                bad_epochs += 1
                if bad_epochs >= cfg.train.patience:
                    log(f"  no improvement for {bad_epochs} evals, stopping early")
                    metrics_f.write(json.dumps(rec) + "\n")
                    metrics_f.flush()
                    break
        metrics_f.write(json.dumps(rec) + "\n")
        metrics_f.flush()
        write_status(run_dir, state="running", epoch=epoch + 1, epochs=cfg.train.epochs, best=best,
                     elapsed=time.time() - t_start)

    metrics_f.close()
    if not (run_dir / "model.pt").exists():
        save_checkpoint(run_dir / "model.pt", model, meta, to_dict(cfg), extra={"epoch": cfg.train.epochs - 1, "best": best})

    log("calibrating thresholds + latent map on val ...")
    write_status(run_dir, state="calibrating")
    from .calibrate import calibrate_run
    calibrate_run(run_dir, prepared=prepared, device=device, n_samples=cfg.train.score_samples, log=log)
    write_status(run_dir, state="finished", elapsed=time.time() - t_start, best=best)
    log(f"done. total {(time.time() - t_start) / 60:.1f} min")
    return run_dir


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None, help="yaml path or a name under configs/")
    ap.add_argument("--prepared", default="default")
    ap.add_argument("--name", default=None, help="run name (default: <config>-<timestamp>)")
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--lr", type=float)
    ap.add_argument("--device", default=None)
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    if args.epochs:
        cfg.train.epochs = args.epochs
    if args.batch_size:
        cfg.train.batch_size = args.batch_size
    if args.lr:
        cfg.train.lr = args.lr
    if args.device:
        cfg.device = args.device
    train(cfg, prepared=args.prepared, run_dir=args.run_dir, name=args.name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
