"""v7: fine-tune the pretrained transaction encoder on the fraud labels (Session B of the plan).

    fraudvae txfm-finetune txfm-base-last --prepared full --sources tabformer,fdh,ieee,capone --name txfm-ft --device cuda:0

The labels had never reached the encoder's weights: pretraining is self-supervised and the
supervised head (txfmhead.py) reads a frozen state. This unfreezes the last two history layers
and their norm, puts a small head on the state after the last event of a labelled window, and
makes one pass over the listed sources' labelled train rows: every fraud row plus legit rows at
--legit-ratio to one, capped per source. The checkpoint with the best mean validation recall at
1% FPR is written to runs/NAME/txfm.pt in the pretraining format, so TxfmScorer, txfm-cache and
the post stage read it unchanged. The fine-tune head itself is not used downstream: the post
stage's booster and txsup head read the exported encoder's block. txfmft.json keeps the numbers.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ..data import schema
from ..paths import RUNS_DIR, ensure_dirs
from .dataset import PreparedSet
from .scoring import auroc, recall_at_fpr
from .txfm import load_txfm, save_checkpoint

FILE = "txfmft.json"
VAL_FRAUD, VAL_LEGIT = 4096, 4096   # validation windows per source: every fraud row up to this, plus this many legit
MIN_VAL_FRAUD = 10                  # a source with fewer fraud rows in its validation draw does not enter the mean


class FinetuneHead(nn.Module):
    """a learned source embedding beside the state after the last event, then one hidden layer"""

    def __init__(self, d_model: int, n_sources: int, src_dim: int = 16, hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.src = nn.Embedding(n_sources + 1, src_dim)
        self.net = nn.Sequential(nn.Linear(d_model + src_dim, hidden), nn.SiLU(), nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, state, src):
        return self.net(torch.cat([state, self.src(src)], dim=1)).squeeze(1)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def _resolve(run) -> Path:
    p = Path(run).expanduser()
    if not p.exists():
        p = RUNS_DIR / str(run)
    if p.is_dir():
        p = p / "txfm.pt"
    if not p.exists():
        raise SystemExit(f"no txfm.pt for {run!r} (a run name under runs/, a directory or a path to txfm.pt)")
    return p


def _train_targets(shard, src_k: int, legit_ratio: float, max_legit: int, rng):
    """every fraud train row of the source plus legit rows at legit_ratio to one (at most max_legit), shuffled"""
    tg = shard.train_by_source.get(src_k)
    if tg is None or len(tg) == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float32)
    lab = shard.labels(tg)
    fraud, legit = tg[lab == 1], tg[lab == 0]
    k = int(min(len(legit), max_legit, round(len(fraud) * legit_ratio)))
    if k < len(legit):
        legit = np.sort(rng.choice(legit, k, replace=False))
    ends = np.concatenate([fraud, legit])
    y = np.concatenate([np.ones(len(fraud), dtype=np.float32), np.zeros(len(legit), dtype=np.float32)])
    perm = rng.permutation(len(ends))
    return ends[perm], y[perm]


def _val_targets(shard, src_k: int, rng):
    """the source's val windows: every fraud row up to VAL_FRAUD, VAL_LEGIT legit, drawn once"""
    tg = shard.val_by_source.get(src_k)
    if tg is None or len(tg) == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float32)
    lab = shard.labels(tg)
    fraud, legit = tg[lab == 1], tg[lab == 0]
    if len(fraud) > VAL_FRAUD:
        fraud = np.sort(rng.choice(fraud, VAL_FRAUD, replace=False))
    if len(legit) > VAL_LEGIT:
        legit = np.sort(rng.choice(legit, VAL_LEGIT, replace=False))
    ends = np.concatenate([fraud, legit])
    return ends, np.concatenate([np.ones(len(fraud), dtype=np.float32), np.zeros(len(legit), dtype=np.float32)])


@torch.no_grad()
def _evaluate(model, head, tok, shard, val: dict, L: int, batch: int, names: dict, use_amp: bool) -> tuple[dict, float]:
    """per source: AUROC and recall at 1% FPR of the head's logit on the validation windows, and
    the mean recall over the sources with enough fraud rows (the number that picks the checkpoint)"""
    model.eval()
    head.eval()
    res = {}
    for k, (ends, y) in val.items():
        if len(ends) == 0:
            continue
        fields = tok.fields_for_source(k, shard.device)
        logits = []
        for s in range(0, len(ends), batch):
            toks, dt, valid = shard.windows(ends[s : s + batch], L)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                states, _ = model(toks, dt, valid, fields)
                logit = head(states[:, L].float(), torch.full((len(toks),), int(k), device=shard.device, dtype=torch.long))
            logits.append(logit.float().cpu().numpy())
        s_ = np.concatenate(logits)
        _, rec = recall_at_fpr(y, s_, 0.01)
        res[names.get(int(k), str(k))] = {"auroc": float(auroc(y, s_)), "recall_at_1pct_fpr": float(rec), "n": int(len(y)), "fraud": int(y.sum())}
    model.train()
    head.train()
    picks = [v["recall_at_1pct_fpr"] for v in res.values() if v["fraud"] >= MIN_VAL_FRAUD and v["recall_at_1pct_fpr"] == v["recall_at_1pct_fpr"]]
    return res, (float(np.mean(picks)) if picks else float("nan"))


def finetune(run, prepared: str, sources: list[str], name: str | None = None, run_dir: Path | None = None, device="cpu", steps: int | None = None,
             legit_ratio: float = 20.0, max_legit: int = 400_000, batch: int = 64, lr: float = 3e-5, head_lr: float = 1e-3,
             weight_decay: float = 0.01, eval_every: int = 2000, patience: int = 3, unfreeze: int = 2, warmup: int = 100,
             seed: int = 1337, log=None) -> Path:
    """returns the run directory holding the fine-tuned txfm.pt and txfmft.json"""
    from ..pretrain import ShardedSequences
    from ..train import Logger, write_status

    ensure_dirs()
    src_path = _resolve(run)
    run_dir = Path(run_dir) if run_dir else RUNS_DIR / (name or f"{src_path.parent.name}-ft")
    run_dir.mkdir(parents=True, exist_ok=True)
    log = log or Logger(run_dir)
    device = torch.device(device)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    unknown = [s for s in sources if s not in schema.SOURCE_INDEX]
    if unknown or not sources:
        raise SystemExit(f"--sources must name known sources (unknown: {unknown}; known: {', '.join(schema.SOURCES)})")
    src_ks = [schema.SOURCE_INDEX[s] for s in sources]
    t_start = time.time()
    model, tok, ck = load_txfm(src_path, device)
    L = int(model.window)
    base_sha = _sha256(src_path)
    log(f"txfm-finetune {run_dir.name} | base {src_path} (sha256 {base_sha[:16]}, step {ck.get('extra', {}).get('step', '?')}) | "
        f"device {device} | sources {', '.join(sources)}")
    data = PreparedSet(prepared, splits=("train", "val"), in_memory=False)
    names = {v: k for k, v in data.meta["source_index"].items()}
    shard = ShardedSequences(data, tok, 0, 1, device, log=log, seed=seed)
    train = {k: _train_targets(shard, k, legit_ratio, max_legit, rng) for k in src_ks}
    train = {k: v for k, v in train.items() if len(v[0])}
    val = {k: _val_targets(shard, k, rng) for k in src_ks}
    if not train:
        raise SystemExit("none of the listed sources has labelled train rows in this prepared set")
    n_fraud = int(sum(v[1].sum() for v in train.values()))
    n_all = int(sum(len(v[0]) for v in train.values()))
    for k, (ends, y) in train.items():
        log(f"  {names.get(k, k)}: {int(y.sum()):,} fraud + {int(len(y) - y.sum()):,} legit train windows; "
            f"val {int(val[k][1].sum()):,} fraud + {int(len(val[k][1]) - val[k][1].sum()):,} legit")
    # frozen but the top of the history encoder and its norm; the head is new
    for p in model.parameters():
        p.requires_grad_(False)
    top = list(model.history.layers)[-max(1, int(unfreeze)):] + [model.hist_norm]
    trainable = []
    for m in top:
        for p in m.parameters():
            p.requires_grad_(True)
            trainable.append(p)
    head = FinetuneHead(model.d_model, max(len(schema.SOURCES), max(src_ks) + 1)).to(device)
    opt = torch.optim.AdamW([{"params": trainable, "lr": lr}, {"params": list(head.parameters()), "lr": head_lr}],
                            weight_decay=weight_decay, betas=(0.9, 0.95))
    pos_weight = torch.tensor(min(max(1.0, (n_all - n_fraud) / max(1, n_fraud)), 20.0), device=device)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    use_amp = device.type == "cuda" and torch.cuda.is_bf16_supported()
    n_steps = int(steps) if steps else int(np.ceil(n_all / batch))
    log(f"model: {sum(p.numel() for p in trainable) / 1e6:.2f}M of {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M params trainable "
        f"({len(top) - 1} history layers + norm), head {sum(p.numel() for p in head.parameters()):,} params; {n_all:,} windows "
        f"({n_fraud:,} fraud, pos_weight {float(pos_weight):.1f}), batch {batch}, {n_steps:,} steps, eval every {eval_every}")
    write_status(run_dir, state="running", epoch=0, epochs=1, step=0, steps=n_steps, pid=None)
    fields = {k: tok.fields_for_source(k, device) for k in train}
    cursors = {k: 0 for k in train}
    best, best_step, bad = float("-inf"), 0, 0
    best_state = None
    best_res = {}
    history = []
    sums, seen, t0 = 0.0, 0, time.time()
    model.train()
    head.train()
    stop = False
    for step in range(1, n_steps + 1):
        remaining = {k: len(train[k][0]) - cursors[k] for k in train if len(train[k][0]) - cursors[k] > 0}
        if not remaining:   # a pass is over and more steps were asked for: shuffle and go again
            for k in train:
                ends, y = train[k]
                perm = rng.permutation(len(ends))
                train[k] = (ends[perm], y[perm])
                cursors[k] = 0
            remaining = {k: len(train[k][0]) for k in train}
        ks = list(remaining)
        p = np.array([remaining[k] for k in ks], dtype=np.float64)
        k = ks[int(rng.choice(len(ks), p=p / p.sum()))]
        ends, y = train[k]
        c = cursors[k]
        b_ends, b_y = ends[c : c + batch], y[c : c + batch]
        cursors[k] = c + len(b_ends)
        scale = min(1.0, step / max(1, warmup))
        opt.param_groups[0]["lr"], opt.param_groups[1]["lr"] = lr * scale, head_lr * scale
        toks, dt, valid = shard.windows(b_ends, L)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            states, _ = model(toks, dt, valid, fields[k])
            logit = head(states[:, L].float(), torch.full((len(toks),), int(k), device=device, dtype=torch.long))
        loss = bce(logit.float(), torch.from_numpy(b_y).to(device))
        if not torch.isfinite(loss):
            log(f"non-finite loss at step {step}; step skipped")
            opt.zero_grad(set_to_none=True)
            continue
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable + list(head.parameters()), 1.0)
        opt.step()
        sums += float(loss)
        seen += 1
        if step % 50 == 0:
            el = time.time() - t0
            log(f"step {step}/{n_steps} loss {sums / seen:.4f} {seen * batch / el:,.0f} windows/s, {(n_steps - step) * el / max(1, seen) / 60:.1f} min left")
            write_status(run_dir, state="running", epoch=0, epochs=1, step=step, steps=n_steps, progress=step / n_steps, loss=sums / seen,
                         elapsed=time.time() - t_start)
        if step % eval_every == 0 or step == n_steps:
            res, mean = _evaluate(model, head, tok, shard, val, L, max(16, batch * 4), names, use_amp)
            history.append({"step": step, "train_loss": sums / max(1, seen), "mean_recall_at_1pct_fpr": mean, "sources": res,
                            "elapsed": time.time() - t_start})
            log(f"eval at step {step}: mean val recall@1%fpr {mean:.4f} | " + ", ".join(
                f"{n} auroc {v['auroc']:.4f} r@1% {v['recall_at_1pct_fpr']:.3f}" for n, v in res.items()))
            if mean == mean and mean > best + 1e-4:
                best, best_step, bad = mean, step, 0
                best_state = {k_: v_.detach().clone() for k_, v_ in model.state_dict().items()}
                best_res = res
                log("  new best")
            else:
                bad += 1
                if bad >= patience:
                    log(f"  no improvement for {bad} evals, stopping")
                    stop = True
            sums, seen, t0 = 0.0, 0, time.time()
        if stop:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    base_extra = ck.get("extra", {})
    base_nll = base_extra.get("best_val_next_nll", base_extra.get("val_next_nll"))
    extra = {"finetuned": True, "finetuned_sources": list(sources), "step": best_step, "steps": n_steps, "unfrozen_history_layers": len(top) - 1,
             "val": best_res, "mean_val_recall_at_1pct_fpr": best if best > float("-inf") else None, "base_checkpoint": str(src_path),
             "base_checkpoint_sha256": base_sha, "base_step": base_extra.get("step")}
    if base_nll is not None:   # the base's validation next-event nll, when its record has one (a snapshot's or an export's)
        extra["base_val_next_nll"] = float(base_nll)
    save_checkpoint(run_dir / "txfm.pt", model, tok, ck.get("prepared_meta", data.meta), ck.get("train_config", {}), extra=extra)
    summary = {"created": time.strftime("%Y-%m-%dT%H:%M:%S"), "prepared": prepared, "base": str(src_path), "base_sha256": base_sha,
               "sources": best_res, "mean_val_recall_at_1pct_fpr": extra["mean_val_recall_at_1pct_fpr"], "best_step": best_step, "steps": n_steps,
               "windows": n_all, "fraud_windows": n_fraud, "legit_ratio": legit_ratio, "max_legit": max_legit, "batch": batch, "lr": lr,
               "head_lr": head_lr, "unfrozen_history_layers": len(top) - 1, "pos_weight": float(pos_weight), "history": history,
               "seconds": round(time.time() - t_start, 1), "device": str(device)}
    (run_dir / FILE).write_text(json.dumps(summary, indent=1))
    write_status(run_dir, state="done", epoch=1, epochs=1, step=n_steps, steps=n_steps, best=best, elapsed=time.time() - t_start)
    from ..manifest import write_manifest

    write_manifest(run_dir, "txfm-finetune", None, prepared, extra={"base_checkpoint_sha256": base_sha, "sources": list(sources)}, log=log)
    log(f"done in {(time.time() - t_start) / 60:.1f} min; best mean val recall@1%fpr {best:.4f} at step {best_step}; {run_dir / 'txfm.pt'} written")
    return run_dir


def main(argv=None):
    ap = argparse.ArgumentParser(prog="fraudvae txfm-finetune", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", help="the pretraining run (runs/<run>/txfm.pt), a directory or a txfm.pt path")
    ap.add_argument("--prepared", default="default")
    ap.add_argument("--sources", required=True, help="comma-separated source names whose labelled rows are used")
    ap.add_argument("--name", default=None, help="the new run under runs/ (default: <run>-ft)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--steps", type=int, default=None, help="optimiser steps (default: one pass over the sampled windows)")
    ap.add_argument("--legit-ratio", type=float, default=20.0, help="legit windows per fraud window, per source")
    ap.add_argument("--max-legit", type=int, default=400_000, help="legit windows per source at most")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-5, help="the unfrozen encoder layers' learning rate")
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--eval-every", type=int, default=2000)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--unfreeze", type=int, default=2, help="history-encoder layers trained, from the top")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args(argv)
    finetune(args.run, args.prepared, [s.strip() for s in args.sources.split(",") if s.strip()], name=args.name, device=args.device,
             steps=args.steps, legit_ratio=args.legit_ratio, max_legit=args.max_legit, batch=args.batch, lr=args.lr, head_lr=args.head_lr,
             eval_every=args.eval_every, patience=args.patience, unfreeze=args.unfreeze, seed=args.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
