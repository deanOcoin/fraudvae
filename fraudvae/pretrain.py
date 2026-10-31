"""v6: pretrain the transaction-sequence encoder (model/txfm.py) on a prepared set, no labels.

    fraudvae pretrain --config txfm_small --prepared default --name txfm-a
    torchrun --nproc_per_node 8 -m fraudvae.pretrain --config txfm_base --prepared v6 --name txfm-b

Writes runs/<name>/{txfm.pt, txfm.json, metrics.jsonl, status.json, train.log}. The post stage
of a run with `txfm.run: <name>` and `boost.txfm_features: true` then reads txfm.pt.

The data path is the reason this is a separate loop and not another `train` option. The v5
history encoder gathered each row's window on the cpu, which made an H200 wait on numpy. Here
the rows a process needs are read from the prepared memmaps once, tokenised into int16 ids,
and kept on the device; a training window is then an on-device gather from the sequence index.
Under torchrun every process takes a disjoint share of the customer blocks (so its windows
never leave its own rows) and the model is wrapped in DistributedDataParallel; a single
process on a laptop runs the same code with world size 1.

A training window is `window` consecutive events of one customer ending on a row of the train
split; every position in it is a target, so one window trains up to `window` next-event
predictions. Validation windows end on val rows (their past may reach into train, which is
older, as it should). Nothing here reads a label.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import sys
import time
from pathlib import Path

import numpy as np
import torch

from . import __version__
from .config import load_config, to_dict
from .model.dataset import PreparedSet
from .model.txfm import (Tokeniser, TransactionEncoder, lr_cosine, mask_tokens, param_count, save_checkpoint)
from .paths import RUNS_DIR, ensure_dirs
from .train import Logger, pick_device, write_status

TOKEN_DTYPE_LIMIT = 32000   # int16 token ids when the vocabulary allows (half the memory on the device)


def _dist_init():
    """torchrun sets these; without them this is one process"""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1:
        import torch.distributed as dist

        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend)
        rank, world, local = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"]), int(os.environ.get("LOCAL_RANK", 0))
        if torch.cuda.is_available():
            torch.cuda.set_device(local)
        return dist, rank, world, local
    return None, 0, 1, 0


def _sample_rows(split, n, rng):
    idx = np.sort(rng.choice(len(split), min(n, len(split)), replace=False))
    return split.num[idx], split.mask[idx]


class ShardedSequences:
    """the rows of this process's customer blocks, tokenised and resident on the device, and the
    positions in seq_order that may end a training / validation window"""

    def __init__(self, data: PreparedSet, tok: Tokeniser, rank: int, world: int, device, log=print, max_rows_per_source=None,
                 seed=0):
        seq = data.seq
        if seq is None:
            raise SystemExit("the prepared set has no sequence index (seq_order.npy); re-run prepare")
        order = np.asarray(seq.order)
        start = np.asarray(seq.start)
        n_all = len(order)
        # a block is identified by its start position in seq_order; blocks are dealt by hash
        start_sorted = start[order]                      # start position of the block each position belongs to
        mine = (start_sorted % world) == rank if world > 1 else np.ones(n_all, dtype=bool)
        # only train and val rows can end a window and, the splits being by time within a block,
        # the rows before them are never test rows: the test split stays on disk
        offsets = seq.offsets
        keep_splits = [s_ for s_ in ("train", "val") if s_ in offsets]
        in_use = np.zeros(n_all, dtype=bool)
        for s_ in keep_splits:
            lo, hi = offsets[s_], offsets[s_] + seq.counts[s_]
            in_use |= (order >= lo) & (order < hi)
        mine &= in_use
        del in_use
        if max_rows_per_source:
            # a laptop probe: keep a prefix of each source's blocks. the row's source is read from
            # the split arrays below; approximate by keeping the first max_rows * n_sources blocks
            keep_rows = int(max_rows_per_source) * max(1, len(data.meta["sources"]))
            if mine.sum() > keep_rows:
                cut = np.flatnonzero(mine)[keep_rows]
                mine[cut:] = False
                # do not cut a block in half
                mine[start_sorted == start_sorted[cut]] = False
        self.positions = np.flatnonzero(mine).astype(np.int64)      # seq positions this rank holds (sorted)
        gids = order[self.positions]
        self.n = len(gids)
        tr_lo, tr_hi = offsets["train"], offsets["train"] + seq.counts["train"]
        va_lo, va_hi = offsets.get("val", -1), offsets.get("val", -1) + seq.counts.get("val", 0)
        self.train_targets = self.positions[(gids >= tr_lo) & (gids < tr_hi)]
        self.val_targets = self.positions[(gids >= va_lo) & (gids < va_hi)] if "val" in offsets else self.train_targets[:0]
        self.block_start = torch.from_numpy(start_sorted[self.positions]).to(device)
        # tokenise in chunks: sorted reads from the memmaps, ids straight to the device
        n_fields = tok.n_fields
        dtype = torch.int16 if tok.vocab < TOKEN_DTYPE_LIMIT else torch.int32
        self.tokens = torch.empty((self.n, n_fields), dtype=dtype, device=device)
        self.dt_log = torch.empty(self.n, dtype=torch.float32, device=device)
        chunk = 1 << 18
        t0 = time.time()
        prev_ts = None
        prev_start = None
        for s in range(0, self.n, chunk):
            g = gids[s : s + chunk]
            num, mask, cat, ts = seq.gather_rows(g)
            src = np.zeros(len(g), dtype=np.int64)
            for lo, hi, name in seq.ranges:
                sp = seq.splits.get(name)
                if sp is None:
                    continue
                sel = np.flatnonzero((g >= lo) & (g < hi))
                if len(sel):
                    local = g[sel] - lo
                    u, inv = np.unique(local, return_inverse=True)
                    src[sel] = sp.src[u][inv].astype(np.int64)
            bs = start_sorted[self.positions[s : s + chunk]]
            # seconds since the previous event of the same block; the first event of a block has none
            ts_prev = np.concatenate([[prev_ts if prev_ts is not None else np.nan], ts[:-1]])
            bs_prev = np.concatenate([[prev_start if prev_start is not None else -1], bs[:-1]])
            dt = np.where(bs_prev == bs, ts - ts_prev, -1.0)
            dt = np.where(np.isnan(dt), -1.0, dt)
            prev_ts, prev_start = float(ts[-1]), int(bs[-1])
            t = tok.encode(torch.from_numpy(num).to(device), torch.from_numpy(mask).to(device), torch.from_numpy(cat).to(device),
                           torch.from_numpy(src).to(device), torch.from_numpy(dt.astype(np.float32)).to(device))
            self.tokens[s : s + len(g)] = t.to(dtype)
            dtt = torch.from_numpy(dt.astype(np.float32)).to(device)
            self.dt_log[s : s + len(g)] = torch.where(dtt >= 0, torch.log1p(dtt.clamp(min=0)), torch.zeros_like(dtt))
        self.pos_t = torch.from_numpy(self.positions).to(device)
        self.device = device
        gb = (self.tokens.numel() * self.tokens.element_size() + self.dt_log.numel() * 4) / 2**30
        log(f"rank {rank}/{world}: {self.n:,} rows of {n_all:,} resident ({gb:.2f} GB as token ids), "
            f"{len(self.train_targets):,} train targets, {len(self.val_targets):,} val targets, tokenised in {time.time() - t0:.0f}s")

    def windows(self, end_positions: np.ndarray, L: int):
        """windows of L events ending at the given seq positions (oldest first):
        tokens [B, L, F] int64, dt_log [B, L], valid [B, L]"""
        end = torch.from_numpy(np.asarray(end_positions, dtype=np.int64)).to(self.device)
        q = end[:, None] - (L - 1) + torch.arange(L, device=self.device)[None]
        local_end = torch.searchsorted(self.pos_t, end)
        local = local_end[:, None] - (L - 1) + torch.arange(L, device=self.device)[None]
        bstart = self.block_start[local_end]
        valid = (q >= bstart[:, None]) & (local >= 0)
        local = local.clamp(min=0)
        toks = self.tokens[local].long()
        dt = self.dt_log[local]
        # the oldest event of a window that starts mid-block still has a previous event; it keeps its dt
        return toks, dt, valid


def _loss(model, tok, toks, dt, valid, cfg, gen=None):
    obs = tok.observed(toks) & valid.unsqueeze(-1)
    corrupted, masked = mask_tokens(toks, valid, tok.mask_id, cfg.mask_token, cfg.mask_event, cfg.mask_field, generator=gen)
    states, tok_out = model(corrupted, dt, valid)
    # the next-event targets are the true tokens; the states have seen only earlier events, which
    # were corrupted - that is a regulariser, not a leak
    nll = model.next_event_nll(states, toks, obs)
    next_loss = nll.sum() / obs.sum().clamp(min=1)
    mlm = model.masked_nll(states, tok_out, toks, masked)
    return next_loss + cfg.mlm_weight * mlm, next_loss.detach(), mlm.detach(), int(obs.sum())


@torch.no_grad()
def _evaluate(model, tok, shard: ShardedSequences, targets: np.ndarray, L: int, batch: int):
    model.eval()
    tot, n = 0.0, 0
    per_field = None
    for s in range(0, len(targets), batch):
        toks, dt, valid = shard.windows(targets[s : s + batch], L)
        obs = tok.observed(toks) & valid.unsqueeze(-1)
        states, _ = model(toks, dt, valid)
        nll = model.next_event_nll(states, toks, obs)
        # the last position is the row the window was chosen for; score that one, like a deployment would
        last = nll[:, L - 1]
        o = obs[:, L - 1]
        tot += float(last.sum())
        n += int(o.sum())
        pf = torch.stack([last.sum(0), o.sum(0).float()])
        per_field = pf if per_field is None else per_field + pf
    model.train()
    fields = {}
    if per_field is not None:
        for j, name in enumerate(tok.field_names):
            c = float(per_field[1, j])
            if c > 0:
                fields[name] = round(float(per_field[0, j]) / c, 4)
    return (tot / max(n, 1)), fields


def pretrain(cfg, prepared="default", name=None, run_dir: Path | None = None):
    dist, rank, world, local = _dist_init()
    tc = cfg.txfm
    ensure_dirs()
    torch.manual_seed(tc.seed + rank)
    rng = np.random.default_rng(tc.seed + rank)
    device = pick_device(cfg.device)
    if device.type == "cuda" and world > 1:
        device = torch.device("cuda", local)
    name = name or f"{cfg.name}-{time.strftime('%Y%m%d-%H%M')}"
    run_dir = Path(run_dir) if run_dir else RUNS_DIR / name
    if rank == 0:
        run_dir.mkdir(parents=True, exist_ok=True)
    if dist:
        dist.barrier()
    log = Logger(run_dir) if rank == 0 else (lambda msg: None)
    log(f"fraudvae {__version__} | pretrain {run_dir.name} | device {device} x{world} | torch {torch.__version__} | host {socket.gethostname()}")

    data = PreparedSet(prepared, splits=("train", "val"), in_memory=False)
    meta = data.meta
    tr = data["train"]
    n_sources = max(len(meta["source_index"]) and max(meta["source_index"].values()), 0)
    from .data import schema

    n_sources = max(n_sources, len(schema.SOURCES))
    # quantile edges from a sample of train rows; the same seed everywhere, so every rank builds the same tokeniser
    num_s, mask_s = _sample_rows(tr, 500_000, np.random.default_rng(tc.seed))
    tok = Tokeniser.fit(num_s, mask_s, data.numeric_names, [c["size"] for c in meta["categorical"]], n_sources,
                        n_bins=tc.n_bins, dt_bins=tc.dt_bins)
    del num_s, mask_s
    log(f"tokeniser: {tok.n_fields} fields, vocabulary {tok.vocab} ({tok.D} numerics x {tc.n_bins} buckets, "
        f"{tok.C} categoricals, {n_sources} sources, {tc.dt_bins} dt buckets)")

    shard = ShardedSequences(data, tok, rank, world, device, log=log, max_rows_per_source=tc.max_rows_per_source, seed=tc.seed)
    if len(shard.train_targets) == 0:
        raise SystemExit("no training rows on this rank")

    model = TransactionEncoder(tok.vocab, tok.n_fields, tok.sizes, tok.D, d_model=tc.d_model, event_layers=tc.event_layers,
                               history_layers=tc.history_layers, heads=tc.heads, window=tc.window, dropout=tc.dropout).to(device)
    n_params = param_count(model)
    log(f"model: {n_params / 1e6:.2f}M params, d {tc.d_model}, event layers {tc.event_layers}, history layers {tc.history_layers}, "
        f"window {tc.window}, batch {tc.batch_size} windows x {world}")
    ddp = model
    if dist:
        from torch.nn.parallel import DistributedDataParallel

        ddp = DistributedDataParallel(model, device_ids=[local] if device.type == "cuda" else None)
    opt = torch.optim.AdamW(ddp.parameters(), lr=tc.lr, weight_decay=tc.weight_decay, betas=(0.9, 0.95))
    use_amp = tc.amp and device.type == "cuda" and torch.cuda.is_bf16_supported()

    if rank == 0:
        info = {"name": run_dir.name, "kind": "pretrain", "started": time.strftime("%Y-%m-%dT%H:%M:%S"), "config": to_dict(cfg),
                "prepared": prepared, "prepared_meta": {k: meta[k] for k in ("name", "created", "counts", "sources", "max_rows_per_source")},
                "device": str(device), "world_size": world, "device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else platform.processor() or platform.machine(),
                "host": socket.gethostname(), "torch": torch.__version__, "params": n_params, "pid": os.getpid(),
                "schema_version": meta.get("schema_version"), "fraudvae": __version__, "vocab": tok.vocab, "n_fields": tok.n_fields}
        (run_dir / "run.json").write_text(json.dumps(info, indent=1))
        write_status(run_dir, state="running", epoch=0, epochs=1, step=0, steps=tc.steps, pid=os.getpid())
    metrics_f = open(run_dir / "metrics.jsonl", "a") if rank == 0 else None

    val_targets = shard.val_targets
    if len(val_targets) > tc.eval_windows:
        val_targets = np.sort(np.random.default_rng(tc.seed).choice(val_targets, tc.eval_windows, replace=False))
    best, bad, t_start, t0 = float("inf"), 0, time.time(), time.time()
    sums = {"loss": 0.0, "next": 0.0, "mlm": 0.0, "tokens": 0}
    seen_steps = 0
    model.train()
    for step in range(1, tc.steps + 1):
        for g in opt.param_groups:
            g["lr"] = lr_cosine(step - 1, tc.lr, tc.warmup, tc.steps)
        ends = rng.choice(shard.train_targets, tc.batch_size, replace=True)
        toks, dt, valid = shard.windows(ends, tc.window)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            loss, next_l, mlm_l, n_tok = _loss(ddp, tok, toks, dt, valid, tc) if not dist else _loss(ddp.module, tok, toks, dt, valid, tc)
        if not torch.isfinite(loss):
            log(f"non-finite loss at step {step}; step skipped")
            opt.zero_grad(set_to_none=True)
            continue
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if dist:
            # the loss was computed on the module, so sync the gradients by hand (one all-reduce per step)
            for p in ddp.module.parameters():
                if p.grad is not None:
                    dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)
        if not torch.isfinite(gnorm):
            log(f"non-finite gradient at step {step}; step skipped")
            opt.zero_grad(set_to_none=True)
            continue
        opt.step()
        seen_steps += 1
        sums["loss"] += float(loss)
        sums["next"] += float(next_l)
        sums["mlm"] += float(mlm_l)
        sums["tokens"] += n_tok
        if step % 50 == 0 and rank == 0:
            el = time.time() - t0
            log(f"step {step}/{tc.steps} loss {sums['loss'] / seen_steps:.4f} next {sums['next'] / seen_steps:.4f} "
                f"mlm {sums['mlm'] / seen_steps:.4f} lr {lr_cosine(step - 1, tc.lr, tc.warmup, tc.steps):.2e} "
                f"{sums['tokens'] * world / el:,.0f} tok/s, {(tc.steps - step) * el / max(1, seen_steps) / 60:.1f} min left")
            write_status(run_dir, state="running", epoch=0, epochs=1, step=step, steps=tc.steps, progress=step / tc.steps,
                         loss=sums["loss"] / seen_steps, elapsed=time.time() - t_start)
        if step % tc.eval_every == 0 or step == tc.steps:
            val_nll, fields = _evaluate(model, tok, shard, val_targets, tc.window, max(16, tc.batch_size // 2)) if len(val_targets) else (float("nan"), {})
            if dist:
                v = torch.tensor([val_nll if val_nll == val_nll else 0.0, 1.0 if val_nll == val_nll else 0.0], device=device)
                dist.all_reduce(v)
                val_nll = float(v[0] / v[1]) if v[1] > 0 else float("nan")
            rec = {"step": step, "train_loss": sums["loss"] / max(1, seen_steps), "train_next": sums["next"] / max(1, seen_steps),
                   "train_mlm": sums["mlm"] / max(1, seen_steps), "val_next_nll": val_nll, "val_fields": fields,
                   "tokens_per_sec": sums["tokens"] * world / max(1e-6, time.time() - t0), "elapsed": time.time() - t_start}
            if rank == 0:
                metrics_f.write(json.dumps(rec) + "\n")
                metrics_f.flush()
                worst = sorted(fields.items(), key=lambda kv: -kv[1])[:3]
                log(f"eval at step {step}: val next-event nll {val_nll:.4f} per observed token (train {rec['train_next']:.4f}); "
                    f"hardest fields {', '.join(f'{k} {v:.2f}' for k, v in worst)}")
            improved = val_nll == val_nll and val_nll < best - 1e-4
            if improved or val_nll != val_nll:
                best = min(best, val_nll) if val_nll == val_nll else best
                bad = 0
                if rank == 0:
                    save_checkpoint(run_dir / "txfm.pt", model, tok, meta, to_dict(cfg),
                                    extra={"step": step, "best_val_next_nll": best, "val_fields": fields})
                    log("  new best, txfm.pt saved")
            else:
                bad += 1
                if bad >= tc.patience:
                    log(f"  no improvement for {bad} evals, stopping")
                    break
            sums = {"loss": 0.0, "next": 0.0, "mlm": 0.0, "tokens": 0}
            seen_steps = 0
            t0 = time.time()
    if rank == 0:
        if not (run_dir / "txfm.pt").exists():
            save_checkpoint(run_dir / "txfm.pt", model, tok, meta, to_dict(cfg), extra={"step": tc.steps, "best_val_next_nll": best})
        summary = {"created": time.strftime("%Y-%m-%dT%H:%M:%S"), "prepared": prepared, "params": n_params, "vocab": tok.vocab,
                   "n_fields": tok.n_fields, "window": tc.window, "d_model": tc.d_model, "steps": tc.steps, "world_size": world,
                   "best_val_next_nll": best if best < float("inf") else None, "seconds": round(time.time() - t_start, 1),
                   "rows_resident_rank0": int(shard.n)}
        (run_dir / "txfm.json").write_text(json.dumps(summary, indent=1))
        write_status(run_dir, state="done", epoch=1, epochs=1, step=tc.steps, steps=tc.steps, best=best, elapsed=time.time() - t_start)
        metrics_f.close()
        log(f"done in {(time.time() - t_start) / 60:.1f} min; best val next-event nll {best:.4f}")
    if dist:
        dist.barrier()
        dist.destroy_process_group()
    return run_dir


def main(argv=None):
    ap = argparse.ArgumentParser(description="pretrain the v6 transaction-sequence encoder")
    ap.add_argument("--config", default=None, help="yaml path or a name under configs/ (txfm_small, txfm_base)")
    ap.add_argument("--prepared", default="default")
    ap.add_argument("--name", default=None)
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--steps", type=int)
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--lr", type=float)
    ap.add_argument("--device", default=None)
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    if args.steps:
        cfg.txfm.steps = args.steps
    if args.batch_size:
        cfg.txfm.batch_size = args.batch_size
    if args.lr:
        cfg.txfm.lr = args.lr
    if args.device:
        cfg.device = args.device
    pretrain(cfg, prepared=args.prepared, name=args.name, run_dir=args.run_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
