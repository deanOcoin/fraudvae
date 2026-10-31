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
    """up to n rows, the same number from every source, so the quantile edges of a field that
    only a small source carries are fit on that source's values and not on a handful of them"""
    src = np.asarray(split.src)
    ids = np.unique(src)
    per = max(1, n // max(1, len(ids)))
    take = [rng.choice(np.flatnonzero(src == s_), min(per, int((src == s_).sum())), replace=False) for s_ in ids]
    idx = np.sort(np.concatenate(take))
    return split.num[idx], split.mask[idx]


def _field_presence(split, D: int, C: int, chunk: int = 4_000_000) -> dict:
    """{source index: [field indices]}: the numerics observed and the categoricals ever set in
    at least one row of that source (hour and dow are categoricals here and always set). the
    tokeniser packs the event encoder's input with these lists"""
    n = len(split)
    seen = {}
    for s0 in range(0, n, chunk):
        src = np.asarray(split.src[s0 : s0 + chunk])
        mask = np.asarray(split.mask[s0 : s0 + chunk])
        cat = np.asarray(split.cat[s0 : s0 + chunk])
        for s_ in np.unique(src):
            rows = src == s_
            have = np.concatenate([mask[rows].any(0), (cat[rows] != 0).any(0)])
            seen[int(s_)] = have if int(s_) not in seen else (seen[int(s_)] | have)
    return {s_: [int(f) for f in np.flatnonzero(v)] for s_, v in seen.items()}


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
        src_all = np.zeros(self.n, dtype=np.int16)
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
            src_all[s : s + len(g)] = src
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
        # the targets by source: a training step draws its windows from one source, so the
        # event encoder runs at that source's packed width (and the sampler can balance sources)
        src_of = lambda targets: src_all[np.searchsorted(self.positions, targets)]  # noqa: E731
        self.train_by_source = {int(k): self.train_targets[src_of(self.train_targets) == k] for k in np.unique(src_of(self.train_targets))}
        self.val_by_source = {int(k): self.val_targets[src_of(self.val_targets) == k] for k in np.unique(src_of(self.val_targets))} if len(self.val_targets) else {}
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


def _loss(model, tok, toks, dt, valid, cfg, gen=None, fields=None):
    obs = tok.observed(toks) & valid.unsqueeze(-1)
    corrupted, masked = mask_tokens(toks, valid, tok.mask_id, cfg.mask_token, cfg.mask_event, cfg.mask_field, generator=gen)
    states, tok_out = model(corrupted, dt, valid, fields)
    # the next-event targets are the true tokens; the states have seen only earlier events, which
    # were corrupted - that is a regulariser, not a leak
    nll = model.next_event_nll(states, toks, obs)
    next_loss = nll.sum() / obs.sum().clamp(min=1)
    mlm = model.masked_nll(states, tok_out, toks, masked, fields)
    encoded = toks.shape[0] * toks.shape[1] * (toks.shape[2] if fields is None else len(fields))
    return next_loss + cfg.mlm_weight * mlm, next_loss.detach(), mlm.detach(), int(obs.sum()), encoded


@torch.no_grad()
def _evaluate(model, tok, shard: ShardedSequences, targets: dict, L: int, batch: int):
    """val next-event nll per observed token over the given windows ({source: end positions};
    the last event of each window, like a deployment would), per field and per source. the
    windows of one source go through together, at that source's packed width"""
    model.eval()
    tot, n = 0.0, 0
    per_field = None
    by_src = {}
    src_f = tok.n_fields - 2
    off = int(tok.offsets[src_f])
    for src_k, tg in targets.items():
        fields = tok.fields_for_source(src_k, shard.device)
        for s in range(0, len(tg), batch):
            toks, dt, valid = shard.windows(tg[s : s + batch], L)
            obs = tok.observed(toks) & valid.unsqueeze(-1)
            states, _ = model(toks, dt, valid, fields)
            nll = model.next_event_nll(states, toks, obs)
            # the last position is the row the window was chosen for; score that one, like a deployment would
            last = nll[:, L - 1]
            o = obs[:, L - 1]
            tot += float(last.sum())
            n += int(o.sum())
            pf = torch.stack([last.sum(0), o.sum(0).float()])
            per_field = pf if per_field is None else per_field + pf
            srcs = toks[:, L - 1, src_f] - off
            for k in torch.unique(srcs).tolist():
                m = srcs == k
                a = by_src.setdefault(int(k), [0.0, 0])
                a[0] += float(last[m].sum())
                a[1] += int(o[m].sum())
    model.train()
    fields = {}
    if per_field is not None:
        for j, name in enumerate(tok.field_names):
            c = float(per_field[1, j])
            if c > 0:
                fields[name] = round(float(per_field[0, j]) / c, 4)
    sources = {k: round(v[0] / v[1], 4) for k, v in sorted(by_src.items()) if v[1] > 0}
    return (tot / max(n, 1)), fields, sources


def pretrain(cfg, prepared="default", name=None, run_dir: Path | None = None, resume: bool = False):
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
    state = None
    if resume:
        sp = run_dir / "state.pt"
        if not sp.exists():
            raise SystemExit(f"nothing to resume: {sp} is missing")
        state = torch.load(sp, map_location="cpu", weights_only=False)
        tok = Tokeniser(**state["tokeniser"])
        log(f"resuming {run_dir.name} from step {state['step']} (best val next-event nll {state['best']:.4f})")
    else:
        # quantile edges from a sample of train rows; the same seed everywhere, so every rank builds the same tokeniser
        num_s, mask_s = _sample_rows(tr, 500_000, np.random.default_rng(tc.seed))
        presence = _field_presence(tr, len(data.numeric_names), len(meta["categorical"])) if tc.pack else None
        tok = Tokeniser.fit(num_s, mask_s, data.numeric_names, [c["size"] for c in meta["categorical"]], n_sources,
                            n_bins=tc.n_bins, dt_bins=tc.dt_bins, source_fields=presence)
        del num_s, mask_s
    src_names = {v: k for k, v in meta["source_index"].items()}
    log(f"tokeniser: {tok.n_fields} fields, vocabulary {tok.vocab} ({tok.D} numerics x {tc.n_bins} buckets, "
        f"{tok.C} categoricals, {n_sources} sources, {tc.dt_bins} dt buckets)")
    if tok.source_fields:
        log("packing: fields per source " + ", ".join(f"{src_names.get(k, k)} {len(v)}" for k, v in sorted(tok.source_fields.items())))

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
    fwd = torch.compile(model, dynamic=True) if tc.compile else model
    # a step's windows come from one source: the event encoder then runs at that source's packed
    # width, and sources are drawn with p ~ rows^alpha (alpha 1 reproduces drawing rows at random)
    src_ids = np.array(sorted(shard.train_by_source))
    src_w = np.array([len(shard.train_by_source[k]) for k in src_ids], dtype=np.float64) ** tc.source_alpha
    src_p = src_w / src_w.sum()
    log("source mix per step (alpha %.2f): " % tc.source_alpha + ", ".join(f"{src_names.get(int(k), k)} {p:.3f}" for k, p in zip(src_ids, src_p)))
    first_step, best, bad = 1, float("inf"), 0
    if state is not None:
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        first_step, best, bad = int(state["step"]) + 1, float(state["best"]), int(state["bad"])
        rng = np.random.default_rng(tc.seed + rank + first_step)
        del state

    if rank == 0 and not resume:
        info = {"name": run_dir.name, "kind": "pretrain", "started": time.strftime("%Y-%m-%dT%H:%M:%S"), "config": to_dict(cfg),
                "prepared": prepared, "prepared_meta": {k: meta[k] for k in ("name", "created", "counts", "sources", "max_rows_per_source")},
                "device": str(device), "world_size": world, "device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else platform.processor() or platform.machine(),
                "host": socket.gethostname(), "torch": torch.__version__, "params": n_params, "pid": os.getpid(),
                "schema_version": meta.get("schema_version"), "fraudvae": __version__, "vocab": tok.vocab, "n_fields": tok.n_fields,
                "packing": bool(tok.source_fields), "fields_per_source": {src_names.get(k, str(k)): len(v) for k, v in sorted(tok.source_fields.items())},
                "source_alpha": tc.source_alpha, "source_mix": {src_names.get(int(k), str(k)): round(float(p), 4) for k, p in zip(src_ids, src_p)},
                "compile": bool(tc.compile), "amp": bool(use_amp)}
        (run_dir / "run.json").write_text(json.dumps(info, indent=1))
    if rank == 0:
        write_status(run_dir, state="running", epoch=0, epochs=1, step=first_step - 1, steps=tc.steps, pid=os.getpid())
    metrics_f = open(run_dir / "metrics.jsonl", "a") if rank == 0 else None

    # validation windows: the same number from every source, so the number that picks the
    # checkpoint is not the biggest source's alone
    vrng = np.random.default_rng(tc.seed)
    per = max(1, tc.eval_windows // max(1, len(shard.val_by_source)))
    val_targets = {k: np.sort(vrng.choice(v, min(per, len(v)), replace=False)) for k, v in shard.val_by_source.items()}
    t_start, t0 = time.time(), time.time()
    sums = {"loss": 0.0, "next": 0.0, "mlm": 0.0, "tokens": 0, "encoded": 0}
    totals = {"tokens": 0, "encoded": 0, "events": 0}
    seen_steps = 0
    model.train()
    for step in range(first_step, tc.steps + 1):
        for g in opt.param_groups:
            g["lr"] = lr_cosine(step - 1, tc.lr, tc.warmup, tc.steps)
        src_k = int(src_ids[rng.choice(len(src_ids), p=src_p)])
        ends = rng.choice(shard.train_by_source[src_k], tc.batch_size, replace=True)
        toks, dt, valid = shard.windows(ends, tc.window)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            loss, next_l, mlm_l, n_tok, n_enc = _loss(fwd, tok, toks, dt, valid, tc, fields=tok.fields_for_source(src_k, device))
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
        sums["encoded"] += n_enc
        totals["tokens"] += n_tok
        totals["encoded"] += n_enc
        totals["events"] += int(valid.sum())
        if step % 50 == 0 and rank == 0:
            el = time.time() - t0
            log(f"step {step}/{tc.steps} loss {sums['loss'] / seen_steps:.4f} next {sums['next'] / seen_steps:.4f} "
                f"mlm {sums['mlm'] / seen_steps:.4f} lr {lr_cosine(step - 1, tc.lr, tc.warmup, tc.steps):.2e} "
                f"{sums['tokens'] * world / el:,.0f} tok/s ({sums['encoded'] * world / el:,.0f} encoded/s), "
                f"{(tc.steps - step) * el / max(1, seen_steps) / 60:.1f} min left")
            write_status(run_dir, state="running", epoch=0, epochs=1, step=step, steps=tc.steps, progress=step / tc.steps,
                         loss=sums["loss"] / seen_steps, elapsed=time.time() - t_start)
        if step % tc.eval_every == 0 or step == tc.steps:
            val_nll, fields, by_src = _evaluate(model, tok, shard, val_targets, tc.window, max(16, tc.batch_size // 2)) if val_targets else (float("nan"), {}, {})
            if dist:
                v = torch.tensor([val_nll if val_nll == val_nll else 0.0, 1.0 if val_nll == val_nll else 0.0], device=device)
                dist.all_reduce(v)
                val_nll = float(v[0] / v[1]) if v[1] > 0 else float("nan")
            rec = {"step": step, "train_loss": sums["loss"] / max(1, seen_steps), "train_next": sums["next"] / max(1, seen_steps),
                   "train_mlm": sums["mlm"] / max(1, seen_steps), "val_next_nll": val_nll, "val_fields": fields,
                   "val_sources": {src_names.get(k, str(k)): v for k, v in by_src.items()},
                   "tokens_per_sec": sums["tokens"] * world / max(1e-6, time.time() - t0),
                   "encoded_per_sec": sums["encoded"] * world / max(1e-6, time.time() - t0), "elapsed": time.time() - t_start}
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
                                    extra={"step": step, "best_val_next_nll": best, "val_fields": fields,
                                           "val_sources": rec["val_sources"]})
                    log("  new best, txfm.pt saved")
            else:
                bad += 1
            if rank == 0:
                # everything a --resume needs to carry on from this eval
                torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step, "best": best, "bad": bad,
                            "tokeniser": tok.config()}, run_dir / "state.pt")
            if bad >= tc.patience:
                log(f"  no improvement for {bad} evals, stopping")
                break
            sums = {"loss": 0.0, "next": 0.0, "mlm": 0.0, "tokens": 0, "encoded": 0}
            seen_steps = 0
            t0 = time.time()
    if dist:
        tt = torch.tensor([totals["tokens"], totals["encoded"], totals["events"]], dtype=torch.float64, device=device)
        dist.all_reduce(tt)
        totals = {"tokens": int(tt[0]), "encoded": int(tt[1]), "events": int(tt[2])}
    if rank == 0:
        if not (run_dir / "txfm.pt").exists():
            save_checkpoint(run_dir / "txfm.pt", model, tok, meta, to_dict(cfg), extra={"step": tc.steps, "best_val_next_nll": best})
        summary = {"created": time.strftime("%Y-%m-%dT%H:%M:%S"), "prepared": prepared, "params": n_params, "vocab": tok.vocab,
                   "n_fields": tok.n_fields, "window": tc.window, "d_model": tc.d_model, "steps": tc.steps, "world_size": world,
                   "best_val_next_nll": best if best < float("inf") else None, "seconds": round(time.time() - t_start, 1),
                   "rows_resident_rank0": int(shard.n), "packing": bool(tok.source_fields), "source_alpha": tc.source_alpha,
                   "first_step": first_step, "observed_tokens_seen": totals["tokens"], "encoded_tokens_seen": totals["encoded"],
                   "event_visits": totals["events"]}
        if (run_dir / "txfm.json").exists() and resume:
            prev = json.loads((run_dir / "txfm.json").read_text())
            for k in ("observed_tokens_seen", "encoded_tokens_seen", "event_visits", "seconds"):
                summary[k] = round(summary[k] + prev.get(k, 0), 1)
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
    ap.add_argument("--resume", action="store_true", help="carry on from the run's state.pt (same --name / --run-dir)")
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
    pretrain(cfg, prepared=args.prepared, name=args.name, run_dir=args.run_dir, resume=args.resume)
    return 0


if __name__ == "__main__":
    sys.exit(main())
