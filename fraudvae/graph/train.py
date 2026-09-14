"""Fit one entity-graph model per source in the post stage.

Per source: snapshot A (the train period minus its last fit_fraction) is the graph the
encoder learns on; the transaction head is fitted on the rows of that last part, whose
entities are looked up in A - a row never sees its own label through its own entities.
Snapshot B (whole train period) gives the embeddings the validation rows are scored with,
snapshot C (train + val) the ones test rows and the server use. Written to
runs/<run>/graph/<source>.pt plus a graph.json summary.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ..config import GraphConfig
from ..data import schema
from ..model.scoring import auroc
from ..paths import PROCESSED_DIR
from .build import KINDS, N_KINDS, Snapshot, cuts, load_rows, snapshot
from .model import EdgeHead, SAGE, adjacency, gather_slots


class NodeIndex:
    """(kind, code) -> node id without a python dict: one sorted code array per kind"""

    def __init__(self, node_kind: np.ndarray, node_code: np.ndarray):
        self.node_kind = np.asarray(node_kind, dtype=np.int8)
        self.node_code = np.asarray(node_code, dtype=np.int64)
        self.per_kind = {}
        for k in range(N_KINDS):
            ids = np.flatnonzero(self.node_kind == k)
            order = np.argsort(self.node_code[ids], kind="stable")
            self.per_kind[k] = (self.node_code[ids][order], ids[order])

    def lookup(self, ent: np.ndarray) -> np.ndarray:
        ent = np.asarray(ent, dtype=np.int64)
        out = np.full(ent.shape, -1, dtype=np.int64)
        for k in range(N_KINDS):
            codes, ids = self.per_kind[k]
            c = ent[:, k]
            ok = c >= 0
            if not ok.any() or len(codes) == 0:
                continue
            pos = np.searchsorted(codes, c[ok])
            pos = np.clip(pos, 0, len(codes) - 1)
            hit = codes[pos] == c[ok]
            res = np.where(hit, ids[pos], -1)
            out[ok, k] = res
        return out

    @property
    def n(self):
        return len(self.node_kind)


def _embed(encoder: SAGE, snap: Snapshot, x_mean, x_std, device) -> torch.Tensor:
    x = torch.from_numpy((snap.x - x_mean) / x_std).float().to(device)
    adj = adjacency(snap.edge_src, snap.edge_dst, snap.edge_w, snap.n_nodes, device=device)
    encoder.eval()
    with torch.no_grad():
        return encoder(x, adj)


def train_source(key: str, cfg: GraphConfig, device="cpu", log=print, seed=1337) -> dict | None:
    """returns the per-source state to save (None when the source has no usable graph)"""
    t0 = time.time()
    path = PROCESSED_DIR / f"{key}.parquet"
    if not path.exists():
        return None
    rows = load_rows(path)
    if not any((rows[k].to_numpy() >= 0).any() for k in KINDS):
        log(f"  graph {key}: no entity columns, skipping")
        return None
    c = cuts(rows, cfg.fit_fraction)
    snap_a = snapshot(key, rows, c["A"], cfg.label_delay_days, cfg.min_degree, cfg.max_nodes)
    fit = rows.iloc[c["fit_rows"]]
    ent_fit = fit[KINDS].to_numpy().astype(np.int64)
    y_fit = fit["is_fraud"].to_numpy().astype(np.float32)
    ids_fit = snap_a.lookup(ent_fit)
    usable = (ids_fit >= 0).any(1)
    if snap_a.n_nodes < 10 or usable.sum() < 500 or y_fit[usable].sum() < 10:
        log(f"  graph {key}: too little structure to learn from ({snap_a.n_nodes} nodes, {int(usable.sum())} usable fit rows), skipping")
        return None
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    idx = np.flatnonzero(usable)
    legit = idx[y_fit[idx] == 0]
    if len(legit) > cfg.max_legit_fit:
        legit = rng.choice(legit, cfg.max_legit_fit, replace=False)
    idx = np.sort(np.concatenate([legit, idx[y_fit[idx] == 1]]))
    pos_weight = min(float((y_fit[idx] == 0).sum() / max(1, (y_fit[idx] == 1).sum())), 50.0)

    x_mean = snap_a.x.mean(0)
    x_std = np.maximum(snap_a.x.std(0), 1e-3)
    x_a = torch.from_numpy((snap_a.x - x_mean) / x_std).float().to(device)
    adj_a = adjacency(snap_a.edge_src, snap_a.edge_dst, snap_a.edge_w, snap_a.n_nodes, device=device)
    encoder = SAGE(x_a.shape[1], cfg.hidden, cfg.out_dim, cfg.layers, cfg.dropout).to(device)
    head = EdgeHead(cfg.out_dim).to(device)
    opt = torch.optim.AdamW(list(encoder.parameters()) + list(head.parameters()), lr=cfg.lr, weight_decay=1e-5)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))
    log(f"  graph {key}: snapshot A {snap_a.n_nodes:,} nodes / {len(snap_a.edge_src) // 2:,} edges, "
        f"{len(idx):,} fit rows ({int(y_fit[idx].sum()):,} fraud), pos_weight {pos_weight:.0f}")
    y_t = torch.from_numpy(y_fit).to(device)
    # a small fit set (fraudecom's few hundred rows on a known device) would get one step per
    # epoch; repeat it so every epoch is at least a handful of updates
    reps = max(1, int(np.ceil(cfg.batch_size * 8 / len(idx))))
    for ep in range(cfg.epochs):
        encoder.train()
        head.train()
        order = rng.permutation(np.tile(idx, reps))
        tot, n = 0.0, 0
        for s in range(0, len(order), cfg.batch_size):
            b = order[s : s + cfg.batch_size]
            emb = encoder(x_a, adj_a)
            slots, known = gather_slots(emb, ids_fit[b])
            loss = loss_fn(head(slots, known), y_t[b])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += loss.item() * len(b)
            n += len(b)
        log(f"  graph {key} epoch {ep + 1}/{cfg.epochs}: loss {tot / max(n, 1):.4f}")

    # snapshots B and C: the embeddings val rows / test rows + the server use
    snap_b = snapshot(key, rows, c["B"], cfg.label_delay_days, cfg.min_degree, cfg.max_nodes)
    snap_c = snapshot(key, rows, c["C"], cfg.label_delay_days, cfg.min_degree, cfg.max_nodes)
    emb_b = _embed(encoder, snap_b, x_mean, x_std, device)
    emb_c = _embed(encoder, snap_c, x_mean, x_std, device)
    head.eval()
    val = rows[rows["split"].to_numpy() == "val"]
    val_auroc, coverage = None, 0.0
    if len(val):
        ids_v = snap_b.lookup(val[KINDS].to_numpy().astype(np.int64))
        ok = (ids_v >= 0).any(1)
        coverage = float(ok.mean())
        yv = val["is_fraud"].to_numpy()[ok]
        if ok.sum() > 100 and yv.sum() > 0 and (yv == 0).sum() > 0:
            with torch.no_grad():
                slots, known = gather_slots(emb_b, ids_v[ok])
                lv = head(slots, known).float().cpu().numpy()
            val_auroc = auroc(yv, lv)
    log(f"  graph {key}: val auroc {val_auroc if val_auroc is None else round(val_auroc, 4)} on {coverage:.0%} of val rows "
        f"(B {snap_b.n_nodes:,} nodes, C {snap_c.n_nodes:,} nodes, {time.time() - t0:.0f}s)")
    return {
        "key": key, "encoder": encoder.state_dict(), "encoder_config": encoder.config(), "head": head.state_dict(),
        "head_hidden": head.hidden, "x_mean": x_mean.astype(np.float32), "x_std": x_std.astype(np.float32),
        "cuts": {"A": c["A"], "B": c["B"], "C": c["C"]},
        "val": {"kind": snap_b.node_kind, "code": snap_b.node_code, "emb": emb_b.half().cpu().numpy()},
        "test": {"kind": snap_c.node_kind, "code": snap_c.node_code, "emb": emb_c.half().cpu().numpy()},
        "stats": {"nodes_a": int(snap_a.n_nodes), "edges_a": int(len(snap_a.edge_src) // 2), "nodes_b": int(snap_b.n_nodes),
                  "nodes_c": int(snap_c.n_nodes), "fit_rows": int(len(idx)), "fit_fraud": int(y_fit[idx].sum()),
                  "val_auroc": val_auroc, "val_coverage": coverage, "seconds": round(time.time() - t0, 1)},
    }


def train_graph(run_dir: Path, prepared_meta: dict, cfg: GraphConfig, device="cpu", log=print, seed=1337) -> dict:
    """every source of the prepared set that has a graph; writes runs/<run>/graph/<source>.pt and graph.json"""
    run_dir = Path(run_dir)
    out = run_dir / "graph"
    out.mkdir(parents=True, exist_ok=True)
    sources = [k for k in prepared_meta.get("source_index", {}) if k in schema.SOURCE_INDEX]
    summary = {"created": time.strftime("%Y-%m-%dT%H:%M:%S"), "config": cfg.__dict__, "sources": {}}
    for key in sources:
        try:
            state = train_source(key, cfg, device=device, log=log, seed=seed)
        except Exception as e:  # noqa: BLE001 - one source's graph must not sink the run
            log(f"  graph {key}: failed ({type(e).__name__}: {e}), skipping")
            state = None
        if state is None:
            continue
        torch.save(state, out / f"{key}.pt")
        summary["sources"][key] = state["stats"]
    vals = [s["val_auroc"] for s in summary["sources"].values() if s.get("val_auroc") is not None]
    summary["val_auroc"] = float(np.mean(vals)) if vals else None
    (run_dir / "graph.json").write_text(json.dumps(summary, indent=1, default=str))
    log("  graph val auroc: " + " ".join(f"{k}={v['val_auroc']:.3f}" for k, v in summary["sources"].items() if v.get("val_auroc") is not None))
    return summary
