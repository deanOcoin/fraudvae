"""The public fraud-graph benchmarks, run on the same encoder the transaction graphs use.
Node classification with the standard splits of each dataset; what they validate is that the
GraphSAGE in model.py learns what the literature says a GraphSAGE learns on these graphs,
nothing about our banks. Results land in a json (and docs/GRAPH_BENCHMARKS.md by hand).

  dgraph    DGraph-Fin: 3.7M users, 4.3M edges, 15.5k fraud, the published train/valid/test
            masks (background classes 2 / 3 are in the graph but never scored)
  elliptic  Elliptic: 203k bitcoin transactions, 234k edges, illicit vs licit, time steps
            1-29 train / 30-34 val / 35-49 test (the published temporal split, val carved out)
  yelpchi   YelpChi: 46k reviews, three relations (same user / same product+month / same
            product+star), 14.5% fraud; stratified 40/20/40 like CARE-GNN
  amazon    Amazon: 12k users, three relations, 6.9% fraud; same protocol, the first 3305
            nodes (no labels of their own) left out of the scored sets

usage: fraudvae graph-bench --dataset elliptic [--out results.json] [--device cuda]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ..model.scoring import auprc, auroc
from ..paths import RAW_DIR
from .model import SAGE, NodeHead, adjacency


def _sym(src, dst, n):
    """undirected, unweighted, deduplicated"""
    s = np.concatenate([src, dst]).astype(np.int64)
    d = np.concatenate([dst, src]).astype(np.int64)
    key = np.unique(s * n + d)
    return key // n, key % n


def load_dgraph(raw=RAW_DIR):
    d = np.load(Path(raw) / "dgraph" / "dgraphfin.npz")
    x = d["x"].astype(np.float32)
    y = d["y"].astype(np.int64)
    src, dst = _sym(d["edge_index"][:, 0], d["edge_index"][:, 1], len(x))
    return {"x": x, "y": (y == 1).astype(np.float32), "splits": {"train": d["train_mask"], "val": d["valid_mask"], "test": d["test_mask"]},
            "relations": [(src, dst)], "name": "dgraph"}


def load_elliptic(raw=RAW_DIR):
    import pandas as pd

    root = Path(raw) / "elliptic" / "elliptic_bitcoin_dataset"
    feats = pd.read_csv(root / "elliptic_txs_features.csv", header=None)
    classes = pd.read_csv(root / "elliptic_txs_classes.csv")
    edges = pd.read_csv(root / "elliptic_txs_edgelist.csv")
    tx = feats[0].to_numpy()
    idx = {int(t): i for i, t in enumerate(tx)}
    step = feats[1].to_numpy()
    x = feats.iloc[:, 2:].to_numpy(dtype=np.float32)
    lab = classes.set_index("txId")["class"].reindex(tx).astype(str).to_numpy()
    y = np.where(lab == "1", 1.0, 0.0).astype(np.float32)
    known = lab != "unknown"
    src = edges["txId1"].map(idx).to_numpy()
    dst = edges["txId2"].map(idx).to_numpy()
    src, dst = _sym(src, dst, len(x))
    splits = {"train": np.flatnonzero(known & (step <= 29)), "val": np.flatnonzero(known & (step >= 30) & (step <= 34)),
              "test": np.flatnonzero(known & (step >= 35))}
    return {"x": x, "y": y, "splits": splits, "relations": [(src, dst)], "name": "elliptic"}


def _load_mat(path, relations, drop_first=0, seed=2):
    import scipy.io as sio
    import scipy.sparse as sp

    m = sio.loadmat(path)
    x = sp.csr_matrix(m["features"]).toarray().astype(np.float32)
    y = np.asarray(m["label"]).reshape(-1).astype(np.float32)
    n = len(y)
    rels = []
    for r in relations:
        a = sp.coo_matrix(m[r])
        rels.append(_sym(a.row, a.col, n))
    rng = np.random.default_rng(seed)
    nodes = np.arange(drop_first, n)
    tr, va, te = [], [], []
    for cls in (0, 1):
        c = nodes[y[nodes] == cls]
        c = rng.permutation(c)
        a, b = int(0.4 * len(c)), int(0.6 * len(c))
        tr.append(c[:a]); va.append(c[a:b]); te.append(c[b:])
    splits = {"train": np.sort(np.concatenate(tr)), "val": np.sort(np.concatenate(va)), "test": np.sort(np.concatenate(te))}
    return {"x": x, "y": y, "splits": splits, "relations": rels}


def load_yelpchi(raw=RAW_DIR):
    d = _load_mat(Path(raw) / "yelpchi" / "YelpChi.mat", ["net_rur", "net_rtr", "net_rsr"])
    d["name"] = "yelpchi"
    return d


def load_amazon(raw=RAW_DIR):
    d = _load_mat(Path(raw) / "amazon" / "Amazon.mat", ["net_upu", "net_usu", "net_uvu"], drop_first=3305)
    d["name"] = "amazon"
    return d


LOADERS = {"dgraph": load_dgraph, "elliptic": load_elliptic, "yelpchi": load_yelpchi, "amazon": load_amazon}


def run_bench(data: dict, device="cpu", hidden=64, out_dim=32, layers=2, dropout=0.2, lr=5e-3, epochs=300, patience=40,
              log=print, seed=0) -> dict:
    torch.manual_seed(seed)
    t0 = time.time()
    x, y, sp = data["x"], data["y"], data["splits"]
    mean = x[sp["train"]].mean(0)
    std = np.maximum(x[sp["train"]].std(0), 1e-6)
    xt = torch.from_numpy((x - mean) / std).float().to(device)
    yt = torch.from_numpy(y).to(device)
    n = len(y)
    adjs = [adjacency(s, d, np.ones(len(s), dtype=np.float32), n, device=device) for s, d in data["relations"]]
    enc = SAGE(x.shape[1], hidden, out_dim, layers, dropout, n_relations=len(adjs)).to(device)
    head = NodeHead(out_dim, hidden=max(32, out_dim), dropout=dropout).to(device)
    opt = torch.optim.Adam(list(enc.parameters()) + list(head.parameters()), lr=lr, weight_decay=5e-5)
    tr = torch.from_numpy(sp["train"]).to(device)
    y_tr = yt[tr]
    pos_weight = min(float((y_tr == 0).sum() / max(1, (y_tr == 1).sum())), 50.0)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))
    n_edges = sum(len(s) for s, _ in data["relations"]) // 2
    log(f"  {data['name']}: {n:,} nodes, {n_edges:,} edges, {len(adjs)} relation(s), {x.shape[1]} features; "
        f"train/val/test {len(sp['train']):,}/{len(sp['val']):,}/{len(sp['test']):,}, fraud {y[sp['test']].mean():.1%} of test, pos_weight {pos_weight:.0f}")
    best, best_state, bad = -1.0, None, 0
    for ep in range(epochs):
        enc.train(); head.train()
        logit = head(enc(xt, adjs))
        loss = loss_fn(logit[tr], y_tr)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if ep % 5 == 4 or ep == epochs - 1:
            enc.eval(); head.eval()
            with torch.no_grad():
                lg = head(enc(xt, adjs)).float().cpu().numpy()
            va = auroc(y[sp["val"]], lg[sp["val"]])
            if va > best + 1e-4:
                best, bad = va, 0
                best_state = ({k: v.detach().clone() for k, v in enc.state_dict().items()}, {k: v.detach().clone() for k, v in head.state_dict().items()})
            else:
                bad += 5
            if ep % 50 == 49:
                log(f"  {data['name']} epoch {ep + 1}: loss {loss.item():.4f} val auroc {va:.4f} (best {best:.4f})")
            if bad >= patience:
                break
    if best_state is not None:
        enc.load_state_dict(best_state[0]); head.load_state_dict(best_state[1])
    enc.eval(); head.eval()
    with torch.no_grad():
        lg = head(enc(xt, adjs)).float().cpu().numpy()
    te = sp["test"]
    yt_, lt = y[te], lg[te]
    # f1 at the threshold that flags the same share as the true fraud rate (elliptic reports f1)
    k = int(round(yt_.sum()))
    thr = np.sort(lt)[-k] if k > 0 else np.inf
    pred = lt >= thr
    tp = float((pred & (yt_ == 1)).sum())
    f1 = 2 * tp / max(pred.sum() + yt_.sum(), 1)
    res = {"auroc": auroc(yt_, lt), "auprc": auprc(yt_, lt), "f1_at_base_rate": float(f1), "val_auroc": float(best),
           "n_nodes": int(n), "n_edges": int(n_edges), "n_relations": len(adjs), "epochs": int(ep + 1),
           "test_nodes": int(len(te)), "test_fraud": int(yt_.sum()), "seconds": round(time.time() - t0, 1),
           "model": {"hidden": hidden, "out_dim": out_dim, "layers": layers, "dropout": dropout, "lr": lr}}
    log(f"  {data['name']}: test auroc {res['auroc']:.4f} auprc {res['auprc']:.4f} f1@base-rate {f1:.3f} ({res['seconds']:.0f}s, {res['epochs']} epochs)")
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(prog="fraudvae graph-bench", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", action="append", choices=list(LOADERS), help="repeatable; default all that are on disk")
    ap.add_argument("--out", default=None, help="json to write / update")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--raw-dir", default=str(RAW_DIR))
    a = ap.parse_args(argv)
    names = a.dataset or list(LOADERS)
    results = {}
    if a.out and Path(a.out).exists():
        results = json.loads(Path(a.out).read_text())
    for name in names:
        try:
            data = LOADERS[name](a.raw_dir)
        except FileNotFoundError as e:
            print(f"  {name}: not on disk ({e}), skipping")
            continue
        results[name] = run_bench(data, device=a.device, epochs=a.epochs, hidden=a.hidden)
        results[name]["created"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        if a.out:
            Path(a.out).write_text(json.dumps(results, indent=1))
    return 0
