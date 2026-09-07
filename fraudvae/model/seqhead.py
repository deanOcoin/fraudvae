"""v5: the sequence head. A small supervised classifier on top of the VAE's history read-out
h (what this customer has been doing, read for this transaction) and the VAE's input embedding
of the transaction itself. The history encoder was trained without labels (ELBO on legit rows);
this is the one place its summary meets the fraud labels, and its logit is the "sequence"
signal the blender sees next to the booster and the novelty score.

Why a separate head and not just more booster features: the booster is a ranker over observed
columns; h is a 64-128 d learned vector that trees handle badly and that would make the
booster depend on the VAE. One MLP, trained on the same labelled subsample the booster uses,
keeps the signals separable on the receipt ("the history says...").
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .dataset import PreparedSet
from .scoring import auroc, batches, history_window


class SeqHead(nn.Module):
    def __init__(self, hdim: int, emb_dim: int, hidden: int = 128, dropout: float = 0.1):
        super().__init__()
        self.hdim, self.emb_dim, self.hidden = hdim, emb_dim, hidden
        self.net = nn.Sequential(nn.LayerNorm(hdim + emb_dim), nn.Linear(hdim + emb_dim, hidden), nn.SiLU(),
                                 nn.Dropout(dropout), nn.Linear(hidden, hidden // 2), nn.SiLU(), nn.Linear(hidden // 2, 1))

    def forward(self, h, emb):
        return self.net(torch.cat([h, emb], dim=1)).squeeze(1)

    @torch.no_grad()
    def logit(self, model, num, mask, cat, src, h) -> np.ndarray:
        """the sequence logit for a batch, given the VAE's context h for it"""
        emb = model.embed_inputs(num, mask, cat, src)
        return self.forward(h.float(), emb.float()).float().cpu().numpy()


def load_seqhead(run_dir: Path, device="cpu") -> SeqHead | None:
    p = Path(run_dir) / "seqhead.pt"
    if not p.exists():
        return None
    ck = torch.load(p, map_location=device, weights_only=False)
    head = SeqHead(ck["hdim"], ck["emb_dim"], ck["hidden"])
    head.load_state_dict(ck["state_dict"])
    head.to(device).eval()
    return head


def _sample(split, per_source_legit, rng):
    src, y = split.src[:], split.y[:]
    keep = []
    for s in np.unique(src):
        idx = np.flatnonzero(src == s)
        lg = idx[y[idx] == 0]
        if per_source_legit and len(lg) > per_source_legit:
            lg = rng.choice(lg, per_source_legit, replace=False)
        keep += [lg, idx[y[idx] == 1]]
    return np.sort(np.concatenate(keep))


def train_seqhead(run_dir: Path, prepared: str, model, device="cpu", log=print, max_legit_per_source=200_000,
                  val_legit_per_source=100_000, epochs=3, batch_size=4096, lr=1e-3, seed=1337) -> dict | None:
    """fit the head on the labelled train split (legit subsampled per source), report val AUROC
    per source, write seqhead.pt + seqhead.json. returns None when the model has no history encoder."""
    window = history_window(model)
    if not window:
        return None
    run_dir = Path(run_dir)
    t0 = time.time()
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    data = PreparedSet(prepared, splits=("train", "val"))
    if data.seq is None:
        log("  sequence head: the prepared set has no sequence index, skipping")
        return None
    tr, va = data["train"], data["val"]
    tr_idx = _sample(tr, max_legit_per_source, rng)
    va_idx = _sample(va, val_legit_per_source, rng)
    y_tr = tr.y[tr_idx]
    pos_weight = float((y_tr == 0).sum() / max(1, (y_tr == 1).sum()))
    pos_weight = min(pos_weight, 50.0)   # 1:1000 base rates make the loss all about the few positives
    emb_dim = 2 * model.n_num + len(model.cat_sizes) * model.cat_emb[0].embedding_dim + model.src_emb.embedding_dim
    head = SeqHead(model.hdim, emb_dim).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-5)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))
    model.eval()
    log(f"  sequence head: {len(tr_idx):,} train rows ({int(y_tr.sum()):,} fraud), {len(va_idx):,} val rows, "
        f"h {model.hdim} + embedding {emb_dim}, pos_weight {pos_weight:.0f}")
    for ep in range(epochs):
        head.train()
        tot, n = 0.0, 0
        for b, (num, mask, cat, src), hist in batches(tr, batch_size, idx=tr_idx, shuffle=True, seed=seed + ep,
                                                      device=device, seq=data.seq, split_name="train", window=window):
            with torch.no_grad():
                h = model.context(num, mask, cat, src, hist).float()
                emb = model.embed_inputs(num, mask, cat, src).float()
            yb = torch.from_numpy(tr.y[b].astype(np.float32)).to(device)
            loss = loss_fn(head(h, emb), yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += loss.item() * len(b)
            n += len(b)
        log(f"  sequence head epoch {ep + 1}/{epochs}: loss {tot / max(n, 1):.4f}")
    head.eval()
    logits = np.empty(len(va_idx), dtype=np.float32)
    pos = {int(i): k for k, i in enumerate(va_idx)}
    with torch.no_grad():
        for b, (num, mask, cat, src), hist in batches(va, batch_size * 2, idx=va_idx, device=device, seq=data.seq,
                                                      split_name="val", window=window):
            h = model.context(num, mask, cat, src, hist)
            out = head.logit(model, num, mask, cat, src, h)
            logits[[pos[int(i)] for i in b]] = out
    yv, sv = va.y[va_idx], va.src[va_idx]
    per_source = {data.source_names.get(int(s), str(int(s))): {"auroc": auroc(yv[sv == s], logits[sv == s]),
                                                              "n": int((sv == s).sum()), "fraud": int(yv[sv == s].sum())}
                  for s in np.unique(sv)}
    info = {"created": time.strftime("%Y-%m-%dT%H:%M:%S"), "prepared": prepared, "hdim": model.hdim, "emb_dim": emb_dim,
            "hidden": head.hidden, "window": window, "train_rows": int(len(tr_idx)), "val_rows": int(len(va_idx)),
            "epochs": epochs, "pos_weight": pos_weight, "sources": per_source,
            "val_auroc": float(np.mean([v["auroc"] for v in per_source.values() if v["auroc"] == v["auroc"]])),
            "seconds": round(time.time() - t0, 1)}
    torch.save({"state_dict": head.state_dict(), "hdim": model.hdim, "emb_dim": emb_dim, "hidden": head.hidden}, run_dir / "seqhead.pt")
    (run_dir / "seqhead.json").write_text(json.dumps(info, indent=1))
    log("  sequence head val auroc: " + " ".join(f"{k}={v['auroc']:.3f}" for k, v in per_source.items())
        + f" ({time.time() - t0:.0f}s)")
    return info
