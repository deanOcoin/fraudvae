"""v5: field heads. A field head is a small supervised classifier over one named block of the
common schema - the device block first - that produces its own calibrated signal for the
blender next to the booster, the novelty score and the sequence head.

Why, when the booster already sees every field: the booster's view of a device is tangled up
with everything else on the row and with the bank it was trained on. A head that only sees
the device block learns "what a risky device looks like" across every source that reports
one (IEEE-CIS identity rows, every BAF application, every fraudecom purchase) and transfers
that to a bank that sends a device payload with no labels of its own. It also gives the
receipt a sentence of its own ("the device says..."), with its presence bit telling the
blender when the payload was not there at all.

The head consumes the same standardised numerics / mask / categorical codes the VAE does,
restricted to its block, plus the source embedding (blanked for a slice of the training rows
so the generic mode is trained like the booster's). Rows where none of the block's fields
is observed get NaN: absent, not zero.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ..data import schema
from .dataset import PreparedSet
from .scoring import auroc

# which schema groups each head reads (the NumericFeature / CategoricalFeature .group tags)
HEAD_SPECS = {
    "device": {"groups": ("device",), "label": "Device & identity", "hidden": 128},
}


def head_columns(name: str, num_names: list[str], cat_names: list[str]):
    """(numeric indices, categorical indices) of a head's fields in a prepared set"""
    groups = set(HEAD_SPECS[name]["groups"])
    num_group = {f.name: f.group for f in schema.NUMERIC}
    cat_group = {f.name: f.group for f in schema.CATEGORICAL}
    num_idx = [j for j, n in enumerate(num_names) if num_group.get(n) in groups]
    cat_idx = [j for j, n in enumerate(cat_names) if cat_group.get(n) in groups]
    return num_idx, cat_idx


class FieldHead(nn.Module):
    def __init__(self, name: str, num_idx: list[int], cat_idx: list[int], cat_sizes: list[int], n_sources: int,
                 hidden: int = 128, cat_emb: int = 4, src_emb: int = 4, dropout: float = 0.1):
        super().__init__()
        self.name = name
        self.num_idx = list(num_idx)
        self.cat_idx = list(cat_idx)
        self.cat_sizes = list(cat_sizes)
        self.n_sources = n_sources
        self.hidden = hidden
        self.register_buffer("num_sel", torch.tensor(self.num_idx, dtype=torch.long))
        self.register_buffer("cat_sel", torch.tensor(self.cat_idx, dtype=torch.long))
        self.cat_emb = nn.ModuleList([nn.Embedding(s, cat_emb) for s in self.cat_sizes])
        self.src_emb = nn.Embedding(n_sources + 1, src_emb)
        d_in = 2 * len(self.num_idx) + cat_emb * len(self.cat_idx) + src_emb
        self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.SiLU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, hidden // 2), nn.SiLU(), nn.Linear(hidden // 2, 1))

    def present(self, mask, cat) -> torch.Tensor:
        """bool per row: at least one field of the block observed"""
        p = torch.zeros(mask.shape[0], dtype=torch.bool, device=mask.device)
        if len(self.num_idx):
            p |= mask[:, self.num_sel].sum(1) > 0
        if len(self.cat_idx):
            p |= (cat[:, self.cat_sel] > 0).any(1)
        return p

    def forward(self, num, mask, cat, src):
        parts = []
        if len(self.num_idx):
            m = mask[:, self.num_sel]
            parts += [num[:, self.num_sel] * m, m]
        for k, emb in enumerate(self.cat_emb):
            parts.append(emb(cat[:, self.cat_sel[k]]))
        parts.append(self.src_emb(src.clamp(0, self.n_sources)))
        return self.net(torch.cat(parts, dim=1)).squeeze(1)

    @torch.no_grad()
    def logit(self, num, mask, cat, src) -> np.ndarray:
        """float32 logit per row, NaN where the block is absent"""
        out = self.forward(num, mask, cat, src).float()
        out = torch.where(self.present(mask, cat), out, torch.full_like(out, float("nan")))
        return out.cpu().numpy()

    def config(self) -> dict:
        return {"name": self.name, "num_idx": self.num_idx, "cat_idx": self.cat_idx, "cat_sizes": self.cat_sizes,
                "n_sources": self.n_sources, "hidden": self.hidden}


def head_path(run_dir: Path, name: str) -> Path:
    return Path(run_dir) / f"head_{name}.pt"


def load_head(run_dir: Path, name: str, device="cpu") -> FieldHead | None:
    p = head_path(run_dir, name)
    if not p.exists():
        return None
    ck = torch.load(p, map_location=device, weights_only=False)
    cfg = ck["config"]
    head = FieldHead(cfg["name"], cfg["num_idx"], cfg["cat_idx"], cfg["cat_sizes"], cfg["n_sources"], cfg["hidden"])
    head.load_state_dict(ck["state_dict"])
    return head.to(device).eval()


def load_heads(run_dir: Path, device="cpu") -> dict[str, FieldHead]:
    heads = {}
    for name in HEAD_SPECS:
        head = load_head(run_dir, name, device)
        if head is not None:
            heads[name] = head
    return heads


def _rows_with_block(split, head: FieldHead, chunk=200_000) -> np.ndarray:
    """indices of the rows where the block is observed, without loading the split twice"""
    keep = []
    n = len(split)
    dev = head.num_sel.device
    for s in range(0, n, chunk):
        b = np.arange(s, min(n, s + chunk))
        _, mask, cat, _ = split.batch(b, dev)
        keep.append(b[head.present(mask, cat).cpu().numpy()])
    return np.concatenate(keep) if keep else np.zeros(0, dtype=np.int64)


def train_head(name: str, run_dir: Path, prepared: str, device="cpu", log=print, max_rows=1_000_000,
               epochs=4, batch_size=4096, lr=1e-3, generic_fraction=0.25, seed=1337) -> dict | None:
    """fit one field head on the labelled train rows that carry its block, report val AUROC
    per source, write head_<name>.pt + head_<name>.json. None when no source carries the block."""
    spec = HEAD_SPECS[name]
    run_dir = Path(run_dir)
    t0 = time.time()
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    data = PreparedSet(prepared, splits=("train", "val"))
    num_names = [f["name"] for f in data.meta["numeric"]]
    cat_names = [f["name"] for f in data.meta["categorical"]]
    cat_sizes = [len(f["values"]) for f in data.meta["categorical"]]
    num_idx, cat_idx = head_columns(name, num_names, cat_names)
    n_sources = max(max(data.meta["source_index"].values()), len(schema.SOURCES))
    head = FieldHead(name, num_idx, cat_idx, [cat_sizes[j] for j in cat_idx], n_sources, spec.get("hidden", 128)).to(device)
    tr, va = data["train"], data["val"]
    tr_idx = _rows_with_block(tr, head)
    va_idx = _rows_with_block(va, head)
    if len(tr_idx) < 1000:
        log(f"  {name} head: the block is observed on too few rows ({len(tr_idx):,}), skipping")
        return None
    if len(tr_idx) > max_rows:
        tr_idx = np.sort(rng.choice(tr_idx, max_rows, replace=False))
    y_tr = tr.y[:][tr_idx]
    pos_weight = float((y_tr == 0).sum() / max(1, (y_tr == 1).sum()))
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-5)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))
    log(f"  {name} head: {len(num_idx)} numerics + {len(cat_idx)} categoricals, {len(tr_idx):,} train rows, "
        f"{len(va_idx):,} val rows")
    gen = torch.Generator(device="cpu").manual_seed(seed)
    for ep in range(epochs):
        head.train()
        tot, n = 0.0, 0
        for b, (num, mask, cat, src) in tr.iter_batches(batch_size, idx=tr_idx, shuffle=True, seed=seed + ep, device=device):
            if generic_fraction > 0:
                blank = (torch.rand(len(b), generator=gen) < generic_fraction).to(device)
                src = torch.where(blank, torch.zeros_like(src), src)
            yb = torch.from_numpy(tr.y[b].astype(np.float32)).to(device)
            loss = loss_fn(head(num, mask, cat, src), yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += loss.item() * len(b)
            n += len(b)
        log(f"  {name} head epoch {ep + 1}/{epochs}: loss {tot / max(n, 1):.4f}")
    head.eval()
    logits = np.empty(len(va_idx), dtype=np.float32)
    pos = {int(i): k for k, i in enumerate(va_idx)}
    with torch.no_grad():
        for b, (num, mask, cat, src) in va.iter_batches(batch_size * 2, idx=va_idx, device=device):
            logits[[pos[int(i)] for i in b]] = head.logit(num, mask, cat, src)
    yv, sv = va.y[:][va_idx], va.src[:][va_idx]
    per_source = {}
    for s in np.unique(sv):
        m = sv == s
        per_source[data.source_names.get(int(s), str(int(s)))] = {
            "auroc": auroc(yv[m], logits[m]), "n": int(m.sum()), "fraud": int(yv[m].sum())}
    info = {"created": time.strftime("%Y-%m-%dT%H:%M:%S"), "prepared": prepared, "name": name, "label": spec.get("label", name),
            "numeric": [num_names[j] for j in num_idx], "categorical": [cat_names[j] for j in cat_idx],
            "train_rows": int(len(tr_idx)), "val_rows": int(len(va_idx)), "epochs": epochs, "pos_weight": pos_weight,
            "sources": per_source,
            "val_auroc": float(np.nanmean([v["auroc"] for v in per_source.values()])),
            "seconds": round(time.time() - t0, 1)}
    torch.save({"state_dict": head.state_dict(), "config": head.config()}, head_path(run_dir, name))
    (run_dir / f"head_{name}.json").write_text(json.dumps(info, indent=1))
    log(f"  {name} head val auroc: " + " ".join(f"{k}={v['auroc']:.3f}" for k, v in per_source.items()))
    return info


def train_heads(run_dir: Path, prepared: str, device="cpu", log=print, **kw) -> dict:
    """every head in HEAD_SPECS; returns {name: info} for the ones that could be trained"""
    out = {}
    for name in HEAD_SPECS:
        info = train_head(name, run_dir, prepared, device=device, log=log, **kw)
        if info is not None:
            out[name] = info
    return out
