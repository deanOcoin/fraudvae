"""Heterogeneous-likelihood VAE for tabular transactions, conditioned on the
source (which bank / dataset the row came from) and, since v5, on the
customer's own history.

Encoder sees [standardised numerics, their observed-mask, embedded categoricals,
embedded source, history summary h]. Decoder gets [z, embedded source, h] and
emits a Gaussian (mean, logvar) per numeric and logits per categorical.
Unobserved entries are masked out of the likelihood (HI-VAE style), which is
how nine datasets with different columns live in one model. With h in both
halves the reconstruction error is "surprising for this customer", not only
"surprising for this bank"; without a history encoder (history disabled, or
the v4 checkpoints) it is exactly the v4 model.
"""
from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from .loading import load_checkpoint as read_checkpoint

LOG_2PI = math.log(2 * math.pi)


def mlp(sizes, dropout=0.0):
    layers = []
    for a, b in zip(sizes[:-1], sizes[1:]):
        layers += [nn.Linear(a, b), nn.LayerNorm(b), nn.SiLU()]
        if dropout:
            layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


class HistoryEncoder(nn.Module):
    """v5: the customer's previous transactions as a sequence. Each past row is embedded with
    the VAE's own field embeddings (shared, so "this amount" means the same thing in the
    history and in the row being scored), tagged with how long ago it was, run through a
    small transformer, and summarised by one attention read-out whose query is the row being
    scored - the history is read *for this transaction*. A learned "no history" token is
    always present, so a customer's first transaction (or a bank without customer ids) gets a
    well-defined context instead of an empty attention."""

    def __init__(self, token_in: int, d_model=128, n_layers=2, n_heads=4, window=32, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.window = window
        self.proj = nn.Linear(token_in, d_model)
        self.time = nn.Linear(1, d_model)                   # log-elapsed-time embedding
        self.recency = nn.Embedding(window, d_model)        # 0 = most recent ... window-1 = oldest
        self.nohist = nn.Parameter(torch.zeros(1, 1, d_model))
        # the projected tokens are a sum of three unbounded terms; without this the attention
        # logits grow with the data's outliers and bf16 training blew up at 2e-3
        self.in_norm = nn.LayerNorm(d_model)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=2 * d_model, dropout=dropout,
                                           batch_first=True, norm_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, n_layers)
        self.query = nn.Linear(token_in, d_model)
        self.readout = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tokens, dt, valid, query_in):
        """tokens [B, L, token_in] (oldest first), dt [B, L] (log elapsed), valid [B, L] bool,
        query_in [B, token_in] (the current row's input embedding) -> h [B, d_model]"""
        B, L, _ = tokens.shape
        rec = torch.arange(L - 1, -1, -1, device=tokens.device).clamp(max=self.window - 1)
        x = self.in_norm(self.proj(tokens) + self.time(dt.unsqueeze(-1)) + self.recency(rec)[None])
        x = torch.cat([self.nohist.expand(B, 1, -1), x], dim=1)
        pad = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=tokens.device), ~valid], dim=1)
        x = self.encoder(x, src_key_padding_mask=pad)
        q = self.query(query_in).unsqueeze(1)
        h, _ = self.readout(q, x, x, key_padding_mask=pad, need_weights=False)
        return self.norm(h.squeeze(1))


class TabularVAE(nn.Module):
    def __init__(self, n_num: int, cat_sizes: list[int], n_sources: int,
                 latent_dim=20, hidden=(512, 512, 256), cat_embed_dim=8, source_embed_dim=6,
                 dropout=0.05, min_logvar=-7.0, max_logvar=4.0, history: dict | None = None):
        super().__init__()
        self.n_num = n_num
        self.cat_sizes = list(cat_sizes)
        self.latent_dim = latent_dim
        self.min_logvar = min_logvar
        self.max_logvar = max_logvar

        self.cat_emb = nn.ModuleList([nn.Embedding(k, cat_embed_dim) for k in cat_sizes])
        self.src_emb = nn.Embedding(n_sources + 1, source_embed_dim)   # +1 for the generic token (0)

        enc_in = 2 * n_num + len(cat_sizes) * cat_embed_dim + source_embed_dim
        # v5: an optional history encoder; its summary h conditions both halves of the VAE,
        # so the reconstruction (and the surprise) is relative to this customer's past
        self.history_cfg = dict(history) if history and history.get("enabled", True) and history.get("d_model", 0) else None
        self.hist = None
        hdim = 0
        if self.history_cfg:
            hdim = int(self.history_cfg["d_model"])
            self.hist = HistoryEncoder(enc_in, d_model=hdim, n_layers=int(self.history_cfg.get("layers", 2)),
                                       n_heads=int(self.history_cfg.get("heads", 4)), window=int(self.history_cfg.get("window", 32)),
                                       dropout=float(self.history_cfg.get("dropout", 0.1)))
        self.hdim = hdim
        hidden = list(hidden)
        self.encoder = mlp([enc_in + hdim] + hidden, dropout)
        self.to_latent = nn.Linear(hidden[-1], 2 * latent_dim)

        dec_hidden = hidden[::-1]
        self.decoder = mlp([latent_dim + source_embed_dim + hdim] + dec_hidden, dropout)
        self.num_head = nn.Linear(dec_hidden[-1], 2 * n_num)
        self.cat_heads = nn.ModuleList([nn.Linear(dec_hidden[-1], k) for k in cat_sizes])

        # decoder logvar starts near 0 (unit variance) instead of random, otherwise epoch 1 is a mess
        nn.init.zeros_(self.num_head.bias)
        with torch.no_grad():
            self.num_head.weight[n_num:] *= 0.1

    # -- pieces ----------------------------------------------------------------

    # standardised numerics beyond this many sigmas are "extreme" and stay extreme: as model
    # inputs they would swamp the first layer, as reconstruction targets one such cell's
    # Gaussian NLL ((150 sigma)^2 * e^7) would own the batch's gradient. bounding both keeps
    # the surprise of such a row maximal without the numerics of the rest going to pieces
    NUM_CLIP = 20.0

    def embed_inputs(self, num, mask, cat, src):
        num = num.clamp(-self.NUM_CLIP, self.NUM_CLIP)
        parts = [num * mask, mask]
        for j, emb in enumerate(self.cat_emb):
            parts.append(emb(cat[:, j]))
        parts.append(self.src_emb(src))
        return torch.cat(parts, dim=1)

    def context(self, num, mask, cat, src, hist=None):
        """h [B, hdim] from the history window, or the "no history" read-out when none is given.
        hist = (h_num, h_mask, h_cat, h_dt, h_valid) as SequenceIndex.history returns them."""
        if self.hist is None:
            return None
        x_cur = self.embed_inputs(num, mask, cat, src)
        B = num.shape[0]
        if hist is None:
            L = 1
            tokens = x_cur.new_zeros(B, L, x_cur.shape[1])
            dt = x_cur.new_zeros(B, L)
            valid = torch.zeros(B, L, dtype=torch.bool, device=num.device)
        else:
            h_num, h_mask, h_cat, dt, valid = hist
            L = h_num.shape[1]
            tokens = self.embed_inputs(h_num.reshape(B * L, -1), h_mask.reshape(B * L, -1), h_cat.reshape(B * L, -1),
                                       src.repeat_interleave(L)).reshape(B, L, -1)
        return self.hist(tokens, dt, valid, x_cur)

    def encode(self, num, mask, cat, src, h=None):
        x = self.embed_inputs(num, mask, cat, src)
        if self.hist is not None:
            x = torch.cat([x, h if h is not None else self.context(num, mask, cat, src)], dim=1)
        hdn = self.encoder(x)
        mu, logvar = self.to_latent(hdn).chunk(2, dim=1)
        return mu, logvar.clamp(-10.0, 6.0)

    @staticmethod
    def reparameterize(mu, logvar):
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def decode(self, z, src, h=None):
        parts = [z, self.src_emb(src)]
        if self.hist is not None:
            parts.append(h)
        hdn = self.decoder(torch.cat(parts, dim=1))
        out = self.num_head(hdn)
        num_mu, num_logvar = out[:, : self.n_num], out[:, self.n_num :]
        num_logvar = num_logvar.clamp(self.min_logvar, self.max_logvar)
        cat_logits = [head(hdn) for head in self.cat_heads]
        return num_mu, num_logvar, cat_logits

    def forward(self, num, mask, cat, src, hist=None):
        h = self.context(num, mask, cat, src, hist)
        mu, logvar = self.encode(num, mask, cat, src, h)
        z = self.reparameterize(mu, logvar)
        num_mu, num_logvar, cat_logits = self.decode(z, src, h)
        return {"mu": mu, "logvar": logvar, "z": z, "num_mu": num_mu, "num_logvar": num_logvar, "cat_logits": cat_logits, "h": h}

    # -- likelihoods -----------------------------------------------------------

    def nll_terms(self, out, num, mask, cat):
        """per-row, per-feature negative log likelihoods. unobserved -> 0"""
        num = num.clamp(-self.NUM_CLIP, self.NUM_CLIP)
        num_nll = 0.5 * (out["num_logvar"] + (num - out["num_mu"]) ** 2 / torch.exp(out["num_logvar"]) + LOG_2PI)
        num_nll = num_nll * mask
        cat_nlls = []
        for j, logits in enumerate(out["cat_logits"]):
            target = cat[:, j].long()
            observed = (target != 0).float()
            nll = F.cross_entropy(logits.float(), target, reduction="none") * observed
            cat_nlls.append(nll)
        cat_nll = torch.stack(cat_nlls, dim=1) if cat_nlls else num_nll.new_zeros((num.shape[0], 0))
        return num_nll, cat_nll

    @staticmethod
    def kl_terms(mu, logvar):
        return -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())   # per dim

    def config(self):
        return {
            "n_num": self.n_num, "cat_sizes": self.cat_sizes, "n_sources": self.src_emb.num_embeddings - 1,
            "latent_dim": self.latent_dim, "hidden": [m.out_features for m in self.encoder if isinstance(m, nn.Linear)],
            "cat_embed_dim": self.cat_emb[0].embedding_dim if len(self.cat_emb) else 0,
            "source_embed_dim": self.src_emb.embedding_dim,
            "dropout": next((m.p for m in self.encoder if isinstance(m, nn.Dropout)), 0.0),
            "min_logvar": self.min_logvar, "max_logvar": self.max_logvar,
            "history": self.history_cfg,
        }


def build_model(meta: dict, model_cfg) -> TabularVAE:
    cat_sizes = [c["size"] for c in meta["categorical"]]
    from ..data import schema

    # always size the source embedding for the whole registry, so a run trained on a subset
    # of sources can still be asked about any of them (and the scorer's one-hots line up)
    n_sources = max(meta["source_index"].values()) if meta.get("source_index") else len(schema.SOURCES)
    n_sources = max(n_sources, len(schema.SOURCES))
    hist = getattr(model_cfg, "history", None)
    history = None
    if hist is not None and getattr(hist, "enabled", False):
        history = {"enabled": True, "d_model": hist.d_model, "layers": hist.layers, "heads": hist.heads,
                   "window": hist.window, "dropout": hist.dropout}
    return TabularVAE(
        n_num=len(meta["numeric"]), cat_sizes=cat_sizes, n_sources=n_sources,
        latent_dim=model_cfg.latent_dim, hidden=tuple(model_cfg.hidden),
        cat_embed_dim=model_cfg.cat_embed_dim, source_embed_dim=model_cfg.source_embed_dim,
        dropout=model_cfg.dropout, min_logvar=model_cfg.min_logvar, max_logvar=model_cfg.max_logvar,
        history=history,
    )


def save_checkpoint(path: Path, model: TabularVAE, meta: dict, cfg: dict, extra: dict | None = None):
    torch.save({
        "model_config": model.config(),
        "state_dict": model.state_dict(),
        "prepared_meta": meta,
        "train_config": cfg,
        "extra": extra or {},
    }, path)


def load_checkpoint(path: Path, device="cpu") -> tuple[TabularVAE, dict]:
    ck = read_checkpoint(path, device)
    mc = ck["model_config"]
    model = TabularVAE(**mc)
    model.load_state_dict(ck["state_dict"])
    model.to(device).eval()
    return model, ck
