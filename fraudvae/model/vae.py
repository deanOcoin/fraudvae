"""Heterogeneous-likelihood VAE for tabular transactions, conditioned on the
source (which bank / dataset the row came from).

Encoder sees [standardised numerics, their observed-mask, embedded categoricals,
embedded source]. Decoder gets [z, embedded source] and emits a Gaussian
(mean, logvar) per numeric and logits per categorical. Unobserved entries are
masked out of the likelihood (HI-VAE style), which is how nine datasets with
different columns live in one model.
"""
from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

LOG_2PI = math.log(2 * math.pi)


def mlp(sizes, dropout=0.0):
    layers = []
    for a, b in zip(sizes[:-1], sizes[1:]):
        layers += [nn.Linear(a, b), nn.LayerNorm(b), nn.SiLU()]
        if dropout:
            layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


class TabularVAE(nn.Module):
    def __init__(self, n_num: int, cat_sizes: list[int], n_sources: int,
                 latent_dim=20, hidden=(512, 512, 256), cat_embed_dim=8, source_embed_dim=6,
                 dropout=0.05, min_logvar=-7.0, max_logvar=4.0):
        super().__init__()
        self.n_num = n_num
        self.cat_sizes = list(cat_sizes)
        self.latent_dim = latent_dim
        self.min_logvar = min_logvar
        self.max_logvar = max_logvar

        self.cat_emb = nn.ModuleList([nn.Embedding(k, cat_embed_dim) for k in cat_sizes])
        self.src_emb = nn.Embedding(n_sources + 1, source_embed_dim)   # +1 for the generic token (0)

        enc_in = 2 * n_num + len(cat_sizes) * cat_embed_dim + source_embed_dim
        hidden = list(hidden)
        self.encoder = mlp([enc_in] + hidden, dropout)
        self.to_latent = nn.Linear(hidden[-1], 2 * latent_dim)

        dec_hidden = hidden[::-1]
        self.decoder = mlp([latent_dim + source_embed_dim] + dec_hidden, dropout)
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

    def encode(self, num, mask, cat, src):
        hdn = self.encoder(self.embed_inputs(num, mask, cat, src))
        mu, logvar = self.to_latent(hdn).chunk(2, dim=1)
        return mu, logvar.clamp(-10.0, 6.0)

    @staticmethod
    def reparameterize(mu, logvar):
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def decode(self, z, src):
        hdn = self.decoder(torch.cat([z, self.src_emb(src)], dim=1))
        out = self.num_head(hdn)
        num_mu, num_logvar = out[:, : self.n_num], out[:, self.n_num :]
        num_logvar = num_logvar.clamp(self.min_logvar, self.max_logvar)
        cat_logits = [head(hdn) for head in self.cat_heads]
        return num_mu, num_logvar, cat_logits

    def forward(self, num, mask, cat, src):
        mu, logvar = self.encode(num, mask, cat, src)
        z = self.reparameterize(mu, logvar)
        num_mu, num_logvar, cat_logits = self.decode(z, src)
        return {"mu": mu, "logvar": logvar, "z": z, "num_mu": num_mu, "num_logvar": num_logvar, "cat_logits": cat_logits}

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

    @torch.no_grad()
    def elbo_per_field(self, num, mask, cat, src, n_samples=8):
        """Monte Carlo estimate of the per-field reconstruction NLL (averaged over n_samples
        draws of z) plus the row's KL. returns (num_nll [B, n_num], cat_nll [B, n_cat], kl [B])"""
        mu, logvar = self.encode(num, mask, cat, src)
        num_acc, cat_acc = None, None
        for _ in range(n_samples):
            z = self.reparameterize(mu, logvar)
            num_mu, num_logvar, cat_logits = self.decode(z, src)
            out = {"num_mu": num_mu, "num_logvar": num_logvar, "cat_logits": cat_logits}
            num_nll, cat_nll = self.nll_terms(out, num, mask, cat)
            num_acc = num_nll if num_acc is None else num_acc + num_nll
            cat_acc = cat_nll if cat_acc is None else cat_acc + cat_nll
        kl = self.kl_terms(mu, logvar).sum(dim=1)
        return num_acc / n_samples, cat_acc / n_samples, kl

    def config(self):
        return {
            "n_num": self.n_num, "cat_sizes": self.cat_sizes, "n_sources": self.src_emb.num_embeddings - 1,
            "latent_dim": self.latent_dim, "hidden": [m.out_features for m in self.encoder if isinstance(m, nn.Linear)],
            "cat_embed_dim": self.cat_emb[0].embedding_dim if len(self.cat_emb) else 0,
            "source_embed_dim": self.src_emb.embedding_dim,
            "dropout": next((m.p for m in self.encoder if isinstance(m, nn.Dropout)), 0.0),
            "min_logvar": self.min_logvar, "max_logvar": self.max_logvar,
        }


def build_model(meta: dict, model_cfg) -> TabularVAE:
    cat_sizes = [c["size"] for c in meta["categorical"]]
    from ..data import schema

    # always size the source embedding for the whole registry, so a run trained on a subset
    # of sources can still be asked about any of them (and the scorer's one-hots line up)
    n_sources = max(meta["source_index"].values()) if meta.get("source_index") else len(schema.SOURCES)
    n_sources = max(n_sources, len(schema.SOURCES))
    return TabularVAE(
        n_num=len(meta["numeric"]), cat_sizes=cat_sizes, n_sources=n_sources,
        latent_dim=model_cfg.latent_dim, hidden=tuple(model_cfg.hidden),
        cat_embed_dim=model_cfg.cat_embed_dim, source_embed_dim=model_cfg.source_embed_dim,
        dropout=model_cfg.dropout, min_logvar=model_cfg.min_logvar, max_logvar=model_cfg.max_logvar,
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
    ck = torch.load(path, map_location=device, weights_only=False)
    mc = ck["model_config"]
    model = TabularVAE(**mc)
    model.load_state_dict(ck["state_dict"])
    model.to(device).eval()
    return model, ck
