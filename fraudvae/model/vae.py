"""Heterogeneous-likelihood VAE for tabular transactions, conditioned on the
source (which bank / dataset the row came from).

Encoder sees [standardised numerics, their observed-mask, embedded categoricals,
embedded source]. Decoder gets [z, embedded source] and emits a Gaussian
(mean, logvar) per numeric and logits per categorical. Unobserved entries are
masked out of the likelihood (HI-VAE style).
"""
from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

LOG_2PI = math.log(2 * math.pi)


def mlp(sizes):
    layers = []
    for a, b in zip(sizes[:-1], sizes[1:]):
        layers += [nn.Linear(a, b), nn.ReLU()]
    return nn.Sequential(*layers)


class TabularVAE(nn.Module):
    def __init__(self, n_num: int, cat_sizes: list[int], n_sources: int,
                 latent_dim=16, hidden=(256, 256), cat_embed_dim=8, source_embed_dim=6):
        super().__init__()
        self.n_num = n_num
        self.cat_sizes = list(cat_sizes)
        self.latent_dim = latent_dim

        self.cat_emb = nn.ModuleList([nn.Embedding(k, cat_embed_dim) for k in cat_sizes])
        self.src_emb = nn.Embedding(n_sources + 1, source_embed_dim)   # +1 for the generic token (0)

        enc_in = 2 * n_num + len(cat_sizes) * cat_embed_dim + source_embed_dim
        hidden = list(hidden)
        self.encoder = mlp([enc_in] + hidden)
        self.to_latent = nn.Linear(hidden[-1], 2 * latent_dim)

        dec_hidden = hidden[::-1]
        self.decoder = mlp([latent_dim + source_embed_dim] + dec_hidden)
        self.num_head = nn.Linear(dec_hidden[-1], 2 * n_num)
        self.cat_heads = nn.ModuleList([nn.Linear(dec_hidden[-1], k) for k in cat_sizes])

    def encode(self, num, mask, cat, src):
        parts = [num * mask, mask]
        for j, emb in enumerate(self.cat_emb):
            parts.append(emb(cat[:, j]))
        parts.append(self.src_emb(src))
        hdn = self.encoder(torch.cat(parts, dim=1))
        mu, logvar = self.to_latent(hdn).chunk(2, dim=1)
        return mu, logvar

    def decode(self, z, src):
        hdn = self.decoder(torch.cat([z, self.src_emb(src)], dim=1))
        out = self.num_head(hdn)
        num_mu, num_logvar = out[:, : self.n_num], out[:, self.n_num :]
        cat_logits = [head(hdn) for head in self.cat_heads]
        return num_mu, num_logvar, cat_logits

    def forward(self, num, mask, cat, src):
        mu, logvar = self.encode(num, mask, cat, src)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        num_mu, num_logvar, cat_logits = self.decode(z, src)
        return {"mu": mu, "logvar": logvar, "z": z, "num_mu": num_mu, "num_logvar": num_logvar, "cat_logits": cat_logits}

    def neg_elbo(self, out, num, mask, cat):
        """per-row negative ELBO, unobserved entries contribute nothing"""
        num_nll = 0.5 * (out["num_logvar"] + (num - out["num_mu"]) ** 2 / torch.exp(out["num_logvar"]) + LOG_2PI)
        rec = (num_nll * mask).sum(dim=1)
        for j, logits in enumerate(out["cat_logits"]):
            target = cat[:, j].long()
            rec = rec + F.cross_entropy(logits, target, reduction="none") * (target != 0).float()
        kl = -0.5 * (1 + out["logvar"] - out["mu"].pow(2) - out["logvar"].exp()).sum(dim=1)
        return rec + kl

    def config(self):
        return {
            "n_num": self.n_num, "cat_sizes": self.cat_sizes, "n_sources": self.src_emb.num_embeddings - 1,
            "latent_dim": self.latent_dim, "hidden": [m.out_features for m in self.encoder if isinstance(m, nn.Linear)],
            "cat_embed_dim": self.cat_emb[0].embedding_dim if len(self.cat_emb) else 0,
            "source_embed_dim": self.src_emb.embedding_dim,
        }


def build_model(meta: dict, model_cfg) -> TabularVAE:
    cat_sizes = [c["size"] for c in meta["categorical"]]
    n_sources = len(meta["source_index"])
    return TabularVAE(
        n_num=len(meta["numeric"]), cat_sizes=cat_sizes, n_sources=n_sources,
        latent_dim=model_cfg.latent_dim, hidden=tuple(model_cfg.hidden),
        cat_embed_dim=model_cfg.cat_embed_dim, source_embed_dim=model_cfg.source_embed_dim,
    )


def save_checkpoint(path: Path, model: TabularVAE, meta: dict, cfg: dict):
    torch.save({
        "model_config": model.config(),
        "state_dict": model.state_dict(),
        "prepared_meta": meta,
        "train_config": cfg,
    }, path)


def load_checkpoint(path: Path, device="cpu") -> tuple[TabularVAE, dict]:
    ck = torch.load(path, map_location=device, weights_only=False)
    model = TabularVAE(**ck["model_config"])
    model.load_state_dict(ck["state_dict"])
    model.to(device).eval()
    return model, ck
