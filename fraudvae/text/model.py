"""A fastText-shaped classifier: hashed bag -> EmbeddingBag mean -> small MLP -> logit, with a
temperature fitted on validation so the probability means what it says, and the base rate
of its own training data so the output can be read as a likelihood ratio."""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn

from .features import BUCKETS, encode_batch


class TextHead(nn.Module):
    def __init__(self, name: str, buckets: int = BUCKETS, dim: int = 64, hidden: int = 64, dropout: float = 0.2):
        super().__init__()
        self.name, self.buckets, self.dim, self.hidden = name, buckets, dim, hidden
        self.bag = nn.EmbeddingBag(buckets, dim, mode="mean")
        self.net = nn.Sequential(nn.LayerNorm(dim), nn.Dropout(dropout), nn.Linear(dim, hidden), nn.SiLU(), nn.Linear(hidden, 1))
        # platt scaling fitted on validation: calibrated logit = logit / temperature + bias. the
        # class weighting in training shifts the raw logits by about log(pos_weight); the bias
        # takes that back out so the probability means what it says
        self.register_buffer("temperature", torch.ones(1))
        self.register_buffer("bias", torch.zeros(1))
        self.register_buffer("base_logit", torch.zeros(1))   # logit of the training base rate

    def forward(self, ids, offsets):
        return self.net(self.bag(ids, offsets)).squeeze(1)

    @torch.no_grad()
    def score(self, texts: list[str]) -> dict:
        """calibrated probabilities and likelihood ratios (log odds against the head's own base
        rate) for a list of texts"""
        ids, offsets = encode_batch(texts, self.buckets)
        dev = self.temperature.device
        logit = self.forward(torch.from_numpy(ids).to(dev), torch.from_numpy(offsets).to(dev)).float() / self.temperature + self.bias
        prob = torch.sigmoid(logit).cpu().numpy()
        llr = (logit - self.base_logit).cpu().numpy()
        return {"prob": prob.astype(np.float32), "llr": llr.astype(np.float32)}

    def config(self) -> dict:
        return {"name": self.name, "buckets": self.buckets, "dim": self.dim, "hidden": self.hidden}


def fit_platt(logits: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """(temperature, bias) such that sigmoid(logit / T + b) is calibrated on held-out rows
    (minimum NLL), with the usual platt smoothing of the targets"""
    lg = torch.from_numpy(np.asarray(logits, dtype=np.float64))
    y = np.asarray(y, dtype=np.float64)
    n_pos, n_neg = y.sum(), len(y) - y.sum()
    yt = torch.from_numpy(np.where(y > 0.5, (n_pos + 1) / (n_pos + 2), 1.0 / (n_neg + 2)))
    params = torch.zeros(2, dtype=torch.float64, requires_grad=True)   # log T, bias
    opt = torch.optim.LBFGS([params], lr=0.1, max_iter=200)

    def closure():
        opt.zero_grad()
        loss = nn.functional.binary_cross_entropy_with_logits(lg / torch.exp(params[0]) + params[1], yt)
        loss.backward()
        return loss

    opt.step(closure)
    return float(math.exp(params[0].item())), float(params[1].item())
