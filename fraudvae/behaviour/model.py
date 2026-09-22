"""One verifier per modality: a small MLP that maps a feature window to a unit-norm embedding,
trained to tell the dataset's users apart (softmax over users on the cosine, a scaled
"normalised softmax" that makes the embedding space metric), used at serve time by comparing
a probe session's embedding with the account holder's enrolled one. The cosine similarity is
turned into a likelihood ratio by platt scaling fitted on genuine / impostor trials of users
the verifier never trained on, so the number means the same thing for a new deployment."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class Verifier(nn.Module):
    def __init__(self, modality: str, d_in: int, emb: int = 64, hidden: int = 256, n_users: int = 0, dropout: float = 0.2, scale: float = 16.0):
        super().__init__()
        self.modality, self.d_in, self.emb_dim, self.hidden, self.scale = modality, d_in, emb, hidden, scale
        self.register_buffer("x_mean", torch.zeros(d_in))
        self.register_buffer("x_std", torch.ones(d_in))
        self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.SiLU(), nn.Dropout(dropout), nn.Linear(hidden, hidden), nn.SiLU(),
                                 nn.Dropout(dropout), nn.Linear(hidden, emb))
        self.classes = nn.Parameter(torch.randn(max(n_users, 1), emb) * 0.05)   # one prototype per training user
        # cosine -> log odds genuine:impostor, fitted on held-out users
        self.register_buffer("platt_a", torch.ones(1))
        self.register_buffer("platt_b", torch.zeros(1))
        self.register_buffer("eer", torch.zeros(1))

    def embed(self, x):
        x = (x - self.x_mean) / self.x_std
        x = x.clamp(-10, 10)
        return nn.functional.normalize(self.net(x), dim=-1)

    def logits(self, x):
        """user classification logits on the normalised embedding (training only)"""
        return self.scale * self.embed(x) @ nn.functional.normalize(self.classes, dim=-1).t()

    @torch.no_grad()
    def session_embedding(self, windows: np.ndarray) -> np.ndarray | None:
        """mean of the window embeddings, unit norm; None without windows"""
        if windows is None or len(windows) == 0:
            return None
        x = torch.from_numpy(np.asarray(windows, dtype=np.float32)).to(self.x_mean.device)
        e = self.embed(x).mean(0)
        return nn.functional.normalize(e, dim=0).cpu().numpy()

    def llr(self, cosine: float) -> float:
        return float(self.platt_a.item() * cosine + self.platt_b.item())

    def config(self) -> dict:
        return {"modality": self.modality, "d_in": self.d_in, "emb": self.emb_dim, "hidden": self.hidden,
                "n_users": int(self.classes.shape[0]), "scale": self.scale}


def equal_error_rate(genuine: np.ndarray, impostor: np.ndarray) -> tuple[float, float]:
    """(eer, threshold) from genuine / impostor similarity scores"""
    scores = np.concatenate([genuine, impostor])
    labels = np.concatenate([np.ones(len(genuine)), np.zeros(len(impostor))])
    order = np.argsort(-scores)
    s, l = scores[order], labels[order]
    tp = np.cumsum(l)
    fp = np.cumsum(1 - l)
    far = fp / max(len(impostor), 1)                 # impostors accepted
    frr = 1 - tp / max(len(genuine), 1)              # genuine rejected
    k = int(np.argmin(np.abs(far - frr)))
    return float((far[k] + frr[k]) / 2), float(s[k])
