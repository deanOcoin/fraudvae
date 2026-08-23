"""Turn a trained VAE into an anomaly scorer.

Two scores come out of the same forward pass:

  raw   = sum_f nll_f(x) + KL(q(z|x) || p(z))            a Monte Carlo -ELBO
  norm  = sum_f obs_f * (nll_f - mu_f) / sd_f / sqrt(n_obs) + (KL - mu_kl) / sd_kl

where mu_f / sd_f are the legit-validation mean/std of feature f's NLL for the
row's source. raw is dominated by whatever feature has the widest likelihood
(hour, day of week...), norm asks "how surprising is each field compared to
normal traffic from this bank". Calibration picks whichever has the higher
validation AUROC and the UI reports both.
"""
from __future__ import annotations

import numpy as np
import torch

from ..data import schema
from .vae import TabularVAE

N_SOURCES = len(schema.SOURCES)


@torch.no_grad()
def score_batch(model: TabularVAE, num, mask, cat, src, n_samples=8):
    """returns dict of numpy arrays: raw, kl, nll_num [N,D], nll_cat [N,C], mu [N,latent],
    dec_mu (decoder mean, standardised units) [N,D], cat_prob (prob of the observed class) [N,C]"""
    mu, logvar = model.encode(num, mask, cat, src)
    kl = model.kl_terms(mu, logvar).sum(1)
    nll_num = torch.zeros_like(num)
    nll_cat = torch.zeros(num.shape[0], len(model.cat_sizes), device=num.device)
    dec_mu = torch.zeros_like(num)
    cat_prob = torch.zeros_like(nll_cat)
    for _ in range(n_samples):
        z = mu if n_samples == 1 else model.reparameterize(mu, logvar)
        num_mu, num_logvar, cat_logits = model.decode(z, src)
        out = {"num_mu": num_mu, "num_logvar": num_logvar, "cat_logits": cat_logits}
        a, b = model.nll_terms(out, num, mask, cat)
        nll_num += a
        nll_cat += b
        dec_mu += num_mu
        for j, logits in enumerate(cat_logits):
            p = torch.softmax(logits.float(), dim=1)
            cat_prob[:, j] += p.gather(1, cat[:, j : j + 1]).squeeze(1)
    nll_num /= n_samples
    nll_cat /= n_samples
    dec_mu /= n_samples
    cat_prob /= n_samples
    raw = nll_num.sum(1) + nll_cat.sum(1) + kl
    return {
        "raw": raw.cpu().numpy(),
        "kl": kl.cpu().numpy(),
        "nll_num": nll_num.cpu().numpy(),
        "nll_cat": nll_cat.cpu().numpy(),
        "observed": torch.cat([mask, (cat != 0).float()], dim=1).cpu().numpy(),
        "mu": mu.cpu().numpy(),
        "dec_mu": dec_mu.cpu().numpy(),
        "cat_prob": cat_prob.cpu().numpy(),
    }


class Scorer:
    """holds per-source NLL baselines and computes the normalised score.

    baselines = {"by_source": {str(src_idx): {"feat_mean": [...], "feat_std": [...], "kl_mean": f, "kl_std": f}},
                 "generic": {...same...}}  (generic is used for src 0 and for sources without stats)"""

    def __init__(self, baselines: dict, n_features: int, n_sources: int = N_SOURCES):
        self.baselines = baselines
        self.n_sources = n_sources
        F = n_features
        self.feat_mean = np.zeros((n_sources + 1, F), dtype=np.float32)
        self.feat_std = np.ones((n_sources + 1, F), dtype=np.float32)
        self.kl_mean = np.zeros(n_sources + 1, dtype=np.float32)
        self.kl_std = np.ones(n_sources + 1, dtype=np.float32)
        gen = baselines.get("generic")
        for s in range(n_sources + 1):
            b = baselines.get("by_source", {}).get(str(s)) or gen
            if b is None:
                continue
            self.feat_mean[s] = b["feat_mean"]
            self.feat_std[s] = np.maximum(b["feat_std"], 1e-4)
            self.kl_mean[s] = b["kl_mean"]
            self.kl_std[s] = max(b["kl_std"], 1e-4)

    def norm(self, r: dict, src: np.ndarray) -> np.ndarray:
        src = np.clip(np.asarray(src).astype(int), 0, self.n_sources)
        nll = np.concatenate([r["nll_num"], r["nll_cat"]], axis=1)
        obs = r["observed"]
        z = (nll - self.feat_mean[src]) / self.feat_std[src] * obs
        n_obs = np.maximum(obs.sum(1), 1.0)
        zkl = (r["kl"] - self.kl_mean[src]) / self.kl_std[src]
        return (z.sum(1) / np.sqrt(n_obs) + zkl).astype(np.float32)

    def feature_z(self, r: dict, src: int) -> np.ndarray:
        nll = np.concatenate([r["nll_num"], r["nll_cat"]], axis=1)
        src = min(max(int(src), 0), self.n_sources)
        return (nll - self.feat_mean[src]) / self.feat_std[src]

    @staticmethod
    def fit_baselines(nll: np.ndarray, kl: np.ndarray) -> dict:
        return {"feat_mean": nll.mean(0).round(6).tolist(), "feat_std": np.maximum(nll.std(0), 1e-4).round(6).tolist(),
                "kl_mean": float(kl.mean()), "kl_std": float(max(kl.std(), 1e-4))}


def score_split(model, split, batch_size=8192, n_samples=8, device="cpu", idx=None, keep_features=True,
                progress=None, scorer: Scorer | None = None, src_override=None, keep_mu=True):
    """score every row of a PreparedSplit (or the given idx). results come back in the order
    of idx (or row order). src_override=0 scores everything in the 'generic' context."""
    model.eval()
    chunks = {"raw": [], "kl": []}
    if keep_mu:
        chunks["mu"] = []
    if keep_features:
        chunks["nll_num"] = []
        chunks["nll_cat"] = []
        chunks["observed"] = []
    if scorer is not None:
        chunks["norm"] = []
    n_done = 0
    for b, (num, mask, cat, src) in split.iter_batches(batch_size, idx=idx, device=device):
        if src_override is not None:
            src = torch.full_like(src, int(src_override))
        r = score_batch(model, num, mask, cat, src, n_samples)
        if scorer is not None:
            r["norm"] = scorer.norm(r, src.cpu().numpy())
        for k in chunks:
            chunks[k].append(r[k])
        n_done += len(b)
        if progress and n_done % (batch_size * 20) == 0:
            progress(n_done)
    return {k: np.concatenate(v) if v else np.zeros((0,)) for k, v in chunks.items()}


def recall_at_fpr(y, s, fpr=0.01):
    """threshold at the (1-fpr) quantile of legit scores, recall of fraud above it"""
    legit = s[y == 0]
    if len(legit) == 0:
        return float("nan"), float("nan")
    thr = float(np.quantile(legit, 1 - fpr))
    fraud = s[y == 1]
    rec = float((fraud > thr).mean()) if len(fraud) else float("nan")
    return thr, rec


def auroc(y, s):
    from sklearn.metrics import roc_auc_score

    y = np.asarray(y)
    if len(y) == 0 or y.min() == y.max():
        return float("nan")
    return float(roc_auc_score(y, s))


def auprc(y, s):
    from sklearn.metrics import average_precision_score

    y = np.asarray(y)
    if len(y) == 0 or y.sum() == 0:
        return float("nan")
    return float(average_precision_score(y, s))
