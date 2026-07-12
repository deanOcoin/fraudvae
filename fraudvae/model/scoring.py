"""Turn a trained VAE into an anomaly scorer.

  score = sum_f nll_f(x) + KL(q(z|x) || p(z))            a Monte Carlo -ELBO

averaged over n_samples draws of z, with the per-field NLLs kept so the UI can show
which fields made a row look unusual.
"""
from __future__ import annotations

import numpy as np
import torch

from .vae import TabularVAE


@torch.no_grad()
def score_batch(model: TabularVAE, num, mask, cat, src, n_samples=8):
    """returns dict of numpy arrays: raw, kl, nll_num [N,D], nll_cat [N,C]"""
    mu, logvar = model.encode(num, mask, cat, src)
    kl = model.kl_terms(mu, logvar).sum(1)
    nll_num = torch.zeros_like(num)
    nll_cat = torch.zeros(num.shape[0], len(model.cat_sizes), device=num.device)
    for _ in range(n_samples):
        z = model.reparameterize(mu, logvar)
        num_mu, num_logvar, cat_logits = model.decode(z, src)
        out = {"num_mu": num_mu, "num_logvar": num_logvar, "cat_logits": cat_logits}
        a, b = model.nll_terms(out, num, mask, cat)
        nll_num += a
        nll_cat += b
    nll_num /= n_samples
    nll_cat /= n_samples
    raw = nll_num.sum(1) + nll_cat.sum(1) + kl
    return {
        "raw": raw.cpu().numpy(),
        "kl": kl.cpu().numpy(),
        "nll_num": nll_num.cpu().numpy(),
        "nll_cat": nll_cat.cpu().numpy(),
    }


def score_split(model, split, batch_size=8192, n_samples=8, device="cpu", idx=None):
    """score every row of a PreparedSplit (or the given idx), in that order"""
    model.eval()
    chunks = {"raw": [], "kl": [], "nll_num": [], "nll_cat": []}
    for b, (num, mask, cat, src) in split.iter_batches(batch_size, idx=idx, device=device):
        r = score_batch(model, num, mask, cat, src, n_samples)
        for k in chunks:
            chunks[k].append(r[k])
    return {k: np.concatenate(v) for k, v in chunks.items()}


def auroc(y, s):
    from sklearn.metrics import roc_auc_score

    y = np.asarray(y)
    if y.min() == y.max():
        return float("nan")
    return float(roc_auc_score(y, s))
