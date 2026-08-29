"""Turn a trained VAE into an anomaly scorer, and combine it with the booster.

Two VAE scores come out of the same forward pass:

  raw   = sum_f nll_f(x) + KL(q(z|x) || p(z))            a Monte Carlo -ELBO
  norm  = sum_f obs_f * (nll_f - mu_f) / sd_f / sqrt(n_obs) + (KL - mu_kl) / sd_kl

where mu_f / sd_f are the legit-validation mean/std of feature f's NLL for the
row's source. raw is dominated by whatever feature has the widest likelihood
(hour, day of week...), norm asks "how surprising is each field compared to
normal traffic from this bank". Calibration picks whichever has the higher
validation AUROC and the UI reports both.

The final p(fraud) is the blender (see Scorer.blend_features): the booster's
logit, the VAE's standardised score and its KL term with a per-source weight,
plus the per-field surprises and the latent mean, through one logistic regression.
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
    """holds per-source NLL baselines and computes the normalised score, plus the
    blender: a logistic regression that turns [booster logit, VAE score z, KL z,
    per-field surprises, latent mean] into the final p(fraud), with per-source
    weights for the two headline signals (fit in calibrate.py on labelled val rows).

    baselines = {"by_source": {str(src_idx): {"feat_mean": [...], "feat_std": [...], "kl_mean": f, "kl_std": f}},
                 "generic": {...same...}}  (generic is used for src 0 and for sources without stats)
    blend = {"coef": [...], "intercept": f, "score_type": "raw"|"norm",
             "score_mean": [S+1], "score_std": [S+1]}   legit mean/std of the chosen VAE score per source idx"""

    def __init__(self, baselines: dict, n_features: int, n_sources: int = N_SOURCES, blend: dict | None = None):
        self.baselines = baselines
        # a run calibrated before a source was added to the schema has fewer baselines than
        # today's SOURCES; size the tables for whichever is larger and let unknown sources
        # fall back to the generic row (see _src)
        known = [int(k) for k in baselines.get("by_source", {})]
        self.n_sources = max([n_sources] + known)
        n_sources = self.n_sources
        self.blend = None
        if blend and blend.get("coef"):
            self.set_blend(blend)
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

    def set_blend(self, blend: dict):
        self.blend = blend
        self.blend_coef = np.asarray(blend["coef"], dtype=np.float32)
        self.blend_intercept = float(blend["intercept"])
        self.score_mean = np.asarray(blend["score_mean"], dtype=np.float32)
        self.score_std = np.maximum(np.asarray(blend["score_std"], dtype=np.float32), 1e-6)

    def _src(self, src, n_known: int) -> np.ndarray:
        """source indices with anything the tables were not fit for mapped to generic (0)"""
        src = np.asarray(src).astype(int)
        return np.where((src < 0) | (src > n_known), 0, src)

    def norm(self, r: dict, src: np.ndarray) -> np.ndarray:
        src = self._src(src, self.n_sources)
        nll = np.concatenate([r["nll_num"], r["nll_cat"]], axis=1)
        obs = r["observed"]
        z = (nll - self.feat_mean[src]) / self.feat_std[src] * obs
        n_obs = np.maximum(obs.sum(1), 1.0)
        zkl = (r["kl"] - self.kl_mean[src]) / self.kl_std[src]
        return (z.sum(1) / np.sqrt(n_obs) + zkl).astype(np.float32)

    def feature_z(self, r: dict, src: int) -> np.ndarray:
        nll = np.concatenate([r["nll_num"], r["nll_cat"]], axis=1)
        src = int(self._src([src], self.n_sources)[0])
        return (nll - self.feat_mean[src]) / self.feat_std[src]

    # -- blender ---------------------------------------------------------------------

    # bounds on the blender's inputs. the per-source z's are unbounded in principle (a card with
    # a 10,000x normal amount) and the far tail drifts between validation and the newest test
    # months; a logistic weight fit on the bulk then does silly things out there. clipping keeps
    # every extreme row "very suspicious" without letting it dominate the sum.
    BLEND_CLIP = {"b": (-15.0, 15.0), "v": (-5.0, 15.0), "zkl": (-5.0, 15.0), "field": (-5.0, 15.0)}

    def blend_features(self, r: dict, src: np.ndarray, booster_logit: np.ndarray | None, score_type: str | None = None,
                       score_mean=None, score_std=None) -> np.ndarray:
        """[booster logit, VAE score z, KL z, source one-hot, each of the three x one-hot,
        per-field surprises, latent mean].
        the interaction columns are what lets the blender trust the booster more for one bank
        and the VAE more for another. every signal is clipped (BLEND_CLIP)."""
        kind = score_type or (self.blend["score_type"] if self.blend else "raw")
        s_mean = self.score_mean if score_mean is None else np.asarray(score_mean, dtype=np.float32)
        s_std = self.score_std if score_std is None else np.maximum(np.asarray(score_std, dtype=np.float32), 1e-6)
        # the one-hot is as wide as the blender was fit (score_mean has one entry per source + generic)
        n_oh = len(s_mean) - 1
        src = self._src(src, n_oh)
        n = len(src)
        score = r["norm"] if kind == "norm" else r["raw"]
        if kind == "norm" and "norm" not in r:
            score = self.norm(r, src)
        v = np.clip((np.asarray(score, dtype=np.float32) - s_mean[src]) / s_std[src], *self.BLEND_CLIP["v"])[:, None]
        b = np.zeros((n, 1), dtype=np.float32) if booster_logit is None else \
            np.clip(np.asarray(booster_logit, dtype=np.float32), *self.BLEND_CLIP["b"])[:, None]
        zkl = np.clip((r["kl"] - self.kl_mean[src]) / self.kl_std[src], *self.BLEND_CLIP["zkl"])[:, None]
        onehot = np.zeros((n, n_oh + 1), dtype=np.float32)
        onehot[np.arange(n), src] = 1.0
        nll = np.concatenate([r["nll_num"], r["nll_cat"]], axis=1)
        fz = np.clip((nll - self.feat_mean[src]) / self.feat_std[src], *self.BLEND_CLIP["field"]) * r["observed"]
        parts = [b, v, zkl, onehot, b * onehot, v * onehot, zkl * onehot, fz, r["mu"]]
        return np.concatenate(parts, axis=1).astype(np.float32)

    @staticmethod
    def blend_feature_names(n_sources: int, n_fields: int, latent: int) -> list[str]:
        oh = [f"src{i}" for i in range(n_sources + 1)]
        names = ["booster", "vae_z", "kl_z"] + oh + [f"booster*{o}" for o in oh] + [f"vae_z*{o}" for o in oh] + [f"kl_z*{o}" for o in oh]
        return names + [f"field{i}_z" for i in range(n_fields)] + [f"mu{i}" for i in range(latent)]

    def blend_prob(self, r: dict, src: np.ndarray, booster_logit: np.ndarray | None) -> np.ndarray:
        x = self.blend_features(r, src, booster_logit)
        logit = x @ self.blend_coef + self.blend_intercept
        return (1.0 / (1.0 + np.exp(-np.clip(logit, -30, 30)))).astype(np.float32)

    @staticmethod
    def fit_baselines(nll: np.ndarray, kl: np.ndarray) -> dict:
        return {"feat_mean": nll.mean(0).round(6).tolist(), "feat_std": np.maximum(nll.std(0), 1e-4).round(6).tolist(),
                "kl_mean": float(kl.mean()), "kl_std": float(max(kl.std(), 1e-4))}


def score_split(model, split, batch_size=8192, n_samples=8, device="cpu", idx=None, keep_features=True,
                progress=None, scorer: Scorer | None = None, src_override=None, booster=None, keep_mu=True):
    """score every row of a PreparedSplit (or the given idx). results come back in the order
    of idx (or row order). src_override=0 scores everything in the 'generic' context. with a
    booster the per-row booster logit comes back as 'booster', and if the scorer has a blender,
    'blend' too."""
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
    if booster is not None:
        chunks["booster"] = []
    if scorer is not None and scorer.blend:
        chunks["blend"] = []
    n_done = 0
    for b, (num, mask, cat, src) in split.iter_batches(batch_size, idx=idx, device=device):
        if src_override is not None:
            src = torch.full_like(src, int(src_override))
        r = score_batch(model, num, mask, cat, src, n_samples)
        s_np = src.cpu().numpy()
        if scorer is not None:
            r["norm"] = scorer.norm(r, s_np)
        if booster is not None:
            wide = split.wide(b) if (booster.wide_names and getattr(split, "n_wide", 0)) else None
            r["booster"] = booster.logit(num.cpu().numpy(), mask.cpu().numpy(), cat.cpu().numpy(), s_np, wide=wide)
        if scorer is not None and scorer.blend:
            r["blend"] = scorer.blend_prob(r, s_np, r.get("booster"))
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
    s = np.asarray(s, dtype=np.float64)
    if len(y) == 0 or y.min() == y.max():
        return float("nan")
    if not np.isfinite(s).all():
        # a non-finite score is a broken model, not a ranking: score it as unknown rather than crash
        ok = np.isfinite(s)
        if ok.sum() < 2 or y[ok].min() == y[ok].max():
            return float("nan")
        s = np.where(ok, s, np.nanmax(s[ok]) + 1.0)   # NaN scores rank as "most anomalous"
    return float(roc_auc_score(y, s))


def auprc(y, s):
    from sklearn.metrics import average_precision_score

    y = np.asarray(y)
    if len(y) == 0 or y.sum() == 0:
        return float("nan")
    return float(average_precision_score(y, s))
