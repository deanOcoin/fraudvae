"""Loaded-model state + turning a raw transaction dict into a scored result.

One run is "active" at a time. Everything the UI needs for a prediction
(standardisation stats, vocab, thresholds) is read from the run directory,
so switching runs is just load_run(name).
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import numpy as np
import torch

from ..data import schema
from ..data.prepare import DERIVED_CATS, transform_numeric
from ..model.scoring import score_batch
from ..model.vae import load_checkpoint
from ..paths import RUNS_DIR

_lock = threading.Lock()


class ActiveModel:
    def __init__(self, run_dir: Path, device="cpu"):
        self.run_dir = Path(run_dir)
        self.name = self.run_dir.name
        self.device = torch.device(device)
        self.model, ck = load_checkpoint(self.run_dir / "model.pt", device=self.device)
        self.meta = ck["prepared_meta"]
        self.train_config = ck.get("train_config", {})
        cal_p = self.run_dir / "calibration.json"
        self.cal = json.loads(cal_p.read_text()) if cal_p.exists() else None
        ev_p = self.run_dir / "eval.json"
        self.eval = json.loads(ev_p.read_text()) if ev_p.exists() else None
        self.numeric = self.meta["numeric"]
        self.categorical = self.meta["categorical"]
        self.num_index = {f["name"]: i for i, f in enumerate(self.numeric)}
        self.cat_index = {c["name"]: i for i, c in enumerate(self.categorical)}
        self.n_samples = (self.cal or {}).get("n_samples", 8)
        self.loaded_at = time.time()

    # -- encoding -------------------------------------------------------------------

    def encode(self, features: dict, source: str | None):
        """raw dict -> (num, mask, cat, src) tensors with batch dim 1"""
        D, C = len(self.numeric), len(self.categorical)
        num = np.zeros((1, D), dtype=np.float32)
        mask = np.zeros((1, D), dtype=np.float32)
        cat = np.zeros((1, C), dtype=np.int64)
        for f in self.numeric:
            v = features.get(f["name"])
            if v is None:
                continue
            j = self.num_index[f["name"]]
            t = transform_numeric(np.array([float(v)]), f["transform"])[0]
            num[0, j] = (t - f["mean"]) / f["std"]
            mask[0, j] = 1.0
        for c in self.categorical:
            v = features.get(c["name"])
            j = self.cat_index[c["name"]]
            if v is None:
                continue
            if c["name"] in DERIVED_CATS:
                k = int(v)
                if 0 <= k < DERIVED_CATS[c["name"]]:
                    cat[0, j] = k + 1
            else:
                cat[0, j] = c["values"].index(str(v)) if str(v) in c["values"] else 0
        src_idx = schema.SOURCE_INDEX.get(source, 0) if source else 0
        dev = self.device
        return (torch.from_numpy(num).to(dev), torch.from_numpy(mask).to(dev),
                torch.from_numpy(cat).to(dev), torch.tensor([src_idx], dtype=torch.long, device=dev), src_idx)

    # -- scoring ------------------------------------------------------------------------

    def _calibration_for(self, source: str | None):
        """(block for this source or generic, kind)"""
        if not self.cal:
            return None, "none"
        if source and source in self.cal.get("sources", {}):
            return self.cal["sources"][source], source
        return self.cal.get("generic"), "generic"

    def score(self, features: dict, source: str | None = None, fpr_target: str = "fpr_0.01") -> dict:
        num, mask, cat, src, src_idx = self.encode(features, source)
        with _lock:
            r = score_batch(self.model, num, mask, cat, src, n_samples=self.n_samples)
        score = float(r["raw"][0])
        kl = float(r["kl"][0])
        out = {"score": score, "kl": kl, "source": source if src_idx else "generic", "n_samples": self.n_samples,
               "fpr_target": fpr_target}
        cal, kind = self._calibration_for(source if src_idx else None)
        out["calibration"] = kind
        if cal:
            thr = cal["thresholds"]
            chosen = thr.get(fpr_target) or thr.get("fpr_0.01")
            out.update({
                "z_score": (score - cal["legit_mean"]) / cal["legit_std"],
                "legit_percentile": _percentile(cal["legit_quantiles"], score),
                "threshold": chosen["threshold"] if chosen else None,
                "flag": bool(chosen and score > chosen["threshold"]),
                "thresholds": thr,
            })
        else:
            out.update({"z_score": None, "legit_percentile": None, "flag": None, "thresholds": None})

        nll_num, nll_cat = r["nll_num"][0], r["nll_cat"][0]
        contribs = []
        for j, f in enumerate(self.numeric):
            if mask[0, j].item() == 0:
                continue
            contribs.append({"feature": f["name"], "kind": "numeric", "nll": float(nll_num[j])})
        for j, c in enumerate(self.categorical):
            k = int(cat[0, j].item())
            if k == 0:
                continue
            contribs.append({"feature": c["name"], "kind": "categorical", "nll": float(nll_cat[j]), "observed": c["values"][k]})
        contribs.sort(key=lambda d: -d["nll"])
        out["contributions"] = contribs
        return out

    def score_frame(self, df, source_col="source", fpr_target="fpr_0.01", batch_size=4096):
        """vectorised scoring of a unified-schema DataFrame (raw units). returns a DataFrame with
        score, z and flag per row"""
        import pandas as pd

        n = len(df)
        D, C = len(self.numeric), len(self.categorical)
        num = np.zeros((n, D), dtype=np.float32)
        mask = np.zeros((n, D), dtype=np.float32)
        for f in self.numeric:
            v = pd.to_numeric(df[f["name"]], errors="coerce").to_numpy(dtype=np.float64)
            ok = np.isfinite(v)
            j = self.num_index[f["name"]]
            t = transform_numeric(np.nan_to_num(v), f["transform"])
            num[:, j] = np.where(ok, (t - f["mean"]) / f["std"], 0.0)
            mask[:, j] = ok
        cat = np.zeros((n, C), dtype=np.int64)
        for c in self.categorical:
            j = self.cat_index[c["name"]]
            col = df[c["name"]]
            if c["name"] in DERIVED_CATS:
                v = pd.to_numeric(col, errors="coerce").to_numpy()
                ok = np.isfinite(v) & (v >= 0) & (v < DERIVED_CATS[c["name"]])
                cat[:, j] = np.where(ok, np.nan_to_num(v).astype(int) + 1, 0)
            else:
                lut = {name: i for i, name in enumerate(c["values"])}
                cat[:, j] = col.astype(str).map(lut).fillna(0).astype(int).to_numpy()
        src = df[source_col].astype(str).map(lambda s: schema.SOURCE_INDEX.get(s, 0)).fillna(0).to_numpy().astype(np.int64)

        raw = np.zeros(n, dtype=np.float32)
        with _lock:
            for s in range(0, n, batch_size):
                sl = slice(s, min(s + batch_size, n))
                r = score_batch(self.model, torch.from_numpy(num[sl]).to(self.device), torch.from_numpy(mask[sl]).to(self.device),
                                torch.from_numpy(cat[sl]).to(self.device), torch.from_numpy(src[sl]).to(self.device),
                                n_samples=self.n_samples)
                raw[sl] = r["raw"]
        out = pd.DataFrame({"score": raw})
        inv = {v: k for k, v in schema.SOURCE_INDEX.items()}
        z = np.full(n, np.nan, dtype=np.float32)
        flag = np.zeros(n, dtype=bool)
        for code in np.unique(src):
            m = src == code
            cal, _ = self._calibration_for(inv.get(int(code)) if code else None)
            if not cal:
                continue
            z[m] = (raw[m] - cal["legit_mean"]) / cal["legit_std"]
            thr = cal["thresholds"].get(fpr_target) or cal["thresholds"].get("fpr_0.01")
            if thr:
                flag[m] = raw[m] > thr["threshold"]
        out["z"] = z
        out["flag"] = flag
        return out


def _percentile(quantiles: dict, value: float) -> float:
    """where does value sit among legit scores, interpolating the stored quantiles"""
    pts = sorted((float(k), v) for k, v in quantiles.items())
    if value <= pts[0][1]:
        return 0.0
    for (q0, v0), (q1, v1) in zip(pts[:-1], pts[1:]):
        if v0 <= value <= v1:
            frac = 0 if v1 == v0 else (value - v0) / (v1 - v0)
            return 100 * (q0 + frac * (q1 - q0))
    return 100.0


# ---- registry -------------------------------------------------------------------------------

_active: ActiveModel | None = None


def list_runs() -> list[dict]:
    out = []
    for d in sorted(RUNS_DIR.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        info = {"name": d.name, "has_model": (d / "model.pt").exists(), "has_calibration": (d / "calibration.json").exists(),
                "has_eval": (d / "eval.json").exists()}
        if (d / "run.json").exists():
            info["run"] = json.loads((d / "run.json").read_text())
        if (d / "eval.json").exists():
            ev = json.loads((d / "eval.json").read_text())
            info["mean_test_auroc"] = ev.get("mean_auroc")
            info["test_sources"] = {k: v["auroc"] for k, v in ev.get("sources", {}).items()}
        out.append(info)
    return out


def pick_default_run() -> str | None:
    runs = [r for r in list_runs() if r["has_model"]]
    if not runs:
        return None
    runs.sort(key=lambda r: (r.get("mean_test_auroc") or 0, r["name"]), reverse=True)
    return runs[0]["name"]


def load_run(name: str, device="cpu") -> ActiveModel:
    global _active
    _active = ActiveModel(RUNS_DIR / name, device=device)
    return _active


def active() -> ActiveModel | None:
    return _active
