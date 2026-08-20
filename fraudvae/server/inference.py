"""Loaded-model state + turning a raw transaction dict into a scored result.

One run is "active" at a time. Everything the UI needs for a prediction
(standardisation stats, vocab, thresholds, PCA for the latent map) is read
from the run directory, so switching runs is just load_run(name).
"""
from __future__ import annotations

import json
import math
import threading
import time
from pathlib import Path

import numpy as np
import torch

from ..data import schema
from ..data.prepare import DERIVED_CATS, transform_numeric
from ..model.scoring import Scorer, score_batch
from ..model.vae import load_checkpoint
from ..paths import RUNS_DIR

_lock = threading.Lock()


class IncompatibleRun(RuntimeError):
    pass


def compatible_schema(v) -> bool:
    return v is not None and int(v) == schema.SCHEMA_VERSION


def run_schema_version(run_dir: Path) -> int | None:
    """runs record the schema they were trained on in run.json (v4+); older runs have nothing"""
    p = Path(run_dir) / "run.json"
    if not p.exists():
        return None
    try:
        r = json.loads(p.read_text())
    except json.JSONDecodeError:
        return None
    v = r.get("schema_version") or (r.get("prepared_meta") or {}).get("schema_version")
    return int(v) if v else 3


class ActiveModel:
    def __init__(self, run_dir: Path, device="cpu"):
        self.run_dir = Path(run_dir)
        self.name = self.run_dir.name
        self.device = torch.device(device)
        self.model, ck = load_checkpoint(self.run_dir / "model.pt", device=self.device)
        self.meta = ck["prepared_meta"]
        self.train_config = ck.get("train_config", {})
        self.extra = ck.get("extra", {})
        cal_p = self.run_dir / "calibration.json"
        self.cal = json.loads(cal_p.read_text()) if cal_p.exists() else None
        ev_p = self.run_dir / "eval.json"
        self.eval = json.loads(ev_p.read_text()) if ev_p.exists() else None
        run_p = self.run_dir / "run.json"
        self.run = json.loads(run_p.read_text()) if run_p.exists() else {}
        self.numeric = self.meta["numeric"]
        self.categorical = self.meta["categorical"]
        self.num_index = {f["name"]: i for i, f in enumerate(self.numeric)}
        self.cat_index = {c["name"]: i for i, c in enumerate(self.categorical)}
        self.n_samples = (self.cal or {}).get("n_samples", 8)
        self.score_type = (self.cal or {}).get("score_type", "raw")
        self.schema_version = run_schema_version(self.run_dir)
        if not compatible_schema(self.schema_version):
            raise IncompatibleRun(f"run {self.name} was trained on schema v{self.schema_version}, "
                                  f"this code is v{schema.SCHEMA_VERSION}; retrain (or check out the matching version)")
        n_feat = len(self.numeric) + len(self.categorical)
        self.scorer = Scorer(self.cal["baselines"], n_feat) if self.cal and "baselines" in self.cal else None
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
            if v is None or v == "":
                continue
            try:
                x = float(v)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(x):
                continue
            j = self.num_index[f["name"]]
            t = transform_numeric(np.array([x]), f["transform"])[0]
            num[0, j] = (t - f["mean"]) / f["std"]
            mask[0, j] = 1.0
        for c in self.categorical:
            v = features.get(c["name"])
            j = self.cat_index[c["name"]]
            if v is None or v == "":
                continue
            if c["name"] in DERIVED_CATS:
                try:
                    k = int(float(v))
                except (TypeError, ValueError):
                    continue
                if 0 <= k < DERIVED_CATS[c["name"]]:
                    cat[0, j] = k + 1
            else:
                if isinstance(v, (int, np.integer)) and not isinstance(v, bool):
                    cat[0, j] = int(v) if 0 <= int(v) < c["size"] else 0
                else:
                    cat[0, j] = c["values"].index(str(v)) if str(v) in c["values"] else 0
        src_idx = schema.SOURCE_INDEX.get(source, 0) if source else 0
        dev = self.device
        return (torch.from_numpy(num).to(dev), torch.from_numpy(mask).to(dev),
                torch.from_numpy(cat).to(dev), torch.tensor([src_idx], dtype=torch.long, device=dev), src_idx)

    def decode_numeric(self, j: int, standardized: float) -> float:
        f = self.numeric[j]
        t = standardized * f["std"] + f["mean"]
        if f["transform"] == "log1p":
            return float(math.expm1(max(min(t, 60), -60)))
        if f["transform"] == "signed_log1p":
            return float(math.copysign(math.expm1(min(abs(t), 60)), t))
        return float(t)

    # -- scoring ------------------------------------------------------------------------

    def _calibration_for(self, source: str | None):
        """(block for this source or generic, kind)"""
        if not self.cal:
            return None, "none"
        if source and source in self.cal.get("sources", {}):
            return self.cal["sources"][source], source
        if self.cal.get("generic"):
            return self.cal["generic"], "generic"
        return None, "none"

    def score(self, features: dict, source: str | None = None, fpr_target: str = "fpr_0.01") -> dict:
        num, mask, cat, src, src_idx = self.encode(features, source)
        with _lock:
            r = score_batch(self.model, num, mask, cat, src, n_samples=self.n_samples)
        raw = float(r["raw"][0])
        kl = float(r["kl"][0])
        nll_num, nll_cat = r["nll_num"][0], r["nll_cat"][0]
        norm = float(self.scorer.norm(r, np.array([src_idx]))[0]) if self.scorer else None
        feat_z = self.scorer.feature_z(r, src_idx)[0] if self.scorer else np.zeros(len(nll_num) + len(nll_cat))
        score = norm if (self.score_type == "norm" and norm is not None) else raw

        out = {"score": score, "score_type": self.score_type, "score_raw": raw, "score_norm": norm, "kl": kl,
               "source": source if src_idx else "generic", "n_samples": self.n_samples, "fpr_target": fpr_target}
        cal, kind = self._calibration_for(source if src_idx else None)
        out["calibration"] = kind
        if cal:
            thr = cal["thresholds"]
            op = fpr_target if fpr_target in thr else "fpr_0.01"
            chosen = thr.get(op)
            out.update({
                "z_score": (score - cal["legit_mean"]) / cal["legit_std"],
                "legit_percentile": _percentile(cal["legit_quantiles"], score),
                "operating_point": op, "threshold": chosen["threshold"] if chosen else None,
                "flag": bool(chosen and score > chosen["threshold"]),
                "expected_recall": chosen.get("recall") if chosen else None,
                "expected_precision": chosen.get("precision") if chosen else None,
                "context_auroc": cal.get("auroc"),
                "thresholds": thr,
            })
        else:
            out.update({"z_score": None, "legit_percentile": None, "flag": None, "thresholds": None})

        contribs = []
        for j, f in enumerate(self.numeric):
            if mask[0, j].item() == 0:
                continue
            x = float(num[0, j].item())
            exp_std = float(r["dec_mu"][0, j])
            contribs.append({
                "feature": f["name"], "label": f.get("label") or f["name"], "kind": "numeric", "group": f.get("group"),
                "nll": float(nll_num[j]), "z": float(feat_z[j]),
                "observed": self.decode_numeric(j, x), "expected": self.decode_numeric(j, exp_std),
                "unit": f.get("unit", ""),
            })
        D = len(self.numeric)
        for j, c in enumerate(self.categorical):
            k = int(cat[0, j].item())
            if k == 0:
                continue
            contribs.append({
                "feature": c["name"], "label": c.get("label") or c["name"], "kind": "categorical", "group": c.get("group"),
                "nll": float(nll_cat[j]), "z": float(feat_z[D + j]),
                "observed": c["values"][k], "prob_observed": float(r["cat_prob"][0, j]),
            })
        contribs.sort(key=lambda d: -d["z"])
        out["contributions"] = contribs
        out["observed_features"] = len(contribs)

        if self.cal and "latent_pca" in self.cal:
            pca = self.cal["latent_pca"]
            mu_vec = r["mu"][0] - np.array(pca["mean"])
            xy = np.array(pca["components"]) @ mu_vec
            out["latent_xy"] = [float(xy[0]), float(xy[1])]
        out["latent_mu"] = r["mu"][0].round(4).tolist()
        return out

    def score_frame(self, df, source_col="source", fpr_target="fpr_0.01", batch_size=4096):
        """vectorised scoring of a unified-schema DataFrame (raw units). returns a DataFrame with
        score, z, percentile and flag per row"""
        import pandas as pd

        n = len(df)
        D, C = len(self.numeric), len(self.categorical)
        num = np.zeros((n, D), dtype=np.float32)
        mask = np.zeros((n, D), dtype=np.float32)
        for f in self.numeric:
            if f["name"] not in df:
                continue
            v = pd.to_numeric(df[f["name"]], errors="coerce").to_numpy(dtype=np.float64)
            ok = np.isfinite(v)
            j = self.num_index[f["name"]]
            t = transform_numeric(np.nan_to_num(v), f["transform"])
            num[:, j] = np.where(ok, (t - f["mean"]) / f["std"], 0.0)
            mask[:, j] = ok
        cat = np.zeros((n, C), dtype=np.int64)
        for c in self.categorical:
            if c["name"] not in df:
                continue
            j = self.cat_index[c["name"]]
            col = df[c["name"]]
            if c["name"] in DERIVED_CATS:
                v = pd.to_numeric(col, errors="coerce").to_numpy()
                ok = np.isfinite(v) & (v >= 0) & (v < DERIVED_CATS[c["name"]])
                cat[:, j] = np.where(ok, np.nan_to_num(v).astype(int) + 1, 0)
            elif pd.api.types.is_numeric_dtype(col):
                v = col.fillna(0).to_numpy().astype(int)
                cat[:, j] = np.where((v >= 0) & (v < c["size"]), v, 0)
            else:
                lut = {name: i for i, name in enumerate(c["values"])}
                cat[:, j] = col.astype(str).map(lut).fillna(0).astype(int).to_numpy()
        if source_col in df:
            src = df[source_col].astype(str).map(lambda s: schema.SOURCE_INDEX.get(s, 0)).fillna(0).to_numpy().astype(np.int64)
        else:
            src = np.zeros(n, dtype=np.int64)

        raw = np.zeros(n, dtype=np.float32)
        norm = np.zeros(n, dtype=np.float32)
        with _lock:
            for s in range(0, n, batch_size):
                sl = slice(s, min(s + batch_size, n))
                num_t, mask_t = torch.from_numpy(num[sl]).to(self.device), torch.from_numpy(mask[sl]).to(self.device)
                cat_t, src_t = torch.from_numpy(cat[sl]).to(self.device), torch.from_numpy(src[sl]).to(self.device)
                r = score_batch(self.model, num_t, mask_t, cat_t, src_t, n_samples=self.n_samples)
                raw[sl] = r["raw"]
                if self.scorer:
                    norm[sl] = self.scorer.norm(r, src[sl])
        score = norm if (self.score_type == "norm" and self.scorer) else raw
        out = pd.DataFrame({"score": score, "score_raw": raw})
        if self.scorer:
            out["score_norm"] = norm
        inv = {v: k for k, v in schema.SOURCE_INDEX.items()}
        z = np.full(n, np.nan, dtype=np.float32)
        flag = np.zeros(n, dtype=bool)
        pct = np.full(n, np.nan, dtype=np.float32)
        for code in np.unique(src):
            m = src == code
            cal, _ = self._calibration_for(inv.get(int(code)) if code else None)
            if not cal:
                continue
            z[m] = (score[m] - cal["legit_mean"]) / cal["legit_std"]
            thr = cal["thresholds"].get(fpr_target) or cal["thresholds"].get("fpr_0.01")
            if thr:
                flag[m] = score[m] > thr["threshold"]
            pct[m] = [_percentile(cal["legit_quantiles"], float(v)) for v in score[m]]
        out["z"] = z
        out["percentile"] = pct
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
    if not RUNS_DIR.exists():
        return out
    for d in sorted(RUNS_DIR.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if not d.is_dir():
            continue
        v = run_schema_version(d)
        info = {"name": d.name, "has_model": (d / "model.pt").exists(), "has_calibration": (d / "calibration.json").exists(),
                "has_eval": (d / "eval.json").exists(), "schema_version": v, "compatible": compatible_schema(v)}
        for f in ("run.json", "status.json"):
            p = d / f
            if p.exists():
                try:
                    info[f.split(".")[0]] = json.loads(p.read_text())
                except json.JSONDecodeError:
                    pass
        if (d / "eval.json").exists():
            try:
                ev = json.loads((d / "eval.json").read_text())
                info["mean_test_auroc"] = ev.get("mean_auroc")
                info["generic_test_auroc"] = (ev.get("generic") or {}).get("auroc")
                info["test_sources"] = {k: v["auroc"] for k, v in ev.get("sources", {}).items()}
                info["score_type"] = ev.get("score_type")
            except json.JSONDecodeError:
                pass
        if "run" in info:
            full = info["run"]
            cfg = full.get("config", {})
            info["run"] = {k: full.get(k) for k in ("name", "started", "device", "device_name", "host", "params", "prepared", "prepared_meta")}
            info["config_name"] = cfg.get("name")
            info["epochs"] = cfg.get("train", {}).get("epochs")
        out.append(info)
    return out


def pick_default_run() -> str | None:
    runs = [r for r in list_runs() if r["has_model"] and r["has_calibration"] and r["compatible"]]
    if not runs:
        return None
    runs.sort(key=lambda r: (r.get("mean_test_auroc") or 0, r["name"]), reverse=True)
    return runs[0]["name"]


def load_run(name: str, device="cpu") -> ActiveModel:
    global _active
    d = RUNS_DIR / name
    if not (d / "model.pt").exists():
        raise FileNotFoundError(f"run {name} has no model.pt")
    _active = ActiveModel(d, device=device)
    return _active


def active() -> ActiveModel | None:
    return _active
