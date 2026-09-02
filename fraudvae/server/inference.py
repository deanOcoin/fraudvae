"""Loaded-model state + turning a raw transaction dict into a scored result.

One run is "active" at a time: the VAE, the booster and the blender that
calibration fit over both. Everything the UI needs for a prediction
(standardisation stats, vocab, thresholds, PCA for the latent map) is read
from the run directory, so switching runs is just load_run(name).

A result has three layers: the blend's p(fraud) and verdict (what the app
flags on), the booster's probability with its top feature pushes, and the
VAE's novelty score with per-field surprises.
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
from ..model.booster import Booster, BoosterUnavailable, Specialists
from ..model.scoring import Scorer, score_batch
from ..graph.build import KINDS as ENTITY_KINDS
from ..graph.scorer import GraphScorer
from ..model.heads import load_heads
from ..model.vae import load_checkpoint
from ..paths import RUNS_DIR
from .payloads import merge_payloads

_lock = threading.Lock()


class IncompatibleRun(RuntimeError):
    pass


# schema v5 only added fields (entity graph block, device block) and a run encodes its inputs
# from its own checkpoint's feature list, so a v4 run still scores v5-era data, ignoring the
# new fields. v3 and earlier changed existing fields and are out.
MIN_COMPATIBLE_SCHEMA = 4


def compatible_schema(v) -> bool:
    return v is not None and MIN_COMPATIBLE_SCHEMA <= int(v) <= schema.SCHEMA_VERSION


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
        self.scorer = Scorer(self.cal["baselines"], n_feat, blend=self.cal.get("blend")) if self.cal and "baselines" in self.cal else None
        try:
            self.booster = Booster.load(self.run_dir)
            self.booster_error = None
        except BoosterUnavailable as e:
            # serve the novelty half rather than nothing; the ui says why the booster is missing
            self.booster, self.booster_error = None, str(e)
        # the blend needs the booster's logit, so without it the verdict falls back to novelty
        self.has_blend = bool(self.scorer and self.scorer.blend and self.booster is not None)
        # v5: the per-source specialists ("local" signal), None for a run without any
        try:
            self.specialists = Specialists.load(self.run_dir) if self.booster is not None else None
        except BoosterUnavailable:
            self.specialists = None
        # v5: the field heads (device ...), each a blend signal with a receipt block of its own
        self.heads = load_heads(self.run_dir, device=self.device)
        self.head_info = {}
        for name in self.heads:
            p = self.run_dir / f"head_{name}.json"
            self.head_info[name] = json.loads(p.read_text()) if p.exists() else {}
        # v5: the entity-graph models (one per source with entities); None for a run without any
        self.graph = GraphScorer.load(self.run_dir, device=self.device)
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

    def _calibration_for(self, source: str | None, block: dict | None = None, name: str | None = None):
        """(block for this source or generic, kind). block defaults to the VAE calibration;
        name="booster" / "seq" picks that signal's block and never falls back to the VAE's"""
        if name is not None:
            block = (self.cal or {}).get(name) or {}
        block = self.cal if block is None else block
        if not block:
            return None, "none"
        if source and source in block.get("sources", {}):
            return block["sources"][source], source
        if block.get("generic"):
            return block["generic"], "generic"
        return None, "none"

    @staticmethod
    def _operating(cal: dict, score: float, fpr_target: str) -> dict:
        thr = cal["thresholds"]
        op = fpr_target if fpr_target in thr else "fpr_0.01"
        chosen = thr.get(op)
        return {
            "z_score": (score - cal["legit_mean"]) / cal["legit_std"],
            "legit_percentile": _percentile(cal["legit_quantiles"], score),
            "operating_point": op, "threshold": chosen["threshold"] if chosen else None,
            "flag": bool(chosen and score > chosen["threshold"]),
            "expected_recall": chosen.get("recall") if chosen else None,
            "expected_precision": chosen.get("precision") if chosen else None,
            "context_auroc": cal.get("auroc"),
        }

    @staticmethod
    def _entity_row(entities: dict | None) -> np.ndarray:
        """{kind: code} -> [1, 6] int64 codes, -1 where not given / not an integer"""
        row = np.full((1, len(ENTITY_KINDS)), -1, dtype=np.int64)
        for k, kind in enumerate(ENTITY_KINDS):
            v = (entities or {}).get(kind)
            if v in (None, ""):
                continue
            try:
                row[0, k] = int(float(v))
            except (TypeError, ValueError):
                continue
        return row

    def score(self, features: dict, source: str | None = None, fpr_target: str = "fpr_0.01", wide: dict | None = None,
              device: dict | None = None, entities: dict | None = None) -> dict:
        """wide: the source's own raw columns keyed by raw name ({'V1': 3.0, 'DeviceInfo': 12}), the
        v5 wide block; only the booster reads it and only for a source that has one.
        device: the device telemetry payload (see payloads.py), mapped onto the device block.
        entities: {customer, counterparty, device, email, address, card: <entity code in the source>}
        for the graph module (codes as the source's build assigned them; unknown ones are simply
        entities the graph has not seen)."""
        features = merge_payloads(features, device=device)
        num, mask, cat, src, src_idx = self.encode(features, source)
        with _lock:
            r = score_batch(self.model, num, mask, cat, src, n_samples=self.n_samples)
            for name, head in self.heads.items():
                r[name] = head.logit(num, mask, cat, src)
        ent = self._entity_row(entities)
        if self.graph is not None and src_idx:
            r["graph"] = self.graph.logit_rows(np.array([src_idx]), ent, which="test")
        raw = float(r["raw"][0])
        kl = float(r["kl"][0])
        nll_num, nll_cat = r["nll_num"][0], r["nll_cat"][0]
        norm = float(self.scorer.norm(r, np.array([src_idx]))[0]) if self.scorer else None
        feat_z = self.scorer.feature_z(r, src_idx)[0] if self.scorer else np.zeros(len(nll_num) + len(nll_cat))
        vae_score = norm if (self.score_type == "norm" and norm is not None) else raw
        r["norm"] = np.array([norm if norm is not None else raw], dtype=np.float32)
        ctx = source if src_idx else None

        out = {"source": source if src_idx else "generic", "n_samples": self.n_samples, "fpr_target": fpr_target}

        # -- novelty: the VAE alone ------------------------------------------------------
        nov = {"score": vae_score, "score_type": self.score_type, "score_raw": raw, "score_norm": norm, "kl": kl}
        cal, kind = self._calibration_for(ctx)
        nov["calibration"] = kind
        if cal:
            nov.update(self._operating(cal, vae_score, fpr_target))
            nov["thresholds"] = cal["thresholds"]
        else:
            nov.update({"z_score": None, "legit_percentile": None, "flag": None, "thresholds": None})
        out["novelty"] = nov

        # -- supervised: the booster -----------------------------------------------------
        b_logit = None
        if self.booster is not None:
            num_np, mask_np, cat_np = num.cpu().numpy(), mask.cpu().numpy(), cat.cpu().numpy()
            # v5: the source's raw columns, if the caller has them ({'V1': 3.0} keyed by the raw name)
            wide_np = self.booster.wide_vector({f"{ctx}.{k}": v for k, v in (wide or {}).items()}) if self.booster.wide_names else None
            b_logit = self.booster.logit(num_np, mask_np, cat_np, np.array([src_idx]), threads=1, wide=wide_np)
            p_sup = float(1.0 / (1.0 + math.exp(-max(min(float(b_logit[0]), 30), -30))))
            sup = {"prob": p_sup, "logit": float(b_logit[0])}
            bcal, bkind = self._calibration_for(ctx, name="booster")
            sup["calibration"] = bkind
            if bcal:
                sup.update(self._operating(bcal, p_sup, fpr_target))
            contrib = self.booster.contributions(num_np, mask_np, cat_np, np.array([src_idx]), wide=wide_np)[0]
            sup["bias"] = float(contrib[-1])
            sup["pushes"] = self._booster_pushes(contrib[:-1], num, mask, cat, features, wide=wide_np)
            out["supervised"] = sup
            # v5: this bank's own specialist, when it has one
            if self.specialists is not None:
                r["local"] = self.specialists.logit(num_np, mask_np, cat_np, np.array([src_idx]), wide=wide_np, threads=1)
                lb = {"present": bool(np.isfinite(r["local"][0])), "label": "Bank specialist", "fields": []}
                if lb["present"]:
                    lg = float(r["local"][0])
                    lb.update({"logit": lg, "prob": float(1.0 / (1.0 + math.exp(-max(min(lg, 30), -30))))})
                    lcal, lkind = self._calibration_for(ctx, name="local")
                    lb["calibration"] = lkind
                    if lcal:
                        lb.update(self._operating(lcal, lb["prob"], fpr_target))
                out["local"] = lb

        # -- v5 field heads: the device block (and whatever lands next) through its own head -----
        for name, head in self.heads.items():
            fields = [self.numeric[j]["name"] for j in head.num_idx if mask[0, j].item() > 0] + \
                     [self.categorical[j]["name"] for j in head.cat_idx if cat[0, j].item() > 0]
            hb = {"present": bool(np.isfinite(r[name][0])), "fields": fields, "label": self.head_info[name].get("label", name)}
            if hb["present"]:
                lg = float(r[name][0])
                hb.update({"logit": lg, "prob": float(1.0 / (1.0 + math.exp(-max(min(lg, 30), -30))))})
                hcal, hkind = self._calibration_for(ctx, name=name)
                hb["calibration"] = hkind
                if hcal:
                    hb.update(self._operating(hcal, hb["prob"], fpr_target))
            out[name] = hb

        # -- v5 graph: the entities' embeddings through the transaction head --------------------
        if self.graph is not None:
            g = self.graph.sources.get(src_idx)
            known = g.known(ent, "test")[0] if g is not None else []
            gb = {"present": bool("graph" in r and np.isfinite(r["graph"][0])), "fields": known, "label": "Entity graph",
                  "sent": [k for k in ENTITY_KINDS if (entities or {}).get(k) not in (None, "")],
                  "available": g is not None}
            if gb["present"]:
                lg = float(r["graph"][0])
                gb.update({"logit": lg, "prob": float(1.0 / (1.0 + math.exp(-max(min(lg, 30), -30))))})
                gcal, gkind = self._calibration_for(ctx, name="graph")
                gb["calibration"] = gkind
                if gcal:
                    gb.update(self._operating(gcal, gb["prob"], fpr_target))
            out["graph"] = gb

        # -- the blend: what we flag on ---------------------------------------------------
        if self.has_blend:
            p = float(self.scorer.blend_prob(r, np.array([src_idx]), b_logit)[0])
            bl = {"prob": p}
            blcal, blkind = self._calibration_for(ctx, name="blend")
            bl["calibration"] = blkind
            if blcal:
                bl.update(self._operating(blcal, p, fpr_target))
                bl["thresholds"] = blcal["thresholds"]
            out["blend"] = bl
            # flat copies of the headline so simple clients / the batch csv don't need to dig
            out.update({"p_fraud": p, "flag": bl.get("flag"), "legit_percentile": bl.get("legit_percentile"),
                        "z_score": bl.get("z_score"), "calibration": blkind, "operating_point": bl.get("operating_point"),
                        "verdict_from": "blend"})
        else:
            out.update({"p_fraud": None, "flag": nov.get("flag"), "legit_percentile": nov.get("legit_percentile"),
                        "z_score": nov.get("z_score"), "calibration": kind, "operating_point": nov.get("operating_point"),
                        "verdict_from": "novelty"})
        # legacy flat fields the batch endpoint and tests still read
        out.update({"score": vae_score, "score_type": self.score_type, "score_raw": raw, "score_norm": norm, "kl": kl,
                    "thresholds": nov.get("thresholds")})

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

    def _booster_pushes(self, contrib: np.ndarray, num, mask, cat, features: dict, top=12, wide=None) -> list[dict]:
        """the booster's per-feature log-odds contributions (SHAP values), largest first, with the
        observed value in original units. unobserved numerics still get a contribution (missingness
        is information to a tree) and are labelled as such."""
        D = len(self.numeric)
        items = []
        for j, f in enumerate(self.numeric):
            c = float(contrib[j])
            observed = mask[0, j].item() > 0
            items.append({"feature": f["name"], "label": f.get("label") or f["name"], "kind": "numeric",
                          "contribution": c, "observed": self.decode_numeric(j, float(num[0, j].item())) if observed else None,
                          "unit": f.get("unit", ""), "is_observed": bool(observed)})
        for j, c_ in enumerate(self.categorical):
            k = int(cat[0, j].item())
            items.append({"feature": c_["name"], "label": c_.get("label") or c_["name"], "kind": "categorical",
                          "contribution": float(contrib[D + j]), "observed": c_["values"][k] if k else None,
                          "is_observed": bool(k)})
        C = len(self.categorical)
        items.append({"feature": "source", "label": "Bank context", "kind": "categorical",
                      "contribution": float(contrib[D + C]), "observed": None, "is_observed": True})
        # v5 wide block: the source's own raw columns, labelled by their raw name
        for j, name in enumerate(self.booster.wide_names if self.booster is not None else []):
            c = float(contrib[D + C + 1 + j])
            if c == 0.0:
                continue
            val = None if wide is None else float(wide[0, j])
            observed = val is not None and val == val
            items.append({"feature": name, "label": name.split(".", 1)[-1] + " (raw)", "kind": "wide",
                          "contribution": c, "observed": val if observed else None, "is_observed": bool(observed)})
        items.sort(key=lambda d: -abs(d["contribution"]))
        total = float(np.abs(contrib).sum()) or 1.0
        for it in items:
            it["share"] = abs(it["contribution"]) / total
        return items[:top]

    def score_frame(self, df, source_col="source", fpr_target="fpr_0.01", batch_size=4096):
        """vectorised scoring of a unified-schema DataFrame (raw units). returns a DataFrame with
        p_fraud + flag (the blend), booster_prob, novelty_score / novelty_flag (the VAE)"""
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
        # v5 wide block for a batch csv: either a 'wide' column of json dicts (the samples file) or
        # columns named like the booster's wide features ('ieee.V1'); anything else is missing
        wide = None
        if self.booster is not None and self.booster.wide_names:
            wide = np.full((n, len(self.booster.wide_names)), np.nan, dtype=np.float32)
            if "wide" in df:
                for i, cell in enumerate(df["wide"].tolist()):
                    if isinstance(cell, str) and cell.startswith("{"):
                        d = json.loads(cell)
                        key = str(df[source_col].iloc[i]) if source_col in df else ""
                        wide[i] = self.booster.wide_vector({f"{key}.{k}": v for k, v in d.items()})[0]
            for k in df.columns:
                if k in self.booster.wide_index:
                    wide[:, self.booster.wide_index[k]] = pd.to_numeric(df[k], errors="coerce").to_numpy(dtype=np.float32)

        raw = np.zeros(n, dtype=np.float32)
        norm = np.zeros(n, dtype=np.float32)
        boost = np.full(n, np.nan, dtype=np.float32)
        headp = {name: np.full(n, np.nan, dtype=np.float32) for name in self.heads}
        graphp = np.full(n, np.nan, dtype=np.float32)
        localp = np.full(n, np.nan, dtype=np.float32)
        ent = None
        if self.graph is not None and any(k in df for k in ENTITY_KINDS):
            ent = np.full((n, len(ENTITY_KINDS)), -1, dtype=np.int64)
            for k, kind in enumerate(ENTITY_KINDS):
                if kind in df:
                    v = pd.to_numeric(df[kind], errors="coerce").to_numpy()
                    ent[:, k] = np.where(np.isfinite(v), np.nan_to_num(v, nan=-1).astype(np.int64), -1)
        blend = np.full(n, np.nan, dtype=np.float32)
        with _lock:
            for s in range(0, n, batch_size):
                sl = slice(s, min(s + batch_size, n))
                num_t, mask_t = torch.from_numpy(num[sl]).to(self.device), torch.from_numpy(mask[sl]).to(self.device)
                cat_t, src_t = torch.from_numpy(cat[sl]).to(self.device), torch.from_numpy(src[sl]).to(self.device)
                r = score_batch(self.model, num_t, mask_t, cat_t, src_t, n_samples=self.n_samples)
                raw[sl] = r["raw"]
                b = None
                for name, head in self.heads.items():
                    r[name] = head.logit(num_t, mask_t, cat_t, src_t)
                    headp[name][sl] = 1.0 / (1.0 + np.exp(-np.clip(r[name], -30, 30)))
                if ent is not None:
                    r["graph"] = self.graph.logit_rows(src[sl], ent[sl], which="test")
                    graphp[sl] = 1.0 / (1.0 + np.exp(-np.clip(r["graph"], -30, 30)))
                if self.scorer:
                    norm[sl] = self.scorer.norm(r, src[sl])
                    r["norm"] = norm[sl]
                if self.booster is not None:
                    b = self.booster.logit(num[sl], mask[sl], cat[sl], src[sl], wide=(wide[sl] if wide is not None else None))
                    boost[sl] = 1.0 / (1.0 + np.exp(-np.clip(b, -30, 30)))
                    if self.specialists is not None:
                        r["local"] = self.specialists.logit(num[sl], mask[sl], cat[sl], src[sl], wide=(wide[sl] if wide is not None else None))
                        localp[sl] = 1.0 / (1.0 + np.exp(-np.clip(r["local"], -30, 30)))
                if self.has_blend:
                    blend[sl] = self.scorer.blend_prob(r, src[sl], b)
        nov = norm if (self.score_type == "norm" and self.scorer) else raw
        out = pd.DataFrame({"p_fraud": blend, "booster_prob": boost, "novelty_score": nov, "novelty_raw": raw})
        for name in self.heads:
            out[f"{name}_prob"] = headp[name]
        if ent is not None:
            out["graph_prob"] = graphp
        if self.specialists is not None:
            out["local_prob"] = localp
        if self.scorer:
            out["novelty_norm"] = norm
        inv = {v: k for k, v in schema.SOURCE_INDEX.items()}

        def _apply(block, scores, prefix):
            z = np.full(n, np.nan, dtype=np.float32)
            flag = np.zeros(n, dtype=bool)
            pct = np.full(n, np.nan, dtype=np.float32)
            if block:
                for code in np.unique(src):
                    m = src == code
                    cal, _ = self._calibration_for(inv.get(int(code)) if code else None, block)
                    if not cal:
                        continue
                    z[m] = (scores[m] - cal["legit_mean"]) / cal["legit_std"]
                    thr = cal["thresholds"].get(fpr_target) or cal["thresholds"].get("fpr_0.01")
                    if thr:
                        flag[m] = scores[m] > thr["threshold"]
                    pct[m] = [_percentile(cal["legit_quantiles"], float(v)) for v in scores[m]]
            out[f"{prefix}z"] = z
            out[f"{prefix}percentile"] = pct
            out[f"{prefix}flag"] = flag

        _apply(self.cal, nov, "novelty_")
        if self.has_blend:
            _apply(self.cal.get("blend"), blend, "")
            out["flag"] = out.pop("flag")   # keep the verdict as the last column
        else:
            out["flag"] = out["novelty_flag"]
            out["percentile"] = out["novelty_percentile"]
        # legacy names
        out["score"] = nov
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
                "has_eval": (d / "eval.json").exists(), "has_booster": (d / "booster.txt").exists(),
                "remote": (d / "remote.json").exists(), "schema_version": v,
                "compatible": compatible_schema(v)}
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
                info["mean"] = ev.get("mean") or {"vae": ev.get("mean_auroc"), "booster": None, "blend": None}
                info["generic_test_auroc"] = (ev.get("generic") or {}).get("auroc")
                info["test_sources"] = {k: v["auroc"] for k, v in ev.get("sources", {}).items()}
                info["score_type"] = ev.get("score_type")
            except json.JSONDecodeError:
                pass
        if (d / "calibration.json").exists():
            try:
                cal = json.loads((d / "calibration.json").read_text())
                gen = (cal.get("blend") or {}).get("generic") or cal.get("generic") or {}
                info["generic_val_auroc"] = gen.get("auroc")
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
