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
import os
import threading
import time
from pathlib import Path

import numpy as np
import torch

from ..data import schema
from ..data.prepare import DERIVED_CATS, transform_numeric
from ..data.schema import ENTITY_LABEL_DELAY_DAYS as SLOW_DELAY_DAYS, ENTITY_LABEL_DELAY_FAST_DAYS as FAST_DELAY_DAYS
from ..model.booster import Booster, BoosterUnavailable, Specialists, fast_columns
from ..model.scoring import Scorer, history_window, score_batch, label_delay_wide
from ..graph.build import KINDS as ENTITY_KINDS
from ..graph.scorer import GraphScorer
from ..model.heads import load_heads
from ..behaviour.scorer import BehaviourScorer, merge_enrolment
from ..text.scorer import TextScorer
from ..model.seqhead import load_seqhead
from ..model.vae import load_checkpoint
from ..paths import APP_HOME, RUNS_DIR
from .entity_store import EntityStore
from .payloads import merge_payloads

_lock = threading.Lock()


class IncompatibleRun(RuntimeError):
    pass


# a run encodes its inputs from its own checkpoint's feature list (names, transforms,
# category values), and looks its bank up in its own source list, so any earlier run scores
# today's transactions with the fields it knows and nothing else: a v4 run ignores the
# entity-graph and device blocks, a v3 run (no schema version recorded) also lacks the
# customer-context block. They load as "legacy" runs - the receipt says what they could not
# see - which is what the model timeline needs: the same transaction on every generation.
MIN_COMPATIBLE_SCHEMA = 3
LEGACY_SCHEMA = 4   # below this a run is served with a "legacy" note


def compatible_schema(v) -> bool:
    return (3 if v is None else int(v)) >= MIN_COMPATIBLE_SCHEMA and (3 if v is None else int(v)) <= schema.SCHEMA_VERSION


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


_ENTITY_WORDS = {"cpty": "counterparty", "cust": "customer", "addr": "address", "device": "device", "email": "email", "card": "card"}


def _wide_label(col: str) -> str:
    """a wide column's receipt label: the raw name for a source's own columns, words for the
    one-week fraud-history block (cust_fraud_share_fast_w30 -> customer fraud share, week-old
    labels, last 30 days)"""
    parts = col.split("_")
    if len(parts) >= 4 and parts[1] == "fraud" and parts[2] in ("count", "share") and parts[3] == "fast":
        who = _ENTITY_WORDS.get(parts[0], parts[0])
        window = f", last {parts[4][1:]} days" if len(parts) > 4 and parts[4].startswith("w") else ""
        return f"{who} fraud {parts[2]}, week-old labels{window}"
    return col + " (raw)"


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
        # the banks this run was trained with, in the order its source embedding knows them
        # (index 0 is the generic context); a bank it never saw is scored as generic
        self.sources = list(self.meta.get("sources") or schema.SOURCES)
        self.source_index = {name: i + 1 for i, name in enumerate(self.sources)}
        self.n_samples = (self.cal or {}).get("n_samples", 8)
        self.score_type = (self.cal or {}).get("score_type", "raw")
        self.schema_version = run_schema_version(self.run_dir)
        if not compatible_schema(self.schema_version):
            raise IncompatibleRun(f"run {self.name} was trained on schema v{self.schema_version}, "
                                  f"this code is v{schema.SCHEMA_VERSION}; retrain (or check out the matching version)")
        self.legacy = (self.schema_version or 3) < LEGACY_SCHEMA
        self.schema_note = (f"trained on schema v{self.schema_version or 3}: scores with the {len(self.numeric)} numeric and "
                            f"{len(self.categorical)} categorical fields it knows, and sees {len(self.sources)} banks") \
            if (self.schema_version or 3) < schema.SCHEMA_VERSION else None
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
        # v5: the customer-history side. window 0 = a model without a history encoder
        self.window = history_window(self.model)
        self.seqhead = load_seqhead(self.run_dir, device=self.device) if self.window else None
        self.store = EntityStore(APP_HOME / "entity_store.sqlite", self.name) if self.window else None
        # v5: the field heads (device ...), each a blend signal with a receipt block of its own
        self.heads = load_heads(self.run_dir, device=self.device)
        self.head_info = {}
        for name in self.heads:
            p = self.run_dir / f"head_{name}.json"
            self.head_info[name] = json.loads(p.read_text()) if p.exists() else {}
        # v5: the entity-graph models (one per source with entities); None for a run without any
        self.graph = GraphScorer.load(self.run_dir, device=self.device)
        # v5: the text heads; their likelihood ratios are fused into the final probability
        self.text = TextScorer.load(self.run_dir, device=self.device)
        # v5: the behaviour verifiers; enrolments live in the entity store per customer
        self.behaviour = BehaviourScorer.load(self.run_dir, device=self.device)
        if self.store is None and self.behaviour is not None:
            self.store = EntityStore(APP_HOME / "entity_store.sqlite", self.name)
        self.fusion_weights = {"text": float((self.train_config.get("text") or {}).get("fusion_weight", 0.5)),
                               "behaviour": float((self.train_config.get("behaviour") or {}).get("fusion_weight", 1.0))}
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
        src_idx = self.source_index.get(source, 0) if source else 0
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

    def _fuse(self, blend_prob: float, llrs: dict[str, list[float]]) -> dict | None:
        """blend probability + external likelihood ratios -> final probability. each ratio is
        clipped to +-6 (one text field cannot say more than e^6), summed per module, weighted."""
        terms = {}
        for module, vals in llrs.items():
            vals = [float(np.clip(v, -6.0, 6.0)) for v in vals if v == v]
            if vals:
                terms[module] = {"llr": float(sum(vals)), "weight": self.fusion_weights.get(module, 1.0), "n": len(vals)}
        if not terms:
            return None
        p = min(max(blend_prob, 1e-6), 1 - 1e-6)
        base = math.log(p / (1 - p))
        final = base + sum(t["llr"] * t["weight"] for t in terms.values())
        return {"blend_logit": base, "terms": terms, "logit": final, "prob": float(1.0 / (1.0 + math.exp(-max(min(final, 30), -30))))}

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

    def history_tensors(self, rows: list, source: str | None, cur_ts: float):
        """v5: past transactions -> the window tensors the history encoder wants (oldest first,
        padded at the front). rows: [(ts, num, mask, cat)] in model space, as the entity store
        hands them out, or dicts {"features": {...}, "ts": epoch_seconds | "secs_before": s}
        in raw units, as the API accepts them."""
        L = self.window
        D, C = len(self.numeric), len(self.categorical)
        h_num = np.zeros((1, L, D), dtype=np.float32)
        h_mask = np.zeros((1, L, D), dtype=np.float32)
        h_cat = np.zeros((1, L, C), dtype=np.int64)
        h_dt = np.zeros((1, L), dtype=np.float32)
        h_valid = np.zeros((1, L), dtype=bool)
        enc = []
        for row in rows[-L:]:
            if isinstance(row, dict):
                feats = row.get("features", row)
                n_, m_, c_, _, _ = self.encode(feats, source)
                ts = float(row["ts"]) if row.get("ts") is not None else cur_ts - float(row.get("secs_before", 0.0))
                enc.append((ts, n_.cpu().numpy()[0], m_.cpu().numpy()[0], c_.cpu().numpy()[0]))
            else:
                enc.append(row)
        for k, (ts, n_, m_, c_) in enumerate(enc):
            pos = L - len(enc) + k
            h_num[0, pos], h_mask[0, pos], h_cat[0, pos] = n_, m_, c_
            h_dt[0, pos] = math.log1p(max(cur_ts - ts, 0.0)) / 10.0
            h_valid[0, pos] = True
        dev = self.device
        return (torch.from_numpy(h_num).to(dev), torch.from_numpy(h_mask).to(dev), torch.from_numpy(h_cat).to(dev),
                torch.from_numpy(h_dt).to(dev), torch.from_numpy(h_valid).to(dev)), len(enc)

    def score(self, features: dict, source: str | None = None, fpr_target: str = "fpr_0.01", wide: dict | None = None,
              history: list | None = None, customer_id: str | None = None, ts: float | None = None,
              remember: bool = True, device: dict | None = None, entities: dict | None = None,
              text: dict | None = None, session: dict | None = None, label_delay_days: float | None = None) -> dict:
        """wide: the source's own raw columns keyed by raw name ({'V1': 3.0, 'DeviceInfo': 12}), the
        v5 wide block; only the booster reads it and only for a source that has one.
        history: the customer's previous transactions, oldest first (dicts with 'features' and
        'ts' or 'secs_before'); customer_id: lets the server's entity store supply the history
        instead, and remember this transaction for next time (remember=False to only read).
        ts: this transaction's epoch seconds (default: now), for the elapsed-time features.
        device: the device telemetry payload (see payloads.py), mapped onto the device block.
        entities: {customer, counterparty, device, email, address, card: <entity code in the source>}
        for the graph module (codes as the source's build assigned them; unknown ones are simply
        entities the graph has not seen).
        text: {url, description, memo, profile ...}: free text for the text heads; their likelihood
        ratios are added to the blend's log-odds (the fusion step) rather than blended, because no
        training row ever carried text.
        session: raw interaction events per modality (see behaviour/scorer.py); compared with the
        customer's enrolled behaviour profile (customer_id) and enrolled for next time."""
        features = merge_payloads(features, device=device)
        num, mask, cat, src, src_idx = self.encode(features, source)
        cur_ts = float(ts) if ts is not None else time.time()
        hist, n_hist, hist_from = None, 0, None
        if self.window:
            if history:
                hist, n_hist = self.history_tensors(list(history), source, cur_ts)
                hist_from = "request"
            elif customer_id is not None and self.store is not None:
                past = self.store.last(source or "", customer_id, cur_ts, self.window)
                if past:
                    hist, n_hist = self.history_tensors(past, source, cur_ts)
                hist_from = "entity store"
        with _lock:
            r = score_batch(self.model, num, mask, cat, src, n_samples=self.n_samples, hist=hist)
            if self.seqhead is not None:
                r["seq"] = self.seqhead.logit(self.model, num, mask, cat, src, torch.from_numpy(r["h"]).to(self.device))
            for name, head in self.heads.items():
                r[name] = head.logit(num, mask, cat, src)
        ent = self._entity_row(entities)
        if self.graph is not None and src_idx:
            r["graph"] = self.graph.logit_rows(np.array([src_idx]), ent, which="test")
        if customer_id is not None and self.store is not None and remember:
            self.store.append(source or "", customer_id, cur_ts, num.cpu().numpy()[0], mask.cpu().numpy()[0], cat.cpu().numpy()[0])
        raw = float(r["raw"][0])
        kl = float(r["kl"][0])
        nll_num, nll_cat = r["nll_num"][0], r["nll_cat"][0]
        norm = float(self.scorer.norm(r, np.array([src_idx]))[0]) if self.scorer else None
        feat_z = self.scorer.feature_z(r, src_idx)[0] if self.scorer else np.zeros(len(nll_num) + len(nll_cat))
        vae_score = norm if (self.score_type == "norm" and norm is not None) else raw
        r["norm"] = np.array([norm if norm is not None else raw], dtype=np.float32)
        ctx = source if src_idx else None

        out = {"source": source if src_idx else "generic", "n_samples": self.n_samples, "fpr_target": fpr_target,
               # which generation scored this, and what an older one could not see
               "model": {"run": self.name, "schema_version": self.schema_version or 3, "legacy": self.legacy,
                         "note": self.schema_note,
                         "bank_known": bool(src_idx) or not source,
                         "fields_used": int(mask.sum().item()) + int((cat > 0).sum().item())}}

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
            # v5: a deployment whose fraud labels take longer than a week to arrive must not be
            # scored with the one-week label block; blank it (the booster trained on that regime too)
            fast_cols = fast_columns(self.booster.wide_names) if self.booster.wide_names else []
            fast_given = bool(wide_np is not None and fast_cols and np.isfinite(wide_np[0, fast_cols]).any())
            if label_delay_days and fast_cols:
                wide_np = label_delay_wide(wide_np, self.booster.wide_names, label_delay_days)
            out["labels"] = {"latency_days": float(label_delay_days) if label_delay_days else None,
                             "fast_block": "blanked" if (label_delay_days and label_delay_days > FAST_DELAY_DAYS and fast_given)
                             else ("used" if fast_given else "not sent"),
                             "note": f"the one-week-delay fraud history (wide *_fast columns) counts only for labels known within "
                                     f"{FAST_DELAY_DAYS:g} days; the schema's delayed block assumes {SLOW_DELAY_DAYS:g}"}
            hist_np = r["h"] if (self.booster.hist_dim and r.get("h") is not None) else None
            b_logit = self.booster.logit(num_np, mask_np, cat_np, np.array([src_idx]), threads=1, wide=wide_np, hist=hist_np)
            p_sup = float(1.0 / (1.0 + math.exp(-max(min(float(b_logit[0]), 30), -30))))
            sup = {"prob": p_sup, "logit": float(b_logit[0])}
            bcal, bkind = self._calibration_for(ctx, name="booster")
            sup["calibration"] = bkind
            if bcal:
                sup.update(self._operating(bcal, p_sup, fpr_target))
            contrib = self.booster.contributions(num_np, mask_np, cat_np, np.array([src_idx]), wide=wide_np, hist=hist_np)[0]
            sup["bias"] = float(contrib[-1])
            sup["pushes"] = self._booster_pushes(contrib[:-1], num, mask, cat, features, wide=wide_np)
            out["supervised"] = sup
            # v5: this bank's own specialist, when it has one
            if self.specialists is not None:
                r["local"] = self.specialists.logit(num_np, mask_np, cat_np, np.array([src_idx]), wide=wide_np, threads=1,
                                                    hist=r.get("h"))
                lb = {"present": bool(np.isfinite(r["local"][0])), "label": "Bank specialist", "fields": []}
                if lb["present"]:
                    lg = float(r["local"][0])
                    lb.update({"logit": lg, "prob": float(1.0 / (1.0 + math.exp(-max(min(lg, 30), -30))))})
                    lcal, lkind = self._calibration_for(ctx, name="local")
                    lb["calibration"] = lkind
                    if lcal:
                        lb.update(self._operating(lcal, lb["prob"], fpr_target))
                out["local"] = lb

        # -- v5 sequence: the history read-out through the sequence head -------------------
        if self.window:
            sq = {"n_history": int(n_hist), "window": int(self.window), "history_from": hist_from}
            if "seq" in r:
                lg = float(r["seq"][0])
                sq.update({"logit": lg, "prob": float(1.0 / (1.0 + math.exp(-max(min(lg, 30), -30))))})
                scal, skind = self._calibration_for(ctx, name="seq")
                sq["calibration"] = skind
                if scal:
                    sq.update(self._operating(scal, sq["prob"], fpr_target))
            out["sequence"] = sq

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

        # -- v5 text: free text through the text heads, as likelihood ratios ----------------------
        text_terms = self.text.score(text) if (self.text is not None and text) else {}
        if self.text is not None:
            out["text"] = {"present": bool(text_terms), "fields": text_terms, "label": "Text",
                           "sent": [k for k, v in (text or {}).items() if v not in (None, "")] if isinstance(text, dict) else []}

        # -- v5 behaviour: the session against the customer's enrolled profile --------------------
        beh_terms = {}
        if self.behaviour is not None:
            embs = self.behaviour.embeddings(session) if session else {}
            profile, n_sessions = (self.store.profile(source or "", customer_id) if (customer_id is not None and self.store is not None)
                                   else (None, 0))
            beh_terms = self.behaviour.compare(embs, profile) if embs else {}
            out["behaviour"] = {"present": any("llr" in v for v in beh_terms.values()), "fields": beh_terms, "label": "Behaviour",
                               "sent": [k for k, v in (session or {}).items() if v] if isinstance(session, dict) else [],
                               "enrolled_sessions": n_sessions, "enrolled_modalities": sorted(profile) if profile else []}
            if embs and customer_id is not None and self.store is not None and remember:
                self.store.set_profile(source or "", customer_id, merge_enrolment(profile, embs), n_sessions + 1)
                out["behaviour"]["enrolled_now"] = sorted(embs)

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
            # v5 fusion: external modules (text, behaviour) were never present on a training row, so
            # their evidence is added to the blend's log-odds with a fixed weight instead of being
            # blended; the operating point stays the blend's (the probability means the same thing)
            fused = self._fuse(p, {"text": [t["llr"] for t in text_terms.values()],
                                   "behaviour": [v["llr"] for v in beh_terms.values() if "llr" in v]})
            if fused is not None:
                out["fusion"] = fused
                p_final = fused["prob"]
                flag = bool(blcal and p_final > (bl.get("threshold") or 1.0)) if blcal else None
                verdict = "blend+" + "+".join(fused["terms"])
            else:
                p_final, flag, verdict = p, bl.get("flag"), "blend"
            # flat copies of the headline so simple clients / the batch csv don't need to dig
            out.update({"p_fraud": p_final, "flag": flag, "legit_percentile": bl.get("legit_percentile"),
                        "z_score": bl.get("z_score"), "calibration": blkind, "operating_point": bl.get("operating_point"),
                        "verdict_from": verdict})
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
            items.append({"feature": name, "label": _wide_label(name.split(".", 1)[-1]), "kind": "wide",
                          "contribution": c, "observed": val if observed else None, "is_observed": bool(observed)})
        # v5: the history encoder's read-out, when the booster reads it: one line for the block
        hd = self.booster.hist_dim if self.booster is not None else 0
        if hd:
            start = D + C + 1 + len(self.booster.wide_names)
            block = contrib[start : start + hd]
            items.append({"feature": "history_readout", "label": f"customer history summary ({hd} dims, {int((block != 0).sum())} used)",
                          "kind": "history", "contribution": float(block.sum()), "observed": None, "is_observed": True})
        items.sort(key=lambda d: -abs(d["contribution"]))
        total = float(np.abs(contrib).sum()) or 1.0
        for it in items:
            it["share"] = abs(it["contribution"]) / total
        return items[:top]

    def score_frame(self, df, source_col="source", fpr_target="fpr_0.01", batch_size=4096, label_delay_days=None):
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
            src = df[source_col].astype(str).map(lambda s: self.source_index.get(s, 0)).fillna(0).to_numpy().astype(np.int64)
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
            # v5: a deployment whose labels take longer than a week never had the one-week block
            wide = label_delay_wide(wide, self.booster.wide_names, label_delay_days)

        raw = np.zeros(n, dtype=np.float32)
        norm = np.zeros(n, dtype=np.float32)
        boost = np.full(n, np.nan, dtype=np.float32)
        seqp = np.full(n, np.nan, dtype=np.float32)
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
                # a csv carries no history: every row gets the "no history" context (a trained state)
                r = score_batch(self.model, num_t, mask_t, cat_t, src_t, n_samples=self.n_samples)
                raw[sl] = r["raw"]
                b = None
                if self.seqhead is not None:
                    r["seq"] = self.seqhead.logit(self.model, num_t, mask_t, cat_t, src_t, torch.from_numpy(r["h"]).to(self.device))
                    seqp[sl] = 1.0 / (1.0 + np.exp(-np.clip(r["seq"], -30, 30)))
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
                    b = self.booster.logit(num[sl], mask[sl], cat[sl], src[sl], wide=(wide[sl] if wide is not None else None),
                                           hist=(r.get("h") if self.booster.hist_dim else None))
                    boost[sl] = 1.0 / (1.0 + np.exp(-np.clip(b, -30, 30)))
                    if self.specialists is not None:
                        r["local"] = self.specialists.logit(num[sl], mask[sl], cat[sl], src[sl], wide=(wide[sl] if wide is not None else None),
                                                            hist=r.get("h"))
                        localp[sl] = 1.0 / (1.0 + np.exp(-np.clip(r["local"], -30, 30)))
                if self.has_blend:
                    blend[sl] = self.scorer.blend_prob(r, src[sl], b)
        nov = norm if (self.score_type == "norm" and self.scorer) else raw
        out = pd.DataFrame({"p_fraud": blend, "booster_prob": boost, "novelty_score": nov, "novelty_raw": raw})
        if self.seqhead is not None:
            out["sequence_prob"] = seqp
        for name in self.heads:
            out[f"{name}_prob"] = headp[name]
        if ent is not None:
            out["graph_prob"] = graphp
        if self.specialists is not None:
            out["local_prob"] = localp
        if self.scorer:
            out["novelty_norm"] = norm
        inv = {v: k for k, v in self.source_index.items()}

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
# v5: more than one generation loaded at once, so a transaction can be scored on each. the
# default (active) model never gets evicted; the others go least-recently-used first
_loaded: dict[str, ActiveModel] = {}
_MAX_LOADED = int(os.environ.get("FRAUDVAE_MAX_LOADED", "8"))
_lock = threading.Lock()


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
    """make a run the default model (the one /api/score uses when no run is named)"""
    global _active
    _active = get_model(name, device=device)
    return _active


def get_model(name: str, device="cpu") -> ActiveModel:
    """a loaded run by name, loading it on first use and evicting the least recently used
    extra model past the cap; the default model is never evicted"""
    with _lock:
        m = _loaded.get(name)
        if m is not None:
            m.last_used = time.time()
            return m
    d = RUNS_DIR / name
    if not (d / "model.pt").exists():
        raise FileNotFoundError(f"run {name} has no model.pt")
    m = ActiveModel(d, device=device)
    m.last_used = time.time()
    with _lock:
        _loaded[name] = m
        extras = [n for n in _loaded if _active is None or n != _active.name]
        while len(_loaded) > _MAX_LOADED and extras:
            victim = min(extras, key=lambda n: getattr(_loaded[n], "last_used", 0.0))
            if _active is not None and victim == _active.name:
                break
            _loaded.pop(victim, None)
            extras.remove(victim)
    return m


def loaded_models() -> list[str]:
    return list(_loaded)


def active() -> ActiveModel | None:
    return _active
