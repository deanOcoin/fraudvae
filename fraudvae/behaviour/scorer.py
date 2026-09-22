"""Serve the verifiers. A `session` payload carries raw events per modality:

  {"mouse":     [[t, x, y, kind], ...]                      kind: move / down / up / drag / scroll
   "keystroke": [[t, key, direction], ...]                  direction: down / up (or 0 / 1)
   "touch":     [[t, x, y, pressure, size, action], ...]   action: 0 down / 2 move / 1 up
   "motion":    [[t, x, y, z], ...]}                        accelerometer or gyroscope samples

Each modality becomes a session embedding. Against an enrolment (the account holder's stored
embeddings, kept per customer by the entity store and updated as a running mean of the
sessions the server has seen for that customer) the cosine similarity is turned into a
likelihood ratio; without an enrolment there is nothing to compare with, the session only
enrols. The ratios are what the fusion step adds to the verdict."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .features import keystroke_features, motion_features, mouse_features, touch_features
from .model import Verifier

MIN_WINDOWS = 1


def _col(rows, k, default=None):
    out = []
    for r in rows:
        if isinstance(r, dict):
            out.append(r.get(k, default))
        else:
            out.append(r[k] if k < len(r) else default)
    return out


def session_windows(payload: dict) -> dict[str, np.ndarray]:
    """payload -> {modality: windows}; a modality with junk or too little data is left out"""
    out = {}
    if not isinstance(payload, dict):
        return out
    try:
        if payload.get("mouse"):
            rows = payload["mouse"]
            keys = ("t", "x", "y", "kind") if isinstance(rows[0], dict) else (0, 1, 2, 3)
            w, _ = mouse_features(_col(rows, keys[0]), _col(rows, keys[1]), _col(rows, keys[2]), _col(rows, keys[3], "move"))
            if len(w) >= MIN_WINDOWS:
                out["mouse"] = w
        if payload.get("keystroke"):
            rows = payload["keystroke"]
            keys = ("t", "key", "direction") if isinstance(rows[0], dict) else (0, 1, 2)
            w, _ = keystroke_features(_col(rows, keys[0]), _col(rows, keys[1]), _col(rows, keys[2], 0))
            if len(w) >= MIN_WINDOWS:
                out["keystroke"] = w
        if payload.get("touch"):
            rows = payload["touch"]
            keys = ("t", "x", "y", "pressure", "size", "action") if isinstance(rows[0], dict) else (0, 1, 2, 3, 4, 5)
            pressure = _col(rows, keys[3], 1.0)
            size = _col(rows, keys[4], 0.0)
            action = _col(rows, keys[5], None)
            w, _ = touch_features(_col(rows, keys[0]), _col(rows, keys[1]), _col(rows, keys[2]),
                                  [1.0 if v is None else v for v in pressure], [0.0 if v is None else v for v in size],
                                  None if all(a is None for a in action) else [2 if a is None else a for a in action])
            if len(w) >= MIN_WINDOWS:
                out["touch"] = w
        if payload.get("motion"):
            rows = payload["motion"]
            keys = ("t", "x", "y", "z") if isinstance(rows[0], dict) else (0, 1, 2, 3)
            w, _ = motion_features(_col(rows, keys[0]), _col(rows, keys[1]), _col(rows, keys[2]), _col(rows, keys[3]))
            if len(w) >= MIN_WINDOWS:
                out["motion"] = w
    except (TypeError, ValueError, IndexError, KeyError):
        return out
    return out


class BehaviourScorer:
    def __init__(self, run_dir: Path, device="cpu"):
        self.verifiers: dict[str, Verifier] = {}
        self.info: dict[str, dict] = {}
        for p in sorted((Path(run_dir) / "behaviour").glob("*.pt")):
            ck = torch.load(p, map_location=device, weights_only=False)
            cfg = ck["config"]
            v = Verifier(cfg["modality"], cfg["d_in"], cfg["emb"], cfg["hidden"], cfg["n_users"], scale=cfg.get("scale", 16.0))
            v.load_state_dict(ck["state_dict"])
            v.to(device).eval()
            self.verifiers[cfg["modality"]] = v
            self.info[cfg["modality"]] = ck.get("info", {})

    def __bool__(self):
        return bool(self.verifiers)

    @staticmethod
    def load(run_dir: Path, device="cpu") -> "BehaviourScorer | None":
        if not (Path(run_dir) / "behaviour").exists():
            return None
        b = BehaviourScorer(run_dir, device)
        return b if b else None

    def embeddings(self, payload: dict) -> dict[str, dict]:
        """{modality: {embedding: [..], windows: n}} for the modalities the payload carries and a verifier knows"""
        out = {}
        for modality, w in session_windows(payload).items():
            v = self.verifiers.get(modality)
            if v is None:
                continue
            e = v.session_embedding(w)
            if e is not None:
                out[modality] = {"embedding": e.astype(np.float32), "windows": int(len(w))}
        return out

    def compare(self, embs: dict[str, dict], enrolment: dict[str, list] | None) -> dict[str, dict]:
        """{modality: {cosine, llr, windows, enrolled}}; without an enrolment for a modality there is no ratio"""
        out = {}
        for modality, e in embs.items():
            v = self.verifiers[modality]
            ref = (enrolment or {}).get(modality)
            rec = {"windows": e["windows"], "enrolled": ref is not None, "eer": float(v.eer.item()),
                   "label": {"mouse": "Mouse dynamics", "keystroke": "Keystroke timing", "touch": "Touch gestures",
                             "motion": "Device motion"}.get(modality, modality)}
            if ref is not None:
                r = np.asarray(ref, dtype=np.float32)
                r /= max(float(np.linalg.norm(r)), 1e-9)
                cos = float(e["embedding"] @ r)
                rec.update({"cosine": cos, "llr": v.llr(cos)})
            out[modality] = rec
        return out

    def describe(self) -> dict:
        return {m: {k: info.get(k) for k in ("datasets", "train_users", "open_set", "balabit")} for m, info in self.info.items()}


def merge_enrolment(old: dict | None, embs: dict[str, dict], weight_old: float = 0.8) -> dict:
    """running mean of session embeddings per modality (unit norm), the stored profile"""
    out = dict(old or {})
    for modality, e in embs.items():
        new = np.asarray(e["embedding"], dtype=np.float32)
        if modality in out:
            prev = np.asarray(out[modality], dtype=np.float32)
            new = weight_old * prev + (1 - weight_old) * new
        new = new / max(float(np.linalg.norm(new)), 1e-9)
        out[modality] = [round(float(x), 6) for x in new]
    return out
