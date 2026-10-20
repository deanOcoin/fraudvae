"""Serve the text heads: a `text` payload ({url, description, profile, memo ...}) is routed to
the heads that know that kind of text, each answers with a calibrated probability and a
likelihood ratio, and the ratios are what the fusion step adds to the blend."""
from __future__ import annotations

from pathlib import Path

from .model import TextHead
from .train import TEXT_SETS
from ..model.loading import load_checkpoint

# payload key -> head name. "memo" and "note" are free text: the scam-text head reads them
PAYLOAD_ROUTES = {"url": "malurl", "description": "fakejob", "memo": "fakejob", "note": "fakejob", "message": "fakejob",
                  "profile": "twitterbot", "bio": "twitterbot"}


class TextScorer:
    def __init__(self, run_dir: Path, device="cpu"):
        self.heads: dict[str, TextHead] = {}
        self.info: dict[str, dict] = {}
        for p in sorted((Path(run_dir) / "text").glob("*.pt")):
            ck = load_checkpoint(p, device)
            cfg = ck["config"]
            head = TextHead(cfg["name"], cfg["buckets"], cfg["dim"], cfg["hidden"])
            head.load_state_dict({k: (v.float() if v.is_floating_point() else v) for k, v in ck["state_dict"].items()})
            head.to(device).eval()
            self.heads[cfg["name"]] = head
            self.info[cfg["name"]] = ck.get("info", {})

    def __bool__(self):
        return bool(self.heads)

    @staticmethod
    def load(run_dir: Path, device="cpu") -> "TextScorer | None":
        if not (Path(run_dir) / "text").exists():
            return None
        t = TextScorer(run_dir, device)
        return t if t else None

    def score(self, payload: dict | None) -> dict:
        """{field: {head, prob, llr, label}} for every payload field a head can read"""
        out = {}
        if not isinstance(payload, dict):
            return out
        for field, value in payload.items():
            name = PAYLOAD_ROUTES.get(str(field).lower())
            if name is None or name not in self.heads or value in (None, ""):
                continue
            r = self.heads[name].score([str(value)])
            out[str(field)] = {"head": name, "label": TEXT_SETS.get(name, {}).get("label", name), "prob": float(r["prob"][0]),
                               "llr": float(r["llr"][0]), "test_auroc": self.info.get(name, {}).get("test_auroc")}
        return out

    def describe(self) -> dict:
        return {name: {k: info.get(k) for k in ("label", "payload", "test_auroc", "test_auprc", "n", "positive_rate")}
                for name, info in self.info.items()}
