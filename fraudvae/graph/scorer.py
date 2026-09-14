"""Serve the trained entity-graph models: look a transaction's entities up in the stored
embeddings (snapshot B for validation rows, C for test rows and live traffic), run the
transaction head, hand back the "graph" logit. NaN where the source has no graph or none
of the row's entities is known to the snapshot - the blender treats that as absent."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ..data import schema
from .build import KINDS
from .model import EdgeHead, gather_slots
from .train import NodeIndex


class _SourceGraph:
    def __init__(self, state: dict, device):
        self.key = state["key"]
        self.head = EdgeHead(state["encoder_config"]["out_dim"], state.get("head_hidden", 128))
        self.head.load_state_dict(state["head"])
        self.head.to(device).eval()
        self.emb = {}
        self.index = {}
        for which in ("val", "test"):
            blk = state[which]
            self.index[which] = NodeIndex(blk["kind"], blk["code"])
            # kept in half precision in memory too: paysim's snapshot is millions of nodes
            self.emb[which] = torch.from_numpy(np.asarray(blk["emb"], dtype=np.float16)).to(device)
        self.stats = state.get("stats", {})
        self.cuts = state.get("cuts", {})

    @torch.no_grad()
    def logit(self, ent: np.ndarray, which: str) -> np.ndarray:
        which = which if which in self.emb else "test"
        ids = self.index[which].lookup(ent)
        out = np.full(len(ent), np.nan, dtype=np.float32)
        ok = (ids >= 0).any(1)
        if ok.any():
            slots, known = gather_slots(self.emb[which], ids[ok])
            out[ok] = self.head(slots, known).float().cpu().numpy()
        return out

    def known(self, ent: np.ndarray, which: str = "test") -> list[list[str]]:
        ids = self.index[which if which in self.emb else "test"].lookup(ent)
        return [[KINDS[k] for k in range(len(KINDS)) if row[k] >= 0] for row in ids]


class GraphScorer:
    def __init__(self, run_dir: Path, device="cpu"):
        self.run_dir = Path(run_dir)
        self.device = device
        self.sources: dict[int, _SourceGraph] = {}
        for p in sorted((self.run_dir / "graph").glob("*.pt")):
            state = torch.load(p, map_location="cpu", weights_only=False)
            key = state["key"]
            if key in schema.SOURCE_INDEX:
                self.sources[schema.SOURCE_INDEX[key]] = _SourceGraph(state, device)

    def __bool__(self):
        return bool(self.sources)

    @staticmethod
    def load(run_dir: Path, device="cpu") -> "GraphScorer | None":
        if not (Path(run_dir) / "graph").exists():
            return None
        g = GraphScorer(run_dir, device)
        return g if g else None

    def logit_rows(self, src: np.ndarray, ent: np.ndarray, which: str = "test") -> np.ndarray:
        """[n] source indices + [n, 6] entity codes -> [n] logits, NaN where absent. src 0
        (generic) scores with no graph: a bank we do not know has no entities we know."""
        src = np.asarray(src).astype(int)
        ent = np.asarray(ent, dtype=np.int64)
        out = np.full(len(src), np.nan, dtype=np.float32)
        for s, g in self.sources.items():
            m = src == s
            if m.any():
                out[m] = g.logit(ent[m], which)
        return out

    def describe(self) -> dict:
        return {g.key: g.stats for g in self.sources.values()}
