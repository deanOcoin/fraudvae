"""GraphSAGE without a graph library: the adjacency is a sparse tensor, a layer is
h' = act(W_self h + W_neigh (A_norm h)) with A_norm the row-normalised weighted adjacency
(mean aggregation over neighbours, weighted by how often the pair co-occurred). Two layers
reach two hops: a card's devices' other cards. Inductive by construction, so a snapshot
with new nodes is just another forward pass.

Three heads share the encoder: EdgeHead reads a transaction's entities (one embedding per
entity slot, zeros + a presence bit where the slot is empty or the entity unknown) into the
"graph" logit; NodeHead classifies nodes for the public benchmarks (DGraph, Elliptic,
YelpChi, Amazon) that validate the encoder; both are small MLPs.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .build import N_KINDS, NODE_FEATURES


def adjacency(edge_src: np.ndarray, edge_dst: np.ndarray, edge_w: np.ndarray, n_nodes: int, device="cpu") -> torch.Tensor:
    """row-normalised sparse [N, N] adjacency (mean aggregation with weights). a node with
    no neighbours aggregates zeros."""
    if len(edge_src) == 0:
        idx = torch.zeros((2, 0), dtype=torch.long)
        return torch.sparse_coo_tensor(idx, torch.zeros(0), (n_nodes, n_nodes)).coalesce().to(device)
    src = torch.from_numpy(np.asarray(edge_src, dtype=np.int64))
    dst = torch.from_numpy(np.asarray(edge_dst, dtype=np.int64))
    w = torch.from_numpy(np.asarray(edge_w, dtype=np.float32))
    deg = torch.zeros(n_nodes, dtype=torch.float32).index_add_(0, src, w)
    w = w / deg[src].clamp_min(1e-6)
    return torch.sparse_coo_tensor(torch.stack([src, dst]), w, (n_nodes, n_nodes)).coalesce().to(device)


class SAGELayer(nn.Module):
    """one weight per relation: a review graph has user / product / star relations, the
    entity graph has one"""

    def __init__(self, d_in: int, d_out: int, n_relations: int = 1):
        super().__init__()
        self.self_lin = nn.Linear(d_in, d_out)
        self.neigh_lin = nn.ModuleList([nn.Linear(d_in, d_out, bias=False) for _ in range(n_relations)])

    def forward(self, h, adjs):
        out = self.self_lin(h)
        for lin, adj in zip(self.neigh_lin, adjs):
            out = out + lin(torch.sparse.mm(adj, h))
        return out


class SAGE(nn.Module):
    def __init__(self, d_in: int, hidden: int = 64, out_dim: int = 32, layers: int = 2, dropout: float = 0.1, n_relations: int = 1):
        super().__init__()
        dims = [d_in] + [hidden] * (layers - 1) + [out_dim]
        self.layers = nn.ModuleList([SAGELayer(dims[i], dims[i + 1], n_relations) for i in range(layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(dims[i + 1]) for i in range(layers - 1)])
        self.drop = nn.Dropout(dropout)
        self.d_in, self.hidden, self.out_dim, self.n_layers, self.dropout, self.n_relations = d_in, hidden, out_dim, layers, dropout, n_relations

    def forward(self, x, adj):
        adjs = adj if isinstance(adj, (list, tuple)) else [adj]
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h, adjs)
            if i < len(self.layers) - 1:
                h = self.drop(torch.nn.functional.silu(self.norms[i](h)))
        return h

    def config(self):
        return {"d_in": self.d_in, "hidden": self.hidden, "out_dim": self.out_dim, "layers": self.n_layers, "dropout": self.dropout,
                "n_relations": self.n_relations}


class EdgeHead(nn.Module):
    """[B, 6, D] slot embeddings + [B, 6] known-bits -> logit. an unknown entity is a known
    kind of thing (a brand-new card), which the bit says."""

    def __init__(self, out_dim: int, hidden: int = 128, dropout: float = 0.1):
        super().__init__()
        self.out_dim, self.hidden = out_dim, hidden
        d_in = N_KINDS * out_dim + N_KINDS
        self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.SiLU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, hidden // 2), nn.SiLU(), nn.Linear(hidden // 2, 1))

    def forward(self, slots, known):
        x = torch.cat([(slots * known.unsqueeze(-1)).flatten(1), known], dim=1)
        return self.net(x).squeeze(1)


class NodeHead(nn.Module):
    def __init__(self, out_dim: int, hidden: int = 64, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(out_dim, hidden), nn.SiLU(), nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, h):
        return self.net(h).squeeze(1)


def gather_slots(emb: torch.Tensor, node_ids: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    """[n, 6] node ids (-1 unknown) -> ([n, 6, D] embeddings with zeros for unknown, [n, 6] known float)"""
    ids = torch.from_numpy(np.asarray(node_ids, dtype=np.int64)).to(emb.device)
    known = (ids >= 0).float()
    slots = emb[ids.clamp_min(0)].float() * known.unsqueeze(-1)
    return slots, known


N_NODE_FEATURES = len(NODE_FEATURES)
