"""Entity-graph snapshots from a source's processed parquet.

A snapshot is "everything the bank had seen up to a cut": the entities that appeared in the
rows before the cut, per-node features computed from those rows (how busy, how old, how much
confirmed fraud with the usual 30-day label delay, distinct partners), and weighted edges
between entities that appeared on the same row. Three cuts per source make the temporal
protocol honest:

  A  the first fit_fraction-complement of the train period   -> the encoder and the
     transaction head are fitted on the LAST part of the train period, whose rows are
     not in A (so a row never sees its own label through its entities)
  B  the whole train period                                  -> val rows (calibration)
  C  train + val                                             -> test rows and serving

The encoder is inductive (GraphSAGE), so B and C only need a forward pass. A node absent
from a snapshot is "unknown" to it, which is what a brand-new card is to a bank.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from ..data import schema

KINDS = list(schema.ENTITY_COLUMNS)           # customer, counterparty, device, email, address, card
N_KINDS = len(KINDS)
# node feature names, in order (standardised with snapshot A's stats at train time)
NODE_FEATURES = ([f"kind_{k}" for k in KINDS] +
                 ["log_rows", "log_fraud_delayed", "fraud_share_delayed", "has_delayed", "log_amount_mean",
                  "log_amount_max", "log_age_days", "log_days_since_last", "log_distinct_customers",
                  "log_distinct_counterparties", "log_distinct_devices", "log_distinct_cards"])


@dataclass
class Snapshot:
    key: str
    cut_ts: float
    node_kind: np.ndarray        # [N] int8
    node_code: np.ndarray        # [N] int64 entity code within the source
    x: np.ndarray                # [N, F] float32 node features (raw, standardise before use)
    edge_src: np.ndarray         # [E] int64 (both directions present)
    edge_dst: np.ndarray
    edge_w: np.ndarray           # [E] float32 log1p(count)
    index: dict                  # {(kind, code): node id}

    @property
    def n_nodes(self):
        return len(self.node_kind)

    def lookup(self, ent: np.ndarray) -> np.ndarray:
        """[n, 6] entity codes -> [n, 6] node ids, -1 where unknown to this snapshot"""
        out = np.full(ent.shape, -1, dtype=np.int64)
        for k in range(N_KINDS):
            codes = ent[:, k]
            ok = codes >= 0
            if not ok.any():
                continue
            out[ok, k] = [self.index.get((k, int(c)), -1) for c in codes[ok]]
        return out


COLUMNS = ["ts", "amount", "is_fraud", "split"] + KINDS


def load_rows(path, columns=COLUMNS) -> pd.DataFrame:
    """the columns the graph needs, as a frame sorted by time"""
    pf = pq.ParquetFile(path)
    have = set(pf.schema_arrow.names)
    df = pf.read(columns=[c for c in columns if c in have]).to_pandas()
    for k in KINDS:
        if k not in df:
            df[k] = np.int64(-1)
    df["split"] = df["split"].astype(str)
    return df.sort_values("ts", kind="stable").reset_index(drop=True)


def _node_table(rows: pd.DataFrame, cut_ts: float, delay_days: float, min_degree: int, max_nodes: int):
    """one record per (kind, code) seen in rows, with the feature block"""
    delay_cut = cut_ts - delay_days * 86400.0
    frames = []
    amt = np.log1p(np.clip(rows["amount"].to_numpy(dtype=np.float64), 0, None))
    amt = np.nan_to_num(amt, nan=0.0)
    fraud = rows["is_fraud"].to_numpy().astype(np.float64)
    old = (rows["ts"].to_numpy() < delay_cut).astype(np.float64)
    base = pd.DataFrame({"ts": rows["ts"].to_numpy(), "amt": amt, "fraud_old": fraud * old, "old": old,
                         "customer": rows["customer"].to_numpy(), "counterparty": rows["counterparty"].to_numpy(),
                         "device": rows["device"].to_numpy(), "card": rows["card"].to_numpy()})
    for k, kind in enumerate(KINDS):
        codes = rows[kind].to_numpy()
        ok = codes >= 0
        if not ok.any():
            continue
        sub = base[ok].copy()
        sub["code"] = codes[ok]
        g = sub.groupby("code", sort=False)
        rec = pd.DataFrame({
            "rows": g.size(),
            "fraud_delayed": g["fraud_old"].sum(),
            "old": g["old"].sum(),
            "amt_mean": g["amt"].mean(),
            "amt_max": g["amt"].max(),
            "first": g["ts"].min(),
            "last": g["ts"].max(),
        })
        for partner in ("customer", "counterparty", "device", "card"):
            if partner == kind:
                rec[f"d_{partner}"] = 0.0
            else:
                p = sub[sub[partner] >= 0]
                rec[f"d_{partner}"] = p.groupby("code")[partner].nunique().reindex(rec.index).fillna(0).to_numpy()
        rec["kind"] = k
        rec = rec[rec["rows"] >= min_degree] if min_degree > 1 else rec
        frames.append(rec.reset_index())
    if not frames:
        return pd.DataFrame(columns=["code", "kind", "rows"])
    nodes = pd.concat(frames, ignore_index=True)
    if len(nodes) > max_nodes:
        nodes = nodes.nlargest(max_nodes, "rows")
    return nodes


def _features(nodes: pd.DataFrame, cut_ts: float) -> np.ndarray:
    n = len(nodes)
    x = np.zeros((n, len(NODE_FEATURES)), dtype=np.float32)
    kind = nodes["kind"].to_numpy()
    x[np.arange(n), kind] = 1.0
    o = N_KINDS
    has_old = nodes["old"].to_numpy() > 0
    x[:, o + 0] = np.log1p(nodes["rows"].to_numpy())
    x[:, o + 1] = np.log1p(nodes["fraud_delayed"].to_numpy())
    x[:, o + 2] = np.where(has_old, nodes["fraud_delayed"].to_numpy() / np.maximum(nodes["old"].to_numpy(), 1), 0.0)
    x[:, o + 3] = has_old
    x[:, o + 4] = nodes["amt_mean"].to_numpy()
    x[:, o + 5] = nodes["amt_max"].to_numpy()
    x[:, o + 6] = np.log1p(np.clip((cut_ts - nodes["first"].to_numpy()) / 86400.0, 0, None))
    x[:, o + 7] = np.log1p(np.clip((cut_ts - nodes["last"].to_numpy()) / 86400.0, 0, None))
    x[:, o + 8] = np.log1p(nodes["d_customer"].to_numpy())
    x[:, o + 9] = np.log1p(nodes["d_counterparty"].to_numpy())
    x[:, o + 10] = np.log1p(nodes["d_device"].to_numpy())
    x[:, o + 11] = np.log1p(nodes["d_card"].to_numpy())
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def _edges(rows: pd.DataFrame, lookup) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """weighted co-occurrence edges between the entities of each row: every pair of kinds
    present on the row, counted over the rows, both directions"""
    ids = lookup(rows[KINDS].to_numpy().astype(np.int64))
    pairs = []
    for a in range(N_KINDS):
        for b in range(a + 1, N_KINDS):
            m = (ids[:, a] >= 0) & (ids[:, b] >= 0)
            if m.any():
                pairs.append(np.stack([ids[m, a], ids[m, b]], axis=1))
    if not pairs:
        return np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0, np.float32)
    p = np.concatenate(pairs)
    key = p[:, 0] * (ids.max() + 2) + p[:, 1]
    uniq, cnt = np.unique(key, return_counts=True)
    src = uniq // (ids.max() + 2)
    dst = uniq % (ids.max() + 2)
    w = np.log1p(cnt).astype(np.float32)
    return (np.concatenate([src, dst]), np.concatenate([dst, src]), np.concatenate([w, w]))


def snapshot(key: str, rows: pd.DataFrame, cut_ts: float, delay_days: float = 30.0, min_degree: int = 2,
             max_nodes: int = 4_000_000) -> Snapshot:
    """the graph of everything in rows with ts < cut_ts"""
    rows = rows[rows["ts"].to_numpy() < cut_ts]
    nodes = _node_table(rows, cut_ts, delay_days, min_degree, max_nodes)
    kind = nodes["kind"].to_numpy().astype(np.int8)
    code = nodes["code"].to_numpy().astype(np.int64)
    index = {(int(k), int(c)): i for i, (k, c) in enumerate(zip(kind, code))}
    snap = Snapshot(key, float(cut_ts), kind, code, _features(nodes, cut_ts), np.zeros(0, np.int64), np.zeros(0, np.int64),
                    np.zeros(0, np.float32), index)
    if len(nodes):
        snap.edge_src, snap.edge_dst, snap.edge_w = _edges(rows, snap.lookup)
    return snap


def cuts(rows: pd.DataFrame, fit_fraction: float = 0.2) -> dict:
    """the three cut timestamps (A: before the fit tail of train, B: end of train, C: end of val)
    and the fit rows' positions in `rows`"""
    split = rows["split"].to_numpy()
    ts = rows["ts"].to_numpy()
    train = np.flatnonzero(split == "train")
    val = np.flatnonzero(split == "val")
    n_fit = max(int(len(train) * fit_fraction), 1)
    fit_rows = train[-n_fit:]
    cut_a = float(ts[fit_rows[0]])
    cut_b = float(ts[train[-1]]) + 1e-3 if len(train) else cut_a
    cut_c = float(ts[val[-1]]) + 1e-3 if len(val) else cut_b
    return {"A": cut_a, "B": cut_b, "C": cut_c, "fit_rows": fit_rows}
