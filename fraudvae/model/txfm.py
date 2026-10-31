"""v6: the transaction-sequence encoder, pretrained on every row without a label.

A customer's transactions are a sequence of events. Each event is a bag of (field, value)
tokens: a numeric field becomes "field f in bucket b" with learned per-field quantile edges,
a categorical field "field f is value v", plus the source and the time since the previous
event. An event encoder turns the bag into one vector, a causal history encoder reads the
customer's events oldest to newest, and two heads learn from that without labels:

  next-event   from the state *before* event t, predict every observed token of event t
               (the negative log likelihood is the "surprise": how unexpected this
               transaction is for this customer, field by field)
  masked-field from the event's own tokens plus the state after it, predict tokens that
               were hidden (15% of tokens, 10% of whole events, 10% of whole fields)

What the rest of the model reads from it: the state after the current event (the embedding of
"this customer right now") and the per-field surprise of the current event. Both go to the
booster as features (model/booster.py, `boost.txfm_features`) and onto the receipt. This is
the shape every published payments foundation model has converged on (Revolut's PRAGMA, Visa's
TransactionGPT, Sardine's issuer model); the v5 history encoder was the same idea at 0.1M
parameters trained as a side effect of the VAE, and the plan is in the owner's notes.

The checkpoint (`txfm.pt`) holds the weights, the tokeniser (edges, sizes) and the prepared
meta it was built on, read back with the project's weights-only loader.
"""
from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .loading import load_checkpoint as read_checkpoint

DT_SCALE = 2.0
EVENT_CHUNK = 32768   # events per event-encoder call: a cuda attention kernel refuses 65,536 sequences in one launch          # dt bucket = ceil(log1p(seconds) * DT_SCALE); 40 buckets reach past a year


class Tokeniser:
    """rows -> token ids. fields in order: numerics (n_bins buckets + unobserved), categoricals
    (their own index, 0 = unobserved), the source (0 = generic), the time since the previous
    event (0 = no previous event). every field owns a slice of one vocabulary, so one embedding
    table and one output layer serve them all; MASK is the last id."""

    def __init__(self, num_names, cat_sizes, edges: np.ndarray, n_sources: int, n_bins: int = 32, dt_bins: int = 40,
                 source_fields: dict | None = None):
        self.num_names = list(num_names)
        self.cat_sizes = [int(k) for k in cat_sizes]
        self.edges = np.asarray(edges, dtype=np.float32)            # [D, n_bins - 1], sorted per row
        self.n_sources = int(n_sources)
        self.n_bins = int(n_bins)
        self.dt_bins = int(dt_bins)
        D, C = len(self.num_names), len(self.cat_sizes)
        sizes = [self.n_bins + 1] * D + self.cat_sizes + [self.n_sources + 1, self.dt_bins + 1]
        self.sizes = sizes
        self.offsets = np.concatenate([[0], np.cumsum(sizes)[:-1]]).astype(np.int64)
        self.vocab = int(sum(sizes))
        self.mask_id = self.vocab
        self.n_fields = len(sizes)
        self.field_names = self.num_names + [f"cat_{j}" for j in range(C)] + ["source", "dt"]
        self.D, self.C = D, C
        self._dev = {}
        # packing: the fields a source ever observes (plus source and dt), by source index. the
        # event encoder reads only those for a batch of that source, a third of the schema for
        # most sources; a source without a list, or the blanked source, gets every field
        self.source_fields = {int(k): sorted(set(int(f) for f in v) | {self.n_fields - 2, self.n_fields - 1})
                              for k, v in (source_fields or {}).items()}
        self._fields_cache = {}

    def config(self) -> dict:
        return {"num_names": self.num_names, "cat_sizes": self.cat_sizes, "edges": self.edges, "n_sources": self.n_sources,
                "n_bins": self.n_bins, "dt_bins": self.dt_bins, "source_fields": self.source_fields}

    def fields_for_source(self, src: int, device=None):
        """the sorted field index of one source, None for every field (no lists, no list for
        this source, or the blanked source 0)"""
        if not self.source_fields:
            return None
        lst = self.source_fields.get(int(src))
        if lst is None or len(lst) >= self.n_fields:
            return None
        key = ("src", int(src), str(device))
        if key not in self._fields_cache:
            self._fields_cache[key] = torch.tensor(lst, dtype=torch.long, device=device)
        return self._fields_cache[key]

    def fields_for(self, tokens: torch.Tensor, valid: torch.Tensor | None = None):
        """the fields the event encoder reads for a batch of windows [B, L, F]: the union of the
        lists of the sources present (among the valid events), as a sorted index tensor; None
        when that is every field (no lists, a source without one, or the blanked source)"""
        if not self.source_fields:
            return None
        src_f = self.n_fields - 2
        off = int(self.offsets[src_f])
        st = tokens[..., src_f] if valid is None else tokens[..., src_f][valid]
        ids = tuple(int(t) - off for t in torch.unique(st).tolist())
        key = (ids, str(tokens.device))
        if key not in self._fields_cache:
            fields: set | None = set()
            for i in ids:
                lst = self.source_fields.get(i)
                if lst is None:
                    fields = None
                    break
                fields |= set(lst)
            if fields is None or len(fields) >= self.n_fields:
                self._fields_cache[key] = None
            else:
                self._fields_cache[key] = torch.tensor(sorted(fields), dtype=torch.long, device=tokens.device)
        return self._fields_cache[key]

    @classmethod
    def fit(cls, num: np.ndarray, mask: np.ndarray, num_names, cat_sizes, n_sources: int, n_bins: int = 32, dt_bins: int = 40,
            source_fields: dict | None = None):
        """quantile edges per numeric field from a sample of (standardised) rows, observed cells
        only; a field with fewer distinct values than buckets gets its distinct values as edges.
        source_fields: {source index: [field indices it observes]} for packing, None = no packing"""
        D = num.shape[1]
        edges = np.zeros((D, n_bins - 1), dtype=np.float32)
        qs = np.linspace(0, 1, n_bins + 1)[1:-1]
        for j in range(D):
            v = num[:, j][mask[:, j] > 0]
            if len(v) == 0:
                edges[j] = np.arange(1, n_bins, dtype=np.float32) * 1e-3     # never observed: any value is bucket 0
                continue
            e = np.quantile(v.astype(np.float64), qs)
            # equal edges (a field with few distinct values) must still be increasing for searchsorted
            e = np.maximum.accumulate(e)
            bump = np.arange(n_bins - 1) * 1e-6
            edges[j] = (e + bump).astype(np.float32)
        return cls(num_names, cat_sizes, edges, n_sources, n_bins, dt_bins, source_fields=source_fields)

    def _tensors(self, device):
        d = str(device)
        if d not in self._dev:
            self._dev[d] = (torch.from_numpy(self.edges).to(device), torch.from_numpy(self.offsets).to(device))
        return self._dev[d]

    def encode(self, num, mask, cat, src, dt_secs) -> torch.Tensor:
        """num [..., D] standardised, mask [..., D], cat [..., C] int, src [...] int,
        dt_secs [...] float seconds since the previous event (negative = none) -> tokens [..., F] int64"""
        edges, off = self._tensors(num.device)
        lead = num.shape[:-1]
        n = int(np.prod(lead)) if len(lead) else 1
        x = num.reshape(n, self.D).float().transpose(0, 1).contiguous()          # [D, n]
        b = torch.searchsorted(edges, x).transpose(0, 1)                             # [n, D] in 0..n_bins-1
        obs = mask.reshape(n, self.D) > 0
        num_tok = torch.where(obs, off[: self.D] + 1 + b, off[: self.D].expand(n, -1))
        cat_tok = off[self.D : self.D + self.C] + cat.reshape(n, self.C).long().clamp(min=0)
        src_tok = off[self.D + self.C] + src.reshape(n).long().clamp(0, self.n_sources)
        secs = dt_secs.reshape(n).float()
        dtb = torch.ceil(torch.log1p(secs.clamp(min=0)) * DT_SCALE).long().clamp(1, self.dt_bins)
        dt_tok = off[self.D + self.C + 1] + torch.where(secs >= 0, dtb, torch.zeros_like(dtb))
        toks = torch.cat([num_tok, cat_tok, src_tok[:, None], dt_tok[:, None]], dim=1)
        return toks.reshape(*lead, self.n_fields)

    def observed(self, tokens: torch.Tensor) -> torch.Tensor:
        """[..., F] bool: a token that says "unobserved" (the first id of its field) is not a target"""
        _, off = self._tensors(tokens.device)
        return tokens != off


def causal_mask(n: int, device) -> torch.Tensor:
    return torch.triu(torch.ones(n, n, dtype=torch.bool, device=device), diagonal=1)


class TransactionEncoder(nn.Module):
    def __init__(self, vocab: int, n_fields: int, sizes: list[int], n_num: int, d_model: int = 256, event_layers: int = 2,
                 history_layers: int = 4, heads: int = 4, window: int = 64, dropout: float = 0.1):
        super().__init__()
        self.vocab, self.n_fields, self.sizes = int(vocab), int(n_fields), [int(s) for s in sizes]
        # the leading n_num fields are the numerics: equal slices (n_bins + 1), scored as one batched matmul
        self.n_num = int(n_num)
        assert all(sz == self.sizes[0] for sz in self.sizes[: self.n_num]), "numeric fields must share one bucket count"
        self.d_model, self.event_layers, self.history_layers = int(d_model), int(event_layers), int(history_layers)
        self.heads, self.window, self.dropout_p = int(heads), int(window), float(dropout)
        self.tok = nn.Embedding(vocab + 1, d_model)             # + MASK
        self.field = nn.Embedding(n_fields, d_model)            # which field a token sits in (a MASK needs to know)
        self.evt_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.bos = nn.Parameter(torch.zeros(1, 1, d_model))
        self.in_norm = nn.LayerNorm(d_model)
        if event_layers > 0:
            layer = nn.TransformerEncoderLayer(d_model, heads, dim_feedforward=4 * d_model, dropout=dropout,
                                               batch_first=True, norm_first=True, activation="gelu")
            self.event = nn.TransformerEncoder(layer, event_layers, enable_nested_tensor=False)
        else:
            self.event = None
            self.event_mlp = nn.Sequential(nn.Linear(d_model, 2 * d_model), nn.GELU(), nn.Linear(2 * d_model, d_model))
        self.event_norm = nn.LayerNorm(d_model)
        self.pos = nn.Embedding(window + 1, d_model)
        self.time = nn.Linear(1, d_model)
        hlayer = nn.TransformerEncoderLayer(d_model, heads, dim_feedforward=4 * d_model, dropout=dropout,
                                            batch_first=True, norm_first=True, activation="gelu")
        self.history = nn.TransformerEncoder(hlayer, history_layers, enable_nested_tensor=False)
        self.hist_norm = nn.LayerNorm(d_model)
        self.next_head = nn.Linear(d_model, vocab)
        self.mlm_head = nn.Linear(d_model, vocab)
        offsets = np.concatenate([[0], np.cumsum(self.sizes)[:-1]]).astype(np.int64)
        self.register_buffer("offsets", torch.from_numpy(offsets), persistent=False)
        self.register_buffer("field_size", torch.tensor(self.sizes, dtype=torch.long), persistent=False)
        nn.init.normal_(self.tok.weight, std=0.02)
        nn.init.normal_(self.field.weight, std=0.02)

    def config(self) -> dict:
        return {"vocab": self.vocab, "n_fields": self.n_fields, "sizes": self.sizes, "n_num": self.n_num, "d_model": self.d_model,
                "event_layers": self.event_layers, "history_layers": self.history_layers, "heads": self.heads,
                "window": self.window, "dropout": self.dropout_p}

    # -- pieces ------------------------------------------------------------------------------

    def encode_events(self, tokens, fields=None):
        """tokens [B, L, F] -> event vectors [B, L, d], token outputs [B, L, F', d]. fields: a
        sorted index of the F' fields to read (Tokeniser.fields_for), None = all of them"""
        if fields is not None:
            tokens = tokens[..., fields]
        B, L, Fn = tokens.shape
        fld = self.field.weight[:Fn] if fields is None else self.field.weight[fields]
        x = self.tok(tokens) + fld[None, None]
        x = self.in_norm(x).reshape(B * L, Fn, -1)
        if self.event is not None:
            x = torch.cat([self.evt_token.expand(B * L, 1, -1), x], dim=1)
            if x.shape[0] > EVENT_CHUNK:
                x = torch.cat([self.event(x[s : s + EVENT_CHUNK]) for s in range(0, x.shape[0], EVENT_CHUNK)], dim=0)
            else:
                x = self.event(x)
            evt, tok_out = x[:, 0], x[:, 1:]
        else:
            evt = self.event_mlp(x.mean(1))
            tok_out = x
        return self.event_norm(evt).reshape(B, L, -1), tok_out.reshape(B, L, Fn, -1)

    def encode_history(self, events, dt_log, valid, block=None):
        """events [B, L, d] oldest first, dt_log [B, L] (log1p seconds since the previous event,
        0 for none), valid [B, L] -> states [B, L + 1, d]: states[:, t] has seen events < t.
        block [B, L] ids: a window that holds several customers in a row (the block pass over
        the sequence index); an event then sees only earlier events of its own block"""
        B, L, _ = events.shape
        dev = events.device
        pos = torch.arange(L, device=dev)
        x = events + self.pos(pos + 1)[None] + self.time(dt_log.unsqueeze(-1).float())
        x = torch.cat([self.bos.expand(B, 1, -1) + self.pos.weight[0][None, None], x], dim=1)
        pad = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=dev), ~valid], dim=1)
        mask = causal_mask(L + 1, dev)
        if block is not None:
            blocked = torch.ones(B, L + 1, L + 1, dtype=torch.bool, device=dev)
            blocked[:, 1:, 1:] = mask[1:, 1:][None] | (block[:, :, None] != block[:, None, :])
            blocked[:, :, 0] = False                                   # every event sees the start token
            mask = blocked.repeat_interleave(self.heads, dim=0)
        x = self.hist_norm(self.history(x, mask=mask, src_key_padding_mask=pad))
        # the state before a customer's first event is the start token's state, whatever sits in
        # the slots before it (padding at serving time, another customer's rows in a training
        # window): one trained prior for first transactions, the same in every path
        return torch.cat([x[:, :1], torch.where(valid.unsqueeze(-1), x[:, 1:], x[:, :1])], dim=1)

    def forward(self, tokens, dt_log, valid, fields=None, block=None):
        events, tok_out = self.encode_events(tokens, fields)
        states = self.encode_history(events, dt_log, valid, block)
        return states, tok_out

    # -- losses --------------------------------------------------------------------------------

    def _grouped_nll(self, head: nn.Linear, x, tokens, mask, fields=None):
        """nll of each token under the softmax restricted to its field's slice of the vocabulary.
        x [N, F', d] (an input per field) or [N, d] (one input for every field), tokens [N, F']
        full-vocab ids, mask [N, F'] bool -> nll [N, F'] float32, 0 where mask is false. fields:
        which F' of the F fields these are (a sorted index), None = all of them in order.
        the numeric fields share one slice width, so they go through a single batched matmul
        against the head's first n_num * k rows; the other fields are scored one slice each. no
        [N, F, vocab] tensor is ever formed (that was gigabytes per step)."""
        N, Fn = tokens.shape
        D, k = self.n_num, self.sizes[0]
        W, b = head.weight, head.bias
        out = torch.zeros(N, Fn, dtype=torch.float32, device=tokens.device)
        per_field = x.dim() == 3
        if fields is None:
            nd = min(D, Fn)
            num_ids = None
            ids = list(range(Fn))
        else:
            ids = [int(f) for f in fields.tolist()]
            nd = sum(1 for f in ids if f < D)
            num_ids = fields[:nd]
        if nd:
            xn = x[:, :nd] if per_field else x.unsqueeze(1).expand(N, nd, -1)
            Wn = W[: D * k].reshape(D, k, -1)
            bn = b[: D * k].reshape(D, k)
            offs = self.offsets[:nd]
            if num_ids is not None:
                Wn, bn, offs = Wn[num_ids], bn[num_ids], self.offsets[num_ids]
            logits = torch.einsum("nfd,fkd->nfk", xn, Wn) + bn.unsqueeze(0)
            tgt = (tokens[:, :nd] - offs).clamp(0, k - 1)
            nll = F.cross_entropy(logits.reshape(-1, k).float(), tgt.reshape(-1), reduction="none").reshape(N, nd)
            out[:, :nd] = nll * mask[:, :nd]
        for j in range(nd, Fn):
            m = mask[:, j]
            if not bool(m.any()):
                continue
            f = ids[j]
            o, sz = int(self.offsets[f]), self.sizes[f]
            xf = (x[:, j] if per_field else x)[m]
            lg = xf @ W[o : o + sz].T + b[o : o + sz]
            out[m, j] = F.cross_entropy(lg.float(), (tokens[m, j] - o).clamp(0, sz - 1), reduction="none")
        return out

    def next_event_nll(self, states, tokens, target_mask):
        """per-token nll of event t's tokens from states[:, t] (the state before it). tokens
        [B, L, F], target_mask [B, L, F] bool (observed and valid) -> nll [B, L, F] (0 elsewhere)"""
        B, L, Fn = tokens.shape
        x = states[:, :L].reshape(B * L, -1)
        return self._grouped_nll(self.next_head, x, tokens.reshape(B * L, Fn), target_mask.reshape(B * L, Fn)).reshape(B, L, Fn)

    def masked_nll(self, states, tok_out, targets, masked, fields=None):
        """tokens hidden from the event encoder, predicted from their token output plus the
        state after the event. targets [B, L, F] the true ids, masked [B, L, F] bool, fields the
        index the event encoder was packed with (tok_out is [B, L, F', d]) -> mean nll"""
        if fields is not None:
            targets, masked = targets[..., fields], masked[..., fields]
        B, L, Fn = targets.shape
        if not bool(masked.any()):
            return targets.new_zeros((), dtype=torch.float32)
        x = (tok_out + states[:, 1 : L + 1].unsqueeze(2)).reshape(B * L, Fn, -1)
        nll = self._grouped_nll(self.mlm_head, x, targets.reshape(B * L, Fn), masked.reshape(B * L, Fn), fields)
        return nll.sum() / masked.sum()


def mask_tokens(tokens, valid, mask_id, p_token=0.15, p_event=0.10, p_field=0.10, generator=None):
    """the three masking levels, as in PRAGMA: single tokens, whole events, whole fields across
    the window. returns (corrupted tokens, masked [B, L, F] bool). the source and dt fields
    (the last two) are never masked: they are context, not content."""
    B, L, Fn = tokens.shape
    dev = tokens.device
    r = lambda *s: torch.rand(*s, device=dev, generator=generator)  # noqa: E731
    m = r(B, L, Fn) < p_token
    m |= (r(B, L, 1) < p_event)
    m |= (r(B, 1, Fn) < p_field)
    m[..., -2:] = False
    m &= valid.unsqueeze(-1)
    return torch.where(m, torch.full_like(tokens, mask_id), tokens), m


# -- windows from the v5 history tensors -------------------------------------------------------

def windows_from_history(tok: Tokeniser, num, mask, cat, src, hist, cur_dt_secs=None):
    """the (tokens, dt_log, valid) a scorer needs for a batch of current rows with their v5
    history tensors (SequenceIndex.history: past rows oldest first, h_dt = log1p(secs to the
    current row) / 10). the current row is the last event. hist None = no past."""
    B = num.shape[0]
    dev = num.device
    if hist is None:
        toks = tok.encode(num, mask, cat, src, torch.full((B,), -1.0, device=dev) if cur_dt_secs is None else cur_dt_secs)
        return toks.unsqueeze(1), torch.zeros(B, 1, device=dev), torch.ones(B, 1, dtype=torch.bool, device=dev)
    h_num, h_mask, h_cat, h_dt, h_valid = hist
    L = h_num.shape[1]
    secs_to_cur = torch.expm1(h_dt.float() * 10.0)                      # [B, L], decreasing oldest -> newest
    prev = torch.cat([torch.full((B, 1), float("nan"), device=dev), secs_to_cur[:, :-1]], dim=1)
    dt_past = prev - secs_to_cur                                          # seconds since the previous past event
    first = torch.cat([torch.ones(B, 1, dtype=torch.bool, device=dev), ~h_valid[:, :-1]], dim=1)
    dt_past = torch.where(first | torch.isnan(dt_past), torch.full_like(dt_past, -1.0), dt_past.clamp(min=0))
    any_past = h_valid.any(1)
    last_secs = secs_to_cur[:, -1]
    dt_cur = torch.where(any_past, last_secs.clamp(min=0), torch.full_like(last_secs, -1.0))
    src_rep = src.unsqueeze(1).expand(B, L)
    past = tok.encode(h_num, h_mask, h_cat, src_rep, dt_past)
    cur = tok.encode(num, mask, cat, src, dt_cur)
    toks = torch.cat([past, cur.unsqueeze(1)], dim=1)
    valid = torch.cat([h_valid, torch.ones(B, 1, dtype=torch.bool, device=dev)], dim=1)
    dt_all = torch.cat([dt_past, dt_cur.unsqueeze(1)], dim=1)
    dt_log = torch.where(dt_all >= 0, torch.log1p(dt_all.clamp(min=0)), torch.zeros_like(dt_all))
    return toks, dt_log, valid


@torch.no_grad()
def readout(model: TransactionEncoder, tok: Tokeniser, tokens, dt_log, valid):
    """for the last event of each window: the state after it [B, d] and the per-field surprise
    [B, F - 2] (nll of each observed content field from the state before it; NaN unobserved)
    and the mean surprise [B]. the booster's and the receipt's view of the encoder."""
    states, _ = model(tokens, dt_log, valid, fields=tok.fields_for(tokens))
    L = tokens.shape[1]
    obs = tok.observed(tokens) & valid.unsqueeze(-1)
    tgt = torch.zeros_like(obs)
    tgt[:, L - 1] = obs[:, L - 1]
    nll = model.next_event_nll(states, tokens, tgt)[:, L - 1, : tok.n_fields - 2]
    o = obs[:, L - 1, : tok.n_fields - 2]
    per_field = torch.where(o, nll, torch.full_like(nll, float("nan")))
    total = (nll * o).sum(1) / o.sum(1).clamp(min=1)
    return states[:, L].float(), per_field.float(), total.float()


def feature_names(tok: Tokeniser, d_model: int) -> list[str]:
    return [f"txfm_h{i}" for i in range(d_model)] + [f"txfm_s_{n}" for n in tok.field_names[:-2]] + ["txfm_surprise"]


def feature_dim(tok: Tokeniser, d_model: int) -> int:
    return d_model + (tok.n_fields - 2) + 1


def save_checkpoint(path: Path, model: TransactionEncoder, tok: Tokeniser, meta: dict, cfg: dict, extra: dict | None = None):
    torch.save({"model_config": model.config(), "state_dict": model.state_dict(), "tokeniser": tok.config(),
                "prepared_meta": meta, "train_config": cfg, "extra": extra or {}}, path)


def load_txfm(path: str | Path, device="cpu") -> tuple[TransactionEncoder, Tokeniser, dict]:
    ck = read_checkpoint(path, device)
    model = TransactionEncoder(**ck["model_config"])
    model.load_state_dict(ck["state_dict"])
    model.to(device).eval()
    tok = Tokeniser(**ck["tokeniser"])
    return model, tok, ck


def param_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def lr_cosine(step, lr, warmup, total, floor=0.05):
    if step < warmup:
        return lr * (step + 1) / max(1, warmup)
    p = (step - warmup) / max(1, total - warmup)
    return lr * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * min(1.0, p))))


def cache_dir_for(prepared: str, txfm_pt) -> Path:
    """the cache of a (prepared set, encoder) pair lives with the set, keyed by the checkpoint's
    sha256 (a different encoder never reads another's features)"""
    import hashlib

    from ..paths import PREPARED_DIR

    h = hashlib.sha256()
    with open(txfm_pt, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            h.update(chunk)
    return PREPARED_DIR / prepared / "txfm_cache" / h.hexdigest()[:16]


class TxfmScorer:
    """the pretrained encoder as the booster, the post stage and the server use it: one
    feature block per row - the state after the row (d_model), the per-field surprise and the
    mean surprise - computed from the row and its customer history. loads runs/<run>/txfm.pt."""

    def __init__(self, model: TransactionEncoder, tok: Tokeniser, device="cpu", info: dict | None = None, cache_dir=None):
        self.model, self.tok, self.device, self.info = model, tok, torch.device(device), info or {}
        self.window = int(model.window)
        self.d_model = int(model.d_model)
        # a directory holding txfm_<split>.npy, the block pass over every row of a split, written by
        # `fraudvae txfm-cache` (or cache_split + merge_cache): features_split reads it instead of
        # encoding again, so the booster, calibration and evaluation share one pass per split
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.names = feature_names(tok, self.d_model)
        self.dim = len(self.names)

    @classmethod
    def load(cls, run_dir: str | Path, device="cpu"):
        p = Path(run_dir)
        p = p if p.suffix == ".pt" else p / "txfm.pt"
        if not p.exists():
            return None
        model, tok, ck = load_txfm(p, device)
        sc = cls(model, tok, device, info={"extra": ck.get("extra", {}), "train_config": ck.get("train_config", {})})
        sc.path = p
        return sc

    def with_cache(self, prepared: str):
        """read the block pass of this encoder over the named prepared set from its cache
        (cache_dir_for) where `fraudvae txfm-cache` has written it; a split without a cache
        file is encoded on the fly as before"""
        if getattr(self, "path", None) is not None:
            self.cache_dir = cache_dir_for(prepared, self.path)
        return self

    def field_labels(self, categorical_meta=None) -> list[str]:
        """the content fields the surprise block is indexed by, with the categoricals named
        from the prepared meta when given (the tokeniser only knows their count)"""
        names = list(self.tok.field_names[:-2])
        if categorical_meta:
            for j, c in enumerate(categorical_meta):
                k = self.tok.D + j
                if k < len(names):
                    names[k] = c["name"]
        return names

    @torch.no_grad()
    def features(self, num, mask, cat, src, hist=None) -> np.ndarray:
        """[B, dim] float32 for a batch of current rows with their history tensors (any window;
        the last `window - 1` past events are used), or hist None for rows without a past"""
        toks, dt, valid = windows_from_history(self.tok, num, mask, cat, src, hist)
        if toks.shape[1] > self.window:
            toks, dt, valid = toks[:, -self.window :], dt[:, -self.window :], valid[:, -self.window :]
        out = np.empty((toks.shape[0], self.dim), dtype=np.float32)
        # one source per readout, so a row's packed fields never depend on the rows beside it
        for s_ in torch.unique(src).tolist():
            m = src == s_
            h, per_field, total = readout(self.model, self.tok, toks[m], dt[m], valid[m])
            out[m.cpu().numpy()] = np.concatenate([h.cpu().numpy(), per_field.cpu().numpy(), total.cpu().numpy()[:, None]], axis=1)
        return out

    def features_windows(self, split, idx, seq, split_name, device=None, src_override=None, batch_size=8192) -> np.ndarray:
        """[len(idx), dim] in the order of idx, one history window per row (what the server
        does for one transaction); the block pass below is the same model at a fraction of the
        work for a whole split"""
        from .scoring import batches

        device = device or self.device
        idx = np.asarray(idx)
        out = np.empty((len(idx), self.dim), dtype=np.float32)
        pos = {int(i): k for k, i in enumerate(idx)}
        for b, (num, mask, cat, src), hist in batches(split, batch_size, idx=idx, device=device, seq=seq, split_name=split_name,
                                                      window=self.window if seq is not None else 0):
            if src_override is not None:
                src = torch.full_like(src, int(src_override))
            f = self.features(num, mask, cat, src, hist)
            out[[pos[int(i)] for i in b]] = f
        return out

    @torch.no_grad()
    def features_split(self, split, idx, seq, split_name, device=None, src_override=None, batch_size=256, stride=None) -> np.ndarray:
        """[len(idx), dim] in the order of idx: the block pass. consecutive events of the sequence
        index go through the causal encoder in segments of `window` events, several customers
        per segment behind a block mask, and every row reads its state and surprise off the
        position it sits at. a segment starts `window - stride` events before the first row it
        serves, so a row sees at least that much of its history (all of it when the customer's
        block is shorter); with stride window // 4 each event is encoded about four times
        instead of once per row. rows without a sequence index fall back to one window each."""
        if seq is None or not split_name:
            return self.features_windows(split, idx, seq, split_name, device=device, src_override=src_override)
        if src_override is None and self.cache_dir is not None and (self.cache_dir / f"txfm_{split_name}.npy").exists():
            cached = np.load(self.cache_dir / f"txfm_{split_name}.npy", mmap_mode="r")
            if cached.shape == (len(split), self.dim):
                idx = np.asarray(idx)
                order = np.argsort(idx, kind="stable")
                out = np.empty((len(idx), self.dim), dtype=np.float32)
                out[order] = cached[idx[order]]
                return out
        device = torch.device(device or self.device)
        L = self.window
        S = int(stride or max(1, L // 4))
        idx = np.asarray(idx)
        n_all = len(seq.order)
        p = seq.positions(split_name, idx).astype(np.int64)
        order = np.argsort(p, kind="stable")
        ps = p[order]
        row_src = np.asarray(split.src)[idx[order]].astype(np.int64)
        # segments serve rows of one source each, and a batch holds segments of one source, so
        # the packed field set is the rows' own source's whatever sits around them
        seg_start, seg_src = [], []
        seg_of, t_of = np.empty(len(ps), dtype=np.int64), np.empty(len(ps), dtype=np.int64)
        a, cur, cur_src = 0, -1, -1
        for i in range(len(ps)):
            q = int(ps[i])
            if cur < 0 or q >= a + L or row_src[i] != cur_src:
                a = max(0, q - (L - S))
                cur, cur_src = cur + 1, int(row_src[i])
                seg_start.append(a)
                seg_src.append(cur_src)
            seg_of[i] = cur
            t_of[i] = q - a
        seg_start = np.asarray(seg_start, dtype=np.int64)
        seg_src = np.asarray(seg_src, dtype=np.int64)
        batches_ = []
        s0 = 0
        while s0 < len(seg_start):
            s1 = min(s0 + batch_size, len(seg_start))
            change = np.flatnonzero(seg_src[s0 + 1 : s1] != seg_src[s0])
            if len(change):
                s1 = s0 + 1 + int(change[0])
            batches_.append((s0, s1))
            s0 = s1
        out = np.empty((len(idx), self.dim), dtype=np.float32)
        Fn = self.tok.n_fields
        start_of = np.asarray(seq.start)
        order_arr = np.asarray(seq.order)
        row_lo = 0
        use_amp = device.type == "cuda"
        for s0, s1 in batches_:
            starts = seg_start[s0:s1]
            B = len(starts)
            fields = None if src_override is not None else self.tok.fields_for_source(int(seg_src[s0]), device)
            pos = starts[:, None] + np.arange(-1, L, dtype=np.int64)[None, :]        # the previous event first
            inside = (pos >= 0) & (pos < n_all)
            gids = np.where(inside, order_arr[np.clip(pos, 0, n_all - 1)], -1)
            num, mask, cat, ts = seq.gather_rows(gids.ravel())
            src = seq.gather_src(gids.ravel()) if src_override is None else np.full(gids.size, int(src_override), dtype=np.int64)
            bstart = np.where(inside, start_of[np.clip(gids, 0, None)], -2).astype(np.int64)
            ts = ts.reshape(B, L + 1)
            same_prev = (bstart[:, 1:] == bstart[:, :-1]) & inside[:, 1:] & inside[:, :-1]
            dt = np.where(same_prev, ts[:, 1:] - ts[:, :-1], -1.0).astype(np.float32)
            t = lambda x: torch.from_numpy(np.ascontiguousarray(x)).to(device)  # noqa: E731
            toks = self.tok.encode(t(num.reshape(B, L + 1, -1)[:, 1:]), t(mask.reshape(B, L + 1, -1)[:, 1:]),
                                   t(cat.reshape(B, L + 1, -1)[:, 1:]), t(src.reshape(B, L + 1)[:, 1:]), t(dt))
            dt_t = t(dt)
            dt_log = torch.where(dt_t >= 0, torch.log1p(dt_t.clamp(min=0)), torch.zeros_like(dt_t))
            valid = t(inside[:, 1:])
            block = t(bstart[:, 1:])
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                states, _ = self.model(toks, dt_log, valid, fields, block=block)
                obs = self.tok.observed(toks) & valid.unsqueeze(-1)
                # a block's first event is predicted from the start token's state, as in training
                # and at serving time, not from the previous customer's last event
                first = t(bstart[:, 1:] != bstart[:, :-1])
                before = torch.where(first.unsqueeze(-1), states[:, :1], states[:, :L])
                nll = self.model.next_event_nll(torch.cat([before, states[:, L:]], dim=1), toks, obs)
            row_hi = int(np.searchsorted(seg_of, s0 + B))
            rows = np.arange(row_lo, row_hi)
            row_lo = row_hi
            b_i = torch.from_numpy(seg_of[rows] - s0).to(device)
            t_i = torch.from_numpy(t_of[rows]).to(device)
            h = states[b_i, t_i + 1].float()
            nf = nll[b_i, t_i, : Fn - 2].float()
            o = obs[b_i, t_i, : Fn - 2]
            per_field = torch.where(o, nf, torch.full_like(nf, float("nan")))
            total = (nf * o).sum(1) / o.sum(1).clamp(min=1)
            out[order[rows]] = np.concatenate([h.cpu().numpy(), per_field.cpu().numpy(), total.cpu().numpy()[:, None]], axis=1)
        assert row_lo == len(ps)
        return out

    def cache_split(self, split, seq, split_name: str, cache_dir, part: int = 0, parts: int = 1, device=None, batch_size=256,
                    stride=None, log=print) -> Path:
        """the block pass over every row of a split, or over the part-th of `parts` slices of
        its rows in sequence order (one gpu each), to cache_dir/txfm_<split>.<part>of<parts>.npy
        as (row ids, features); merge_cache assembles the parts"""
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        idx = np.arange(len(split))
        idx = idx[np.argsort(seq.positions(split_name, idx), kind="stable")]
        lo, hi = len(idx) * part // parts, len(idx) * (part + 1) // parts
        mine = idx[lo:hi]
        t0 = time.time()
        feats = self.features_split(split, mine, seq, split_name, device=device, batch_size=batch_size, stride=stride)
        path = cache_dir / f"txfm_{split_name}.{part}of{parts}.npy"
        np.save(path, feats)
        # the row ids beside the features, as integers: a float32 id is exact only below 2^24,
        # and a 30M-row split rounded ids in a part file once
        np.save(cache_dir / f"txfm_{split_name}.{part}of{parts}.rows.npy", mine.astype(np.int64))
        log(f"  txfm cache: {split_name} part {part + 1}/{parts}, {len(mine):,} rows in {time.time() - t0:.0f}s -> {path.name}")
        return path

    @staticmethod
    def merge_cache(cache_dir, split_name: str, n_rows: int) -> Path:
        """assemble txfm_<split>.npy [n_rows, dim] from every part present; every row must be
        covered exactly once"""
        cache_dir = Path(cache_dir)
        parts = sorted(p for p in cache_dir.glob(f"txfm_{split_name}.*of*.npy") if not p.name.endswith(".rows.npy"))
        if not parts:
            raise SystemExit(f"no cache parts for {split_name} under {cache_dir}")
        first = np.load(parts[0], mmap_mode="r")
        out = np.lib.format.open_memmap(cache_dir / f"txfm_{split_name}.tmp.npy", "w+", np.float32, (n_rows, first.shape[1]))
        seen = np.zeros(n_rows, dtype=bool)
        for p_ in parts:
            a = np.load(p_, mmap_mode="r")
            rows = np.load(p_.with_name(p_.name[:-4] + ".rows.npy"))
            for s in range(0, len(rows), 1_000_000):
                out[rows[s:s + 1_000_000]] = a[s:s + 1_000_000]
            seen[rows] = True
        if not seen.all():
            raise SystemExit(f"cache parts for {split_name} cover {int(seen.sum()):,} of {n_rows:,} rows")
        out.flush()
        del out
        dest = cache_dir / f"txfm_{split_name}.npy"
        (cache_dir / f"txfm_{split_name}.tmp.npy").rename(dest)
        for p_ in parts:
            p_.unlink()
            p_.with_name(p_.name[:-4] + ".rows.npy").unlink()
        return dest

    def explain(self, feats_row: np.ndarray, labels: list[str], top: int = 5) -> dict:
        """the receipt's view of one feature row: mean surprise and the most surprising fields"""
        s = feats_row[self.d_model : self.d_model + len(labels)]
        order = [int(j) for j in np.argsort(-np.nan_to_num(s, nan=-1.0)) if s[j] == s[j]][:top]
        return {"surprise": float(feats_row[-1]) if feats_row[-1] == feats_row[-1] else None,
                "n_fields": int(np.isfinite(s).sum()),
                "most_surprising": [{"field": labels[j], "nll": float(s[j])} for j in order]}
