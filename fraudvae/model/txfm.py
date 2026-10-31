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
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .loading import load_checkpoint as read_checkpoint

DT_SCALE = 2.0          # dt bucket = ceil(log1p(seconds) * DT_SCALE); 40 buckets reach past a year


class Tokeniser:
    """rows -> token ids. fields in order: numerics (n_bins buckets + unobserved), categoricals
    (their own index, 0 = unobserved), the source (0 = generic), the time since the previous
    event (0 = no previous event). every field owns a slice of one vocabulary, so one embedding
    table and one output layer serve them all; MASK is the last id."""

    def __init__(self, num_names, cat_sizes, edges: np.ndarray, n_sources: int, n_bins: int = 32, dt_bins: int = 40):
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

    def config(self) -> dict:
        return {"num_names": self.num_names, "cat_sizes": self.cat_sizes, "edges": self.edges, "n_sources": self.n_sources,
                "n_bins": self.n_bins, "dt_bins": self.dt_bins}

    @classmethod
    def fit(cls, num: np.ndarray, mask: np.ndarray, num_names, cat_sizes, n_sources: int, n_bins: int = 32, dt_bins: int = 40):
        """quantile edges per numeric field from a sample of (standardised) rows, observed cells
        only; a field with fewer distinct values than buckets gets its distinct values as edges"""
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
        return cls(num_names, cat_sizes, edges, n_sources, n_bins, dt_bins)

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

    def encode_events(self, tokens):
        """tokens [B, L, F] -> event vectors [B, L, d], token outputs [B, L, F, d]"""
        B, L, Fn = tokens.shape
        x = self.tok(tokens) + self.field.weight[None, None, :Fn]
        x = self.in_norm(x).reshape(B * L, Fn, -1)
        if self.event is not None:
            x = torch.cat([self.evt_token.expand(B * L, 1, -1), x], dim=1)
            x = self.event(x)
            evt, tok_out = x[:, 0], x[:, 1:]
        else:
            evt = self.event_mlp(x.mean(1))
            tok_out = x
        return self.event_norm(evt).reshape(B, L, -1), tok_out.reshape(B, L, Fn, -1)

    def encode_history(self, events, dt_log, valid):
        """events [B, L, d] oldest first, dt_log [B, L] (log1p seconds since the previous event,
        0 for none), valid [B, L] -> states [B, L + 1, d]: states[:, t] has seen events < t"""
        B, L, _ = events.shape
        pos = torch.arange(L, device=events.device)
        x = events + self.pos(pos + 1)[None] + self.time(dt_log.unsqueeze(-1).float())
        x = torch.cat([self.bos.expand(B, 1, -1) + self.pos.weight[0][None, None], x], dim=1)
        pad = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=events.device), ~valid], dim=1)
        x = self.history(x, mask=causal_mask(L + 1, events.device), src_key_padding_mask=pad)
        return self.hist_norm(x)

    def forward(self, tokens, dt_log, valid):
        events, tok_out = self.encode_events(tokens)
        states = self.encode_history(events, dt_log, valid)
        return states, tok_out

    # -- losses --------------------------------------------------------------------------------

    def _grouped_nll(self, head: nn.Linear, x, tokens, mask):
        """nll of each token under the softmax restricted to its field's slice of the vocabulary.
        x [N, F, d] (an input per field) or [N, d] (one input for every field), tokens [N, F]
        full-vocab ids, mask [N, F] bool -> nll [N, F] float32, 0 where mask is false.
        the numeric fields share one slice width, so they go through a single batched matmul
        against the head's first n_num * k rows; the other fields are scored one slice each. no
        [N, F, vocab] tensor is ever formed (that was gigabytes per step)."""
        N, Fn = tokens.shape
        D, k = self.n_num, self.sizes[0]
        W, b = head.weight, head.bias
        out = torch.zeros(N, Fn, dtype=torch.float32, device=tokens.device)
        per_field = x.dim() == 3
        if D:
            xn = x[:, :D] if per_field else x.unsqueeze(1).expand(N, D, -1)
            Wn = W[: D * k].reshape(D, k, -1)
            logits = torch.einsum("nfd,fkd->nfk", xn, Wn) + b[: D * k].reshape(1, D, k)
            tgt = (tokens[:, :D] - self.offsets[:D]).clamp(0, k - 1)
            nll = F.cross_entropy(logits.reshape(-1, k).float(), tgt.reshape(-1), reduction="none").reshape(N, D)
            out[:, :D] = nll * mask[:, :D]
        for f in range(D, Fn):
            m = mask[:, f]
            if not bool(m.any()):
                continue
            o, sz = int(self.offsets[f]), self.sizes[f]
            xf = (x[:, f] if per_field else x)[m]
            lg = xf @ W[o : o + sz].T + b[o : o + sz]
            out[m, f] = F.cross_entropy(lg.float(), (tokens[m, f] - o).clamp(0, sz - 1), reduction="none")
        return out

    def next_event_nll(self, states, tokens, target_mask):
        """per-token nll of event t's tokens from states[:, t] (the state before it). tokens
        [B, L, F], target_mask [B, L, F] bool (observed and valid) -> nll [B, L, F] (0 elsewhere)"""
        B, L, Fn = tokens.shape
        x = states[:, :L].reshape(B * L, -1)
        return self._grouped_nll(self.next_head, x, tokens.reshape(B * L, Fn), target_mask.reshape(B * L, Fn)).reshape(B, L, Fn)

    def masked_nll(self, states, tok_out, targets, masked):
        """tokens hidden from the event encoder, predicted from their token output plus the
        state after the event. targets [B, L, F] the true ids, masked [B, L, F] bool -> mean nll"""
        B, L, Fn = targets.shape
        if not bool(masked.any()):
            return targets.new_zeros((), dtype=torch.float32)
        x = (tok_out + states[:, 1 : L + 1].unsqueeze(2)).reshape(B * L, Fn, -1)
        nll = self._grouped_nll(self.mlm_head, x, targets.reshape(B * L, Fn), masked.reshape(B * L, Fn))
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
    states, _ = model(tokens, dt_log, valid)
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


class TxfmScorer:
    """the pretrained encoder as the booster, the post stage and the server use it: one
    feature block per row - the state after the row (d_model), the per-field surprise and the
    mean surprise - computed from the row and its customer history. loads runs/<run>/txfm.pt."""

    def __init__(self, model: TransactionEncoder, tok: Tokeniser, device="cpu", info: dict | None = None):
        self.model, self.tok, self.device, self.info = model, tok, torch.device(device), info or {}
        self.window = int(model.window)
        self.d_model = int(model.d_model)
        self.names = feature_names(tok, self.d_model)
        self.dim = len(self.names)

    @classmethod
    def load(cls, run_dir: str | Path, device="cpu"):
        p = Path(run_dir)
        p = p if p.suffix == ".pt" else p / "txfm.pt"
        if not p.exists():
            return None
        model, tok, ck = load_txfm(p, device)
        return cls(model, tok, device, info={"extra": ck.get("extra", {}), "train_config": ck.get("train_config", {})})

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
        h, per_field, total = readout(self.model, self.tok, toks, dt, valid)
        return np.concatenate([h.cpu().numpy(), per_field.cpu().numpy(), total.cpu().numpy()[:, None]], axis=1).astype(np.float32)

    def features_split(self, split, idx, seq, split_name, device=None, src_override=None, batch_size=8192) -> np.ndarray:
        """[len(idx), dim] in the order of idx, histories from the sequence index (rows are
        visited in customer order, like score_split, and put back)"""
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

    def explain(self, feats_row: np.ndarray, labels: list[str], top: int = 5) -> dict:
        """the receipt's view of one feature row: mean surprise and the most surprising fields"""
        s = feats_row[self.d_model : self.d_model + len(labels)]
        order = [int(j) for j in np.argsort(-np.nan_to_num(s, nan=-1.0)) if s[j] == s[j]][:top]
        return {"surprise": float(feats_row[-1]) if feats_row[-1] == feats_row[-1] else None,
                "n_fields": int(np.isfinite(s).sum()),
                "most_surprising": [{"field": labels[j], "nll": float(s[j])} for j in order]}
