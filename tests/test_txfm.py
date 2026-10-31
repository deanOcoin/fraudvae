"""v6: the transaction-sequence encoder - the tokeniser, the model's causality and losses, a
short pretraining on the synthetic prepared set, and a booster that reads the encoder's block."""
import json

import numpy as np
import pytest
import torch

from fraudvae.model.txfm import Tokeniser, TransactionEncoder, TxfmScorer, mask_tokens, readout, windows_from_history


def _tok(D=5, C=3, S=11, n=300):
    rng = np.random.default_rng(0)
    num = rng.normal(size=(n, D)).astype(np.float32)
    mask = (rng.random((n, D)) > 0.2).astype(np.float32)
    return Tokeniser.fit(num, mask, [f"n{i}" for i in range(D)], [4, 7, 3], S, n_bins=8, dt_bins=10)


def test_tokeniser_fields_own_disjoint_slices():
    tok = _tok()
    assert tok.n_fields == 5 + 3 + 2
    assert tok.vocab == sum(tok.sizes) and tok.mask_id == tok.vocab
    assert list(tok.offsets) == list(np.concatenate([[0], np.cumsum(tok.sizes)[:-1]]))
    B = 6
    num = torch.randn(B, 5)
    mask = torch.ones(B, 5)
    mask[0, 0] = 0
    toks = tok.encode(num, mask, torch.randint(0, 3, (B, 3)), torch.full((B,), 3), torch.tensor([-1.0, 0.0, 60.0, 3600.0, 86400.0, 3e7]))
    assert toks.shape == (B, tok.n_fields)
    # an unobserved numeric is the first id of its field; observed ones never are
    assert toks[0, 0] == tok.offsets[0] and (toks[1:, 0] > tok.offsets[0]).all()
    # every token sits inside its field's slice
    for f in range(tok.n_fields):
        assert (toks[:, f] >= tok.offsets[f]).all() and (toks[:, f] < tok.offsets[f] + tok.sizes[f]).all()
    # no previous event -> dt token 0 of its field; longer gaps -> larger buckets, monotone
    dt = toks[:, -1] - tok.offsets[-1]
    assert dt[0] == 0 and (dt[1:].diff() >= 0).all() and dt[-1] <= tok.dt_bins
    assert tok.observed(toks)[0, 0].item() is False and tok.observed(toks)[1, 0].item() is True


def test_encoder_is_causal_and_losses_are_finite():
    torch.manual_seed(0)
    tok = _tok()
    m = TransactionEncoder(tok.vocab, tok.n_fields, tok.sizes, tok.D, d_model=32, event_layers=1, history_layers=2, heads=4, window=6).eval()
    B, L = 3, 6
    toks = tok.encode(torch.randn(B, L, 5), (torch.rand(B, L, 5) > 0.3).float(), torch.randint(0, 3, (B, L, 3)),
                      torch.randint(1, 12, (B, L)), torch.rand(B, L) * 1e5)
    valid = torch.ones(B, L, dtype=torch.bool)
    valid[0, :2] = False
    dt = torch.rand(B, L) * 10
    with torch.no_grad():
        states, tok_out = m(toks, dt, valid)
        assert states.shape == (B, L + 1, 32) and tok_out.shape == (B, L, tok.n_fields, 32)
        toks2 = toks.clone()
        toks2[:, 4, 0] = tok.offsets[0] + 3
        s2, _ = m(toks2, dt, valid)
    # states[:, t] has seen events < t only: changing event 4 leaves states 0..4 alone and moves 5
    assert torch.allclose(states[:, :5], s2[:, :5], atol=1e-5)
    assert not torch.allclose(states[:, 5], s2[:, 5])
    obs = tok.observed(toks) & valid.unsqueeze(-1)
    nll = m.next_event_nll(states, toks, obs)
    assert nll.shape == toks.shape and torch.isfinite(nll).all() and (nll[~obs] == 0).all()
    corrupted, masked = mask_tokens(toks, valid, tok.mask_id, generator=torch.Generator().manual_seed(1))
    assert not masked[..., -2:].any() and not masked[0, :2].any()
    assert (corrupted[masked] == tok.mask_id).all()
    with torch.no_grad():
        _, to2 = m(corrupted, dt, valid)
        mlm = m.masked_nll(states, to2, toks, masked)
    assert torch.isfinite(mlm)
    # the readout scores the last event from the state before it
    h, per_field, total = readout(m, tok, toks, dt, valid)
    assert h.shape == (B, 32) and per_field.shape == (B, tok.n_fields - 2) and total.shape == (B,)
    assert torch.isnan(per_field[~obs[:, -1, :-2]]).all() and torch.isfinite(per_field[obs[:, -1, :-2]]).all()


def test_windows_from_history_match_the_v5_tensors():
    tok = _tok()
    B, L = 2, 3
    secs = torch.tensor([[3e5, 2e5, 1e4], [5e3, 4e3, 3e3]])
    h = (torch.randn(B, L, 5), torch.ones(B, L, 5), torch.randint(0, 3, (B, L, 3)), torch.log1p(secs) / 10,
         torch.tensor([[True, True, True], [False, True, True]]))
    toks, dt_log, valid = windows_from_history(tok, torch.randn(B, 5), torch.ones(B, 5), torch.randint(0, 3, (B, 3)), torch.full((B,), 2), h)
    assert toks.shape == (B, L + 1, tok.n_fields) and valid.tolist() == [[True] * 4, [False, True, True, True]]
    # dt of the current row = seconds since the newest past row; the oldest valid row has no previous
    assert abs(float(dt_log[0, 3]) - np.log1p(1e4)) < 1e-3 and abs(float(dt_log[0, 1]) - np.log1p(1e5)) < 1e-3
    assert float(dt_log[0, 0]) == 0.0 and float(dt_log[1, 1]) == 0.0
    t0, d0, v0 = windows_from_history(tok, torch.randn(B, 5), torch.ones(B, 5), torch.randint(0, 3, (B, 3)), torch.full((B,), 2), None)
    assert t0.shape == (B, 1, tok.n_fields) and v0.all() and (d0 == 0).all()


@pytest.fixture(scope="session")
def pretrained_txfm(prepared_set):
    from fraudvae.config import load_config
    from fraudvae.pretrain import pretrain

    cfg = load_config("txfm_small")
    cfg.device = "cpu"
    cfg.txfm.d_model = 32
    cfg.txfm.event_layers = 1
    cfg.txfm.history_layers = 1
    cfg.txfm.window = 8
    cfg.txfm.steps = 40
    cfg.txfm.batch_size = 32
    cfg.txfm.warmup = 5
    cfg.txfm.eval_every = 20
    cfg.txfm.eval_windows = 128
    return pretrain(cfg, prepared=prepared_set, name="test-txfm")


def test_pretrain_writes_a_loadable_encoder(pretrained_txfm, prepared_set):
    run_dir = pretrained_txfm
    assert (run_dir / "txfm.pt").exists() and (run_dir / "txfm.json").exists()
    info = json.loads((run_dir / "txfm.json").read_text())
    assert info["steps"] == 40 and info["best_val_next_nll"] is not None and info["best_val_next_nll"] > 0
    metrics = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()]
    assert len(metrics) == 2 and all(np.isfinite(m["val_next_nll"]) for m in metrics)
    status = json.loads((run_dir / "status.json").read_text())
    assert status["state"] == "done"
    sc = TxfmScorer.load(run_dir, device="cpu")
    assert sc.window == 8 and sc.d_model == 32 and sc.dim == 32 + (sc.tok.n_fields - 2) + 1
    # features from the prepared set, with and without the sequence index
    from fraudvae.model.dataset import PreparedSet

    data = PreparedSet(prepared_set, splits=("val",))
    va = data["val"]
    idx = np.arange(min(300, len(va)))
    f = sc.features_split(va, idx, data.seq, "val", device="cpu")
    assert f.shape == (len(idx), sc.dim) and np.isfinite(f[:, :32]).all() and np.isfinite(f[:, -1]).all()
    f0 = sc.features_split(va, idx, None, None, device="cpu")
    assert f0.shape == f.shape
    labels = sc.field_labels(data.meta["categorical"])
    ex = sc.explain(f[0], labels)
    assert ex["surprise"] is not None and len(ex["most_surprising"]) <= 5 and all(e["field"] in labels for e in ex["most_surprising"])

