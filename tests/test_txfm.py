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


def test_packing_reads_only_a_sources_fields():
    """with per-source field lists the event encoder sees only the fields a source observes: a
    token outside the list cannot move the states, a list of every field is a no-op, the losses
    index the packed layout, and the blanked source gets every field"""
    torch.manual_seed(0)
    tok = _tok()
    F = tok.n_fields
    tok.source_fields = {3: [0, 1, 5], 4: [2, 6, 7]}
    tok.source_fields = {k: sorted(set(v) | {F - 2, F - 1}) for k, v in tok.source_fields.items()}
    m = TransactionEncoder(tok.vocab, F, tok.sizes, tok.D, d_model=32, event_layers=1, history_layers=2, heads=4, window=6).eval()
    B, L = 4, 6
    toks = tok.encode(torch.randn(B, L, 5), torch.ones(B, L, 5), torch.randint(0, 3, (B, L, 3)), torch.full((B, L), 3), torch.rand(B, L) * 1e5)
    valid = torch.ones(B, L, dtype=torch.bool)
    dt = torch.rand(B, L)
    fields = tok.fields_for(toks)
    assert fields.tolist() == [0, 1, 5, F - 2, F - 1]
    with torch.no_grad():
        s1, to1 = m(toks, dt, valid, fields)
        assert to1.shape == (B, L, 5, 32)
        toks2 = toks.clone()
        toks2[:, 2, 3] = tok.offsets[3] + 2          # a field the source never observes
        s2, _ = m(toks2, dt, valid, fields)
        assert torch.allclose(s1, s2)
        toks3 = toks.clone()
        toks3[:, 2, 1] = (toks3[:, 2, 1] - tok.offsets[1] + 1) % tok.sizes[1] + tok.offsets[1]   # one the source does observe
        s3, _ = m(toks3, dt, valid, fields)
        assert not torch.allclose(s1[:, 3:], s3[:, 3:])
        # every field listed = no packing
        s_all, to_all = m(toks, dt, valid, torch.arange(F))
        s_none, to_none = m(toks, dt, valid, None)
        assert torch.allclose(s_all, s_none) and torch.allclose(to_all, to_none)
    corrupted, masked = mask_tokens(toks, valid, tok.mask_id, generator=torch.Generator().manual_seed(2))
    with torch.no_grad():
        st, to = m(corrupted, dt, valid, fields)
        mlm = m.masked_nll(st, to, toks, masked, fields)
        full = m.masked_nll(*m(corrupted, dt, valid, None), toks, masked, None)
    assert torch.isfinite(mlm) and torch.isfinite(full)
    # a batch that mixes the two sources reads the union; the blanked source reads everything
    mixed = toks.clone()
    mixed[:2, :, F - 2] = tok.offsets[F - 2] + 4
    assert tok.fields_for(mixed).tolist() == sorted({0, 1, 2, 5, 6, 7, F - 2, F - 1})
    blank = toks.clone()
    blank[0, :, F - 2] = tok.offsets[F - 2]
    assert tok.fields_for(blank) is None
    h, per_field, total = readout(m, tok, toks, dt, valid)
    assert h.shape == (B, 32) and torch.isfinite(total).all()
    # the lists travel with the checkpoint
    cfg = tok.config()
    assert Tokeniser(**cfg).source_fields == tok.source_fields


def test_event_encoder_chunks_large_batches(monkeypatch):
    """the event encoder sees at most EVENT_CHUNK events per call; chunked and whole give the
    same vectors"""
    import fraudvae.model.txfm as T

    torch.manual_seed(0)
    tok = _tok()
    m = TransactionEncoder(tok.vocab, tok.n_fields, tok.sizes, tok.D, d_model=32, event_layers=1, history_layers=1, heads=4, window=6).eval()
    toks = tok.encode(torch.randn(5, 6, 5), torch.ones(5, 6, 5), torch.randint(0, 3, (5, 6, 3)), torch.full((5, 6), 3), torch.rand(5, 6))
    with torch.no_grad():
        whole, tw = m.encode_events(toks)
        monkeypatch.setattr(T, "EVENT_CHUNK", 7)
        chunked, tc = m.encode_events(toks)
    assert torch.allclose(whole, chunked, atol=1e-5) and torch.allclose(tw, tc, atol=1e-5)


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
    info = json.loads((run_dir / "run.json").read_text())
    assert info["packing"] is True and set(info["fields_per_source"]) == set(info["source_mix"]) and len(info["source_mix"]) == 2
    assert abs(sum(info["source_mix"].values()) - 1) < 1e-6
    summary = json.loads((run_dir / "txfm.json").read_text())
    assert summary["event_visits"] > 0 and summary["encoded_tokens_seen"] >= summary["observed_tokens_seen"] > 0
    assert (run_dir / "state.pt").exists()
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


def test_pretrain_resumes_from_its_state(pretrained_txfm, prepared_set):
    """--resume carries on from state.pt: more steps, the same tokeniser, the summary adds up"""
    from fraudvae.config import load_config
    from fraudvae.pretrain import pretrain

    before = json.loads((pretrained_txfm / "txfm.json").read_text())
    cfg = load_config("txfm_small")
    cfg.device = "cpu"
    cfg.txfm.d_model, cfg.txfm.event_layers, cfg.txfm.history_layers, cfg.txfm.window = 32, 1, 1, 8
    cfg.txfm.steps, cfg.txfm.batch_size, cfg.txfm.warmup, cfg.txfm.eval_every, cfg.txfm.eval_windows = 60, 32, 5, 20, 128
    run_dir = pretrain(cfg, prepared=prepared_set, run_dir=pretrained_txfm, resume=True)
    after = json.loads((run_dir / "txfm.json").read_text())
    assert after["first_step"] == 41 and after["steps"] == 60
    assert after["event_visits"] > before["event_visits"]
    lines = [json.loads(ln) for ln in (run_dir / "metrics.jsonl").read_text().splitlines()]
    assert [r["step"] for r in lines][-2:] == [40, 60] and all("val_sources" in r for r in lines)
    sc = TxfmScorer.load(run_dir, device="cpu")
    assert sc.tok.source_fields and sc.model.window == 8


def test_block_pass_matches_one_window_per_row(pretrained_txfm, prepared_set):
    """the block pass (several customers per segment behind a block mask, each row read off the
    position it sits at) gives the window path's features exactly when a segment starts
    window - 1 events before the row (stride 1), and the same shapes and finite values at the
    default stride, where a row sees at least three quarters of a window of its history"""
    from fraudvae.model.dataset import PreparedSet

    data = PreparedSet(prepared_set, splits=("val",), in_memory=True)
    sc = TxfmScorer.load(pretrained_txfm, device="cpu")
    val = data["val"]
    idx = np.random.default_rng(0).choice(len(val), 200, replace=False)
    one = sc.features_windows(val, idx, data.seq, "val")
    exact = sc.features_split(val, idx, data.seq, "val", stride=1, batch_size=16)
    assert exact.shape == one.shape == (200, sc.dim)
    assert np.array_equal(np.isnan(exact), np.isnan(one))
    # a row whose customer block starts inside the window sees the same events with the same
    # gaps in both paths; with a longer history the window path has no gap for its oldest event
    # (the server does not know the event before the window), the block pass has the real one
    hist = data.seq.positions("val", idx) - np.asarray(data.seq.start)[data.seq.global_ids("val", idx)]
    short = hist < sc.window - 1
    assert short.sum() >= 20
    _d = np.abs(np.nan_to_num(exact) - np.nan_to_num(one)).max(1)
    _src = np.asarray(val.src)[idx]
    _pos = data.seq.positions("val", idx)
    assert np.allclose(np.nan_to_num(exact[short]), np.nan_to_num(one[short]), atol=1e-4), np.nanmax(np.abs(exact[short] - one[short]))
    assert np.corrcoef(exact[:, -1], one[:, -1])[0, 1] > 0.99
    fast = sc.features_split(val, idx, data.seq, "val", batch_size=16)
    assert fast.shape == one.shape and np.isfinite(fast[:, -1]).all() and np.array_equal(np.isnan(fast), np.isnan(one))
    # the surprise of a row does not depend on which segment served it beyond its history length
    assert np.corrcoef(fast[:, -1], one[:, -1])[0, 1] > 0.9
    # the generic pass blanks the source and reads every field
    gen = sc.features_split(val, idx[:20], data.seq, "val", src_override=0, batch_size=4)
    assert gen.shape == (20, sc.dim) and np.isfinite(gen[:, -1]).all()
    # the cache: the whole split in one pass equals the block pass over every row, read back
    # for any idx; built in parts the segments start elsewhere (another valid placement of
    # each row inside its window), so that one is checked for agreement, not equality
    full = sc.features_split(val, np.arange(len(val)), data.seq, "val", batch_size=16)
    cache = pretrained_txfm / "cache-test"
    sc.cache_split(val, data.seq, "val", cache, batch_size=16)
    assert sc.merge_cache(cache, "val", len(val)) == cache / "txfm_val.npy"
    sc.cache_dir = cache
    cached = sc.features_split(val, idx, data.seq, "val")
    assert cached.shape == fast.shape and np.allclose(np.nan_to_num(cached), np.nan_to_num(full[idx]), atol=1e-4)
    assert np.array_equal(np.isnan(cached), np.isnan(fast))
    (cache / "txfm_val.npy").unlink()
    for part in range(3):
        sc.cache_split(val, data.seq, "val", cache, part=part, parts=3, batch_size=16)
    sc.merge_cache(cache, "val", len(val))
    parted = sc.features_split(val, idx, data.seq, "val")
    assert np.corrcoef(parted[:, -1], cached[:, -1])[0, 1] > 0.99 and np.array_equal(np.isnan(parted), np.isnan(cached))
    # a blanked source never reads the cache
    assert np.allclose(np.nan_to_num(sc.features_split(val, idx[:20], data.seq, "val", src_override=0, batch_size=4)), np.nan_to_num(gen), atol=1e-4)


def test_booster_reads_the_encoder_block(pretrained_txfm, prepared_set, tmp_path):
    from fraudvae.config import load_config
    from fraudvae.model.booster import Booster, train_booster
    from fraudvae.model.dataset import PreparedSet
    from fraudvae.model.scoring import score_split
    from fraudvae.model.txfm import TxfmScorer

    cfg = load_config("smoke")
    cfg.boost.txfm_features = True
    cfg.boost.max_rounds = 30
    cfg.boost.early_stopping = 10
    sc = TxfmScorer.load(pretrained_txfm, device="cpu")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    info = train_booster(run_dir, prepared_set, cfg.boost, log=lambda m: None, seed=1, model=None, device="cpu", txfm=sc)
    assert info["txfm_dim"] == sc.dim and info["features"][-sc.dim:] == sc.names
    bst = Booster.load(run_dir)
    assert bst.txfm_dim == sc.dim and bst.txfm_names == sc.names
    data = PreparedSet(prepared_set, splits=("val",))
    va = data["val"]
    b = np.arange(50)
    # without the block the booster still answers (NaN = "no encoder here"), with it the logits differ
    z0 = bst.logit(va.num[b], va.mask[b], va.cat[b], va.src[b])
    feats = sc.features_split(va, b, data.seq, "val", device="cpu")
    z1 = bst.logit(va.num[b], va.mask[b], va.cat[b], va.src[b], txfm=feats)
    assert z0.shape == z1.shape == (50,) and np.isfinite(z0).all() and np.isfinite(z1).all()
    # score_split threads the block through when given the scorer, and reports the surprise
    from fraudvae.model.vae import TabularVAE

    vae = TabularVAE(va.num.shape[1], [c["size"] for c in data.meta["categorical"]], 11, latent_dim=4, hidden=(16,)).eval()
    r = score_split(vae, va, batch_size=32, n_samples=1, idx=b, booster=bst, txfm=sc, seq=data.seq, split_name="val", keep_features=False)
    assert r["booster"].shape == (50,) and np.allclose(r["booster"], z1, atol=1e-4) and np.isfinite(r["txfm_surprise"]).all()


def test_merge_cache_keeps_row_ids_above_two_to_the_24(tmp_path):
    """the part files keep their row ids as integers: a 30M-row split once had its ids stored as
    float32, which is exact only below 16,777,216, and the merge indexed past the end"""
    from fraudvae.model.txfm import TxfmScorer

    n = 2 ** 24 + 3
    rows = np.arange(n, dtype=np.int64)
    feats = (rows % 1000).astype(np.float32)[:, None]
    np.save(tmp_path / "txfm_train.0of1.npy", feats)
    np.save(tmp_path / "txfm_train.0of1.rows.npy", rows)
    dest = TxfmScorer.merge_cache(tmp_path, "train", n)
    out = np.load(dest, mmap_mode="r")
    assert out.shape == (n, 1)
    assert float(out[2 ** 24 + 1, 0]) == float((2 ** 24 + 1) % 1000)
    assert float(out[n - 1, 0]) == float((n - 1) % 1000)
    assert not list(tmp_path.glob("*of*.npy"))


def test_gate_first_blanks_rows_without_history(pretrained_txfm, prepared_set):
    """with gate_first the block is NaN for a customer's first event (the encoder has seen no
    history, so its block only echoes the row's own fields) and unchanged for every other row,
    through the block pass, the cache path's gather and the server's window path alike"""
    from fraudvae.model.dataset import PreparedSet
    from fraudvae.model.txfm import TxfmScorer

    data = PreparedSet(prepared_set, splits=("train",), in_memory=True)
    tr = data["train"]
    idx = np.arange(min(300, len(tr)))
    sc = TxfmScorer.load(pretrained_txfm, device="cpu")
    plain = sc.features_split(tr, idx, data.seq, "train", batch_size=16)
    sc.gate_first = True
    gated = sc.features_split(tr, idx, data.seq, "train", batch_size=16)
    hist = data.seq.positions("train", idx) - np.asarray(data.seq.start)[data.seq.global_ids("train", idx)]
    first = hist == 0
    assert first.any() and (~first).any()
    assert np.isnan(gated[first]).all()
    assert np.array_equal(gated[~first], plain[~first], equal_nan=True)   # NaN = a field the source never observes
    one = sc.features_windows(tr, idx, data.seq, "train")
    assert np.isnan(one[first]).all()
    assert np.isfinite(one[~first, : sc.d_model]).all()   # the state is always finite; a surprise is NaN where the field is unobserved
