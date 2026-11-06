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


def test_block_routed_to_named_sources(pretrained_txfm, prepared_set, tmp_path):
    """v7: with txfm.sources the block is NaN for every row of a source it is not routed to and
    unchanged elsewhere, on the block pass and the window path alike; the booster records the
    routing by name and a scorer loaded from that run applies the same one"""
    import shutil

    from fraudvae.config import load_config
    from fraudvae.data import schema
    from fraudvae.model.booster import train_booster
    from fraudvae.model.dataset import PreparedSet
    from fraudvae.model.txfm import TxfmScorer

    data = PreparedSet(prepared_set, splits=("train", "val"), in_memory=True)
    va = data["val"]
    src_all = np.asarray(va.src)
    # the split lays sources out one after another: take rows of both
    idx = np.concatenate([np.flatnonzero(src_all == schema.SOURCE_INDEX["fdh"])[:200],
                          np.flatnonzero(src_all == schema.SOURCE_INDEX["capone"])[:200]])
    src = src_all[idx]
    assert {int(s) for s in np.unique(src)} == {schema.SOURCE_INDEX["fdh"], schema.SOURCE_INDEX["capone"]}
    sc = TxfmScorer.load(pretrained_txfm, device="cpu")
    plain = sc.features_split(va, idx, data.seq, "val", batch_size=16)
    sc.set_sources(["fdh"])
    assert sc.sources == {schema.SOURCE_INDEX["fdh"]} and sc.source_names == ["fdh"]
    routed = sc.features_split(va, idx, data.seq, "val", batch_size=16)
    fdh = src == schema.SOURCE_INDEX["fdh"]
    assert fdh.any() and (~fdh).any()
    assert np.isnan(routed[~fdh]).all() and np.array_equal(routed[fdh], plain[fdh], equal_nan=True)
    one = sc.features_windows(va, idx, data.seq, "val")
    assert np.isnan(one[~fdh]).all() and np.isfinite(one[fdh, : sc.d_model]).all()
    with pytest.raises(SystemExit):
        sc.set_sources(["no_such_bank"])
    assert sc.set_sources(None).sources is None and sc.set_sources([]).sources is None
    # the booster records the routing, and a scorer loaded beside that booster applies it
    sc.set_sources(["fdh"])
    cfg = load_config("smoke")
    cfg.boost.txfm_features, cfg.boost.max_rounds, cfg.boost.early_stopping = True, 20, 10
    run_dir = tmp_path / "routed"
    run_dir.mkdir()
    shutil.copy2(pretrained_txfm / "txfm.pt", run_dir / "txfm.pt")
    info = train_booster(run_dir, prepared_set, cfg.boost, log=lambda m: None, seed=1, model=None, device="cpu", txfm=sc)
    assert info["txfm_sources"] == ["fdh"] and info["txfm_dim"] == sc.dim
    again = TxfmScorer.load(run_dir, device="cpu")
    assert again.sources == {schema.SOURCE_INDEX["fdh"]} and again.source_names == ["fdh"]
    assert np.isnan(again.features_split(va, idx, data.seq, "val", batch_size=16)[~fdh]).all()


def test_booster_reads_a_compressed_block(pretrained_txfm, prepared_set, tmp_path):
    """v7: boost.txfm_pca hands the trees k principal components of the encoder's block plus the
    mean surprise instead of its hundreds of columns; the compression is fitted by the shared
    booster, saved in the run, read by the specialists, and applied to the raw block wherever
    a booster is handed one (score_split, the server). a row with no block stays NaN"""
    import shutil

    from fraudvae.config import load_config
    from fraudvae.model.booster import BlockPCA, Booster, Specialists, train_booster, train_specialists
    from fraudvae.model.dataset import PreparedSet
    from fraudvae.model.scoring import score_split
    from fraudvae.model.txfm import TxfmScorer
    from fraudvae.model.vae import TabularVAE

    cfg = load_config("smoke")
    cfg.boost.txfm_features, cfg.boost.txfm_pca, cfg.boost.max_rounds, cfg.boost.early_stopping = True, 4, 20, 10
    sc = TxfmScorer.load(pretrained_txfm, device="cpu")
    run_dir = tmp_path / "pca"
    run_dir.mkdir()
    shutil.copy2(pretrained_txfm / "txfm.pt", run_dir / "txfm.pt")
    info = train_booster(run_dir, prepared_set, cfg.boost, log=lambda m: None, seed=1, model=None, device="cpu", txfm=sc)
    assert info["txfm_pca"] == 4 and info["txfm_dim"] == 5 and info["txfm_in_dim"] == sc.dim
    assert info["txfm_names"] == ["txfm_pc0", "txfm_pc1", "txfm_pc2", "txfm_pc3", "txfm_surprise"] and info["features"][-5:] == info["txfm_names"]
    assert 0 < info["txfm_pca_explained"] <= 1 and (run_dir / BlockPCA.FILE).exists()
    bst = Booster.load(run_dir)
    assert bst.pca is not None and bst.pca.k == 4 and bst.pca.in_dim == sc.dim and bst.txfm_dim == 5
    data = PreparedSet(prepared_set, splits=("val",))
    va = data["val"]
    b = np.arange(60)
    raw = sc.features_split(va, b, data.seq, "val", device="cpu")
    comp = bst.pca.transform(raw)
    assert comp.shape == (60, 5) and np.isfinite(comp).all() and np.array_equal(comp[:, -1], raw[:, -1])
    raw_nan = raw.copy()
    raw_nan[3] = np.nan
    assert np.isnan(bst.pca.transform(raw_nan)[3]).all() and np.isfinite(bst.pca.transform(raw_nan)[4]).all()
    # raw in or compressed in, the same logit; and the scoring path (which also hands the booster
    # the rows' wide block) agrees with the direct call given the same block
    z_raw = bst.logit(va.num[b], va.mask[b], va.cat[b], va.src[b], wide=va.wide(b), txfm=raw)
    z_comp = bst.logit(va.num[b], va.mask[b], va.cat[b], va.src[b], wide=va.wide(b), txfm=comp)
    assert np.allclose(z_raw, z_comp) and np.isfinite(z_raw).all()
    vae = TabularVAE(va.num.shape[1], [c["size"] for c in data.meta["categorical"]], 11, latent_dim=4, hidden=(16,)).eval()
    r = score_split(vae, va, batch_size=32, n_samples=1, idx=b, booster=bst, txfm=sc, seq=data.seq, split_name="val", keep_features=False)
    assert np.allclose(r["booster"], z_raw, atol=1e-4)
    # the specialists read the same five columns from the file the booster wrote
    summary = train_specialists(run_dir, prepared_set, cfg.boost, log=lambda m: None, seed=1, min_fraud=10, model=None, device="cpu", txfm=sc)
    assert summary["sources"] and summary["txfm_dim"] == 5
    sp = Specialists.load(run_dir)
    assert sp is not None and all(b_.pca is not None and b_.txfm_dim == 5 for b_ in sp.boosters.values())
    local = sp.logit(va.num[b], va.mask[b], va.cat[b], va.src[b], txfm=raw)
    assert np.isfinite(local).any()
    # the compression object alone: k is capped by what the sample can carry
    small = BlockPCA.fit(raw[:3], 10, seed=0)
    assert small.k <= 2 and small.transform(raw[:3]).shape == (3, small.k + 1)


def test_supervised_head_on_the_block(pretrained_txfm, prepared_set, tmp_path):
    """v7: one head per source with enough labelled fraud, on the encoder's block; its logit is
    the txsup signal - a number on that source's rows with a block, NaN elsewhere - and
    score_split threads it through beside the block"""
    import shutil

    from fraudvae.data import schema
    from fraudvae.model.dataset import PreparedSet
    from fraudvae.model.scoring import EXTRA_SIGNALS, score_split
    from fraudvae.model.txfm import TxfmScorer
    from fraudvae.model.txfmhead import TxfmHead, load_txfmhead, train_txfmhead
    from fraudvae.model.vae import TabularVAE

    assert "txsup" in EXTRA_SIGNALS
    sc = TxfmScorer.load(pretrained_txfm, device="cpu")
    run_dir = tmp_path / "head"
    run_dir.mkdir()
    shutil.copy2(pretrained_txfm / "txfm.pt", run_dir / "txfm.pt")
    quiet = lambda m: None  # noqa: E731
    # the synthetic sources have about 230 labelled fraud rows each in train: below the default floor
    assert train_txfmhead(run_dir, prepared_set, sc, log=quiet, min_fraud=300, epochs=2, seed=1) is None
    assert not (run_dir / "txfmhead.pt").exists()
    info = train_txfmhead(run_dir, prepared_set, sc, log=quiet, max_legit_per_source=3000, val_legit_per_source=1000, min_fraud=50,
                          epochs=4, patience=2, seed=1)
    assert info is not None and set(info["source_names"]) == {"fdh", "capone"} and info["signal"] == "txsup"
    for v in info["sources"].values():
        assert 0 <= v["auroc"] <= 1 and v["best_epoch"] >= 1 and v["train_fraud"] >= 50
    head = load_txfmhead(run_dir)
    assert isinstance(head, TxfmHead) and set(head.sources) == {schema.SOURCE_INDEX["fdh"], schema.SOURCE_INDEX["capone"]}
    data = PreparedSet(prepared_set, splits=("val",), in_memory=True)
    va = data["val"]
    src_all = np.asarray(va.src)
    idx = np.concatenate([np.flatnonzero(src_all == schema.SOURCE_INDEX["fdh"])[:40], np.flatnonzero(src_all == schema.SOURCE_INDEX["capone"])[:40]])
    feats = sc.features_split(va, idx, data.seq, "val", device="cpu")
    lg = head.logit(feats, src_all[idx])
    assert lg.shape == (80,) and np.isfinite(lg).all()
    # a source without a head, or a row without a block, gets no number
    assert np.isnan(head.logit(feats, np.zeros(80, dtype=np.int64))).all()
    blank = feats.copy()
    blank[:5] = np.nan
    lg_blank = head.logit(blank, src_all[idx])
    assert np.isnan(lg_blank[:5]).all() and np.allclose(lg_blank[5:], lg[5:])
    # through score_split, beside the block; in the generic context (source blanked) the signal is absent
    vae = TabularVAE(va.num.shape[1], [c["size"] for c in data.meta["categorical"]], 11, latent_dim=4, hidden=(16,)).eval()
    r = score_split(vae, va, batch_size=32, n_samples=1, idx=idx, txfm=sc, txhead=head, seq=data.seq, split_name="val", keep_features=False)
    assert r["txsup"].shape == (80,) and np.allclose(r["txsup"], lg, atol=1e-5)
    rg = score_split(vae, va, batch_size=32, n_samples=1, idx=idx, txfm=sc, txhead=head, seq=data.seq, split_name="val", keep_features=False,
                     src_override=0)
    assert np.isnan(rg["txsup"]).all()


def test_pretrain_snapshots_and_probes(prepared_set):
    """v7: txfm.save_every writes a loadable checkpoint on a step cadence whatever the validation
    says, and txfm.probe logs at every eval how linearly separable one source's fraud is in the
    frozen state - the downstream curve a pretraining is gated on"""
    from fraudvae.config import load_config
    from fraudvae.pretrain import linear_probe, pretrain

    cfg = load_config("txfm_small")
    cfg.device = "cpu"
    cfg.txfm.d_model, cfg.txfm.event_layers, cfg.txfm.history_layers, cfg.txfm.window = 32, 1, 1, 8
    cfg.txfm.steps, cfg.txfm.batch_size, cfg.txfm.warmup, cfg.txfm.eval_every, cfg.txfm.eval_windows = 40, 32, 5, 20, 128
    cfg.txfm.save_every, cfg.txfm.probe, cfg.txfm.probe_windows = 20, "fdh", 256
    run_dir = pretrain(cfg, prepared=prepared_set, name="test-txfm-probe")
    for step in (20, 40):
        p = run_dir / f"txfm_step{step}.pt"
        assert p.exists()
        sc = TxfmScorer.load(p, device="cpu")
        assert sc.info["extra"]["snapshot"] is True and sc.info["extra"]["step"] == step and sc.window == 8
    metrics = [json.loads(ln) for ln in (run_dir / "metrics.jsonl").read_text().splitlines()]
    assert [m["step"] for m in metrics] == [20, 40] and all(m["probe_source"] == "fdh" and m["probe_windows"] > 0 for m in metrics)
    assert all(np.isfinite(m["probe_auroc"]) and 0 <= m["probe_auroc"] <= 1 for m in metrics)
    # the probe itself: a separable representation reads high, a thin class reads NaN
    rng = np.random.default_rng(0)
    y = (rng.random(600) < 0.3).astype(int)
    x = rng.normal(size=(600, 8))
    x[:, 0] += 3.0 * y
    assert linear_probe(x, y, seed=0) > 0.95
    assert np.isnan(linear_probe(x[:30], np.r_[np.ones(5, dtype=int), np.zeros(25, dtype=int)], seed=0))
    with pytest.raises(SystemExit):
        cfg.txfm.probe = "no_such_bank"
        pretrain(cfg, prepared=prepared_set, name="test-txfm-probe-bad")


def test_permuted_block_is_a_shuffle_within_each_source(pretrained_txfm, prepared_set, tmp_path):
    """v7: the dilution control. with permute on, the block pass hands every row another row's
    block of the same source: within a source the rows are a permutation of the plain block's
    (the same marginals), they differ from it, and the same call gives the same shuffle; the gate
    and the routing still read the row's own place; the booster records the choice and a scorer
    loaded beside it applies the same one"""
    import shutil
    from collections import Counter

    from fraudvae.config import load_config
    from fraudvae.data import schema
    from fraudvae.model.booster import train_booster
    from fraudvae.model.dataset import PreparedSet
    from fraudvae.model.txfm import TxfmScorer

    data = PreparedSet(prepared_set, splits=("train", "val"), in_memory=True)
    va = data["val"]
    src_all = np.asarray(va.src)
    idx = np.concatenate([np.flatnonzero(src_all == schema.SOURCE_INDEX["fdh"])[:200],
                          np.flatnonzero(src_all == schema.SOURCE_INDEX["capone"])[:200]])
    src = src_all[idx]
    sc = TxfmScorer.load(pretrained_txfm, device="cpu")
    assert sc.permute is False
    plain = sc.features_split(va, idx, data.seq, "val", batch_size=16)
    sc.permute = True
    shuffled = sc.features_split(va, idx, data.seq, "val", batch_size=16)
    assert shuffled.shape == plain.shape and not np.array_equal(shuffled, plain, equal_nan=True)
    assert np.array_equal(shuffled, sc.features_split(va, idx, data.seq, "val", batch_size=16), equal_nan=True)
    rows = lambda x: Counter(np.nan_to_num(r, nan=-1e9).tobytes() for r in x)   # noqa: E731
    for s in np.unique(src):
        m = src == s
        assert rows(shuffled[m]) == rows(plain[m])
    # the gate and the routing still apply to the row's own history and source
    sc.gate_first = True
    sc.set_sources(["fdh"])
    both = sc.features_split(va, idx, data.seq, "val", batch_size=16)
    fdh = src == schema.SOURCE_INDEX["fdh"]
    hist = data.seq.positions("val", idx) - np.asarray(data.seq.start)[data.seq.global_ids("val", idx)]
    assert np.isnan(both[~fdh]).all() and np.isnan(both[hist == 0]).all()
    assert rows(both[fdh & (hist > 0)]) <= rows(plain[fdh])
    sc.gate_first = False
    sc.set_sources(None)
    # the booster records the control, and a scorer loaded beside that booster applies it
    cfg = load_config("smoke")
    cfg.boost.txfm_features, cfg.boost.max_rounds, cfg.boost.early_stopping = True, 20, 10
    run_dir = tmp_path / "permuted"
    run_dir.mkdir()
    shutil.copy2(pretrained_txfm / "txfm.pt", run_dir / "txfm.pt")
    info = train_booster(run_dir, prepared_set, cfg.boost, log=lambda m: None, seed=1, model=None, device="cpu", txfm=sc)
    assert info["txfm_permuted"] is True
    again = TxfmScorer.load(run_dir, device="cpu")
    assert again.permute is True
    assert np.array_equal(again.features_split(va, idx, data.seq, "val", batch_size=16), shuffled, equal_nan=True)


def test_finetune_exports_an_encoder_with_only_its_top_layers_changed(pretrained_txfm, prepared_set, tmp_path):
    """v7: txfm-finetune writes a txfm.pt in the pretraining format that TxfmScorer loads and the
    block pass reads unchanged; only the unfrozen history layers and their norm differ from the
    base weights; the summary carries per-source validation numbers and the base checkpoint's sha"""
    from fraudvae.model.dataset import PreparedSet
    from fraudvae.model.txfm import load_txfm
    from fraudvae.model.txfmft import FILE, finetune

    base, _, _ = load_txfm(pretrained_txfm / "txfm.pt", "cpu")
    out = finetune(pretrained_txfm, prepared_set, ["fdh", "capone"], run_dir=tmp_path / "ft", device="cpu", steps=20, batch=8,
                   eval_every=10, legit_ratio=3, max_legit=200, log=lambda m: None)
    assert out == tmp_path / "ft" and (out / "txfm.pt").exists() and (out / FILE).exists()
    sc = TxfmScorer.load(out, device="cpu")
    assert sc is not None and sc.window == base.window and sc.d_model == base.d_model
    before, after = base.state_dict(), sc.model.state_dict()
    changed = [k for k in before if not torch.equal(before[k], after[k])]
    assert changed, "the fine-tune moved nothing"
    assert all(k.startswith("history.layers.") or k.startswith("hist_norm.") for k in changed), changed
    frozen = [k for k in before if k not in changed]
    assert any(k.startswith("tok.") for k in frozen) and any(k.startswith("event") for k in frozen) and any(k.startswith("next_head") for k in frozen)
    _, _, ck = load_txfm(out / "txfm.pt", "cpu")
    assert ck["extra"]["finetuned_sources"] == ["fdh", "capone"] and len(ck["extra"]["base_checkpoint_sha256"]) == 64
    assert ck["extra"]["unfrozen_history_layers"] == 1   # the fixture's encoder has one history layer
    summary = json.loads((out / FILE).read_text())
    assert set(summary["sources"]) == {"fdh", "capone"} and all(0 <= v["auroc"] <= 1 and v["fraud"] > 0 for v in summary["sources"].values())
    assert summary["steps"] == 20 and len(summary["history"]) == 2 and summary["base_sha256"] == ck["extra"]["base_checkpoint_sha256"]
    data = PreparedSet(prepared_set, splits=("val",))
    f = sc.features_split(data["val"], np.arange(50), data.seq, "val")
    assert f.shape == (50, sc.dim) and np.isfinite(f[:, : sc.d_model]).all()
