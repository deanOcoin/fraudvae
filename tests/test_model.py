import numpy as np
import pytest
import torch

from fraudvae.config import ModelConfig
from fraudvae.model.scoring import Scorer, score_batch
from fraudvae.model.vae import TabularVAE, build_model, load_checkpoint, save_checkpoint


def tiny_model(n_num=6, cat_sizes=(4, 3)):
    return TabularVAE(n_num=n_num, cat_sizes=list(cat_sizes), n_sources=8, latent_dim=4, hidden=(16, 16), cat_embed_dim=3, source_embed_dim=2)


def batch(n=10, n_num=6, cat_sizes=(4, 3), seed=0):
    g = torch.Generator().manual_seed(seed)
    num = torch.randn(n, n_num, generator=g)
    mask = (torch.rand(n, n_num, generator=g) > 0.3).float()
    cat = torch.stack([torch.randint(0, k, (n,), generator=g) for k in cat_sizes], dim=1)
    src = torch.randint(0, 9, (n,), generator=g)
    return num, mask, cat, src


def test_forward_shapes():
    m = tiny_model()
    num, mask, cat, src = batch()
    out = m(num, mask, cat, src)
    assert out["mu"].shape == (10, 4)
    assert out["num_mu"].shape == (10, 6)
    assert len(out["cat_logits"]) == 2 and out["cat_logits"][0].shape == (10, 4)
    nll_num, nll_cat = m.nll_terms(out, num, mask, cat)
    assert nll_num.shape == (10, 6) and nll_cat.shape == (10, 2)


def test_unobserved_entries_contribute_nothing():
    m = tiny_model()
    num, mask, cat, src = batch()
    mask[:, 0] = 0
    cat[:, 1] = 0
    out = m(num, mask, cat, src)
    nll_num, nll_cat = m.nll_terms(out, num, mask, cat)
    assert torch.all(nll_num[:, 0] == 0)
    assert torch.all(nll_cat[:, 1] == 0)
    assert torch.all(m.kl_terms(out["mu"], out["logvar"]).sum(1) >= 0)


def test_masked_value_does_not_change_output():
    # the encoder sees num * mask, so garbage behind a zero mask must be invisible
    m = tiny_model().eval()
    num, mask, cat, src = batch()
    mask[:, 2] = 0
    with torch.no_grad():
        a = m.encode(num, mask, cat, src)[0]
        num2 = num.clone()
        num2[:, 2] = 1e3
        b = m.encode(num2, mask, cat, src)[0]
    assert torch.allclose(a, b)


def test_checkpoint_roundtrip(tmp_path):
    m = tiny_model()
    meta = {"numeric": [{"name": f"f{i}"} for i in range(6)], "categorical": [{"size": 4}, {"size": 3}], "source_index": {"fdh": 1}}
    save_checkpoint(tmp_path / "m.pt", m, meta, {"x": 1}, extra={"epoch": 3})
    m2, ck = load_checkpoint(tmp_path / "m.pt")
    num, mask, cat, src = batch()
    with torch.no_grad():
        assert torch.allclose(m.eval().encode(num, mask, cat, src)[0], m2.encode(num, mask, cat, src)[0])
    assert ck["extra"]["epoch"] == 3 and ck["prepared_meta"]["source_index"] == {"fdh": 1}


def test_build_model_from_meta():
    meta = {"numeric": [{"name": f"f{i}"} for i in range(5)], "categorical": [{"size": 7}], "source_index": {"fdh": 1, "ieee": 8}}
    m = build_model(meta, ModelConfig(latent_dim=3, hidden=[8]))
    assert m.n_num == 5 and m.cat_sizes == [7] and m.latent_dim == 3


def test_score_batch_and_normalised_scorer():
    m = tiny_model().eval()
    num, mask, cat, src = batch(n=32)
    r = score_batch(m, num, mask, cat, src, n_samples=3)
    assert r["raw"].shape == (32,) and r["nll_num"].shape == (32, 6) and r["observed"].shape == (32, 8)
    assert np.all(r["cat_prob"] >= 0) and np.all(r["cat_prob"] <= 1)
    nll = np.concatenate([r["nll_num"], r["nll_cat"]], axis=1)
    base = {"generic": Scorer.fit_baselines(nll, r["kl"])}
    sc = Scorer(base, n_features=8)
    norm = sc.norm(r, src.numpy())
    assert norm.shape == (32,) and np.isfinite(norm).all()
    # a row identical to the baseline mean should score about zero
    z = sc.feature_z(r, 0)
    assert abs(z.mean()) < 1.0


def test_openmp_guard_only_objects_to_torchs_bundled_copy(tmp_path, monkeypatch):
    """the macOS crash is torch's own libomp next to another one. a symlinked (fixed)
    torch copy or a second runtime that isn't torch's must pass."""
    import os
    import sys
    import types

    from fraudvae.model import booster

    fake_torch = tmp_path / "torch"
    (fake_torch / "lib").mkdir(parents=True)
    bundled = fake_torch / "lib" / "libomp.dylib"
    bundled.write_bytes(b"not really a dylib")
    mod = types.ModuleType("torch")
    mod.__file__ = str(fake_torch / "__init__.py")
    monkeypatch.setitem(sys.modules, "torch", mod)
    brew = "/opt/homebrew/Cellar/libomp/23.1.3/lib/libomp.dylib"

    monkeypatch.setattr(booster, "_loaded_openmp_runtimes", lambda: [os.path.realpath(bundled), brew])
    with pytest.raises(booster.BoosterUnavailable, match="fix-omp"):
        booster._check_single_openmp()

    # only torch's copy loaded: fine
    monkeypatch.setattr(booster, "_loaded_openmp_runtimes", lambda: [os.path.realpath(bundled)])
    booster._check_single_openmp()

    # torch points at the homebrew copy (what scripts/macos_openmp.sh does): two paths, one file, fine
    bundled.unlink()
    target = tmp_path / "brew_libomp.dylib"
    target.write_bytes(b"shared")
    bundled.symlink_to(target)
    monkeypatch.setattr(booster, "_loaded_openmp_runtimes", lambda: [str(target), "/x/sklearn/.dylibs/libomp.dylib"])
    booster._check_single_openmp()

    # no torch in the process at all: nothing to referee
    monkeypatch.delitem(sys.modules, "torch")
    monkeypatch.setattr(booster, "_loaded_openmp_runtimes", lambda: ["/a/libomp.dylib", "/b/libomp.dylib"])
    booster._check_single_openmp()
