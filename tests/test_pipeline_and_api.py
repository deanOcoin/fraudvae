"""prepare -> train -> calibrate -> evaluate -> serve, on the synthetic corpus from conftest"""
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from fraudvae.paths import PREPARED_DIR


def test_prepare_outputs(prepared_set):
    meta = json.loads((PREPARED_DIR / prepared_set / "meta.json").read_text())
    assert meta["counts"]["train"] == 6000          # capped at 3000 per source
    assert meta["counts"]["val"] + meta["counts"]["test"] == 2 * 6000 * 0.3
    assert len(meta["numeric"]) == 95 and len(meta["categorical"]) == 24   # v5: 97 - hour/dow, 22 + hour/dow
    num = np.load(PREPARED_DIR / prepared_set / "train_num.npy", mmap_mode="r")
    mask = np.load(PREPARED_DIR / prepared_set / "train_mask.npy", mmap_mode="r")
    assert num.shape == (6000, 95)
    assert np.all(num[mask == 0] == 0)
    # v5: capone's raw columns as a wide block, sequence index for everyone
    assert list(meta["wide"]) == ["capone"]
    assert meta["wide"]["capone"]["columns"] == ["raw_a", "raw_b", "raw_cat"]
    wide = np.load(PREPARED_DIR / prepared_set / "train_wide_capone.npy")
    a, b = meta["wide"]["capone"]["offsets"]["train"]
    assert wide.shape == (b - a, 3)
    assert np.isnan(wide[:, 1]).any() and not np.isnan(wide[:, 0]).any()
    for f in ("seq_order.npy", "seq_pos.npy", "seq_start.npy", "train_cust.npy", "train_ts.npy"):
        assert (PREPARED_DIR / prepared_set / f).exists(), f
    # amount observed everywhere, credit limit nowhere in the synthetic data
    names = [f["name"] for f in meta["numeric"]]
    assert mask[:, names.index("amount")].all()
    assert not mask[:, names.index("credit_limit")].any()
    cat = np.load(PREPARED_DIR / prepared_set / "train_cat.npy")
    hour_col = [c["name"] for c in meta["categorical"]].index("hour")
    assert cat[:, hour_col].min() >= 1 and cat[:, hour_col].max() <= 24


def test_training_run_artifacts(trained_run):
    for f in ("model.pt", "metrics.jsonl", "calibration.json", "eval.json", "latent_map.npz", "run.json", "status.json",
              "booster.txt", "booster.json"):
        assert (trained_run / f).exists(), f
    ev = json.loads((trained_run / "eval.json").read_text())
    assert set(ev["sources"]) == {"fdh", "capone"}
    # planted pattern (x6 amounts at night) is easy, even 2 tiny epochs should get it
    assert ev["mean"]["vae"] > 0.7, ev["mean"]
    assert ev["mean"]["booster"] > 0.9 and ev["mean"]["blend"] > 0.9, ev["mean"]
    assert ev["mean_auroc"] == ev["mean"]["blend"]
    assert ev["sources"]["fdh"]["blend"]["auroc"] > 0.9 and ev["sources"]["capone"]["blend"]["auroc"] > 0.9
    cal = json.loads((trained_run / "calibration.json").read_text())
    assert cal["score_type"] in ("raw", "norm")
    assert "generic" in cal and "fpr_0.01" in cal["sources"]["fdh"]["thresholds"]
    assert cal["blend"]["n_features"] == len(cal["blend"]["coef"]) and cal["blend"]["with_booster"]
    assert set(cal["blend"]["sources"]) == {"fdh", "capone"}
    bj = json.loads((trained_run / "booster.json").read_text())
    # v5: [common numerics | common categoricals | source | wide block]
    assert bj["features"][-3:] == ["capone.raw_a", "capone.raw_b", "capone.raw_cat"] and bj["best_iteration"] >= 1
    assert bj["wide_names"] == ["capone.raw_a", "capone.raw_b", "capone.raw_cat"]
    st = json.loads((trained_run / "status.json").read_text())
    assert st["state"] == "finished"


@pytest.fixture(scope="module")
def client(trained_run):
    from fraudvae.server.app import app

    with TestClient(app) as c:
        yield c


def test_meta_and_schema(client):
    meta = client.get("/api/meta").json()
    assert meta["active_run"]["name"] == "test-run"
    schema = client.get("/api/schema").json()
    assert len(schema["numeric"]) == 95
    assert any(s["built"] for s in schema["sources"])


def test_datasets_endpoint(client):
    d = client.get("/api/datasets").json()
    assert d["total_rows"] == 12000
    built = [s for s in d["sources"] if s["built"]]
    assert {s["key"] for s in built} == {"fdh", "capone"}


def test_examples_and_scoring(client):
    ex = client.get("/api/examples?source=fdh&fraud=1").json()[0]
    assert ex["is_fraud"] == 1 and "amount" in ex["features"]
    r = client.post("/api/score", json={"source": "fdh", "features": ex["features"], "fpr_target": "fpr_0.01"}).json()
    for k in ("p_fraud", "flag", "legit_percentile", "novelty", "supervised", "blend", "contributions", "latent_xy", "score"):
        assert k in r, k
    assert r["verdict_from"] == "blend" and 0.0 <= r["p_fraud"] <= 1.0
    assert r["calibration"] == "fdh"
    assert 0.0 <= r["supervised"]["prob"] <= 1.0 and r["supervised"]["pushes"]
    zs = [c["z"] for c in r["contributions"]]
    assert zs == sorted(zs, reverse=True)
    assert r["flag"] is True   # planted fraud: x6 amount at night
    g = client.post("/api/score", json={"source": None, "features": {"amount": 50, "hour": 12}}).json()
    assert g["calibration"] == "generic" and g["observed_features"] == 2


def test_scoring_tolerates_junk(client):
    r = client.post("/api/score", json={"source": "nope", "features": {"amount": "abc", "channel": "spaceship", "hour": 99}})
    assert r.status_code == 200
    assert r.json()["observed_features"] == 0


def test_batch_scoring(client, tmp_path):
    import pandas as pd

    rows = client.get("/api/examples?source=capone&n=20").json()
    df = pd.DataFrame([{**r["features"], "source": r["source"], "is_fraud": r["is_fraud"]} for r in rows])
    p = tmp_path / "x.csv"
    df.to_csv(p, index=False)
    with open(p, "rb") as f:
        res = client.post("/api/score/batch", files={"file": ("x.csv", f, "text/csv")})
    assert res.status_code == 200
    summary = json.loads(res.headers["X-Summary"])
    assert summary["rows"] == 20 and "auroc" in summary
    header = res.text.splitlines()[0].split(",")
    assert {"p_fraud", "booster_prob", "novelty_score", "flag"} <= set(header)


def test_runs_listing_and_metrics(client):
    runs = client.get("/api/runs").json()
    assert runs["active"] == "test-run"
    me = [r for r in runs["runs"] if r["name"] == "test-run"][0]
    assert me["compatible"] and me["has_booster"]
    m = client.get("/api/runs/test-run/metrics").json()
    assert len(m["metrics"]) == 2 and m["eval"]["mean_auroc"] > 0.5
    lat = client.get("/api/runs/test-run/latent").json()
    assert len(lat["xy"]) == len(lat["y"]) > 0


def test_run_loading_edge_cases(client, trained_run):
    """unknown run names are a 404; a run folder with a broken status file is listed but not compatible"""
    assert client.get("/api/runs/no-such-run/metrics").status_code == 404
    assert client.post("/api/runs/no-such-run/activate").status_code == 404
    broken = trained_run.parent / "broken-run"
    broken.mkdir(exist_ok=True)
    (broken / "status.json").write_text("{not json")
    runs = client.get("/api/runs").json()
    me = [r for r in runs["runs"] if r["name"] == "broken-run"]
    assert me and not me[0]["compatible"] and runs["active"] == "test-run"


def test_post_stage_on_a_consistent_set(trained_run, processed_corpus):
    """prepare --stats-from-run gives arrays standardised like the run's own set, and
    `post` redoes booster + calibration + eval on them without touching model.pt"""
    from fraudvae.data.prepare import prepare
    from fraudvae.train import post_main

    before = (trained_run / "model.pt").stat().st_mtime
    prepare(name="t2", max_rows_per_source=2000, stats_from_run=str(trained_run))
    meta = json.loads((PREPARED_DIR / "t2" / "meta.json").read_text())
    own = json.loads((PREPARED_DIR / "t" / "meta.json").read_text())
    assert meta["stats_from_run"] and meta["counts"]["train"] == 4000
    assert [(f["mean"], f["std"]) for f in meta["numeric"]] == [(f["mean"], f["std"]) for f in own["numeric"]]
    assert post_main([str(trained_run), "--prepared", "t2", "--config", "smoke", "--samples", "2"]) == 0
    assert (trained_run / "model.pt").stat().st_mtime == before
    ev = json.loads((trained_run / "eval.json").read_text())
    assert ev["mean"]["blend"] > 0.9


def _customer_timelines(data):
    by_cust = {}
    for name in ("train", "val", "test"):
        sp = data[name]
        for i in range(len(sp)):
            by_cust.setdefault(int(sp.cust[i]), []).append((float(sp.ts[i]), name, i))
    for c in by_cust:
        by_cust[c].sort()
    return by_cust


def test_sequence_index_windows(prepared_set):
    """v5: a row's history window is the previous L rows of the same customer, oldest first,
    whatever split those rows fell into; rows without enough history are padded and masked"""
    from fraudvae.model.dataset import PreparedSet

    data = PreparedSet(prepared_set)
    seq = data.seq
    assert seq is not None
    by_cust = _customer_timelines(data)
    L = 4
    rng = np.random.default_rng(0)
    for name in ("train", "val", "test"):
        sp = data[name]
        pick = rng.choice(len(sp), 25, replace=False)
        h_num, h_mask, h_cat, h_dt, h_valid = seq.history(name, pick, L)
        assert h_num.shape == (25, L, sp.num.shape[1]) and h_valid.shape == (25, L)
        for k, i in enumerate(pick):
            c, t = int(sp.cust[i]), float(sp.ts[i])
            earlier = [r for r in by_cust[c] if (r[0], r[1], r[2]) < (t, name, int(i))]
            n = min(L, len(earlier))
            assert int(h_valid[k].sum()) == n, (name, i)
            assert h_valid[k, L - n:].all() and not h_valid[k, : L - n].any()


def test_history_context_shapes(trained_run, prepared_set):
    """the history read-out exists, differs between 'some history' and 'no history', and the
    model still scores a row without any window"""
    import torch

    from fraudvae.model.dataset import PreparedSet
    from fraudvae.model.scoring import score_batch
    from fraudvae.model.vae import load_checkpoint

    model, _ = load_checkpoint(trained_run / "model.pt")
    assert model.hist is not None and model.hdim == 16
    data = PreparedSet(prepared_set)
    sp = data["val"]
    idx = np.arange(8)
    num, mask, cat, src = sp.batch(idx)
    hist = data.seq.history("val", idx, 4)
    r_with = score_batch(model, num, mask, cat, src, n_samples=1, hist=hist)
    r_without = score_batch(model, num, mask, cat, src, n_samples=1)
    assert r_with["h"].shape == (8, 16) and r_without["h"].shape == (8, 16)
    assert np.isfinite(r_with["raw"]).all() and np.isfinite(r_without["raw"]).all()
    with torch.no_grad():
        out = model(num, mask, cat, src)   # no history at all is a valid forward pass
    assert out["h"].shape == (8, 16)


def test_history_in_the_score_api(client):
    """v5: history comes with the request as past rows; the sequence section reports how many
    were used and the wide block reaches the booster"""
    ex = client.get("/api/examples?source=capone&fraud=0").json()[0]
    assert ex["wide"] and {"raw_a", "raw_cat"} <= set(ex["wide"])   # raw_b may be NaN
    base = {"source": "capone", "features": ex["features"], "wide": ex["wide"]}
    r0 = client.post("/api/score", json={**base, "ts": 1.6e9}).json()
    assert r0["sequence"]["n_history"] == 0 and "prob" in r0["sequence"]
    past = []
    for k in (30, 20, 10):
        past.append({"features": {**ex["features"], "amount": 12.0}, "ts": 1.6e9 - 3600 * k})
    r1 = client.post("/api/score", json={**base, "ts": 1.6e9, "history": past}).json()
    assert r1["sequence"]["n_history"] == 3
    assert r1["novelty"]["score"] != r0["novelty"]["score"]   # the context changed the reconstruction
    # rows in the future of the scored transaction are not history
    late = past + [{"features": ex["features"], "ts": 1.6e9 + 3600}]
    r2 = client.post("/api/score", json={**base, "ts": 1.6e9, "history": late}).json()
    assert r2["sequence"]["n_history"] == 3
    # the wide block shows up in the booster's pushes when it matters
    kinds = {p["kind"] for p in r0["supervised"]["pushes"]}
    assert kinds <= {"numeric", "categorical", "wide"}


def test_device_head_trained_and_evaluated(trained_run):
    """v5: the device head trains on the source that carries the block (fdh in the synthetic
    corpus) and lands in the calibration and the evaluation as its own signal"""
    assert (trained_run / "head_device.pt").exists() and (trained_run / "head_device.json").exists()
    hj = json.loads((trained_run / "head_device.json").read_text())
    assert "device_os" in hj["categorical"] and "session_length_min" in hj["numeric"]
    assert hj["sources"]["fdh"]["auroc"] > 0.6, hj["sources"]
    cal = json.loads((trained_run / "calibration.json").read_text())
    assert "device" in cal["blend"]["signals"]
    assert cal["device"]["sources"]["fdh"]["auroc"] > 0.6 and cal["device"]["sources"]["fdh"]["coverage"] > 0.5
    assert cal["device"]["sources"].get("capone") is None      # capone has no device block at all
    ev = json.loads((trained_run / "eval.json").read_text())
    assert ev["sources"]["fdh"]["device"]["auroc"] > 0.6 and ev["sources"]["capone"].get("device") is None


DEVICE_COLS = ("device_os", "browser", "screen_w", "screen_h", "session_length_min", "device_type", "keep_alive_session")


def test_device_payload_in_the_score_api(client):
    """v5: a device payload (user-agent style strings) becomes the device block, the head
    reports present / fields / a probability, and without it the signal is absent"""
    ex = client.get("/api/examples?source=fdh&fraud=0").json()[0]
    feats = {k: v for k, v in ex["features"].items() if k not in DEVICE_COLS}
    r0 = client.post("/api/score", json={"source": "fdh", "features": feats}).json()
    assert r0["device"]["present"] is False and "prob" not in r0["device"]
    dev = {"os": "Windows 10", "browser": "chrome 63.0 for desktop", "screen": "1920x1080", "session_length_min": 1.5,
           "keep_alive": True}
    r1 = client.post("/api/score", json={"source": "fdh", "features": feats, "device": dev}).json()
    d = r1["device"]
    assert d["present"] is True and 0.0 <= d["prob"] <= 1.0 and d["calibration"] == "fdh"
    assert set(d["fields"]) == {"device_os", "browser", "screen_w", "screen_h", "session_length_min", "keep_alive_session"}
    assert d["label"] == "Device & identity"
    # a value set directly on the features wins over the payload
    r2 = client.post("/api/score", json={"source": "fdh", "features": {**feats, "device_os": "android"}, "device": dev}).json()
    assert any(c["feature"] == "device_os" and c["observed"] == "android" for c in r2["contributions"])
    meta = client.get("/api/meta").json()["active_run"]
    assert "device" in meta["heads"]
