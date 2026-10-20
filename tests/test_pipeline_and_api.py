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
    # v5: a wide block for every source (the short-delay label block comes with the entity
    # features; capone's raw columns on top), sequence index for everyone
    assert list(meta["wide"]) == ["fdh", "capone"]
    fast = {"cpty_fraud_count_fast", "cpty_fraud_share_fast", "cust_fraud_count_fast", "cust_fraud_share_fast"}
    assert fast <= set(meta["wide"]["fdh"]["columns"]) and "raw_a" not in meta["wide"]["fdh"]["columns"]
    assert meta["wide"]["capone"]["columns"][-3:] == ["raw_a", "raw_b", "raw_cat"] and fast <= set(meta["wide"]["capone"]["columns"])
    wide = np.load(PREPARED_DIR / prepared_set / "train_wide_capone.npy")
    a, b = meta["wide"]["capone"]["offsets"]["train"]
    cols = meta["wide"]["capone"]["columns"]
    assert wide.shape == (b - a, len(cols)) and np.isnan(wide[:, cols.index("raw_b")]).any() and not np.isnan(wide[:, cols.index("raw_a")]).any()
    # the short-delay block is causal and bounded: a share in [0, 1] or NaN, and never set on a
    # row whose entity has no labelled rows a week older than it
    share = wide[:, cols.index("cpty_fraud_share_fast")]
    assert np.nanmin(share) >= 0 and np.nanmax(share) <= 1 and np.isnan(share).any()
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
    assert ev["generic"]["blend"]["auroc"] > 0.8
    for key in ("vae", "booster", "blend"):
        assert "fpr_0.01" in ev["sources"]["fdh"][key]["at_val_thresholds"], key
    cal = json.loads((trained_run / "calibration.json").read_text())
    assert cal["score_type"] in ("raw", "norm")
    assert "generic" in cal and "fpr_0.01" in cal["sources"]["fdh"]["thresholds"]
    assert cal["blend"]["n_features"] == len(cal["blend"]["coef"]) and cal["blend"]["with_booster"]
    assert "fdh" in cal["blend"]["sources"] and "generic" in cal["blend"]
    assert cal["booster"]["sources"]["fdh"]["auroc"] > 0.9
    bj = json.loads((trained_run / "booster.json").read_text())
    # v5: [common numerics | common categoricals | source | wide block]
    assert bj["features"][-3:] == ["capone.raw_a", "capone.raw_b", "capone.raw_cat"] and bj["best_iteration"] >= 1
    assert "source" in bj["features"] and bj["features"].index("source") < bj["features"].index("fdh.cpty_fraud_share_fast")
    assert bj["wide_names"][-3:] == ["capone.raw_a", "capone.raw_b", "capone.raw_cat"] and "fdh.cust_fraud_share_fast" in bj["wide_names"]
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
    assert "by_hour" in built[0]["stats"]


def test_examples_and_scoring(client):
    ex = client.get("/api/examples?source=fdh&fraud=1").json()[0]
    assert ex["is_fraud"] == 1 and "amount" in ex["features"]
    r = client.post("/api/score", json={"source": "fdh", "features": ex["features"], "fpr_target": "fpr_0.01"}).json()
    for k in ("p_fraud", "flag", "legit_percentile", "novelty", "supervised", "blend", "contributions", "latent_xy", "score"):
        assert k in r, k
    assert r["verdict_from"] == "blend" and 0.0 <= r["p_fraud"] <= 1.0
    assert r["calibration"] == "fdh" and r["novelty"]["calibration"] == "fdh"
    assert 0.0 <= r["supervised"]["prob"] <= 1.0 and len(r["supervised"]["pushes"]) > 0
    assert abs(r["supervised"]["pushes"][0]["contribution"]) >= abs(r["supervised"]["pushes"][-1]["contribution"])
    assert r["contributions"][0]["z"] >= r["contributions"][-1]["z"]
    assert r["flag"] is True   # planted fraud: x6 amount at night
    # generic context works too, and has its own blend thresholds
    g = client.post("/api/score", json={"source": None, "features": {"amount": 50, "hour": 12}}).json()
    assert g["calibration"] == "generic" and g["blend"]["calibration"] == "generic" and g["observed_features"] == 2


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
    assert summary["rows"] == 20 and "auroc" in summary and summary["verdict_from"] == "blend"
    header = res.text.splitlines()[0]
    for col in ("p_fraud", "booster_prob", "novelty_score", "flag", "novelty_flag"):
        assert col in header, col


def test_runs_listing_and_metrics(client):
    runs = client.get("/api/runs").json()
    assert runs["active"] == "test-run"
    me = [r for r in runs["runs"] if r["name"] == "test-run"][0]
    assert me["compatible"] and me["has_booster"] and me["mean"]["blend"] > 0.9
    m = client.get("/api/runs/test-run/metrics").json()
    assert len(m["metrics"]) == 2 and m["eval"]["mean_auroc"] > 0.5 and m["booster"]["best_iteration"] >= 1
    lat = client.get("/api/runs/test-run/latent").json()
    assert len(lat["xy"]) == len(lat["y"]) > 0


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
    assert ev["mean"]["blend"] > 0.9 and json.loads((trained_run / "status.json").read_text())["state"] == "finished"


def test_sequence_index_windows(prepared_set):
    """v5: a row's history window is the previous L rows of the same customer, oldest first,
    whatever split those rows fell into; rows without enough history are padded and masked"""
    import torch

    from fraudvae.model.dataset import PreparedSet

    data = PreparedSet(prepared_set)
    seq = data.seq
    assert seq is not None
    # ground truth from the per-split customer/time arrays
    rows = []
    for name in ("train", "val", "test"):
        sp = data[name]
        for i in range(len(sp)):
            rows.append((int(sp.cust[i]), float(sp.ts[i]), name, i))
    by_cust = {}
    for c, t, name, i in rows:
        by_cust.setdefault(c, []).append((t, name, i))
    for c in by_cust:
        by_cust[c].sort()
    L = 4
    rng = np.random.default_rng(0)
    for name in ("train", "val", "test"):
        sp = data[name]
        pick = rng.choice(len(sp), 25, replace=False)
        h_num, h_mask, h_cat, h_dt, h_valid = seq.history(name, pick, L)
        assert h_num.shape == (25, L, sp.num.shape[1]) and h_valid.shape == (25, L)
        for k, i in enumerate(pick):
            c, t = int(sp.cust[i]), float(sp.ts[i])
            hist = [r for r in by_cust[c] if r[0] < t or (r[0] == t and (r[1], r[2]) < (name, int(i)))]
            hist = hist[-L:]
            n_valid = int(h_valid[k].sum())
            assert n_valid == len(hist), (name, i, n_valid, len(hist))
            # the valid entries sit at the end (oldest first) and carry the right rows
            for j, (ht, hname, hi) in enumerate(hist):
                pos = L - len(hist) + j
                assert bool(h_valid[k, pos])
                exp = torch.from_numpy(np.ascontiguousarray(data[hname].num[hi]))
                assert torch.allclose(h_num[k, pos], exp), (name, i, pos)
                assert abs(float(h_dt[k, pos]) - np.log1p(t - ht) / 10.0) < 1e-5
            assert not h_valid[k, : L - len(hist)].any()
    # the batched iterator covers every requested row exactly once
    idx = np.sort(rng.choice(len(data["val"]), 300, replace=False))
    seen = np.concatenate([b for b, _, _ in seq.iter_batches("val", 64, L, idx=idx, shuffle=True)])
    assert np.array_equal(np.sort(seen), idx)


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
    has_hist = hist[4].any(1).numpy()
    assert has_hist.any()
    assert not np.allclose(r_with["h"][has_hist], r_without["h"][has_hist])
    assert np.isfinite(r_with["raw"]).all() and np.isfinite(r_without["raw"]).all()
    with torch.no_grad():
        out = model(num, mask, cat, src)   # no history at all is a valid forward pass
    assert out["h"].shape == (8, 16)


def test_history_in_the_score_api(client):
    """v5: history can come with the request, or from the entity store via customer_id; the
    sequence section reports where it came from and the wide block reaches the booster"""
    ex = client.get("/api/examples?source=capone&fraud=0").json()[0]
    assert ex["wide"] and {"raw_a", "raw_cat"} <= set(ex["wide"]) and "cpty_fraud_share_fast" in ex["wide"]   # raw_b may be NaN
    base = {"source": "capone", "features": ex["features"], "wide": ex["wide"]}
    r0 = client.post("/api/score", json={**base, "ts": 1.6e9}).json()
    assert r0["sequence"]["n_history"] == 0 and r0["sequence"]["history_from"] is None
    assert "prob" in r0["sequence"] and r0["sequence"]["calibration"] == "capone"
    past = [{"features": {**ex["features"], "amount": 12.0}, "ts": 1.6e9 - 3600 * k} for k in (30, 20, 10)]
    r1 = client.post("/api/score", json={**base, "ts": 1.6e9, "history": past}).json()
    assert r1["sequence"]["n_history"] == 3 and r1["sequence"]["history_from"] == "request"
    assert r1["novelty"]["score"] != r0["novelty"]["score"]   # the context changed the reconstruction
    # entity store: three remembered transactions, then a fourth sees them as its history
    cid = "cust-42"
    client.delete(f"/api/history/capone/{cid}")
    for k, row in enumerate(past):
        rr = client.post("/api/score", json={**base, "features": row["features"], "ts": row["ts"], "customer_id": cid}).json()
        assert rr["sequence"]["n_history"] == k and rr["sequence"]["history_from"] == "entity store"
    r2 = client.post("/api/score", json={**base, "ts": 1.6e9, "customer_id": cid}).json()
    assert r2["sequence"]["n_history"] == 3 and r2["sequence"]["history_from"] == "entity store"
    # same history either way: the deterministic parts agree (the -elbo itself is a 2-sample monte carlo estimate)
    assert abs(r2["sequence"]["prob"] - r1["sequence"]["prob"]) < 1e-4 and abs(r2["novelty"]["kl"] - r1["novelty"]["kl"]) < 1e-3
    assert client.delete(f"/api/history/capone/{cid}").json()["deleted"] == 4
    # the wide block shows up in the booster's pushes when it matters
    kinds = {p["kind"] for p in r0["supervised"]["pushes"]}
    assert kinds <= {"numeric", "categorical", "wide"}


def test_device_head_trained_and_evaluated(trained_run):
    """v5: the device head trains on the source that carries the block (fdh in the synthetic
    corpus), lands in the calibration and the evaluation as its own signal, and the blender
    has a coefficient for it"""
    assert (trained_run / "head_device.pt").exists() and (trained_run / "head_device.json").exists()
    hj = json.loads((trained_run / "head_device.json").read_text())
    assert "device_os" in hj["categorical"] and "session_length_min" in hj["numeric"]
    assert "fdh" in hj["sources"] and hj["sources"]["fdh"]["auroc"] > 0.6, hj["sources"]
    cal = json.loads((trained_run / "calibration.json").read_text())
    assert "device" in cal["blend"]["signals"] and "device_present" in cal["blend"]["feature_names"]
    assert cal["device"]["sources"]["fdh"]["auroc"] > 0.6 and cal["device"]["sources"]["fdh"]["coverage"] > 0.5
    assert cal["device"]["sources"].get("capone") is None      # capone has no device block at all
    ev = json.loads((trained_run / "eval.json").read_text())
    assert ev["sources"]["fdh"]["device"]["auroc"] > 0.6 and ev["sources"]["capone"].get("device") is None
    assert ev["mean"]["device"] > 0.6


def test_device_payload_in_the_score_api(client):
    """v5: a device payload (user-agent style strings) becomes the device block, the head
    reports present / fields / a probability, and without it the signal is absent"""
    ex = client.get("/api/examples?source=fdh&fraud=0").json()[0]
    # strip everything the device head reads: the device block and the device / email / address graph columns
    feats = {k: v for k, v in ex["features"].items()
             if k not in ("device_os", "browser", "screen_w", "screen_h", "session_length_min", "device_type")
             and not k.startswith(("device_", "email_", "addr_"))}
    r0 = client.post("/api/score", json={"source": "fdh", "features": feats}).json()
    assert r0["device"]["present"] is False and r0["device"]["fields"] == [] and "prob" not in r0["device"]
    dev = {"os": "Windows 10", "browser": "chrome 63.0 for desktop", "screen": "1920x1080", "session_length_min": 1.5,
           "keep_alive": True, "email_domain": "gmail.com"}
    r1 = client.post("/api/score", json={"source": "fdh", "features": feats, "device": dev}).json()
    d = r1["device"]
    assert d["present"] is True and 0.0 <= d["prob"] <= 1.0 and d["calibration"] == "fdh"
    assert {"device_os", "browser", "screen_w", "screen_h", "session_length_min", "keep_alive_session"} <= set(d["fields"])
    assert d["label"] == "Device & identity"
    # the payload reached the model inputs: the device fields are now among the observed features
    obs = {c["feature"] for c in r1["contributions"]}
    assert "session_length_min" in obs and "device_os" in obs
    # a value set directly on the features wins over the payload
    r2 = client.post("/api/score", json={"source": "fdh", "features": {**feats, "device_os": "android"}, "device": dev}).json()
    assert any(c["feature"] == "device_os" and c["observed"] == "android" for c in r2["contributions"])
    meta = client.get("/api/meta").json()["active_run"]
    assert "device" in meta["heads"] and meta["heads"]["device"]["label"] == "Device & identity"


def test_graph_module_trained_and_scored(trained_run, prepared_set):
    """v5: one entity-graph model per synthetic source, a 'graph' signal in calibration /
    evaluation / the blend, and a GraphScorer that scores prepared rows by their entity codes"""
    from fraudvae.graph.scorer import GraphScorer
    from fraudvae.model.dataset import PreparedSet

    gj = json.loads((trained_run / "graph.json").read_text())
    assert set(gj["sources"]) == {"fdh", "capone"} and (trained_run / "graph" / "fdh.pt").exists()
    for s in gj["sources"].values():
        assert s["nodes_a"] > 100 and s["edges_a"] > 100 and s["val_coverage"] > 0.9
    cal = json.loads((trained_run / "calibration.json").read_text())
    assert "graph" in cal["blend"]["signals"] and cal["graph"]["sources"]["fdh"]["coverage"] > 0.9
    ev = json.loads((trained_run / "eval.json").read_text())
    assert ev["sources"]["fdh"]["graph"]["auroc"] > 0.5 and ev["mean"]["graph"] is not None
    g = GraphScorer.load(trained_run)
    data = PreparedSet(prepared_set, splits=("test",))
    te = data["test"]
    idx = np.arange(0, len(te), 7)
    lg = g.logit_rows(te.src[:][idx], np.asarray(te.ent[idx]), which="test")
    assert np.isfinite(lg).mean() > 0.9
    # an all-unknown row has no graph signal; a row of a source without a graph neither
    assert np.isnan(g.logit_rows(np.array([1, 1]), np.array([[-1] * 6, [10 ** 8] * 6]), "test")).all()
    assert np.isnan(g.logit_rows(np.array([0, 9]), np.asarray(te.ent[:2]), "test")).all()


def test_entities_in_the_score_api(client):
    """v5: example rows carry entity codes, the API returns a graph section, and the receipt
    says which entities were known"""
    ex = client.get("/api/examples?source=fdh&fraud=0").json()[0]
    assert ex["entities"] and "customer" in ex["entities"]
    base = {"source": "fdh", "features": ex["features"]}
    r0 = client.post("/api/score", json=base).json()
    assert r0["graph"]["present"] is False and r0["graph"]["sent"] == [] and r0["graph"]["available"] is True
    r1 = client.post("/api/score", json={**base, "entities": ex["entities"]}).json()
    assert r1["graph"]["present"] is True and 0.0 <= r1["graph"]["prob"] <= 1.0
    assert set(r1["graph"]["fields"]) <= set(ex["entities"]) and r1["graph"]["calibration"] == "fdh"
    assert r1["blend"]["prob"] != r0["blend"]["prob"]      # the blend used the graph signal
    r2 = client.post("/api/score", json={**base, "entities": {"customer": 10 ** 8, "device": "junk"}}).json()
    assert r2["graph"]["present"] is False and r2["graph"]["sent"] == ["customer", "device"]
    meta = client.get("/api/meta").json()["active_run"]
    assert "fdh" in meta["graph"] and meta["graph"]["fdh"]["nodes_a"] > 100


def test_text_heads_trained_and_fused(trained_run, client):
    """v5: the text heads train on the tiny text sets in the raw folder, serve through the
    `text` payload, and their likelihood ratios are fused into the final probability"""
    tj = json.loads((trained_run / "text.json").read_text())
    assert set(tj["heads"]) == {"malurl", "fakejob"}
    for name, info in tj["heads"].items():
        assert info["test_auroc"] > 0.8, (name, info["test_auroc"])
        assert 0.0 < info["positive_rate"] < 1.0 and (trained_run / "text" / f"{name}.pt").exists()
    ex = client.get("/api/examples?source=fdh&fraud=0").json()[0]
    base = {"source": "fdh", "features": ex["features"], "entities": ex["entities"]}
    r0 = client.post("/api/score", json=base).json()
    assert r0["text"]["present"] is False and "fusion" not in r0 and r0["verdict_from"] == "blend"
    bad = {"url": "verify-login-secure.ru/account/update", "memo": "urgent urgent fee wire prize free verify account login"}
    r1 = client.post("/api/score", json={**base, "text": bad}).json()
    t = r1["text"]
    assert t["present"] is True and set(t["fields"]) == {"url", "memo"}
    assert t["fields"]["url"]["head"] == "malurl" and t["fields"]["memo"]["head"] == "fakejob"
    assert t["fields"]["url"]["llr"] > 0 and t["fields"]["memo"]["llr"] > 0
    f = r1["fusion"]
    assert f["terms"]["text"]["n"] == 2 and f["terms"]["text"]["weight"] == 0.5
    assert abs(f["blend_logit"] - np.log(r1["blend"]["prob"] / (1 - r1["blend"]["prob"]))) < 1e-4
    assert r1["p_fraud"] == f["prob"] and r1["p_fraud"] > r1["blend"]["prob"] and r1["verdict_from"] == "blend+text"
    good = {"url": "news-recipe-weather.com/blog/music", "memo": "team docs sports shop about blog"}
    r2 = client.post("/api/score", json={**base, "text": good}).json()
    assert r2["text"]["fields"]["url"]["llr"] < 0 and r2["p_fraud"] < r2["blend"]["prob"]
    # keys no head reads are reported, not scored
    r3 = client.post("/api/score", json={**base, "text": {"colour": "blue"}}).json()
    assert r3["text"]["present"] is False and r3["text"]["sent"] == ["colour"]
    meta = client.get("/api/meta").json()["active_run"]
    assert set(meta["text"]) == {"malurl", "fakejob"}


def _mouse_events(rng, speed, jitter, n_strokes=30):
    """the api's session rows for a synthetic mouse user (same generator as the balabit fixture)"""
    from conftest import synthetic_mouse_session

    df = synthetic_mouse_session(rng, speed, jitter, n_strokes=n_strokes)
    kind = {"Move": "move", "Pressed": "down", "Released": "up"}
    return [[float(r["client timestamp"]), int(r["x"]), int(r["y"]), kind[r["state"]]] for _, r in df.iterrows()]


def test_behaviour_verifier_trained_and_fused(trained_run, client):
    """v5: the mouse verifier trains on the synthetic balabit set, the first session of a
    customer enrols, the second is compared, an impostor's session gets a negative ratio"""
    bj = json.loads((trained_run / "behaviour.json").read_text())
    assert "mouse" in bj["modalities"] and (trained_run / "behaviour" / "mouse.pt").exists()
    info = bj["modalities"]["mouse"]
    assert info["balabit"]["auroc"] > 0.8 and info["balabit"]["eer"] < 0.35 and "platt" in info
    rng = np.random.default_rng(7)
    ex = client.get("/api/examples?source=fdh&fraud=0").json()[0]
    base = {"source": "fdh", "features": ex["features"], "customer_id": "beh-cust-1"}
    client.delete("/api/history/fdh/beh-cust-1")
    r0 = client.post("/api/score", json={**base, "session": {"mouse": _mouse_events(rng, 60.0, 1.0)}}).json()
    b0 = r0["behaviour"]
    assert b0["present"] is False and "mouse" in b0["fields"] and b0["fields"]["mouse"]["enrolled"] is False
    assert b0["enrolled_now"] == ["mouse"] and b0["enrolled_sessions"] == 0 and "fusion" not in r0
    r1 = client.post("/api/score", json={**base, "session": {"mouse": _mouse_events(rng, 60.0, 1.0)}}).json()
    b1 = r1["behaviour"]
    assert b1["present"] is True and b1["enrolled_sessions"] == 1 and "cosine" in b1["fields"]["mouse"]
    assert r1["fusion"]["terms"]["behaviour"]["weight"] == 1.0 and r1["verdict_from"] == "blend+behaviour"
    r2 = client.post("/api/score", json={**base, "remember": False, "session": {"mouse": _mouse_events(rng, 40.0, 8.0)}}).json()
    assert r2["behaviour"]["fields"]["mouse"]["cosine"] < b1["fields"]["mouse"]["cosine"]
    assert r2["behaviour"]["fields"]["mouse"]["llr"] < b1["fields"]["mouse"]["llr"]
    # junk session rows are ignored, not fatal
    r3 = client.post("/api/score", json={**base, "remember": False, "session": {"mouse": [[1, 2]], "keystroke": "nope"}}).json()
    assert r3["behaviour"]["present"] is False
    meta = client.get("/api/meta").json()["active_run"]
    assert "mouse" in meta["behaviour"]
    assert client.delete("/api/history/fdh/beh-cust-1").json()["deleted"] == 2


def test_specialist_boosters(trained_run, client):
    """v5: a per-source booster for every synthetic source, the 'local' signal in calibration /
    evaluation / the blend and on the receipt"""
    sj = json.loads((trained_run / "specialists.json").read_text())
    assert set(sj["sources"]) == {"fdh", "capone"}
    for k, v in sj["sources"].items():
        assert v["val_auroc"] > 0.9 and (trained_run / "specialist" / k / "booster.txt").exists()
    cal = json.loads((trained_run / "calibration.json").read_text())
    assert "local" in cal["blend"]["signals"] and cal["local"]["sources"]["fdh"]["auroc"] > 0.9
    ev = json.loads((trained_run / "eval.json").read_text())
    assert ev["sources"]["capone"]["local"]["auroc"] > 0.9 and ev["mean"]["local"] > 0.9
    ex = client.get("/api/examples?source=capone&fraud=1").json()[0]
    r = client.post("/api/score", json={"source": "capone", "features": ex["features"], "wide": ex["wide"]}).json()
    assert r["local"]["present"] is True and r["local"]["calibration"] == "capone" and 0.0 <= r["local"]["prob"] <= 1.0
    assert "threshold" in r["local"] and "local" in cal["blend"]["feature_names"][-20:][0] or "local" in cal["blend"]["signals"]
    g = client.post("/api/score", json={"source": None, "features": ex["features"]}).json()
    assert g["local"]["present"] is False
    meta = client.get("/api/meta").json()["active_run"]
    assert set(meta["specialists"]) == {"fdh", "capone"}


def test_fast_label_block_windows():
    """v5: the one-week-delay fraud history, lifetime and over the last 7 / 30 days before the
    label horizon, computed causally: a fraud 10 days ago is in the fast block of a row today
    and not in the 30-day schema block"""
    from fraudvae.data import entities
    from fraudvae.model.booster import blank_fast, fast_columns

    DAY = 86400.0
    ts = np.array([0, 10 * DAY, 20 * DAY, 50 * DAY, 51 * DAY])
    term = np.array([1, 1, 1, 1, 1])
    y = np.array([0, 1, 0, 0, 0], dtype=np.int8)
    cust = np.array([5, 6, 7, 8, 9])
    f = entities.fast_label_features({"cpty": term}, cust, ts, y)
    assert set(f) >= {"wide__cpty_fraud_share_fast", "wide__cpty_fraud_share_fast_w7", "wide__cpty_fraud_share_fast_w30",
                      "wide__cust_fraud_count_fast"}
    np.testing.assert_allclose(f["wide__cpty_fraud_share_fast"], [np.nan, 0, 0.5, 1 / 3, 1 / 3])
    np.testing.assert_allclose(f["wide__cpty_fraud_share_fast_w7"], [np.nan, 0, 1.0, np.nan, np.nan])
    np.testing.assert_allclose(f["wide__cpty_fraud_share_fast_w30"], [np.nan, 0, 0.5, 0, 0])
    e = entities.entity_features({"cpty": term}, cust, ts, y)
    np.testing.assert_allclose(e["cpty_fraud_share_delayed"], [np.nan, np.nan, np.nan, 1 / 3, 1 / 3])
    # the booster side knows which columns those are and can blank them on a share of rows
    names = ["amount", "fdh.cpty_fraud_share_fast", "fdh.cpty_fraud_count_fast_w7", "fdh.fastidious", "fdh.cust_fraud_share_fast_w30", "x_fast_w"]
    assert fast_columns(names) == [1, 2, 4]
    X = np.ones((1000, len(names)), dtype=np.float32)
    X = blank_fast(X, names, np.random.default_rng(0), 0.3)
    blanked = np.isnan(X[:, 1])
    assert 200 < blanked.sum() < 400 and np.isnan(X[blanked][:, [1, 2, 4]]).all() and np.isfinite(X[:, [0, 3, 5]]).all()
    assert np.isfinite(X[~blanked][:, 1]).all()


def test_label_latency_in_the_score_api(client, trained_run):
    """v5: a deployment whose labels take a month must not be scored with the one-week fraud
    history; the receipt says what happened to it, and the post stage evaluated that regime too"""
    ex = client.get("/api/examples?source=capone&fraud=1").json()[0]
    assert any(k.endswith("_fast") or "_fast_w" in k for k in ex["wide"])
    base = {"source": "capone", "features": ex["features"], "wide": ex["wide"]}
    r7 = client.post("/api/score", json={**base, "label_delay_days": 7}).json()
    r30 = client.post("/api/score", json={**base, "label_delay_days": 30}).json()
    r0 = client.post("/api/score", json=base).json()
    assert r7["labels"]["fast_block"] == "used" and r0["labels"]["fast_block"] == "used"
    assert r30["labels"]["fast_block"] == "blanked" and r30["labels"]["latency_days"] == 30
    assert "supervised" in r30 and 0 <= r30["supervised"]["prob"] <= 1
    # without the fast columns in the request there is nothing to blank
    rn = client.post("/api/score", json={**base, "wide": {k: v for k, v in ex["wide"].items() if "_fast" not in k},
                                          "label_delay_days": 30}).json()
    assert rn["labels"]["fast_block"] == "not sent"
    ev30 = json.loads((trained_run / "eval_delay30.json").read_text())
    assert ev30["label_delay_days"] == 30 and set(ev30["sources"]) == {"fdh", "capone"}
    ev = json.loads((trained_run / "eval.json").read_text())
    assert ev["label_delay_days"] is None and 0 <= ev30["sources"]["fdh"]["blend"]["auroc"] <= 1


def test_history_readout_as_booster_features(trained_run, prepared_set, tmp_path):
    """v5: with boost.history_features the booster (and the specialists) read the frozen
    history encoder's summary of the customer's past; scoring paths that have the encoder
    pass it along, the receipt reports it as one block"""
    from fraudvae.config import load_config
    from fraudvae.model.booster import Booster, Specialists, history_readout, train_booster, train_specialists
    from fraudvae.model.dataset import PreparedSet
    from fraudvae.model.scoring import score_split
    from fraudvae.model.vae import load_checkpoint
    src_run = trained_run
    assert (src_run / "model.pt").exists()
    run_dir = tmp_path / "hist-run"
    run_dir.mkdir()
    for f in ("model.pt", "run.json"):
        (run_dir / f).write_bytes((src_run / f).read_bytes())
    cfg = load_config("smoke")
    cfg.boost.history_features = True
    cfg.boost.max_rounds = 30
    model, _ = load_checkpoint(run_dir / "model.pt", device="cpu")
    data = PreparedSet(prepared_set, splits=("train", "val"))
    h = history_readout(model, data["train"], np.arange(0, 64), data.seq, "train", "cpu")
    assert h.shape == (64, model.hdim) and np.isfinite(h).all()
    info = train_booster(run_dir, prepared_set, cfg.boost, log=lambda *a: None, model=model, device="cpu")
    assert info["hist_dim"] == model.hdim and info["features"][-1] == f"hist_{model.hdim - 1}"
    train_specialists(run_dir, prepared_set, cfg.boost, log=lambda *a: None, min_fraud=50, model=model, device="cpu")
    bst = Booster.load(run_dir)
    sp = Specialists.load(run_dir)
    assert bst.hist_dim == model.hdim and sp is not None
    te = PreparedSet(prepared_set, splits=("test",))
    r = score_split(model, te["test"], batch_size=256, n_samples=1, idx=np.arange(300), booster=bst, specialists=sp,
                    seq=te.seq, split_name="test", keep_features=False, keep_mu=False)
    assert r["booster"].shape == (300,) and np.isfinite(r["booster"]).all() and np.isfinite(r["local"]).any()
    # a caller without the encoder's summary gets a NaN block and still an answer
    sub = te["test"]
    b = np.arange(300)
    plain = bst.logit(sub.num[b], sub.mask[b], sub.cat[b], sub.src[b], wide=sub.wide(b))
    assert plain.shape == (300,) and np.isfinite(plain).all()
    x = bst._x(sub.num[b], sub.mask[b], sub.cat[b], sub.src[b], sub.wide(b), r["h"][:300] if "h" in r else None)
    assert x.shape[1] == len(info["features"])


def test_model_timeline_and_comparison(client, trained_run):
    """the public demo: every loadable generation on a timeline, a transaction scored on a chosen
    run and compared across runs in one request"""
    import shutil

    from fraudvae.paths import RUNS_DIR

    twin = RUNS_DIR / "test-run-twin"
    if not twin.exists():
        shutil.copytree(trained_run, twin)
    models = client.get("/api/models").json()
    names = [m["name"] for m in models["models"]]
    assert "test-run" in names and "test-run-twin" in names and models["active"] == "test-run"
    me = next(m for m in models["models"] if m["name"] == "test-run")
    assert me["version"] == "v5" and me["train_minutes"] is not None and me["mean_test_auroc"] is not None and me["loaded"]
    assert "hardware" in me and "what" in me   # commentary fields exist even for an unknown run (empty)
    ex = client.get("/api/examples?source=fdh&fraud=1").json()[0]
    body = {"source": "fdh", "features": ex["features"], "wide": ex["wide"], "run": "test-run-twin", "compare": ["test-run", "test-run-twin"]}
    r = client.post("/api/score", json=body).json()
    assert r["model"]["run"] == "test-run-twin" and r["model"]["legacy"] is False and r["model"]["bank_known"]
    rows = r["compare"]
    assert [x["run"] for x in rows] == ["test-run-twin", "test-run"]      # the scoring run first, duplicates dropped
    assert rows[0]["verdict_from"] == "blend" and all(0 <= x["p_fraud"] <= 1 for x in rows)
    assert abs(rows[0]["p_fraud"] - r["p_fraud"]) < 1e-6              # the strip's first row is the receipt itself
    assert set(rows[0]) >= {"p_fraud", "flag", "legit_percentile", "signals", "fields_used"}
    # (the twin is a copy of the files on disk; the resident test-run may be older than those
    # files by now, after the post-stage tests, so the two are not compared to each other)
    # an unknown run is a clean 404, the default model answers when no run is named
    assert client.post("/api/score", json={**body, "run": "nope"}).status_code == 404
    r0 = client.post("/api/score", json={"source": "fdh", "features": ex["features"], "wide": ex["wide"]}).json()
    assert r0["model"]["run"] == "test-run"
    meta = client.get("/api/meta").json()
    assert set(meta["loaded_runs"]) >= {"test-run", "test-run-twin"} and meta["public"]["public"] is False


def test_public_mode_gates(client, monkeypatch):
    """FRAUDVAE_PUBLIC: no training / data / default-model changes, no visitor state, paths hidden,
    per-address rate limit, batch rows capped"""
    from fraudvae.server import api_data, public

    # an example row from the build, fetched before public mode hides the build's samples
    ex = client.get("/api/examples?source=fdh&fraud=0").json()[0]
    monkeypatch.setattr(public, "PUBLIC", True)
    monkeypatch.setattr(public, "SCORE_PER_MINUTE", 60)
    monkeypatch.setattr(public, "SCORE_BURST", 3)
    monkeypatch.setattr(public, "BATCH_ROWS", 5)
    public._buckets.clear()
    assert client.post("/api/train/local", json={"config": "smoke"}).status_code == 403
    assert client.post("/api/data/build", json={}).status_code == 403
    assert client.post("/api/runs/test-run/load").status_code == 403
    assert client.delete("/api/history/fdh/c1").status_code == 403
    meta = client.get("/api/meta").json()
    assert meta["host"] is None and meta["data_dir"] is None and meta["public"]["public"] is True
    mx = client.get("/api/runs/test-run/metrics").json()
    assert mx["remote"] is None and "host" not in (mx["run"] or {})
    api_data._examples_cache.clear()
    assert client.get("/api/examples?source=fdh&fraud=0").status_code == 404   # the build's samples are not served publicly
    body = {"source": "fdh", "features": ex["features"], "wide": ex["wide"], "customer_id": "visitor-1", "remember": True, "ts": 1.7e9}
    codes = [client.post("/api/score", json=body).status_code for _ in range(5)]
    assert codes[:3] == [200, 200, 200] and 429 in codes[3:]
    # a forged X-Forwarded-For does not buy a fresh bucket: the address nginx appended counts
    spoof = {"X-Forwarded-For": "9.9.9.9, testclient"}
    assert client.post("/api/score", json=body, headers=spoof).status_code == 429
    assert client.post("/api/score", json=body, headers={"X-Real-IP": "5.5.5.5"}).status_code == 200
    public._buckets.clear()
    r = client.post("/api/score", json=body).json()
    assert r["sequence"]["history_from"] is None     # nothing remembered, nothing read back
    # a batch past the cap is refused before any scoring
    import io

    import pandas as pd

    rows = pd.DataFrame([ex["features"] | {"source": "fdh"}] * 6)
    buf = io.BytesIO(); rows.to_csv(buf, index=False)
    resp = client.post("/api/score/batch?source=fdh", files={"file": ("x.csv", buf.getvalue(), "text/csv")})
    assert resp.status_code == 413
    public._buckets.clear()


def test_examples_export_and_public_examples(client, monkeypatch, tmp_path):
    """data/examples/: only the sources whose licence allows it are written, from the test split;
    the public site reads that directory and nothing else"""
    import dataclasses

    from fraudvae.data import examples as ex
    from fraudvae.data.build import SAMPLES
    from fraudvae.data.registry import SOURCES
    from fraudvae.server import api_data, public

    # nothing in the synthetic corpus (fdh, capone) is redistributable -> nothing written
    assert ex.write_examples(tmp_path / "ex0", samples=SAMPLES) == []
    monkeypatch.setitem(SOURCES, "fdh", dataclasses.replace(SOURCES["fdh"], redistributable=True, license_url="https://x/l"))
    written = ex.write_examples(tmp_path / "ex", n=20, samples=SAMPLES)
    assert written == ["fdh"] and (tmp_path / "ex" / "fdh.parquet").exists() and not (tmp_path / "ex" / "capone.parquet").exists()
    readme = (tmp_path / "ex" / "README.md").read_text()
    assert "fdh" in readme and "https://x/l" in readme and "capone" not in readme.split("| source |")[1].split("\n## ")[0]
    df = ex.load_examples(tmp_path / "ex")
    assert set(df["source"].astype(str)) == {"fdh"} and (df["split"].astype(str) == "test").all() and len(df) <= 20
    assert 0 < int(df["is_fraud"].sum()) < len(df)
    # the workbench keeps reading the build's samples; the public site only the examples directory
    assert client.get("/api/examples?source=capone").status_code == 200
    monkeypatch.setattr(public, "PUBLIC", True)
    monkeypatch.setattr(ex, "EXAMPLES_DIR", tmp_path / "ex")
    api_data._examples_cache.clear()
    row = client.get("/api/examples?source=fdh&fraud=1").json()[0]
    assert row["source"] == "fdh" and row["is_fraud"] == 1 and row["split"] == "test" and row["features"]
    r = client.get("/api/examples?source=capone")
    assert r.status_code == 404 and "licence" in r.json()["detail"] and SOURCES["capone"].url in r.json()["detail"]
    api_data._examples_cache.clear()


def test_models_pack_and_pull(trained_run, tmp_path, monkeypatch):
    """a run packed into a bundle (deterministically, without the training host's details) and
    pulled back, verified against the manifest"""
    import json
    import tarfile

    from fraudvae import models
    from fraudvae.paths import RUNS_DIR

    monkeypatch.setattr(models, "MANIFEST", tmp_path / "pretrained.json")
    (trained_run / "remote.json").write_text(json.dumps({"host": "10.0.0.1", "port": 22}))
    try:
        entry = models.pack("test-run", tmp_path / "dist", RUNS_DIR, tag="t")
        asset = tmp_path / "dist" / "test-run.tar.gz"
        info = entry["assets"]["test-run.tar.gz"]
        assert asset.exists() and info["sha256"] == models.sha256(asset) and info["bytes"] == asset.stat().st_size
        with tarfile.open(asset) as tf:
            names = tf.getnames()
            run = json.loads(tf.extractfile("test-run/run.json").read())
        assert "test-run/model.pt" in names and "test-run/remote.json" not in names and "host" not in run and "pid" not in run
        assert models.pack("test-run", tmp_path / "dist", RUNS_DIR)["assets"]["test-run.tar.gz"]["sha256"] == info["sha256"]
        manifest = json.loads((tmp_path / "pretrained.json").read_text())
        assert manifest["tag"] == "t" and "test-run" in manifest["models"]
        monkeypatch.setattr(models, "MIRROR_URL", f"file://{tmp_path / 'dist'}/{{asset}}")
        dest = models.pull("test-run", tmp_path / "runs2")
        assert (dest / "model.pt").read_bytes() == (trained_run / "model.pt").read_bytes()
        assert not (dest / "remote.json").exists() and (dest / "calibration.json").exists()
        assert models.list_models(tmp_path / "runs2")[0]["on_disk"]
        # a tampered archive is refused
        (tmp_path / "dist" / "test-run.tar.gz").write_bytes(asset.read_bytes()[:-1] + b"\x00")
        import pytest

        with pytest.raises(RuntimeError):
            models.pull("test-run", tmp_path / "runs3")
    finally:
        (trained_run / "remote.json").unlink(missing_ok=True)
