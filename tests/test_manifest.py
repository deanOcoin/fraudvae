"""manifest.json and the clearing of inherited evaluations (fraudvae/manifest.py)."""
import json

import numpy as np

from fraudvae.config import load_config
from fraudvae.manifest import clear_evaluations, config_digest, write_manifest


def _fake_run(tmp_path):
    d = tmp_path / "run"
    d.mkdir()
    (d / "model.pt").write_bytes(b"\x00" * 100)
    (d / "booster.txt").write_text("tree")
    (d / "booster.json").write_text(json.dumps({"features": ["a"]}))
    (d / "head_device.pt").write_bytes(b"\x01" * 10)
    (d / "specialist" / "fdh").mkdir(parents=True)
    (d / "specialist" / "fdh" / "booster.txt").write_text("tree")
    (d / "eval.json").write_text("{}")
    (d / "eval_delay30.json").write_text("{}")
    np.savez(d / "scores_test.npz", y=np.zeros(3))
    (d / "train.log").write_text("kept")
    return d


def test_clear_evaluations_removes_only_the_evaluations(tmp_path):
    d = _fake_run(tmp_path)
    removed = clear_evaluations(d)
    assert removed == ["eval.json", "eval_delay30.json", "scores_test.npz"]
    assert (d / "model.pt").exists() and (d / "train.log").exists() and (d / "booster.json").exists()
    assert clear_evaluations(d) == []


def test_manifest_records_the_artefacts_and_keeps_history(tmp_path):
    d = _fake_run(tmp_path)
    cfg = load_config("smoke")
    rec = write_manifest(d, "post", cfg, prepared=None)
    assert rec["stage"] == "post" and rec["run"] == "run" and rec["fraudvae"] and rec["python"]
    assert set(rec["files"]) == {"model.pt", "booster.txt", "booster.json", "head_device.pt"}
    assert rec["files"]["model.pt"]["bytes"] == 100 and len(rec["files"]["model.pt"]["sha256"]) == 64
    assert rec["specialists"] == ["fdh"] and set(rec["evaluations"]) == {"eval.json", "eval_delay30.json", "scores_test.npz"}
    assert rec["config_sha256"] == config_digest(cfg) and rec["config"]["name"] == "smoke" and rec["history"] == []
    assert "commit" in rec["git"] and rec["prepared"] is None
    on_disk = json.loads((d / "manifest.json").read_text())
    assert on_disk["files"] == rec["files"]
    # the same config hashes the same; a changed one does not
    cfg2 = load_config("smoke")
    assert config_digest(cfg2) == rec["config_sha256"]
    cfg2.boost.num_leaves += 1
    assert config_digest(cfg2) != rec["config_sha256"]
    # a second write keeps the first as history, without its config body, and reflects the new state
    (d / "booster.txt").write_text("another tree")
    rec2 = write_manifest(d, "evaluate", None, prepared="nowhere", extra={"label_delay_days": 30})
    assert rec2["label_delay_days"] == 30 and rec2["config"] is None and rec2["prepared"] == {"name": "nowhere", "meta_sha256": None}
    assert len(rec2["history"]) == 1 and rec2["history"][0]["stage"] == "post" and "config" not in rec2["history"][0]
    assert rec2["files"]["booster.txt"]["sha256"] != rec["files"]["booster.txt"]["sha256"]
