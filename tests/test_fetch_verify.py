"""`fraudvae fetch --verify`: a download checked against the files the shipped generations
were built from - on disk, inside a zip, missing, or a different edition - and the git
clones against their commits."""
import hashlib
import json
import subprocess
import zipfile

from fraudvae.data import fetch


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def test_verify_against_expected(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    good = b"a" * 1000
    other = b"b" * 1000
    (raw / "ulb").mkdir(parents=True)
    (raw / "ulb" / "creditcard.csv").write_bytes(good)                       # matches
    (raw / "paysim").mkdir()
    (raw / "paysim" / "log.csv").write_bytes(other)                           # same size, different bytes
    (raw / "sparkov").mkdir()
    with zipfile.ZipFile(raw / "sparkov" / "download.zip", "w") as z:        # still zipped
        z.writestr("fraudTrain.csv", good)
    (raw / "ieee").mkdir()
    (raw / "ieee" / "train_transaction.csv").write_bytes(good[:500])         # a different edition (size)
    (raw / "tabformer").mkdir()
    (raw / "tabformer" / "transactions.tgz").write_bytes(b"pointer")         # a git-lfs pointer next to the real file
    (raw / "tabformer" / "dl").mkdir()
    (raw / "tabformer" / "dl" / "transactions.tgz").write_bytes(good)
    # a clone at the recorded commit, and one that moved on
    for key in ("fdh", "capone"):
        d = raw / key / "repo"
        d.mkdir(parents=True)
        subprocess.run(["git", "init", "-q", str(d)], check=True)
        subprocess.run(["git", "-C", str(d), "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "--allow-empty", "-m", "x"], check=True)
    fdh_head = subprocess.run(["git", "-C", str(raw / "fdh" / "repo"), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    expected = {"sources": {
        "ulb": [{"path": "ulb/creditcard.csv", "bytes": 1000, "sha256": _sha(good)}],
        "paysim": [{"path": "paysim/log.csv", "bytes": 1000, "sha256": _sha(good)}],
        "sparkov": [{"path": "sparkov/fraudTrain.csv", "bytes": 1000, "sha256": _sha(good)}],
        "ieee": [{"path": "ieee/train_transaction.csv", "bytes": 1000, "sha256": _sha(good)}],
        "banksim": [{"path": "banksim/bs.csv", "bytes": 1000, "sha256": _sha(good)}],
        "tabformer": [{"path": "tabformer/credit_card_transactions-ibm_v2.csv", "bytes": 5000, "sha256": _sha(b"x")},
                      {"path": "tabformer/transactions.tgz", "bytes": 1000, "sha256": _sha(good),
                       "alternative_to": "tabformer/credit_card_transactions-ibm_v2.csv"}],
    }, "repos": {"fdh": {"commit": fdh_head}, "capone": {"commit": "0" * 40}}}
    exp = tmp_path / "expected.json"
    exp.write_text(json.dumps(expected))
    monkeypatch.setattr(fetch, "EXPECTED", exp)
    res = fetch.verify(raw, ["ulb", "paysim", "sparkov", "ieee", "banksim", "tabformer", "fdh", "capone"])
    assert res["ulb"]["ulb/creditcard.csv"] == "matches the trained-on file"
    assert res["paysim"]["paysim/log.csv"] == "same size, different checksum"
    assert res["sparkov"]["sparkov/fraudTrain.csv"].startswith("matches the trained-on file (inside download.zip")
    assert res["ieee"]["ieee/train_transaction.csv"].startswith("size 500")
    assert res["banksim"]["banksim/bs.csv"] == "missing"
    assert res["tabformer"]["tabformer/transactions.tgz"] == "matches the trained-on file"      # the lfs pointer was skipped
    assert res["tabformer"]["tabformer/credit_card_transactions-ibm_v2.csv"].startswith("not extracted yet")
    assert res["fdh"]["clone"] == "matches the trained-on commit" and res["capone"]["clone"].startswith("commit ")
    quick = fetch.verify(raw, ["ulb"], quick=True)
    assert quick["ulb"]["ulb/creditcard.csv"] == "size matches (not hashed)"
    assert fetch.main(["--verify", "--raw-dir", str(raw), "--source", "ulb"]) == 0
    assert fetch.main(["--verify", "--raw-dir", str(raw), "--source", "paysim"]) == 2


def test_expected_manifest_is_complete():
    """every source in the registry has recorded files, and the hashes look like hashes"""
    from fraudvae.data.registry import SOURCES

    exp = json.loads(fetch.EXPECTED.read_text())
    for key in SOURCES:
        files = exp["sources"][key]
        assert files, key
        for f in files:
            assert f["path"].startswith(key + "/") and f["bytes"] > 0 and len(f["sha256"]) == 64
    assert set(exp["repos"]) == {"fdh", "capone"}
