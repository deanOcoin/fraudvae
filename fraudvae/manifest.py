"""manifest.json: what a run's artefacts are, exactly, so a comparison is always between known things.

Written by the stages that leave artefacts behind (pretrain, the post stage, a standalone
boost or evaluation): the code (version, git commit, whether the tree was clean), the
configuration and its hash, the prepared set with the hash of its meta.json, and the sha256
of every model file in the run. Earlier entries stay under "history". The post stage also
clears any evaluation a run inherited (eval*.json, scores_test*.npz) before it writes its
own: the node run once compared a round against a stale file copied in with the checkpoint,
and these two things are what make that impossible.
"""
from __future__ import annotations

import hashlib
import json
import platform
import socket
import subprocess
import time
from pathlib import Path

from . import __version__
from .paths import IN_CHECKOUT, PREPARED_DIR, ROOT

ARTEFACTS = ("model.pt", "txfm.pt", "booster.txt", "booster.json", "txfm_pca.npz", "specialists.json", "seqhead.pt", "seqhead.json",
             "head_*.pt", "head_*.json", "txfmhead.pt", "txfmhead.json", "calibration.json", "latent_map.npz", "graph.json",
             "text.json", "behaviour.json", "run.json", "txfm.json")
EVALUATIONS = ("eval.json", "eval_delay*.json", "scores_test*.npz")
HISTORY = 50


def sha256(path: Path, chunk=1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def git_state() -> dict:
    """the checkout's commit and whether its tracked files were clean; None outside a checkout"""
    if not IN_CHECKOUT or not (ROOT / ".git").exists():
        return {"commit": None, "clean": None}
    try:
        head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10)
        st = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "clean": None}
    return {"commit": head.stdout.strip() or None, "clean": (st.returncode == 0 and not st.stdout.strip()) if head.returncode == 0 else None}


def config_digest(cfg) -> str | None:
    if cfg is None:
        return None
    from .config import to_dict

    d = cfg if isinstance(cfg, dict) else to_dict(cfg)
    return hashlib.sha256(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()


def prepared_record(prepared: str | None) -> dict | None:
    if not prepared:
        return None
    meta = Path(PREPARED_DIR) / prepared / "meta.json"
    if not meta.exists():
        return {"name": prepared, "meta_sha256": None}
    m = json.loads(meta.read_text())
    return {"name": prepared, "meta_sha256": sha256(meta), "created": m.get("created"), "counts": m.get("counts"),
            "sources": m.get("sources"), "max_rows_per_source": m.get("max_rows_per_source"), "stats_from_run": m.get("stats_from_run")}


def clear_evaluations(run_dir: Path, log=None) -> list[str]:
    """remove the evaluations a run holds (its own or inherited with a copied checkpoint) so the
    post stage's are the only ones; returns what was removed"""
    run_dir = Path(run_dir)
    removed = []
    for pat in EVALUATIONS:
        for p in sorted(run_dir.glob(pat)):
            p.unlink()
            removed.append(p.name)
    if removed and log:
        log(f"  cleared inherited evaluation files: {', '.join(removed)}")
    return removed


def write_manifest(run_dir: Path, stage: str, cfg=None, prepared: str | None = None, extra: dict | None = None, log=None) -> dict:
    """record the run as it stands after `stage`; earlier records move under history"""
    run_dir = Path(run_dir)
    path = run_dir / "manifest.json"
    prev = None
    if path.exists():
        try:
            prev = json.loads(path.read_text())
        except (OSError, ValueError):
            prev = None
    files = {}
    for pat in ARTEFACTS:
        for p in sorted(run_dir.glob(pat)):
            files[p.name] = {"sha256": sha256(p), "bytes": p.stat().st_size, "modified": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(p.stat().st_mtime))}
    specialists = sorted(d.name for d in (run_dir / "specialist").glob("*") if (d / "booster.txt").exists()) if (run_dir / "specialist").exists() else []
    evaluations = {}
    for pat in EVALUATIONS:
        for p in sorted(run_dir.glob(pat)):
            evaluations[p.name] = {"bytes": p.stat().st_size, "modified": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(p.stat().st_mtime))}
    from .config import to_dict

    try:
        import torch

        torch_v = torch.__version__
    except Exception:  # noqa: BLE001 - a manifest never fails a run
        torch_v = None
    rec = {"created": time.strftime("%Y-%m-%dT%H:%M:%S"), "stage": stage, "run": run_dir.name, "fraudvae": __version__, "git": git_state(),
           "host": socket.gethostname(), "python": platform.python_version(), "torch": torch_v,
           "config_sha256": config_digest(cfg), "config": None if cfg is None else (cfg if isinstance(cfg, dict) else to_dict(cfg)),
           "prepared": prepared_record(prepared), "files": files, "specialists": specialists, "evaluations": evaluations}
    if extra:
        rec.update(extra)
    history = []
    if prev:
        history = list(prev.get("history") or [])
        history.append({k: v for k, v in prev.items() if k not in ("history", "config")})
    rec["history"] = history[-HISTORY:]
    path.write_text(json.dumps(rec, indent=1, default=str))
    if log:
        log(f"  manifest: {len(files)} artefacts hashed, stage {stage}, commit {(rec['git']['commit'] or 'none')[:10]}")
    return rec
