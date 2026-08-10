"""Get a fresh GPU box ready: look at what it has, install what's missing, push
the code and the processed data.
"""
from __future__ import annotations

import json
import shlex
import tarfile
import tempfile
import time
from pathlib import Path
from typing import Callable

from ..paths import PROCESSED_DIR, ROOT
from .ssh import RemoteError, SSHSession

Log = Callable[[str], None]

# what goes to the server. data/ and runs/ are synced separately, .git never
CODE_INCLUDE = ["fraudvae", "configs", "pyproject.toml", "requirements.txt", "README.md"]


def probe(s: SSHSession) -> dict:
    """hardware + software inventory of the remote"""
    info = {"ok": True, "warnings": []}
    rc, out, _ = s.run("hostname; uname -sm; nproc; free -g | awk '/Mem:/{print $2}'")
    lines = out.strip().splitlines() + ["", "", "", ""]
    info["hostname"], info["os"], info["cpus"], info["ram_gb"] = lines[0], lines[1], _int(lines[2]), _int(lines[3])

    rc, out, _ = s.run("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits 2>/dev/null")
    gpus = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2:
            gpus.append({"name": parts[0], "memory_total_mb": _int(parts[1])})
    info["gpus"] = gpus
    if not gpus:
        info["warnings"].append("no nvidia-smi / no GPU visible; training would run on CPU")

    rc, out, _ = s.run("command -v python3 2>/dev/null")
    python = out.strip() or None
    info["python"] = python
    if python:
        rc, out, _ = s.run(f"{python} -c 'import torch,json; print(json.dumps({{\"torch\": torch.__version__, \"cuda\": torch.cuda.is_available()}}))' 2>/dev/null")
        try:
            info["torch"] = json.loads(out.strip().splitlines()[-1])
        except (json.JSONDecodeError, IndexError):
            info["torch"] = None
    else:
        info["ok"] = False
        info["warnings"].append("no python3 on the remote")
    return info


def _int(x):
    try:
        return int(float(str(x).strip()))
    except (TypeError, ValueError):
        return None


def _code_tarball() -> Path:
    tmp = Path(tempfile.mkstemp(prefix="fraudvae-code-", suffix=".tar.gz")[1])
    with tarfile.open(tmp, "w:gz") as tar:
        for name in CODE_INCLUDE:
            p = ROOT / name
            if not p.exists():
                continue
            tar.add(p, arcname=name, filter=lambda ti: None if "__pycache__" in ti.name else ti)
    return tmp


def bootstrap(s: SSHSession, log: Log = print, remote_dir: str | None = None) -> dict:
    """create the remote dir, make sure python+torch are there. returns the probe"""
    rd = remote_dir or s.host.remote_dir
    s.host.remote_dir = rd
    s.run(f"mkdir -p {shlex.quote(rd)}/data/processed {shlex.quote(rd)}/runs", check=True)
    info = probe(s)
    log(f"remote: {info.get('hostname')} {info.get('os')} | {info.get('cpus')} cpus | {info.get('ram_gb')} GB ram | "
        f"gpus: {', '.join(g['name'] for g in info['gpus']) or 'none'}")
    py = info.get("python")
    if not py:
        raise RemoteError("no python3 on the remote")
    torch_info = info.get("torch")
    if not torch_info or not torch_info.get("cuda"):
        log("no cuda torch found, installing torch...")
        s.run(f"{py} -m pip install --quiet torch", timeout=3600)
    s.host.python = py
    return info


def sync_code(s: SSHSession, log: Log = print) -> None:
    rd = s.host.remote_dir
    tmp = _code_tarball()
    try:
        log("uploading code...")
        s.upload(tmp, f"{rd}/code.tar.gz")
        s.run(f"cd {shlex.quote(rd)} && tar xzf code.tar.gz && rm code.tar.gz", check=True)
    finally:
        tmp.unlink(missing_ok=True)
    py = s.host.python or "python3"
    log("installing python deps...")
    rc, out, err = s.run(f"cd {shlex.quote(rd)} && {py} -m pip install --quiet -r requirements.txt", timeout=1800)
    if rc != 0:
        log(f"pip exited {rc}: {(out or err).strip()[-500:]}")


def sync_data(s: SSHSession, log: Log = print) -> list[str]:
    """push data/processed/*.parquet (+ stats)"""
    rd = s.host.remote_dir
    files = sorted(PROCESSED_DIR.glob("*.parquet")) + [PROCESSED_DIR / "stats.json"]
    sent = []
    for f in files:
        if not f.exists():
            continue
        t0 = time.time()
        s.upload(f, f"{rd}/data/processed/{f.name}")
        sent.append(f.name)
        log(f"uploaded {f.name} ({f.stat().st_size / 1e6:.0f} MB in {time.time() - t0:.0f}s)")
    return sent
