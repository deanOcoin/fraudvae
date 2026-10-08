"""Get a fresh GPU box ready: look at what it has, install what's missing, push
the code and the processed data. Everything is idempotent so re-running after a
dropped connection just continues.
"""
from __future__ import annotations

import json
import shlex
import tarfile
import tempfile
import time
from pathlib import Path
from typing import Callable

from .. import __version__
from ..paths import PROCESSED_DIR, ROOT
from .ssh import RemoteError, SSHSession

Log = Callable[[str], None]

# what goes to the server. data/ and runs/ are synced separately, .git never
CODE_INCLUDE = ["fraudvae", "configs", "pyproject.toml", "requirements.txt", "README.md", "scripts", "Makefile"]
CODE_EXCLUDE_DIRS = {"__pycache__", ".pytest_cache", "node_modules", ".git"}


def probe(s: SSHSession) -> dict:
    """hardware + software inventory of the remote"""
    info = {"ok": True, "warnings": []}
    rc, out, _ = s.run("hostname; uname -sm; nproc; free -g | awk '/Mem:/{print $2}'")
    lines = out.strip().splitlines() + ["", "", "", ""]
    info["hostname"], info["os"], info["cpus"], info["ram_gb"] = lines[0], lines[1], _int(lines[2]), _int(lines[3])

    rc, out, _ = s.run("nvidia-smi --query-gpu=name,memory.total,memory.used,driver_version --format=csv,noheader,nounits 2>/dev/null")
    gpus = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3:
            gpus.append({"name": parts[0], "memory_total_mb": _int(parts[1]), "memory_used_mb": _int(parts[2]),
                         "driver": parts[3] if len(parts) > 3 else ""})
    info["gpus"] = gpus
    if not gpus:
        info["warnings"].append("no nvidia-smi / no GPU visible; training would run on CPU")

    rc, out, _ = s.run("df -BG --output=avail ~ 2>/dev/null | tail -1")
    info["disk_free_gb"] = _int(out.strip().rstrip("G"))

    python = _find_python(s)
    info["python"] = python
    if python:
        rc, out, _ = s.run(f"{python} -c 'import sys,json; print(json.dumps({{\"version\": sys.version.split()[0]}}))'")
        try:
            info["python_version"] = json.loads(out)["version"]
        except (json.JSONDecodeError, KeyError):
            info["python_version"] = ""
        rc, out, _ = s.run(f"{python} -c 'import torch,json; print(json.dumps({{\"torch\": torch.__version__, \"cuda\": torch.cuda.is_available(), \"cuda_version\": torch.version.cuda}}))' 2>/dev/null")
        try:
            info["torch"] = json.loads(out.strip().splitlines()[-1])
        except (json.JSONDecodeError, IndexError):
            info["torch"] = None
        rc, out, _ = s.run(f"{python} -c 'import fraudvae; print(fraudvae.__version__)' 2>/dev/null")
        info["fraudvae_installed"] = out.strip() or None
    else:
        info["ok"] = False
        info["warnings"].append("no python3 on the remote")
    rc, out, _ = s.run("command -v pip3 pip 2>/dev/null | head -1")
    info["pip"] = out.strip() or None
    rc, out, _ = s.run("command -v tmux screen nohup 2>/dev/null")
    info["tools"] = out.strip().splitlines()
    return info


def _int(x):
    try:
        return int(float(str(x).strip()))
    except (TypeError, ValueError):
        return None


def _find_python(s: SSHSession) -> str | None:
    # prefer a python that already has a cuda torch (vast.ai pytorch images keep
    # theirs in a venv that only a login shell has on PATH, hence the bash -lc)
    cands = ["/venv/main/bin/python", "/opt/conda/bin/python", "/usr/local/bin/python3", "/usr/bin/python3"]
    rc, out, _ = s.run("bash -lc 'command -v python3 python' 2>/dev/null")
    cands = [c for c in out.split() if c] + cands
    seen = set()
    fallback = None
    for cand in cands:
        if cand in seen:
            continue
        seen.add(cand)
        rc, out, _ = s.run(f"test -x {shlex.quote(cand)} && {shlex.quote(cand)} -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null")
        if out.strip() == "True":
            return cand
        if rc == 0 or out.strip() == "False":
            fallback = fallback or cand
    if fallback:
        return fallback
    rc, out, _ = s.run("command -v python3 2>/dev/null")
    return out.strip() or None


def _code_tarball() -> Path:
    tmp = Path(tempfile.mkstemp(prefix="fraudvae-code-", suffix=".tar.gz")[1])
    with tarfile.open(tmp, "w:gz") as tar:
        for name in CODE_INCLUDE:
            p = ROOT / name
            if not p.exists():
                continue
            tar.add(p, arcname=name, filter=_filter)
    return tmp


def _filter(ti: tarfile.TarInfo):
    parts = Path(ti.name).parts
    if any(part in CODE_EXCLUDE_DIRS or part.endswith(".pyc") for part in parts):
        return None
    return ti


def bootstrap(s: SSHSession, log: Log = print, remote_dir: str | None = None) -> dict:
    """create the remote dir, make sure python+torch+deps are there. returns the probe"""
    rd = s.expand(remote_dir or s.host.remote_dir)
    s.host.remote_dir = rd
    s.mkdirs(f"{rd}/data/processed")
    s.mkdirs(f"{rd}/runs")
    info = probe(s)
    log(f"remote: {info.get('hostname')} {info.get('os')} | {info.get('cpus')} cpus | {info.get('ram_gb')} GB ram | "
        f"{info.get('disk_free_gb')} GB free | gpus: {', '.join(g['name'] for g in info['gpus']) or 'none'}")
    py = info.get("python")
    if not py:
        raise RemoteError("no python3 on the remote, pick an image with python (any pytorch image works)")
    torch_info = info.get("torch")
    if not torch_info or not torch_info.get("cuda"):
        log("no cuda torch found, installing torch (this is a few GB, grab a coffee)...")
        for line in s.stream(f"{py} -m pip install --quiet torch 2>&1"):
            if line.strip():
                log("  " + line)
        rc, out, _ = s.run(f"{py} -c 'import torch; print(torch.cuda.is_available())'")
        log(f"torch cuda available: {out.strip()}")
    else:
        log(f"torch {torch_info['torch']} with cuda {torch_info.get('cuda_version')} already there")
    s.host.python = py
    return info


def sync_code(s: SSHSession, log: Log = print) -> None:
    rd = s.host.remote_dir
    tmp = _code_tarball()
    try:
        remote_tar = f"{rd}/code.tar.gz"
        log(f"uploading code ({tmp.stat().st_size / 1e6:.1f} MB)...")
        s.upload(tmp, remote_tar, resume=False)
        s.run(f"cd {shlex.quote(rd)} && rm -rf fraudvae configs scripts && tar xzf code.tar.gz && rm code.tar.gz", check=True)
    finally:
        tmp.unlink(missing_ok=True)
    py = s.host.python or "python3"
    log("installing python deps (minus torch)...")
    reqs = [l.strip() for l in (ROOT / "requirements.txt").read_text().splitlines()
            if l.strip() and not l.startswith("#") and not l.startswith("torch") and "pytest" not in l and "asyncssh" not in l]
    cmd = f"cd {shlex.quote(rd)} && {py} -m pip install --quiet {' '.join(shlex.quote(r) for r in reqs)} 2>&1 | tail -3"
    rc, out, err = s.run(cmd, timeout=1800)
    if rc != 0:
        log(f"pip exited {rc}: {(out or err).strip()[-500:]}")
    else:
        log("deps ok")
    rc, out, _ = s.run(f"cd {shlex.quote(rd)} && {py} -c 'import fraudvae; print(fraudvae.__version__)'")
    log(f"remote fraudvae {out.strip() or '?'} (local {__version__})")


def sync_data(s: SSHSession, log: Log = print, sources: list[str] | None = None, progress: Callable | None = None) -> list[str]:
    """push data/processed/*.parquet (+ stats). skips files the remote already has at the same size"""
    rd = s.host.remote_dir
    files = sorted(PROCESSED_DIR.glob("*.parquet")) + [PROCESSED_DIR / "stats.json"]
    if sources:
        files = [f for f in files if f.stem in sources or f.suffix == ".json"]
    sent = []
    for f in files:
        if not f.exists():
            continue
        remote = f"{rd}/data/processed/{f.name}"
        size = f.stat().st_size
        t0 = time.time()
        last = [0.0]

        def cb(done, total, _name=f.name, _t0=t0, _last=last):
            now = time.time()
            if now - _last[0] > 2 or done == total:
                _last[0] = now
                rate = done / max(now - _t0, 1e-6) / 1e6
                log(f"  {_name}: {done / 1e6:.0f}/{total / 1e6:.0f} MB ({rate:.1f} MB/s)")
                if progress:
                    progress(_name, done, total)

        did = s.upload(f, remote, progress=cb)
        if did:
            sent.append(f.name)
            log(f"uploaded {f.name} ({size / 1e6:.0f} MB in {time.time() - t0:.0f}s)")
        else:
            log(f"{f.name} already on remote, skipped")
    return sent


def sync_raw(s: SSHSession, raw_root: Path, log: Log = print, keys: list[str] | None = None, progress: Callable | None = None,
             skip_direct: bool = True) -> dict:
    """v5: push the raw downloads themselves (the zips as they came from kaggle / zenodo / the
    registration forms) to data/raw/<key>/ on the remote, identified by their contents, so the
    v5 build runs on the box instead of the laptop. resumes partial uploads and skips files the
    remote already has at the same size. datasets with a public direct url (brainrun) are left
    for the box to fetch itself (remote_data_pipeline) unless skip_direct=False."""
    from ..data.aux_registry import AUX, scan

    rd = s.host.remote_dir
    found = scan(Path(raw_root))
    if keys:
        found = {k: v for k, v in found.items() if k in keys}
    else:
        found.pop("talkingdata", None)   # 11 GB and optional: only when asked for by name
    if skip_direct:
        found = {k: v for k, v in found.items() if not (k in AUX and AUX[k].direct_urls)}
    total = sum(f.stat().st_size for fs in found.values() for f in fs)
    log(f"raw data under {raw_root}: {len(found)} datasets, {sum(len(v) for v in found.values())} files, {total / 1e9:.1f} GB")
    sent, skipped = [], []
    for key, files in found.items():
        s.mkdirs(f"{rd}/data/raw/{key}")
        for f in files:
            remote = f"{rd}/data/raw/{key}/{f.name}"
            size = f.stat().st_size
            t0 = time.time()
            last = [0.0]

            def cb(done, tot, _name=f"{key}/{f.name}", _t0=t0, _last=last):
                now = time.time()
                if now - _last[0] > 5 or done == tot:
                    _last[0] = now
                    rate = done / max(now - _t0, 1e-6) / 1e6
                    log(f"  {_name}: {done / 1e6:.0f}/{tot / 1e6:.0f} MB ({rate:.1f} MB/s)")
                    if progress:
                        progress(_name, done, tot)

            did = s.upload(f, remote, progress=cb, resume=True)
            if did:
                sent.append(f"{key}/{f.name}")
                log(f"uploaded {key}/{f.name} ({size / 1e6:.0f} MB in {time.time() - t0:.0f}s)")
            else:
                skipped.append(f"{key}/{f.name}")
    log(f"raw sync done: {len(sent)} uploaded, {len(skipped)} already there")
    return {"sent": sent, "skipped": skipped, "datasets": sorted(found)}


def remote_data_pipeline(s: SSHSession, log: Log = print, fetch: bool = True, build: bool = True, prepare: bool = True,
                         prepared_name: str = "full", max_rows_per_source: int | None = None, keys: list[str] | None = None,
                         poll: int = 20) -> dict:
    """the box-side data steps (public downloads + github sources, build, prepare) as ONE
    detached script with a done-marker per step, so a laptop going to sleep mid-build costs
    nothing: re-running this re-attaches to the live script or skips the finished steps.
    Streams the script's log back until it ends."""
    from ..data.aux_registry import AUX

    rd = s.host.remote_dir
    py = s.host.python or "python3"
    jobs = f"{rd}/jobs"
    s.mkdirs(jobs)
    lines = ["#!/bin/bash", f"cd {shlex.quote(rd)} || exit 1", "set -o pipefail"]
    if fetch:
        for key, d in AUX.items():
            if not d.direct_urls or (keys and key not in keys):
                continue
            lines.append(f"mkdir -p data/raw/{key}")
            for url in d.direct_urls:
                name = url.split("/")[-1].split("?")[0]
                lines.append(f"[ -f jobs/fetch_{key}_{name}.done ] || {{ echo \"== wget {key}/{name}\"; "
                             f"wget -c -q -O data/raw/{key}/{shlex.quote(name)} {shlex.quote(url)} && touch jobs/fetch_{key}_{name}.done; }}")
        lines.append(f"[ -f jobs/fetch_github.done ] || {{ echo \"== fraudvae fetch\"; {py} -m fraudvae.data.fetch && touch jobs/fetch_github.done; }}")
    if build:
        extra = "".join(f" --source {shlex.quote(k)}" for k in (keys or []))
        lines.append(f"[ -f jobs/build.done ] || {{ echo \"== build\"; {py} -m fraudvae.data.build{extra} && touch jobs/build.done; }}")
    if prepare:
        extra = f" --max-rows-per-source {int(max_rows_per_source)}" if max_rows_per_source else ""
        lines.append(f"[ -f jobs/prepare_{prepared_name}.done ] || {{ echo \"== prepare {prepared_name}\"; "
                     f"{py} -m fraudvae.data.prepare --name {shlex.quote(prepared_name)}{extra} && touch jobs/prepare_{prepared_name}.done; }}")
    lines.append('echo "== PIPELINE DONE $(date)"')
    script = "\n".join(lines) + "\n"
    s.run(f"cat > {jobs}/data_pipeline.sh <<'FRAUDVAE_EOF'\n{script}FRAUDVAE_EOF\nchmod +x {jobs}/data_pipeline.sh", check=True)
    # already running from an earlier attempt? re-attach instead of starting a second one
    rc, out, _ = s.run(f"cat {jobs}/data_pipeline.pid 2>/dev/null")
    pid = out.strip()
    alive = bool(pid) and s.run(f"kill -0 {pid} 2>/dev/null")[0] == 0
    if alive:
        log(f"data pipeline already running on the box (pid {pid}), re-attaching")
    else:
        rc, out, _ = s.run(f"cd {shlex.quote(rd)} || exit 1; nohup bash jobs/data_pipeline.sh >> jobs/data_pipeline.log 2>&1 < /dev/null & echo $!")
        pid = out.strip().splitlines()[-1] if out.strip() else ""
        s.run(f"echo {pid} > {jobs}/data_pipeline.pid")
        log(f"data pipeline launched on the box (pid {pid}); it survives this connection")
    seen = 0
    while True:
        rc, out, _ = s.run(f"wc -l < {jobs}/data_pipeline.log 2>/dev/null; tail -n +{seen + 1} {jobs}/data_pipeline.log 2>/dev/null | head -200")
        parts = out.split("\n", 1)
        try:
            total = int(parts[0].strip())
        except ValueError:
            total = seen
        new = parts[1].splitlines() if len(parts) > 1 else []
        for line in new:
            if line.strip():
                log("  " + line.rstrip())
        seen = min(seen + len(new), total)
        if any("== PIPELINE DONE" in line for line in new):
            break
        rc, _, _ = s.run(f"kill -0 {pid} 2>/dev/null")
        if rc != 0:
            rc2, tail, _ = s.run(f"tail -3 {jobs}/data_pipeline.log 2>/dev/null")
            if "== PIPELINE DONE" in tail:
                break
            raise RemoteError("the data pipeline on the box stopped before finishing; see jobs/data_pipeline.log there")
        time.sleep(poll)
    rc, out, _ = s.run(f"ls {jobs}/*.done 2>/dev/null")
    return {"done": [Path(x).name for x in out.split()]}
