from __future__ import annotations

# start / watch / collect training runs on the remote box

import json
import shlex
import time
from pathlib import Path
from typing import Callable

from ..paths import APP_HOME, RUNS_DIR
from .ssh import RemoteError, SSHSession

JOBS_FILE = APP_HOME / "remote_jobs.json"
RUN_FILES = ["run.json", "metrics.jsonl", "status.json", "calibration.json", "eval.json", "latent_map.npz", "train.log",
             "booster.json", "booster.txt", "model.pt",
             # v5 modules
             "seqhead.pt", "seqhead.json", "specialists.json", "graph.json", "text.json", "behaviour.json", "graph_bench.json"]
# v5: whole subtrees of the run (per-source specialists and graphs, the text heads and verifiers, the field heads)
RUN_GLOBS = ["head_*.pt", "head_*.json", "specialist/*/booster.txt", "specialist/*/booster.json", "graph/*.pt", "text/*.pt",
             "behaviour/*.pt", "eval_delay*.json"]


def _load_jobs() -> list[dict]:
    if JOBS_FILE.exists():
        try:
            return json.loads(JOBS_FILE.read_text())
        except json.JSONDecodeError:
            pass
    return []


def _save_jobs(jobs):
    JOBS_FILE.parent.mkdir(parents=True, exist_ok=True)
    JOBS_FILE.write_text(json.dumps(jobs, indent=1))


def list_jobs() -> list[dict]:
    return _load_jobs()


def launch(s: SSHSession, run_name: str, config: str = "gpu_full", prepared: str = "full",
           extra_args: str = "", log: Callable = print, stage: str = "train") -> dict:
    """stage 'train' = the whole thing; 'post' = booster + calibration + evaluation on an
    existing remote run (a cheap re-do with a different booster config)"""
    rd = s.host.remote_dir
    py = s.host.python or "python3"
    run_dir = f"{rd}/runs/{run_name}"
    s.mkdirs(run_dir)
    if stage == "post":
        prog = (f"{py} -m fraudvae.cli post {shlex.quote(run_name)} --config {shlex.quote(config)} "
                f"--prepared {shlex.quote(prepared)}")
    else:
        prog = (f"{py} -m fraudvae.train --config {shlex.quote(config)} --prepared {shlex.quote(prepared)} "
                f"--name {shlex.quote(run_name)} {extra_args}")
    # stdin MUST be redirected too or sshd keeps the session open until training ends (classic).
    # and `&` has to bind to the nohup alone: `cd X && nohup ... &` backgrounds a subshell that
    # keeps the ssh channel's stdout, so the command never returns and $! is the subshell
    cmd = (f"cd {shlex.quote(rd)} || exit 1; nohup {prog} > {shlex.quote(run_dir)}/launcher.log 2>&1 < /dev/null & echo $!")
    rc, out, err = s.run(cmd, timeout=60)
    pid = out.strip().splitlines()[-1] if out.strip() else ""
    if not pid.isdigit():
        raise RemoteError(f"launch failed: {err or out}")
    job = {"run": run_name, "pid": int(pid), "host": s.host.host, "port": s.host.port, "user": s.host.user,
           "remote_dir": rd, "config": config, "prepared": prepared, "started": time.time(), "state": "running",
           "stage": stage}
    jobs = [j for j in _load_jobs() if j["run"] != run_name]
    jobs.insert(0, job)
    _save_jobs(jobs)
    log(f"launched {run_name} on {s.host.host} (pid {pid})")
    return job


def status(s: SSHSession, run_name: str) -> dict:
    rd = s.host.remote_dir
    run_dir = f"{rd}/runs/{run_name}"
    out = {"run": run_name, "alive": False, "status": None, "metrics": []}
    jobs = {j["run"]: j for j in _load_jobs()}
    if s.exists(f"{run_dir}/status.json"):
        try:
            out["status"] = json.loads(s.read_text(f"{run_dir}/status.json"))
        except json.JSONDecodeError:
            pass
    # three places a pid can come from: our launch record, the process's own status.json
    # (the trainer and the post stage both write theirs), and pgrep for a run started by
    # hand. a run re-launched for a post stage has a stale launch pid, so try them all
    candidates = []
    for p in (jobs.get(run_name, {}).get("pid"), (out["status"] or {}).get("pid")):
        if p and int(p) not in candidates:
            candidates.append(int(p))
    if not candidates:
        rc, o, _ = s.run(f"pgrep -f 'fraudvae.(train.*--name|cli post) {shlex.quote(run_name)}' | head -1")
        if o.strip().isdigit():
            candidates.append(int(o.strip()))
    live = None
    for p in candidates:
        rc, _, _ = s.run(f"kill -0 {p} 2>/dev/null")
        if rc == 0:
            live = p
            break
    out["alive"] = live is not None
    if run_name not in jobs and candidates:
        jobs[run_name] = {"run": run_name, "pid": live or candidates[0], "host": s.host.host, "port": s.host.port,
                          "user": s.host.user, "remote_dir": rd, "started": time.time(), "state": "running", "recovered": True}
    elif live is not None and run_name in jobs:
        jobs[run_name]["pid"] = live
    if s.exists(f"{run_dir}/metrics.jsonl"):
        for line in s.read_text(f"{run_dir}/metrics.jsonl").splitlines():
            try:
                out["metrics"].append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if run_name in jobs:
        st = out["status"] or {}
        jobs[run_name]["state"] = st.get("state") if st.get("state") in ("finished",) else ("running" if out["alive"] else "stopped")
        _save_jobs(list(jobs.values()))
    return out


def tail_log(s: SSHSession, run_name: str, max_bytes=60_000) -> str:
    rd = s.host.remote_dir
    for name in ("train.log", "launcher.log"):
        p = f"{rd}/runs/{run_name}/{name}"
        if s.exists(p):
            txt = s.read_text(p, max_bytes=max_bytes)
            if txt.strip():
                return txt
    return ""


def stop(s: SSHSession, run_name: str) -> bool:
    """kill whatever is running this run: the launch pid, the pid the process wrote into
    status.json (a post stage has its own), or pgrep as a last resort"""
    jobs = {j["run"]: j for j in _load_jobs()}
    pids = []
    p = jobs.get(run_name, {}).get("pid")
    if p:
        pids.append(int(p))
    st_path = f"{s.host.remote_dir}/runs/{run_name}/status.json"
    if s.exists(st_path):
        try:
            p = json.loads(s.read_text(st_path)).get("pid")
            if p and int(p) not in pids:
                pids.append(int(p))
        except (json.JSONDecodeError, ValueError):
            pass
    rc, o, _ = s.run(f"pgrep -f 'fraudvae.(train.*--name|cli post) {shlex.quote(run_name)}'")
    for line in o.split():
        if line.isdigit() and int(line) not in pids:
            pids.append(int(line))
    ok = False
    for p in pids:
        rc, _, _ = s.run(f"kill {p} 2>/dev/null")
        ok = ok or rc == 0
    return ok


def fetch(s: SSHSession, run_name: str, log: Callable = print, local_root: Path = RUNS_DIR) -> Path:
    """pull a finished run's artifacts into local runs/<run_name>"""
    rd = s.host.remote_dir
    dest = Path(local_root) / run_name
    dest.mkdir(parents=True, exist_ok=True)
    got = []
    names = list(RUN_FILES)
    # expand the v5 subtree globs on the remote (one level of directories is all they use)
    for pattern in RUN_GLOBS:
        rc, out, _ = s.run(f"cd {rd}/runs/{run_name} 2>/dev/null && ls -1 {pattern} 2>/dev/null")
        names += [line.strip() for line in out.splitlines() if line.strip()]
    for name in names:
        remote = f"{rd}/runs/{run_name}/{name}"
        if not s.exists(remote):
            continue
        (dest / name).parent.mkdir(parents=True, exist_ok=True)
        size = s.sftp.stat(remote).st_size
        last = [0.0]

        def cb(done, total, _n=name, _last=last):
            if time.time() - _last[0] > 2:
                _last[0] = time.time()
                log(f"  {_n}: {done / 1e6:.0f}/{total / 1e6:.0f} MB")

        s.download(remote, dest / name, progress=cb if size > 5_000_000 else None)
        got.append(name)
    (dest / "remote.json").write_text(json.dumps({"host": s.host.host, "port": s.host.port, "user": s.host.user,
                                                  "remote_dir": rd, "fetched": time.time(), "files": got}, indent=1))
    log(f"fetched {len(got)} files into {dest}")
    return dest


def fetch_processed(s: SSHSession, log: Callable = print, local_root: Path | None = None) -> list[str]:
    """v5: pull the box's processed data (parquet + wide blocks + stats + samples) into the local
    data/processed, so an app on a laptop that never built v5 itself has the examples, the
    data page and the entity codes the fetched run was trained on. skips same-size files."""
    from ..paths import PROCESSED_DIR

    rd = s.host.remote_dir
    dest = Path(local_root or PROCESSED_DIR)
    dest.mkdir(parents=True, exist_ok=True)
    rc, out, _ = s.run(f"cd {rd}/data/processed 2>/dev/null && ls -1")
    names = [n.strip() for n in out.splitlines() if n.strip() and (n.endswith(".parquet") or n.endswith(".json"))]
    got = []
    for name in names:
        remote = f"{rd}/data/processed/{name}"
        size = s.sftp.stat(remote).st_size
        local = dest / name
        if local.exists() and local.stat().st_size == size:
            continue
        last = [0.0]

        def cb(done, total, _n=name, _last=last):
            if time.time() - _last[0] > 2:
                _last[0] = time.time()
                log(f"  {_n}: {done / 1e6:.0f}/{total / 1e6:.0f} MB")

        s.download(remote, local, progress=cb if size > 5_000_000 else None)
        got.append(name)
        log(f"fetched {name} ({size / 1e6:.0f} MB)")
    log(f"processed data: {len(got)} files fetched, {len(names) - len(got)} already current")
    return got
