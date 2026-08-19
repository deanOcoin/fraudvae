"""/api/remote/*: ssh key, hosts, probe, provision, launch, watch, fetch.

Long operations run as background tasks (see tasks.py); the ui polls them.
Each operation opens its own ssh session, nothing is kept open between calls.
"""
from __future__ import annotations

from pathlib import Path


from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..remote import advisor, jobs, keys, provision
from ..remote.ssh import RemoteError, RemoteHost, SSHSession, forget_host, load_hosts, remember_host, ssh_config_hosts
from . import inference, tasks
from .api_data import _clean

router = APIRouter(prefix="/api/remote")


class HostIn(BaseModel):
    host: str
    port: int = 22
    user: str = "root"
    key_path: str | None = None
    label: str = ""
    remote_dir: str = "~/fraudvae"


def _host(h: HostIn) -> RemoteHost:
    known = {(x.host, x.port, x.user): x for x in load_hosts()}
    rh = known.get((h.host, h.port, h.user)) or RemoteHost(h.host, h.port, h.user)
    rh.key_path = h.key_path or rh.key_path
    rh.label = h.label or rh.label
    if h.remote_dir and h.remote_dir != "~/fraudvae":
        rh.remote_dir = h.remote_dir
    return rh


@router.get("/key")
def get_key(rotate: bool = False):
    if rotate:
        keys.rotate()
    return {"public_key": keys.public_key_text(), "fingerprint": keys.fingerprint(), "path": str(keys.PUBLIC)}


@router.get("/hosts")
def get_hosts():
    return {"remembered": [h.to_dict() for h in load_hosts()], "ssh_config": ssh_config_hosts(),
            "jobs": jobs.list_jobs()}


@router.delete("/hosts")
def delete_host(host: str, port: int = 22, user: str = "root"):
    forget_host(host, port, user)
    return {"ok": True}


@router.get("/advise")
def get_advise(config: str = "gpu_full", prepared: str | None = "default", epochs: int | None = None):
    try:
        return _clean(advisor.advise(config, prepared, full_corpus=True, epochs=epochs))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, f"{type(e).__name__}: {e}")


@router.post("/probe")
def post_probe(h: HostIn):
    rh = _host(h)
    try:
        with SSHSession(rh, timeout=15) as s:
            info = provision.probe(s)
            info["fingerprint"] = s.fingerprint
            info["first_seen"] = s.first_seen
            info["remote_dir"] = s.expand(rh.remote_dir)
    except RemoteError as e:
        raise HTTPException(502, str(e))
    remember_host(rh)
    return _clean(info)


class ProvisionIn(HostIn):
    sources: list[str] | None = None
    sync_data: bool = True
    prepared_name: str = "full"
    max_rows_per_source: int | None = None
    run_prepare: bool = True
    # v5: ship the raw downloads instead of (or as well as) the processed parquet, let the box
    # fetch what has a public url, and build there
    raw_dir: str | None = None
    sync_raw: bool = False
    fetch_direct: bool = False
    run_build: bool = False


@router.post("/provision")
def post_provision(req: ProvisionIn):
    rh = _host(req)

    def work(t: tasks.Task):
        with SSHSession(rh, timeout=20) as s:
            t.log(f"connected to {rh.user}@{rh.host}:{rh.port} ({s.fingerprint})")
            info = provision.bootstrap(s, log=t.log)
            t.meta["probe"] = info
            provision.sync_code(s, log=t.log)
            if req.sync_raw and req.raw_dir:
                provision.sync_raw(s, Path(req.raw_dir).expanduser(), log=t.log, keys=req.sources,
                                   progress=lambda name, done, total: t.meta.update({"upload": {"file": name, "done": done, "total": total}}))
            if req.sync_data:
                provision.sync_data(s, log=t.log, sources=req.sources)
            if req.fetch_direct or req.run_build or (req.run_prepare and (req.sync_data or req.run_build)):
                # the box-side steps run detached there and are only watched from here
                provision.remote_data_pipeline(s, log=t.log, fetch=req.fetch_direct, build=req.run_build,
                                               prepare=req.run_prepare, prepared_name=req.prepared_name,
                                               max_rows_per_source=req.max_rows_per_source, keys=req.sources)
            remember_host(s.host)
            t.log("provisioning done")
            return {"remote_dir": s.host.remote_dir, "python": s.host.python}

    t = tasks.run_thread("provision", f"provision {rh.host}", work)
    return {"task": t.id}


class ExecIn(HostIn):
    cmd: str
    timeout: int = 600


@router.post("/exec")
def post_exec(req: ExecIn):
    """run one shell command on the box and return its output. the app is bound to localhost
    and the box is one the user rented, so this is the same trust as the ssh line they
    pasted; it is what the relay setup and ad-hoc checks go through."""
    rh = _host(req)
    try:
        with SSHSession(rh, timeout=20) as s:
            rc, out, err = s.run(req.cmd, timeout=req.timeout)
    except RemoteError as e:
        raise HTTPException(502, str(e))
    return {"rc": rc, "out": out[-200_000:], "err": err[-20_000:]}


class LaunchIn(HostIn):
    config: str = "gpu_full"
    prepared: str = "full"
    run_name: str | None = None
    epochs: int | None = None
    resume: bool = False
    stage: str = "train"      # or "post": booster + calibration + evaluation on an existing remote run


@router.post("/launch")
def post_launch(req: LaunchIn):
    rh = _host(req)
    if req.stage == "post" and not req.run_name:
        raise HTTPException(400, "stage 'post' needs the name of an existing run")
    run_name = req.run_name or tasks.run_name_from(f"remote-{req.config}")
    extra = f"--epochs {int(req.epochs)}" if req.epochs else ""
    if req.resume:
        extra += f" --resume runs/{run_name}/model.pt"
    try:
        with SSHSession(rh, timeout=20) as s:
            if not rh.python:
                rh.python = provision._find_python(s) or "python3"
            job = jobs.launch(s, run_name, config=req.config, prepared=req.prepared, extra_args=extra, stage=req.stage)
            remember_host(s.host)
    except RemoteError as e:
        raise HTTPException(502, str(e))
    except Exception as e:  # noqa: BLE001 - timeouts etc, show them instead of a bare 500
        raise HTTPException(502, f"launch failed: {type(e).__name__}: {e}")
    return job


class RunIn(HostIn):
    run_name: str


@router.post("/status")
def post_status(req: RunIn):
    rh = _host(req)
    try:
        with SSHSession(rh, timeout=20) as s:
            st = jobs.status(s, req.run_name)
            st["log"] = jobs.tail_log(s, req.run_name, max_bytes=30_000).splitlines()[-200:]
    except RemoteError as e:
        raise HTTPException(502, str(e))
    return _clean(st)


@router.post("/stop")
def post_stop(req: RunIn):
    rh = _host(req)
    try:
        with SSHSession(rh, timeout=20) as s:
            ok = jobs.stop(s, req.run_name)
    except RemoteError as e:
        raise HTTPException(502, str(e))
    return {"stopped": ok}


class FetchIn(RunIn):
    load: bool = True
    processed: bool = False     # v5: also pull the box's processed data (the run's sources, examples, entity codes)


@router.post("/fetch")
def post_fetch(req: FetchIn):
    rh = _host(req)

    def work(t: tasks.Task):
        with SSHSession(rh, timeout=20) as s:
            dest = jobs.fetch(s, req.run_name, log=t.log)
            if req.processed:
                jobs.fetch_processed(s, log=t.log)
        if req.load and (dest / "model.pt").exists():
            m = inference.load_run(dest.name)
            t.log(f"loaded {m.name} as the active model")
        return {"run": dest.name}

    t = tasks.run_thread("fetch", f"fetch {req.run_name}", work)
    return {"task": t.id}
