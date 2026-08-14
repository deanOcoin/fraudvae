"""`fraudvae remote ...` - the Compute tab without the browser.

  fraudvae remote key
  fraudvae remote probe     --host IP --port PORT [--user root]
  fraudvae remote provision --host IP --port PORT [--prepared full] [--no-data]
  fraudvae remote launch    --host IP --port PORT [--config gpu_full] [--prepared full] [--name RUN] [--stage post]
  fraudvae remote status    --host IP --port PORT --name RUN [--log]
  fraudvae remote fetch     --host IP --port PORT --name RUN
  fraudvae remote run       --host IP --port PORT [...]   provision + launch + wait + fetch in one go
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

from . import jobs, keys, provision
from .ssh import RemoteError, RemoteHost, SSHSession, load_hosts, remember_host


def _host_args(ap):
    ap.add_argument("--host", required=True)
    ap.add_argument("--port", type=int, default=22)
    ap.add_argument("--user", default="root")
    ap.add_argument("--remote-dir", default="~/fraudvae")
    ap.add_argument("--key", default=None, help="private key path (default: the app's own key)")


def _host(a) -> RemoteHost:
    known = {(x.host, x.port, x.user): x for x in load_hosts()}
    h = known.get((a.host, a.port, a.user)) or RemoteHost(a.host, a.port, a.user)
    if a.key:
        h.key_path = a.key
    if a.remote_dir != "~/fraudvae":
        h.remote_dir = a.remote_dir
    return h


def cmd_key(a):
    print(keys.public_key_text())
    print(f"# {keys.fingerprint()}", file=sys.stderr)


def cmd_probe(a):
    with SSHSession(_host(a)) as s:
        info = provision.probe(s)
        info["fingerprint"] = s.fingerprint
        remember_host(s.host)
    print(json.dumps(info, indent=1))


def _data_steps(s, a):
    """v5: --raw pushes the raw downloads, fetches the public ones on the box and builds there;
    otherwise the processed parquet is pushed as before. either way prepare runs after."""
    raw = getattr(a, "raw", None)
    if raw:
        provision.sync_raw(s, Path(raw).expanduser())
    if not a.no_data:
        provision.sync_data(s)
    if raw or not a.no_data:
        provision.remote_data_pipeline(s, fetch=bool(raw), build=bool(raw), prepare=True, prepared_name=a.prepared,
                                       max_rows_per_source=a.max_rows_per_source)


def cmd_provision(a):
    with SSHSession(_host(a)) as s:
        provision.bootstrap(s)
        provision.sync_code(s)
        _data_steps(s, a)
        remember_host(s.host)
    print("provisioned", s.host.remote_dir)


def _launch(s, a):
    extra = f"--epochs {a.epochs}" if a.epochs else ""
    if a.resume and a.name:
        extra += f" --resume runs/{a.name}/model.pt"
    stage = getattr(a, "stage", "train")
    if stage == "post" and not a.name:
        raise RemoteError("--stage post needs --name of an existing run")
    return jobs.launch(s, a.name or f"remote-{a.config}-{time.strftime('%Y%m%d-%H%M')}", config=a.config,
                       prepared=a.prepared, extra_args=extra, stage=stage)


def cmd_launch(a):
    with SSHSession(_host(a)) as s:
        if not s.host.python:
            s.host.python = provision._find_python(s) or "python3"
        job = _launch(s, a)
        remember_host(s.host)
    print(json.dumps(job, indent=1))


def cmd_status(a):
    with SSHSession(_host(a)) as s:
        st = jobs.status(s, a.name)
        if a.log:
            print(jobs.tail_log(s, a.name))
    st.pop("metrics", None) if not a.metrics else None
    print(json.dumps(st, indent=1))


def cmd_fetch(a):
    with SSHSession(_host(a)) as s:
        dest = jobs.fetch(s, a.name)
    print(dest)


def cmd_run(a):
    """the whole cycle. polls every --poll seconds until status.json says finished, then fetches"""
    h = _host(a)
    with SSHSession(h) as s:
        provision.bootstrap(s)
        provision.sync_code(s)
        _data_steps(s, a)
        job = _launch(s, a)
        remember_host(s.host)
    name = job["run"]
    print(f"launched {name}, polling every {a.poll}s (ctrl-c just stops polling, the run keeps going)")
    last_epoch = None
    while True:
        time.sleep(a.poll)
        try:
            with SSHSession(h) as s:
                st = jobs.status(s, name)
        except RemoteError as e:
            print(f"  (status check failed: {e}; retrying)")
            continue
        status = st.get("status") or {}
        ep = status.get("epoch")
        if ep != last_epoch:
            m = st["metrics"][-1] if st.get("metrics") else {}
            print(f"  epoch {ep}/{status.get('epochs')} state={status.get('state')} "
                  f"val_auroc={m.get('val_mean_auroc', float('nan')):.4f} rows/s={status.get('rows_per_sec', 0):,.0f}")
            last_epoch = ep
        if status.get("state") == "finished":
            break
        if not st["alive"] and status.get("state") not in ("finished",):
            print("process is gone and the run did not finish; check the log with `remote status --log`")
            return 1
    with SSHSession(h) as s:
        dest = jobs.fetch(s, name)
    print(f"done, run fetched to {dest}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="fraudvae remote", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("key").set_defaults(fn=cmd_key)
    p = sub.add_parser("probe"); _host_args(p); p.set_defaults(fn=cmd_probe)
    p = sub.add_parser("provision"); _host_args(p)
    p.add_argument("--prepared", default="full"); p.add_argument("--no-data", action="store_true")
    p.add_argument("--raw", default=None, help="v5: folder of raw downloads to push and build on the box (e.g. ~/Downloads/fraud-data-raw)")
    p.add_argument("--max-rows-per-source", type=int, default=None); p.set_defaults(fn=cmd_provision)
    for name, fn in (("launch", cmd_launch), ("run", cmd_run)):
        p = sub.add_parser(name); _host_args(p)
        p.add_argument("--config", default="gpu_full"); p.add_argument("--prepared", default="full")
        p.add_argument("--name", default=None); p.add_argument("--epochs", type=int, default=None)
        p.add_argument("--resume", action="store_true")
        p.add_argument("--stage", choices=["train", "post"], default="train",
                       help="post = only booster + calibration + evaluation on an existing remote run")
        if name == "run":
            p.add_argument("--no-data", action="store_true"); p.add_argument("--max-rows-per-source", type=int, default=None)
            p.add_argument("--raw", default=None)
            p.add_argument("--poll", type=int, default=60)
        p.set_defaults(fn=fn)
    p = sub.add_parser("status"); _host_args(p); p.add_argument("--name", required=True)
    p.add_argument("--log", action="store_true"); p.add_argument("--metrics", action="store_true"); p.set_defaults(fn=cmd_status)
    p = sub.add_parser("fetch"); _host_args(p); p.add_argument("--name", required=True); p.set_defaults(fn=cmd_fetch)
    # v5: `fraudvae remote relay --repo DIR [--url URL] run "cmd"` / `put LOCAL REL`, the laptop-off channel
    p = sub.add_parser("relay", add_help=False); p.set_defaults(fn=lambda a: __import__("fraudvae.remote.relay", fromlist=["main"]).main(a.rest))
    p.add_argument("rest", nargs=argparse.REMAINDER)
    a = ap.parse_args(argv)
    try:
        return a.fn(a) or 0
    except RemoteError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
