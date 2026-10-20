"""Pretrained generations: pack a run into a bundle, publish the bundles as release assets,
pull one back onto any machine.

    fraudvae models list                      # what the manifest knows, what is on disk
    fraudvae models pull v5-h200-h            # -> runs/v5-h200-h, verified against the manifest
    fraudvae models pull --all                # every generation on the timeline
    fraudvae models pull v5-h200-h --with-paysim-graph
    fraudvae models pack v5-h200-h            # runs/<name> -> dist/<name>.tar.gz (+ the paysim graph apart)

A bundle is the run directory as the trainer left it, minus what never leaves the machine
that trained it: `remote.json` (the box it was fetched from), launcher logs and pid files.
The v5 bundles also set aside `graph/paysim.pt`, the PaySim entity graph: 822 MB per run for
a head that scores 0.55 on PaySim and gets no weight in the blend, so it ships as a separate
optional asset and the server scores PaySim without the graph head when it is absent (the
receipt says so). `fraudvae/pretrained.json` is the manifest the pull verifies against:
asset names, sizes and sha256 for each generation, written by `pack` and committed with the
release that carries the files."""
from __future__ import annotations

import argparse
import calendar
import gzip
import hashlib
import io
import json
import os
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

from .paths import RUNS_DIR

MANIFEST = Path(__file__).parent / "pretrained.json"
REPO = os.environ.get("FRAUDVAE_REPO", "deanOcoin/fraudvae")
RELEASE_URL = "https://github.com/{repo}/releases/download/{tag}/{asset}"
# a mirror instead of the github release, e.g. https://fraudvae.example.org/models/{asset}
MIRROR_URL = os.environ.get("FRAUDVAE_MODELS_URL")
NEVER_SHIPPED = ("remote.json", "launcher.log", "launcher.pid", "train.pid", "status.pid")
APART = {"graph/paysim.pt": "graph-paysim"}       # file inside the run -> suffix of its own bundle


def sha256(path: Path, chunk=1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def load_manifest() -> dict:
    if not MANIFEST.exists():
        return {"tag": "pretrained", "repo": REPO, "models": {}}
    return json.loads(MANIFEST.read_text())


def save_manifest(m: dict):
    MANIFEST.write_text(json.dumps(m, indent=1, sort_keys=False) + "\n")


def _scrub_run_json(data: bytes) -> bytes:
    """run.json names the training host and process; neither is a visitor's business"""
    try:
        d = json.loads(data)
    except json.JSONDecodeError:
        return data
    for k in ("host", "pid"):
        d.pop(k, None)
    return json.dumps(d, indent=1).encode()


def pack(name: str, out_dir: Path, runs_dir: Path = RUNS_DIR, tag: str | None = None) -> dict:
    """runs/<name> -> out_dir/<name>.tar.gz (+ one bundle per APART file); returns the manifest entry"""
    src = runs_dir / name
    if not (src / "model.pt").exists():
        raise FileNotFoundError(f"{src} has no model.pt")
    out_dir.mkdir(parents=True, exist_ok=True)
    bundles = {}   # bundle name -> list of (relative path, absolute path)
    main = []
    for p in sorted(src.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(src).as_posix()
        if p.name in NEVER_SHIPPED or p.name.endswith(".pid"):
            continue
        if rel in APART:
            bundles.setdefault(f"{name}-{APART[rel]}", []).append((rel, p))
        else:
            main.append((rel, p))
    bundles = {f"{name}": main, **bundles}
    # one timestamp for every member, taken from the run itself, so a copy of the run on
    # another machine (different file mtimes, same bytes) packs to the same archive
    stamp = 0
    try:
        started = json.loads((src / "run.json").read_text()).get("started")
        stamp = int(calendar.timegm(time.strptime(started, "%Y-%m-%dT%H:%M:%S"))) if started else 0
    except (OSError, ValueError, json.JSONDecodeError):
        pass
    entry = {"assets": {}}
    for bname, files in bundles.items():
        out = out_dir / f"{bname}.tar.gz"
        print(f"packing {out.name}: {len(files)} files")
        # deterministic: the same run packed on another machine gives the same bytes and the
        # same checksum (no gzip timestamp, no owner names, members in sorted order)
        with open(out, "wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=3) as gz, \
                tarfile.open(fileobj=gz, mode="w", format=tarfile.PAX_FORMAT) as tf:
            for rel, p in files:
                data = _scrub_run_json(p.read_bytes()) if rel == "run.json" else None
                info = tarfile.TarInfo(f"{name}/{rel}")
                info.size = len(data) if data is not None else p.stat().st_size
                info.mtime = stamp
                info.mode = 0o644
                info.uid = info.gid = 0
                info.uname = info.gname = ""
                with (io.BytesIO(data) if data is not None else open(p, "rb")) as fh:
                    tf.addfile(info, fh)
        entry["assets"][out.name] = {"bytes": out.stat().st_size, "sha256": sha256(out),
                                     "optional": bname != name,
                                     "contents": "the run" if bname == name else f"{[r for r, _ in files][0]} (optional)"}
        print(f"  {out.stat().st_size / 1e6:.0f} MB  sha256 {entry['assets'][out.name]['sha256'][:12]}")
    # what the timeline knows about it, so `list` can describe a bundle before pulling it
    try:
        from .milestones import MILESTONES
        cur = MILESTONES.get(name, {})
        entry.update({k: cur[k] for k in ("version", "label", "hardware", "corpus", "changed") if k in cur})
    except Exception:  # noqa: BLE001
        pass
    try:
        run = json.loads((src / "run.json").read_text())
        entry["schema_version"] = run.get("schema_version", 3)
        entry["params"] = run.get("params")
    except (OSError, json.JSONDecodeError):
        pass
    m = load_manifest()
    if tag:
        m["tag"] = tag
    m["repo"] = REPO
    m.setdefault("models", {})[name] = entry
    # keep the manifest in timeline order, whatever order the bundles were packed in
    try:
        from .milestones import MILESTONES
        m["models"] = dict(sorted(m["models"].items(), key=lambda kv: (MILESTONES.get(kv[0], {}).get("order", 100), kv[0])))
    except Exception:  # noqa: BLE001
        pass
    save_manifest(m)
    return entry


def _download(url: str, dest: Path, expect_bytes: int | None = None):
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "fraudvae-models"})
    with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as out:
        total = int(r.headers.get("Content-Length") or expect_bytes or 0)
        done, t0 = 0, time.time()
        while True:
            b = r.read(1 << 20)
            if not b:
                break
            out.write(b)
            done += len(b)
            if time.time() - t0 > 1.5:
                print(f"\r  {dest.name}: {done / 1e6:.0f}/{total / 1e6:.0f} MB", end="", flush=True)
                t0 = time.time()
    print(f"\r  {dest.name}: {done / 1e6:.0f} MB downloaded")
    tmp.rename(dest)


def pull(name: str, runs_dir: Path = RUNS_DIR, with_paysim_graph=False, keep_archives=False, force=False) -> Path:
    m = load_manifest()
    entry = (m.get("models") or {}).get(name)
    if not entry:
        raise KeyError(f"{name} is not in the manifest ({', '.join(m.get('models', {}))})")
    dest = runs_dir / name
    if (dest / "model.pt").exists() and not force:
        print(f"[{name}] already in {dest} (use --force to re-download)")
        return dest
    cache = runs_dir / ".bundles"
    cache.mkdir(parents=True, exist_ok=True)
    for asset, info in entry["assets"].items():
        if info.get("optional") and not (with_paysim_graph and "paysim" in asset):
            continue
        path = cache / asset
        if not path.exists() or path.stat().st_size != info["bytes"]:
            url = (MIRROR_URL or RELEASE_URL).format(repo=m.get("repo", REPO), tag=m.get("tag", "pretrained"), asset=asset)
            print(f"[{name}] downloading {url}")
            _download(url, path, info["bytes"])
        got = sha256(path)
        if got != info["sha256"]:
            path.unlink(missing_ok=True)
            raise RuntimeError(f"{asset}: checksum mismatch (got {got[:12]}, manifest {info['sha256'][:12]}); deleted, run again")
        print(f"[{name}] {asset} verified, extracting into {runs_dir}")
        with tarfile.open(path, "r:gz") as tf:
            for member in tf.getmembers():
                # the archive only ever holds <name>/...; refuse anything else
                if not member.name.startswith(f"{name}/") or ".." in member.name.split("/"):
                    raise RuntimeError(f"unexpected path in bundle: {member.name}")
            # the data filter (3.12+) refuses links, devices and absolute paths on top of the check above
            tf.extractall(runs_dir, filter="data") if hasattr(tarfile, "data_filter") else tf.extractall(runs_dir)
        if not keep_archives:
            path.unlink(missing_ok=True)
    print(f"[{name}] ready: {dest}")
    return dest


def list_models(runs_dir: Path = RUNS_DIR) -> list[dict]:
    m = load_manifest()
    out = []
    for name, e in (m.get("models") or {}).items():
        main = next((a for a, i in e["assets"].items() if not i.get("optional")), None)
        out.append({"name": name, "version": e.get("version", ""), "label": e.get("label", name),
                    "bytes": e["assets"][main]["bytes"] if main else 0,
                    "optional": [a for a, i in e["assets"].items() if i.get("optional")],
                    "on_disk": (runs_dir / name / "model.pt").exists()})
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(prog="fraudvae models", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="the generations in the manifest and whether each is on disk")
    p = sub.add_parser("pull", help="download a generation into runs/")
    p.add_argument("name", nargs="?")
    p.add_argument("--all", action="store_true", help="every generation in the manifest")
    p.add_argument("--with-paysim-graph", action="store_true", help="also fetch the optional PaySim entity graph (v5, ~800 MB)")
    p.add_argument("--keep-archives", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--runs-dir", default=str(RUNS_DIR))
    p = sub.add_parser("pack", help="write the bundle(s) for a run and record them in the manifest")
    p.add_argument("name", nargs="+")
    p.add_argument("--out", default="dist")
    p.add_argument("--runs-dir", default=str(RUNS_DIR))
    p.add_argument("--tag", default=None, help="the release tag the assets will be attached to")
    args = ap.parse_args(argv)
    if args.cmd == "list":
        for e in list_models():
            print(f"{e['name']:18} {e['version']:3} {e['bytes'] / 1e6:6.0f} MB  {'on disk' if e['on_disk'] else 'not pulled'}  "
                  f"{e['label']}" + (f"  (+ optional: {', '.join(e['optional'])})" if e["optional"] else ""))
        return 0
    if args.cmd == "pull":
        names = list(load_manifest().get("models", {})) if args.all else ([args.name] if args.name else [])
        if not names:
            ap.error("give a generation name or --all")
        for n in names:
            pull(n, Path(args.runs_dir), args.with_paysim_graph, args.keep_archives, args.force)
        return 0
    if args.cmd == "pack":
        for n in args.name:
            pack(n, Path(args.out), Path(args.runs_dir), args.tag)
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
