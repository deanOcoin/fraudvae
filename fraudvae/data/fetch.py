"""Download / locate raw data and write the lock manifest.

    python -m fraudvae.data.fetch              # everything it can get
    python -m fraudvae.data.fetch --source fdh --source capone

Kaggle sources are only pulled automatically if the kaggle CLI is configured
(KAGGLE_USERNAME/KAGGLE_KEY or ~/.kaggle/kaggle.json). Otherwise it prints what
to download and where to put it, and `build` will unzip whatever it finds.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from ..paths import DATA_DIR, RAW_DIR, ensure_dirs
from .adapters.base import available, extract, source_dir
from .registry import ORDER, SOURCES

LOCK = DATA_DIR / "manifest.lock.json"


def sha256(path: Path, chunk=1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _git_clone(url: str, dest: Path):
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", "--depth", "1", url, str(dest)], check=True)


def _download(url: str, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 1024:
        return
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "fraudvae/0.4"})
    with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as out:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        t0 = time.time()
        while True:
            b = r.read(1 << 20)
            if not b:
                break
            out.write(b)
            done += len(b)
            if total and time.time() - t0 > 2:
                print(f"\r  {dest.name}: {done / 1e6:.0f}/{total / 1e6:.0f} MB", end="", flush=True)
                t0 = time.time()
    print()
    tmp.rename(dest)


def _kaggle_configured() -> bool:
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return shutil.which("kaggle") is not None
    return (Path.home() / ".kaggle" / "kaggle.json").exists() and shutil.which("kaggle") is not None


def _kaggle_download(ref: str, dest: Path):
    kind, slug = ref.split("/", 1)
    dest.mkdir(parents=True, exist_ok=True)
    if kind == "competitions":
        cmd = ["kaggle", "competitions", "download", "-c", slug, "-p", str(dest)]
    else:
        cmd = ["kaggle", "datasets", "download", "-d", slug, "-p", str(dest)]
    subprocess.run(cmd, check=True)


def fetch_source(key: str, raw_dir: Path = RAW_DIR, quiet=False) -> bool:
    src = SOURCES[key]
    sdir = source_dir(raw_dir, key)
    if available(raw_dir, key):
        if not quiet:
            print(f"[{key}] already present")
        return True

    if src.access == "github":
        name = src.git_url.rstrip("/").split("/")[-1]
        print(f"[{key}] cloning {src.git_url}")
        _git_clone(src.git_url, sdir / name)
    elif src.access == "github_lfs":
        for rel, url in src.lfs_urls:
            print(f"[{key}] downloading {url}")
            _download(url, sdir / rel)
            extract(sdir / rel, sdir)
        if not available(raw_dir, key) and src.kaggle_ref and _kaggle_configured():
            _kaggle_download(src.kaggle_ref, sdir)
    elif src.access == "kaggle":
        if _kaggle_configured():
            print(f"[{key}] kaggle download {src.kaggle_ref}")
            try:
                _kaggle_download(src.kaggle_ref, sdir)
            except subprocess.CalledProcessError as e:
                print(f"[{key}] kaggle download failed ({e}); falling back to manual")
        if not available(raw_dir, key):
            print(f"[{key}] MISSING. Download {src.url} and drop the zip into {sdir}/ (or {raw_dir}/)")
            return False
    ok = available(raw_dir, key)
    print(f"[{key}] {'ok' if ok else 'still missing'}")
    return ok


def check_links(keys=None, timeout=20) -> list[tuple[str, str, str]]:
    """does every recorded location still answer? (page url, git repository, direct download
    urls, kaggle listing). Returns (source, url, status) rows and prints them; the point is
    to notice a moved dataset before a user does"""
    import subprocess as sp
    import urllib.error

    rows = []

    def head(url):
        req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "fraudvae-fetch-check"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return f"ok {r.status}"
        except urllib.error.HTTPError as e:
            if e.code in (403, 405):     # kaggle answers HEAD with 403/405 for everything; try a GET
                try:
                    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "fraudvae-fetch-check"}), timeout=timeout) as r:
                        return f"ok {r.status}"
                except urllib.error.HTTPError as e2:
                    return f"HTTP {e2.code}"
                except Exception as e2:  # noqa: BLE001
                    return f"error {type(e2).__name__}"
            return f"HTTP {e.code}"
        except Exception as e:  # noqa: BLE001
            return f"error {type(e).__name__}: {e}"

    for key in keys or ORDER:
        src = SOURCES[key]
        rows.append((key, src.url, head(src.url)))
        if src.git_url:
            r = sp.run(["git", "ls-remote", "--exit-code", "-h", src.git_url], capture_output=True, text=True, timeout=timeout * 3)
            rows.append((key, src.git_url + " (git)", "ok" if r.returncode == 0 else f"git exit {r.returncode}"))
        for _, url in src.lfs_urls:
            rows.append((key, url, head(url)))
        if src.kaggle_ref:
            kind, slug = src.kaggle_ref.split("/", 1)
            kurl = f"https://www.kaggle.com/{kind}/{slug}"
            if kurl != src.url:
                rows.append((key, kurl, head(kurl)))
    width = max(len(u) for _, u, _ in rows)
    for key, url, status in rows:
        print(f"{key:10} {url:<{width}}  {status}")
    bad = [r for r in rows if not r[2].startswith("ok")]
    print(f"\n{len(rows) - len(bad)}/{len(rows)} locations answer" + (f"; check: {', '.join(sorted({r[0] for r in bad}))}" if bad else ""))
    return rows


def write_lock(raw_dir: Path = RAW_DIR, keys=None):
    keys = keys or ORDER
    lock = {"raw_dir": str(raw_dir), "generated": time.strftime("%Y-%m-%dT%H:%M:%S"), "sources": {}}
    if LOCK.exists():
        try:
            lock["sources"] = json.loads(LOCK.read_text()).get("sources", {})
        except json.JSONDecodeError:
            pass
    for key in keys:
        if not available(raw_dir, key):
            lock["sources"].pop(key, None)
            continue
        from .adapters.base import require

        files = []
        for rel in SOURCES[key].files:
            p = require(raw_dir, key, rel)   # resolves a|b alternatives
            files.append({"path": str(p.relative_to(raw_dir)), "bytes": p.stat().st_size, "sha256": sha256(p)})
        lock["sources"][key] = {"url": SOURCES[key].url, "license": SOURCES[key].license, "files": files}
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    LOCK.write_text(json.dumps(lock, indent=2))
    return lock


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", action="append", help="limit to these sources (repeatable)")
    ap.add_argument("--raw-dir", default=str(RAW_DIR))
    ap.add_argument("--no-lock", action="store_true", help="skip hashing files into manifest.lock.json")
    ap.add_argument("--check", action="store_true", help="only check that every recorded download location still answers")
    args = ap.parse_args(argv)
    if args.check:
        bad = [r for r in check_links(args.source) if not r[2].startswith("ok")]
        return 0 if not bad else 2
    ensure_dirs()
    raw_dir = Path(args.raw_dir)
    keys = args.source or ORDER
    got = [k for k in keys if fetch_source(k, raw_dir)]
    if not args.no_lock:
        print("hashing raw files...")
        write_lock(raw_dir, got)
    missing = [k for k in keys if k not in got]
    print(f"\n{len(got)}/{len(keys)} sources available" + (f", missing: {', '.join(missing)}" if missing else ""))
    return 0 if not missing else 2


if __name__ == "__main__":
    sys.exit(main())
