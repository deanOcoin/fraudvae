"""Download / locate raw data, write the lock manifest, and check the files against the ones
the shipped generations were built from.

    fraudvae fetch                             # everything it can get, then a verification
    fraudvae fetch --source fdh --source capone
    fraudvae fetch --verify [--quick]          # just the check: size + sha256 per file (quick: sizes)
    fraudvae fetch --check                     # do the recorded download locations still answer?

Every dataset is public. The GitHub-hosted ones need nothing; the Kaggle ones need a free
Kaggle account (IEEE-CIS additionally requires joining its competition and accepting the
rules, and is non-commercial), and Kaggle refuses some VPN exit addresses. They are only
pulled automatically if the kaggle CLI is configured (KAGGLE_USERNAME/KAGGLE_KEY or
~/.kaggle/kaggle.json); otherwise this prints what to download and where to put it, and
`build` unzips whatever it finds. What matters more than where a file came from is that it
is the same file: datasets get re-uploaded and re-cut, and fraudvae/data/expected.json holds
the size and sha256 of every raw file behind the numbers in the docs, which `--verify`
checks. A file that differs will still build, but the results will not be the documented
ones."""
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
EXPECTED = Path(__file__).parent / "expected.json"   # the files the shipped generations were built from


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


def _find_raw(root: Path, name: str, want_bytes: int | None = None) -> Path | None:
    """a raw file by basename anywhere under the source directory (archives may unpack into
    a subfolder; kaggle's zip-in-zip sometimes two). With several candidates - a git-lfs
    pointer next to the real download, say - the one of the expected size wins"""
    if not root.exists():
        return None
    hits = [p for p in [root / name] if p.exists()] + [p for p in sorted(root.rglob(Path(name).name)) if p != root / name]
    hits = [p for p in hits if p.is_file()]
    if not hits:
        return None
    if want_bytes is not None:
        for p in hits:
            if p.stat().st_size == want_bytes:
                return p
    return hits[0]


def _zip_member(root: Path, name: str):
    """(archive, member) when the file is still inside a zip under the source directory"""
    import zipfile

    for z in sorted(root.rglob("*.zip")) if root.exists() else []:
        try:
            with zipfile.ZipFile(z) as zf:
                for m in zf.namelist():
                    if Path(m).name == Path(name).name:
                        return z, m
        except zipfile.BadZipFile:
            continue
    return None


def _sha256_member(archive: Path, member: str, chunk=1 << 22):
    import zipfile

    h = hashlib.sha256()
    n = 0
    with zipfile.ZipFile(archive) as zf, zf.open(member) as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
            n += len(b)
    return h.hexdigest(), n


def _ok(status: str) -> bool:
    return status.startswith(("matches", "not extracted", "size matches", "inside"))


def verify(raw_dir: Path = RAW_DIR, keys=None, quick=False) -> dict:
    """Compare what is on disk with the files the shipped generations were built from
    (fraudvae/data/expected.json): size, then sha256 (quick=True stops at the size). A file
    that is still inside a downloaded zip is hashed in place. For the two sources that are
    git clones the clone's commit is compared instead. Returns {key: {path: status}} and
    prints it; the point is to know, before an hour of building, that the data really is the
    data behind the numbers in the docs."""
    exp = json.loads(EXPECTED.read_text())
    out = {}
    for key in keys or ORDER:
        root = source_dir(raw_dir, key)
        res = out.setdefault(key, {})
        repo = exp.get("repos", {}).get(key)
        if repo:
            clone = next((d for d in root.glob("*/.git")), None) if root.exists() else None
            if clone is None:
                res["clone"] = "missing"
            else:
                r = subprocess.run(["git", "-C", str(clone.parent), "rev-parse", "HEAD"], capture_output=True, text=True)
                head = r.stdout.strip()
                res["clone"] = "matches the trained-on commit" if head == repo["commit"] else f"commit {head[:10]} (trained on {repo['commit'][:10]})"
        files = exp["sources"].get(key, [])
        # alternatives: one of the forms satisfies the other (the tabformer tarball vs its csv)
        alt_for = {f["path"]: f["alternative_to"] for f in files if f.get("alternative_to")}
        for f in files:
            name = f["path"].split("/", 1)[1]
            hit = _find_raw(root, name, f["bytes"])
            if hit is not None:
                size = hit.stat().st_size
                if size != f["bytes"]:
                    res[f["path"]] = f"size {size:,} (trained on {f['bytes']:,}): a different edition"
                elif quick:
                    res[f["path"]] = "size matches (not hashed)"
                else:
                    res[f["path"]] = "matches the trained-on file" if sha256(hit) == f["sha256"] else "same size, different checksum"
                continue
            zm = _zip_member(root, name)
            if zm is not None:
                if quick:
                    res[f["path"]] = f"inside {zm[0].name} (not hashed)"
                else:
                    digest, size = _sha256_member(*zm)
                    res[f["path"]] = ("matches the trained-on file (inside " + zm[0].name + ")") if digest == f["sha256"] \
                        else f"inside {zm[0].name}: differs ({size:,} bytes, trained on {f['bytes']:,})"
                continue
            res[f["path"]] = "missing"
        # an alternative form that is present covers the file it stands for
        for alt, main in alt_for.items():
            if _ok(res.get(alt, "missing")) and res.get(main) == "missing":
                res[main] = f"not extracted yet; {alt.split('/', 1)[1]} is here, build extracts it"
    width = max((len(p) for r in out.values() for p in r), default=10)
    bad = 0
    for key, r in out.items():
        for path, status in r.items():
            bad += 0 if _ok(status) else 1
            print(f"{key:10} {path:<{width}}  {status}")
    print(f"\n{bad} file(s) missing or different" if bad else "\nevery file present matches what the models were trained on")
    return out


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
    ap.add_argument("--verify", action="store_true",
                    help="only compare the files on disk with the ones the shipped generations were built from (size + sha256)")
    ap.add_argument("--quick", action="store_true", help="with --verify: sizes only, no hashing")
    args = ap.parse_args(argv)
    if args.check:
        bad = [r for r in check_links(args.source) if not r[2].startswith("ok")]
        return 0 if not bad else 2
    if args.verify:
        res = verify(Path(args.raw_dir), args.source, quick=args.quick)
        bad = [1 for r in res.values() for st in r.values() if not _ok(st)]
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
    if got:
        print("\nagainst the files the shipped generations were built from:")
        verify(raw_dir, got, quick=args.no_lock)
    return 0 if not missing else 2


if __name__ == "__main__":
    sys.exit(main())
