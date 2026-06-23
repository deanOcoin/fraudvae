"""Shared bits for the per-source adapters: locating raw files (unzipping kaggle
downloads when needed) and a few datetime helpers."""
from __future__ import annotations

import tarfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from ..registry import SOURCES
from ..schema import WIDE_PREFIX


class MissingData(FileNotFoundError):
    pass


def source_dir(raw_dir: Path, key: str) -> Path:
    return Path(raw_dir) / key


def release_memory():
    """give freed memory back to the os. glibc keeps freed arena pages around and arrow's
    pool does the same, which on the 24M row source looked like a 2 GB leak between the
    csv parse and the parquet write. harmless no-op where unsupported"""
    import ctypes
    import gc

    gc.collect()
    try:
        import pyarrow as pa

        pa.default_memory_pool().release_unused()
    except Exception:  # noqa: BLE001
        pass
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:  # noqa: BLE001 - not glibc (macos), nothing to do
        pass


def _find(root: Path, name: str) -> Path | None:
    direct = root / name
    if direct.exists():
        return direct
    hits = list(root.rglob(Path(name).name))
    # prefer the shallowest match
    hits.sort(key=lambda p: len(p.parts))
    return hits[0] if hits else None


def _candidate_archives(raw_dir: Path, key: str):
    src = SOURCES[key]
    sdir = source_dir(raw_dir, key)
    seen = set()
    candidates = []
    if sdir.exists():
        candidates += sorted(sdir.rglob("*"))   # clones nest their archives (DS/transactions.zip)
    if Path(raw_dir).exists():
        candidates += sorted(Path(raw_dir).glob("*"))
    for p in candidates:
        if not p.is_file() or p.suffix.lower() not in (".zip", ".tgz", ".gz", ".tar") or p in seen:
            continue
        if ".git" in p.parts:
            continue
        seen.add(p)
        if p.name in src.kaggle_zip_names or sdir in p.parents:
            yield p
        else:
            # archive dropped at the raw root with some other name: peek inside
            try:
                if archive_contains(p, src.files):
                    yield p
            except Exception:
                pass


def archive_contains(archive: Path, names) -> bool:
    wanted = {Path(alt).name for n in names for alt in n.split("|")}
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as z:
            inside = {Path(n).name for n in z.namelist()}
    elif tarfile.is_tarfile(archive):
        with tarfile.open(archive) as t:
            inside = {Path(n).name for n in t.getnames()}
    else:
        return False
    return bool(wanted & inside)


def extract(archive: Path, dest: Path, members=()):
    """members: basenames to pull out; empty = everything. the amlworld zip is 8 GB of
    which we want 500 MB, so partial extraction matters there"""
    dest.mkdir(parents=True, exist_ok=True)
    want = set(members)
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as z:
            names = [n for n in z.namelist() if not want or Path(n).name in want]
            z.extractall(dest, members=names)
    elif tarfile.is_tarfile(archive):
        with tarfile.open(archive) as t:
            ms = [m for m in t.getmembers() if not want or Path(m.name).name in want]
            t.extractall(dest, members=ms)  # trusted archives only (we checked the sha first)
    else:
        raise ValueError(f"not an archive: {archive}")
    # kaggle sometimes nests zips in zips
    for inner in dest.rglob("*.zip"):
        if inner != archive and zipfile.is_zipfile(inner):
            with zipfile.ZipFile(inner) as z:
                z.extractall(inner.parent)


def require(raw_dir: Path, key: str, name: str) -> Path:
    """Path to a raw file, extracting archives as needed. Raises MissingData.
    name may list alternatives separated by '|' (the kaggle and github copies of
    tabformer call the same csv different things)."""
    root = source_dir(raw_dir, key)
    names = name.split("|")
    for n in names:
        hit = _find(root, n) if root.exists() else None
        if hit is not None:
            return hit
    for archive in _candidate_archives(raw_dir, key):
        extract(archive, root, members=SOURCES[key].extract_members)
        for n in names:
            hit = _find(root, n)
            if hit is not None:
                return hit
    src = SOURCES[key]
    raise MissingData(
        f"{key}: could not find {name!r} under {root}. "
        + (f"Download {src.url} and put the zip in {root} (or {raw_dir})." if src.access == "kaggle"
           else f"Run `fraudvae fetch --source {key}`.")
    )


def available(raw_dir: Path, key: str) -> bool:
    try:
        for f in SOURCES[key].files:
            require(raw_dir, key, f)
        return True
    except MissingData:
        return False


# ---- the wide block ----------------------------------------------------------

def wide_encode(raw: pd.DataFrame, cat_cols, max_levels: int = 300):
    """a source's own raw columns -> {wide__name: float32 array} plus the column descriptions
    build writes to <source>.wide.json. Numerics pass through (NaN = missing); categoricals
    become 1-based frequency-rank codes with NaN for missing and max_levels + 1 for "other".
    Only the booster reads these; the VAE never sees them."""
    cat_cols = set(cat_cols)
    out, desc = {}, []
    for c in raw.columns:
        if c in cat_cols:
            s = raw[c].astype("object")
            vc = s.value_counts()
            vals = [str(v) for v in vc.index[:max_levels]]
            lut = {v: i + 1 for i, v in enumerate(vals)}
            codes = s.map(lambda v: lut.get(str(v), max_levels + 1) if v == v and v is not None else np.nan)
            out[WIDE_PREFIX + c] = codes.to_numpy(dtype=np.float32)
            desc.append({"name": c, "kind": "cat", "values": vals})
        else:
            out[WIDE_PREFIX + c] = pd.to_numeric(raw[c], errors="coerce").to_numpy(dtype=np.float32)
            desc.append({"name": c, "kind": "num"})
    return out, desc


# ---- big sources --------------------------------------------------------------

def chunked_frames(key: str, n: int, columns: dict, chunk: int = 1_000_000):
    """For the sources too big for one pandas frame: columns is a dict of
    full-length numpy arrays (unified names, plus 'split'); yields finalized
    frames of `chunk` rows. txn_id is generated as <key>-<row>.

    A column may also be a scalar, or a callable f(start, stop) -> array for
    things cheap enough to compute per chunk (lookups through a user table),
    which keeps 24M-row columns out of RAM. 'split' may be int8 codes from
    schema.time_split_codes."""
    from ..schema import SPLIT_NAMES, empty_frame, finalize

    split_names = np.array(SPLIT_NAMES, dtype=object)
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        df = empty_frame(stop - start)
        df["txn_id"] = pd.array([f"{key}-{i}" for i in range(start, stop)], dtype="string[pyarrow]")
        for name, arr in columns.items():
            if callable(arr):
                val = arr(start, stop)
            elif np.isscalar(arr):
                val = arr
            else:
                val = arr[start:stop]
            if name == "split" and not np.isscalar(val) and getattr(val, "dtype", None) is not None and val.dtype.kind in "iu":
                val = split_names[np.asarray(val)]
            df[name] = val
        yield finalize(df, key)


# ---- time helpers --------------------------------------------------------

def epoch_seconds(dt: pd.Series) -> np.ndarray:
    # NOT astype(int64)//1e9: pandas 3 parses strings to datetime64[us], which
    # silently gave timestamps 1000x too small. numpy handles any resolution.
    dt = pd.to_datetime(dt, errors="coerce")
    vals = dt.to_numpy().astype("datetime64[s]").astype("int64").astype("float64")
    vals[dt.isna().to_numpy()] = np.nan
    return vals


def hour_dow(dt: pd.Series):
    dt = pd.to_datetime(dt, errors="coerce")
    return dt.dt.hour.astype("float32").to_numpy(), dt.dt.dayofweek.astype("float32").to_numpy()


def yes_no(mask: pd.Series | np.ndarray, unknown: pd.Series | np.ndarray | None = None) -> np.ndarray:
    """bool -> code for the (unknown,no,yes) style categoricals"""
    m = np.asarray(mask, dtype=bool)
    out = np.where(m, 2, 1).astype(np.int8)
    if unknown is not None:
        out[np.asarray(unknown, dtype=bool)] = 0
    return out
