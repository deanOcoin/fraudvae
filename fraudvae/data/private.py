"""v7: adapters that live outside the repository.

A partner's transactions never enter this public tree, yet they should go through the same
`fetch / build / prepare` as every public source. `FRAUDVAE_ADAPTERS` names a directory; every
`<key>.py` in it is a source adapter like the ones under `data/adapters/`, and `<key>` joins the
schema's source list after the public ones, so every public index stands. Such a module exposes

    SOURCE = {"name": ..., "access": "local", "url": ..., "license": ..., "citation": ..., "context": ...,
              "label_rate_hint": ..., "approx_rows": ..., "files": (...), "notes": ...}   # Source(...) fields, key added
    def load(raw_dir) -> DataFrame      # the unified frame, ending in schema.finalize(df, KEY)

Keys are read from the file names alone (no import) when the schema loads, and the module is
imported only when the registry or an adapter is asked for it, so a broken private adapter
breaks `fraudvae build --source <key>` and nothing else. Nothing here is ever a model input:
a run trained with a private source carries its own source count, as every run does.
"""
from __future__ import annotations

import importlib.util
import os
import re
from pathlib import Path

ENV = "FRAUDVAE_ADAPTERS"
_KEY = re.compile(r"^[a-z][a-z0-9_]{1,40}$")


def directory() -> Path | None:
    raw = os.environ.get(ENV)
    if not raw:
        return None
    p = Path(raw).expanduser()
    return p if p.is_dir() else None


def scan(public_keys=()) -> list[str]:
    """the private keys, in name order, from the file names; a key that collides with a public
    source or is not a plain identifier is skipped rather than shadowing anything"""
    d = directory()
    if d is None:
        return []
    out = []
    for p in sorted(d.glob("*.py")):
        key = p.stem
        if key.startswith("_") or not _KEY.match(key) or key in public_keys:
            continue
        out.append(key)
    return out


def module(key: str):
    """import `<key>.py` from the private directory (fresh each call; adapters are stateless)"""
    d = directory()
    if d is None or not (d / f"{key}.py").exists():
        raise KeyError(key)
    spec = importlib.util.spec_from_file_location(f"fraudvae_private_adapters.{key}", d / f"{key}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "load"):
        raise ImportError(f"private adapter {key}: no load(raw_dir) function")
    return mod


def source_fields(key: str) -> dict:
    """the SOURCE dict of a private adapter with its key filled in, for the registry"""
    mod = module(key)
    d = dict(getattr(mod, "SOURCE", {}) or {})
    d.setdefault("key", key)
    d.setdefault("name", key)
    d.setdefault("access", "local")
    d.setdefault("url", "")
    d.setdefault("license", "private")
    d.setdefault("citation", "")
    d.setdefault("context", "a private source")
    d.setdefault("label_rate_hint", 0.0)
    d.setdefault("approx_rows", 0)
    d.setdefault("files", ())
    d["files"] = tuple(d["files"])
    return d
