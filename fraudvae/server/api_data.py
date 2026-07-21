"""/api/datasets, /api/schema, /api/examples, build + prepare triggers"""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..data import schema
from ..data.adapters.base import available
from ..data.build import SAMPLES, STATS
from ..data.fetch import LOCK
from ..data.prepare import DERIVED_CATS, model_categorical_features, model_numeric_features
from ..paths import PREPARED_DIR, PROCESSED_DIR, RAW_DIR
from ..data.registry import ORDER, SOURCES
from . import tasks

router = APIRouter(prefix="/api")


def _clean(o):
    # json can't do NaN and starlette just throws
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
        return None
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    return o


@router.get("/schema")
def get_schema():
    stats = json.loads(STATS.read_text()) if STATS.exists() else {"sources": {}}
    nums = []
    for f in model_numeric_features():
        nums.append({"name": f.name, "label": f.label, "unit": f.unit, "group": f.group, "transform": f.transform,
                     "coverage": {k: s["coverage"].get(f.name, 0) for k, s in stats["sources"].items()}})
    cats = []
    for c in model_categorical_features():
        cats.append({**c, "coverage": {k: s["coverage"].get(c["name"], 1.0 if c["name"] in DERIVED_CATS else 0)
                                       for k, s in stats["sources"].items()}})
    return {
        "schema_version": schema.SCHEMA_VERSION,
        "sources": [{"key": k, "name": SOURCES[k].name, "context": SOURCES[k].context, "built": k in stats["sources"]}
                    for k in ORDER],
        "numeric": nums, "categorical": cats,
        "groups": {"core": "Transaction", "velocity": "Customer history", "merchant": "Merchant side",
                   "account": "Account & customer", "ext": "Source specific", "ulb_pca": "ULB PCA components"},
    }


@router.get("/datasets")
def get_datasets():
    stats = json.loads(STATS.read_text()) if STATS.exists() else {"sources": {}}
    lock = json.loads(LOCK.read_text()) if LOCK.exists() else {"sources": {}}
    out = []
    for k in ORDER:
        s = SOURCES[k]
        entry = {"key": k, "name": s.name, "access": s.access, "url": s.url, "license": s.license, "citation": s.citation,
                 "context": s.context, "approx_rows": s.approx_rows, "label_rate_hint": s.label_rate_hint, "notes": s.notes,
                 "files": list(s.files), "kaggle_ref": s.kaggle_ref,
                 "raw_present": available(RAW_DIR, k), "built": k in stats["sources"],
                 "stats": stats["sources"].get(k), "lock": lock.get("sources", {}).get(k)}
        out.append(entry)
    total_rows = sum(s["rows"] for s in stats["sources"].values())
    total_fraud = sum(s["fraud"] for s in stats["sources"].values())
    prepared = []
    if PREPARED_DIR.exists():
        for d in sorted(PREPARED_DIR.iterdir()):
            m = d / "meta.json"
            if m.exists():
                meta = json.loads(m.read_text())
                prepared.append({k: meta.get(k) for k in ("name", "created", "counts", "fraud_counts", "sources", "max_rows_per_source")})
    return _clean({"sources": out, "total_rows": total_rows, "total_fraud": total_fraud, "generated": stats.get("generated"),
                   "raw_dir": str(RAW_DIR), "processed_dir": str(PROCESSED_DIR), "prepared": prepared,
                   "vocab": {c.name: list(c.values) for c in schema.CATEGORICAL}})


def _row_to_features(row: pd.Series) -> dict:
    """unified-schema parquet row -> the raw-units dict the score form uses"""
    feats = {}
    for f in schema.NUMERIC:
        v = row.get(f.name)
        if v is not None and not (isinstance(v, float) and math.isnan(v)):
            feats[f.name] = round(float(v), 4)
    for c in schema.CATEGORICAL:
        code = int(row.get(c.name, 0) or 0)
        if code:
            feats[c.name] = c.values[code]
    return feats


@router.get("/examples")
def get_examples(source: str | None = None, fraud: int | None = None, n: int = 1, seed: int | None = None):
    if not SAMPLES.exists():
        raise HTTPException(404, "no samples yet, run build")
    df = pd.read_parquet(SAMPLES)
    if source:
        df = df[df["source"] == source]
    if fraud is not None:
        df = df[df["is_fraud"] == fraud]
    if df.empty:
        raise HTTPException(404, "no matching sample rows")
    rows = df.sample(n=min(n, len(df)), random_state=seed)
    out = []
    for _, row in rows.iterrows():
        wide = row.get("wide") if "wide" in row else None
        out.append({"source": str(row["source"]), "txn_id": str(row["txn_id"]), "is_fraud": int(row["is_fraud"]),
                    "split": str(row["split"]), "fraud_scenario": int(row.get("fraud_scenario", 0) or 0),
                    "features": _row_to_features(row),
                    # v5: the source's raw columns for this row, to hand back to /api/score as `wide`
                    "wide": json.loads(wide) if isinstance(wide, str) and wide.startswith("{") else None,
                    # v5: the row's entity codes, to hand back as `entities` for the graph module
                    "entities": {k: int(row[k]) for k in schema.ENTITY_COLUMNS if k in row and row[k] == row[k] and int(row[k]) >= 0}})
    return out


class BuildRequest(BaseModel):
    sources: list[str] | None = None
    max_rows: int | None = None


@router.post("/data/fetch")
def post_fetch(req: BuildRequest):
    args = ["fraudvae.data.fetch"]
    for s in req.sources or []:
        args += ["--source", s]
    t = tasks.run_subprocess("fetch", "fetch raw data", args)
    return {"task": t.id}


@router.post("/data/build")
def post_build(req: BuildRequest):
    args = ["fraudvae.data.build"]
    for s in req.sources or []:
        args += ["--source", s]
    if req.max_rows:
        args += ["--max-rows", str(req.max_rows)]
    t = tasks.run_subprocess("build", "build unified parquet", args)
    return {"task": t.id}


class PrepareRequest(BaseModel):
    name: str = "default"
    sources: list[str] | None = None
    max_rows_per_source: int | None = None


@router.post("/data/prepare")
def post_prepare(req: PrepareRequest):
    args = ["fraudvae.data.prepare", "--name", req.name]
    for s in req.sources or []:
        args += ["--source", s]
    if req.max_rows_per_source:
        args += ["--max-rows-per-source", str(req.max_rows_per_source)]
    t = tasks.run_subprocess("prepare", f"prepare set '{req.name}'", args)
    return {"task": t.id}
