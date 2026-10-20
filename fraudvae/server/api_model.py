from __future__ import annotations

import io
import json

import numpy as np
import pandas as pd
from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from ..data.schema import SCHEMA_VERSION
from ..paths import CONFIG_DIR, PREPARED_DIR, RUNS_DIR
from . import inference, public, tasks
from .api_data import _clean

router = APIRouter(prefix="/api")


def _need_model() -> inference.ActiveModel:
    m = inference.active()
    if m is None:
        raise HTTPException(409, "no model loaded. train one (Model tab) or load a run")
    return m


class ScoreRequest(BaseModel):
    source: str | None = None
    features: dict
    fpr_target: str = "fpr_0.01"
    wide: dict | None = None            # v5: the source's own raw columns, keyed by raw name
    history: list[dict] | None = None   # v5: previous transactions, oldest first: {features, ts | secs_before}
    customer_id: str | None = None      # v5: let the server's entity store supply / remember the history
    ts: float | None = None             # v5: this transaction's epoch seconds (default now)
    remember: bool = True               # v5: append this transaction to the customer's stored history
    device: dict | None = None          # v5: device telemetry payload (os, browser, screen, session...), see payloads.py
    entities: dict | None = None        # v5: {customer, counterparty, device, email, address, card: code} for the graph module
    text: dict | None = None            # v5: {url, description, memo, profile ...} for the text heads
    session: dict | None = None         # v5: {mouse, keystroke, touch, motion: [events]} for the behaviour verifiers
    label_delay_days: float | None = None   # v5: how long this deployment's fraud labels take to arrive; past 7 days the
                                            # one-week fraud-history block is blanked before the booster sees it
    run: str | None = None                  # which generation scores it (default: the active model)
    compare: list[str] | None = None        # other runs to score the same transaction on, for the comparison strip


def _model_named(name: str | None) -> inference.ActiveModel:
    if not name:
        return _need_model()
    try:
        return inference.get_model(name)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except inference.IncompatibleRun as e:
        raise HTTPException(409, str(e))


def _summary(r: dict) -> dict:
    """the few numbers of a receipt that the comparison strip shows"""
    nov = r.get("novelty") or {}
    return {"run": r["model"]["run"], "version_note": r["model"].get("note"), "legacy": r["model"].get("legacy"),
            "bank_known": r["model"].get("bank_known"), "fields_used": r["model"].get("fields_used"),
            "p_fraud": r.get("p_fraud"), "flag": r.get("flag"), "verdict_from": r.get("verdict_from"),
            "legit_percentile": r.get("legit_percentile"), "novelty_percentile": nov.get("legit_percentile"),
            "novelty_flag": nov.get("flag"), "booster_prob": (r.get("supervised") or {}).get("prob"),
            "signals": {k: True for k in ("local", "sequence", "device", "graph", "text", "behaviour")
                        if isinstance(r.get(k), dict) and r[k].get("present")}}


@router.post("/score")
def post_score(req: ScoreRequest, request: Request):
    public.limit("score", request)
    m = _model_named(req.run)
    # a public site keeps no visitor state: histories must come with the request
    remember, customer_id = (False, None) if public.PUBLIC else (req.remember, req.customer_id)
    kw = dict(wide=req.wide, history=req.history, customer_id=customer_id, ts=req.ts, remember=remember,
              device=req.device, entities=req.entities, text=req.text, session=req.session, label_delay_days=req.label_delay_days)
    try:
        out = m.score(req.features, req.source, req.fpr_target, **kw)
        if req.compare:
            rows = [_summary(out)]
            for name in req.compare[: public.COMPARE_MAX]:
                if name == m.name:
                    continue
                other = _model_named(name)
                # the same transaction, read-only: no history remembered under another generation's name
                r = other.score(req.features, req.source, req.fpr_target, **{**kw, "remember": False, "customer_id": None})
                rows.append(_summary(r))
            out["compare"] = rows
        return _clean(out)
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, f"could not score: {type(e).__name__}: {e}")


@router.get("/models")
def get_models():
    """the model timeline: every loadable generation with hardware, training time, numbers and
    what changed, oldest first; which are loaded now and which is the default"""
    from .timeline import timeline

    items = timeline()
    loaded = set(inference.loaded_models())
    act = inference.active()
    for it in items:
        it["loaded"] = it["name"] in loaded
        it["active"] = bool(act and it["name"] == act.name)
    return {"models": items, "active": act.name if act else None}


@router.delete("/history/{source}/{customer_id}")
def delete_history(source: str, customer_id: str):
    """v5: forget a customer's stored history (source 'generic' for the no-bank context)"""
    public.forbid("the customer store")
    m = _need_model()
    if m.store is None:
        raise HTTPException(409, "the active run keeps no customer state (no history encoder, no behaviour verifiers)")
    return {"deleted": m.store.forget("" if source == "generic" else source, customer_id)}


@router.post("/score/batch")
async def post_score_batch(request: Request, file: UploadFile = File(...), source: str | None = None, fpr_target: str = "fpr_0.01",
                           label_delay_days: float | None = None, run: str | None = None):
    public.limit("batch", request)
    m = _model_named(run)
    raw = await file.read()
    cap = public.BATCH_ROWS if public.PUBLIC else 500_000
    if len(raw) > (cap * 4000):
        raise HTTPException(413, f"file too large for {cap:,} rows")
    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, f"not a csv i can read: {e}")
    if len(df) > cap:
        raise HTTPException(413, f"batch too large, {cap:,} rows max" + (" on the public site" if public.PUBLIC else " through the ui"))
    if source and "source" not in df:
        df["source"] = source
    if public.PUBLIC and not public.batch_slots.acquire(blocking=False):
        raise HTTPException(429, "the batch scorer is busy, try again in a minute", headers={"Retry-After": "60"})
    try:
        scored = m.score_frame(df, fpr_target=fpr_target, label_delay_days=label_delay_days)
    finally:
        if public.PUBLIC:
            public.batch_slots.release()
    out = pd.concat([df.reset_index(drop=True), scored], axis=1)
    summary = {"rows": int(len(out)), "flagged": int(out["flag"].sum()), "score_type": m.score_type,
               "verdict_from": "blend" if m.has_blend else "novelty"}
    if "is_fraud" in out:
        from ..model.scoring import auprc, auroc

        y = out["is_fraud"].to_numpy()
        main = out["p_fraud"] if m.has_blend else out["novelty_score"]
        summary["auroc"] = auroc(y, main.to_numpy())
        summary["auprc"] = auprc(y, main.to_numpy())
        summary["novelty_auroc"] = auroc(y, out["novelty_score"].to_numpy())
        if "booster_prob" in out and out["booster_prob"].notna().any():
            summary["booster_auroc"] = auroc(y, out["booster_prob"].to_numpy())
        flagged = out["flag"].to_numpy()
        summary["recall"] = float((flagged & (y == 1)).sum() / max(y.sum(), 1))
        summary["precision"] = float((flagged & (y == 1)).sum() / max(flagged.sum(), 1))
    buf = io.StringIO()
    out.to_csv(buf, index=False)
    headers = {"X-Summary": json.dumps(_clean(summary)), "Content-Disposition": "attachment; filename=scored.csv"}
    return StreamingResponse(iter([buf.getvalue()]), media_type="text/csv", headers=headers)


@router.get("/runs")
def get_runs():
    m = inference.active()
    return _clean({"runs": inference.list_runs(), "active": m.name if m else None})


@router.post("/runs/{name}/load")
def post_load(name: str):
    public.forbid("changing the default model (pick a model per request instead)")
    try:
        m = inference.load_run(name)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except inference.IncompatibleRun as e:
        raise HTTPException(409, str(e))
    return {"active": m.name, "score_type": m.score_type, "has_calibration": m.cal is not None,
            "has_booster": m.booster is not None, "has_blend": m.has_blend}


@router.get("/runs/{name}/metrics")
def get_metrics(name: str):
    d = RUNS_DIR / name
    if not d.exists():
        raise HTTPException(404, "no such run")
    out = {"metrics": [], "status": None, "run": None, "eval": None, "eval_delay": None, "calibration": None, "booster": None,
           "seqhead": None, "specialists": None, "graph": None, "text": None, "behaviour": None, "graph_bench": None,
           "heads": {},
           "schema_version": inference.run_schema_version(d), "compatible": inference.run_schema_version(d) == SCHEMA_VERSION}
    p = d / "metrics.jsonl"
    if p.exists():
        for line in p.read_text().splitlines():
            try:
                out["metrics"].append(json.loads(line))
            except json.JSONDecodeError:
                continue
    for key, fname in (("status", "status.json"), ("run", "run.json"), ("eval", "eval.json"), ("remote", "remote.json"),
                       ("booster", "booster.json"),
                       # v5 modules, each with its own summary
                       ("seqhead", "seqhead.json"), ("specialists", "specialists.json"), ("graph", "graph.json"),
                       ("text", "text.json"), ("behaviour", "behaviour.json"), ("graph_bench", "graph_bench.json")):
        f = d / fname
        if f.exists():
            try:
                out[key] = json.loads(f.read_text())
            except json.JSONDecodeError:
                pass
    # v5: the same test rows scored as a deployment with month-long label latency would see them
    for f in sorted(d.glob("eval_delay*.json")):
        try:
            out["eval_delay"] = json.loads(f.read_text())
        except json.JSONDecodeError:
            pass
    for f in sorted(d.glob("head_*.json")):
        try:
            out["heads"][f.stem[len("head_"):]] = json.loads(f.read_text())
        except json.JSONDecodeError:
            pass
    cal = d / "calibration.json"
    if cal.exists():
        c = json.loads(cal.read_text())
        c.pop("baselines", None)   # big and not interesting for the ui
        out["calibration"] = c
    if public.PUBLIC:
        # where a run was fetched from is the owner's business, not a visitor's
        out["remote"] = None
        if out.get("run"):
            out["run"] = {k: v for k, v in out["run"].items() if k not in ("host", "pid")}
    return _clean(out)


@router.get("/runs/{name}/log")
def get_log(name: str, tail: int = 300):
    d = RUNS_DIR / name
    for fname in ("train.log", "launcher.log"):
        p = d / fname
        if p.exists():
            lines = p.read_text(errors="replace").splitlines()
            return {"lines": lines[-tail:], "n": len(lines)}
    return {"lines": [], "n": 0}


@router.get("/runs/{name}/latent")
def get_latent(name: str):
    p = RUNS_DIR / name / "latent_map.npz"
    if not p.exists():
        raise HTTPException(404, "no latent map for this run")
    z = np.load(p)
    out = {"xy": z["xy"].round(3).tolist(), "y": z["y"].tolist(), "src": z["src"].tolist(), "score": z["score"].round(3).tolist()}
    if "blend" in z:
        out["blend"] = z["blend"].round(4).tolist()
    return out


@router.get("/configs")
def get_configs():
    out = []
    for p in sorted(CONFIG_DIR.glob("*.yaml")):
        out.append({"name": p.stem, "text": p.read_text()})
    prepared = [d.name for d in PREPARED_DIR.iterdir() if (d / "meta.json").exists()] if PREPARED_DIR.exists() else []
    return {"configs": out, "prepared": prepared}


class TrainRequest(BaseModel):
    config: str = "cpu_small"
    prepared: str = "default"
    name: str | None = None
    epochs: int | None = None
    device: str | None = None


@router.post("/train/local")
def post_train_local(req: TrainRequest):
    public.forbid("training")
    name = req.name or tasks.run_name_from(req.config)
    args = ["fraudvae.train", "--config", req.config, "--prepared", req.prepared, "--name", name]
    if req.epochs:
        args += ["--epochs", str(req.epochs)]
    if req.device:
        args += ["--device", req.device]
    (RUNS_DIR / name).mkdir(parents=True, exist_ok=True)
    t = tasks.run_subprocess("train", f"train {name}", args, log_file=RUNS_DIR / name / "launcher.log")
    t.meta["run"] = name
    return {"task": t.id, "run": name}


@router.get("/tasks")
def get_tasks():
    return tasks.list_tasks()


@router.get("/tasks/{task_id}")
def get_task(task_id: str, tail: int = 400):
    t = tasks.get(task_id)
    if not t:
        raise HTTPException(404, "no such task")
    return _clean(t.to_dict(tail=tail))
