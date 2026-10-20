from __future__ import annotations

import io
import json
import re
import threading

import numpy as np
import pandas as pd
from fastapi import APIRouter, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, field_validator

from ..data.schema import SCHEMA_VERSION
from ..paths import CONFIG_DIR, PREPARED_DIR, RUNS_DIR
from . import inference, public, tasks
from .api_data import _clean

router = APIRouter(prefix="/api")

# a run is a directory name and nothing else: no separators, no dot-dot, nothing a path could
# be built from. Everything that takes a run name goes through here
_RUN_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$")


def _run_dir(name: str):
    if not _RUN_NAME.match(name or "") or ".." in name:
        raise HTTPException(404, "no such run")
    d = RUNS_DIR / name
    if not d.is_dir():
        raise HTTPException(404, "no such run")
    return d


def _error_detail(prefix: str, e: Exception) -> str:
    """what a caller is told when scoring fails: the exception on a workbench, where the
    message is for the person running it; a plain sentence on the public site, where the
    message is for whoever is probing it (the detail goes to the server log)"""
    if public.PUBLIC:
        print(f"{prefix}: {type(e).__name__}: {e}")
        return f"{prefix}: the request could not be scored (malformed fields?)"
    return f"{prefix}: {type(e).__name__}: {e}"


# how much one request may carry. nginx caps the body at 6 MB before any of this runs; these
# keep one request's cpu time proportionate to a transaction, whatever is inside the json
MAX_FEATURES = 600
MAX_WIDE = 3000
MAX_HISTORY = 256
MAX_EVENTS = 20_000        # per behaviour modality
MAX_TEXT = 20_000          # characters per text field
MAX_STRING = 2_000         # any other string value


def _bounded_dict(v, limit: int, what: str, max_str: int = MAX_STRING):
    if v is None:
        return v
    if not isinstance(v, dict):
        raise ValueError(f"{what} must be an object")
    if len(v) > limit:
        raise ValueError(f"{what}: at most {limit} keys")
    for k, x in v.items():
        if len(str(k)) > 200:
            raise ValueError(f"{what}: key too long")
        if isinstance(x, str) and len(x) > max_str:
            raise ValueError(f"{what}.{k}: value too long")
    return v


def _need_model() -> inference.ActiveModel:
    m = inference.active()
    if m is None:
        raise HTTPException(409, "no model loaded. train one (Model tab) or load a run")
    return m


class ScoreRequest(BaseModel):
    model_config = {"json_schema_extra": {"examples": [{
        "source": "sparkov",
        "features": {"amount": 412.80, "hour": 2, "dow": 6, "secs_since_prev_txn": 674, "cust_txn_count_24h": 6,
                     "cust_amount_sum_24h": 3189.02, "amount_over_cust_mean": 3.08, "cust_days_since_first": 615.7},
        "fpr_target": "fpr_0.01",
        "history": [{"features": {"amount": 23.10, "hour": 18, "dow": 5}, "ts": 1760900000}],
        "compare": ["gpu_full-4090-v4", "v5-h200-e"],
    }]}}
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

    @field_validator("features")
    @classmethod
    def _v_features(cls, v):
        return _bounded_dict(v, MAX_FEATURES, "features")

    @field_validator("wide")
    @classmethod
    def _v_wide(cls, v):
        return _bounded_dict(v, MAX_WIDE, "wide")

    @field_validator("device", "entities")
    @classmethod
    def _v_small(cls, v):
        return _bounded_dict(v, 100, "payload")

    @field_validator("text")
    @classmethod
    def _v_text(cls, v):
        return _bounded_dict(v, 20, "text", max_str=MAX_TEXT)

    @field_validator("history")
    @classmethod
    def _v_history(cls, v):
        if v is None:
            return v
        if len(v) > MAX_HISTORY:
            raise ValueError(f"history: at most {MAX_HISTORY} previous transactions")
        for row in v:
            if not isinstance(row, dict):
                raise ValueError("history rows must be objects")
            _bounded_dict(row.get("features", row), MAX_FEATURES, "history.features")
        return v

    @field_validator("session")
    @classmethod
    def _v_session(cls, v):
        if v is None:
            return v
        if not isinstance(v, dict) or len(v) > 8:
            raise ValueError("session must be an object of at most 8 modalities")
        for k, events in v.items():
            # a modality that is not a list is junk the scorer ignores; a list has a ceiling
            if isinstance(events, list) and len(events) > MAX_EVENTS:
                raise ValueError(f"session.{k}: at most {MAX_EVENTS} events")
            if isinstance(events, str) and len(events) > MAX_STRING:
                raise ValueError(f"session.{k}: too long")
        return v

    @field_validator("run", "source", "fpr_target", "customer_id")
    @classmethod
    def _v_short(cls, v):
        if v is not None and len(v) > 100:
            raise ValueError("too long")
        return v

    @field_validator("compare")
    @classmethod
    def _v_compare(cls, v):
        if v is not None and (len(v) > 32 or any(len(x) > 100 for x in v)):
            raise ValueError("compare: too many or too long")
        return v


def _model_named(name: str | None) -> inference.ActiveModel:
    if not name:
        return _need_model()
    _run_dir(name)
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


@router.post("/score", summary="Score one transaction", tags=["scoring"])
def post_score(req: ScoreRequest, request: Request):
    """Score a single transaction and get back the full receipt.

    The only required field is `features`: a flat object of the transaction's fields, named
    as the bank's own export names them (`amount`, `hour`, `secs_since_prev_txn`, ...) or with
    the names in `/api/schema`. Anything the model does not know is ignored; anything missing
    is treated as not observed. `source` names the dataset family the bank belongs to
    (`sparkov`, `ieee`, `paysim`, ...) so the right per-bank model and threshold apply; leave
    it out and the generic model scores it.

    What comes back: `p_fraud` (the calibrated chance of fraud), `flag` (whether it crosses
    the threshold for the requested false-positive budget, `fpr_target`: `fpr_0.001`,
    `fpr_0.01` or `fpr_0.05`), `z_score` and `legit_percentile` (how unusual it is next to the
    bank's legitimate traffic), `verdict_from` (which parts decided), `contributions` (the
    features that pushed the verdict and which way), and a block per part of the model -
    `novelty` (the autoencoder), `supervised` (the booster), `local` (the bank's own model),
    `sequence`, `device`, `graph`, `text`, `behaviour` (present when the request carries what
    they need), `blend` (the final combination with every threshold) - plus `compare`, one
    summary per generation, when other generations were asked for.

    Optional context: `wide` (the source's raw columns as they are), `history` (previous
    transactions of the same customer, oldest first, each `{features, ts}`), `device`
    (telemetry: os, browser, screen), `entities` (customer, device, card, email codes for the
    entity graph), `text` (url, description, memo), `session` (mouse / keystroke / touch /
    motion events), `label_delay_days` (how long this deployment's fraud labels take to
    arrive). `run` picks a generation; `compare` lists others to score the same transaction on.

    On the public site a visitor's history is not stored, so `customer_id` and `remember`
    are ignored, and the limits are 60 requests a minute (burst 20) and 6 generations in a
    comparison. 429 or 503 with `Retry-After` past that; 422 when a field is over its size.
    """
    public.limit("score", request)
    m = _model_named(req.run)
    if public.PUBLIC and not _score_slots.acquire(timeout=3.0):
        raise HTTPException(503, "the scorer is busy, try again in a moment", headers={"Retry-After": "2"})
    try:
        return _score(req, m)
    finally:
        if public.PUBLIC:
            _score_slots.release()


def _score(req: ScoreRequest, m: inference.ActiveModel):
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
        raise HTTPException(400, _error_detail("could not score", e))


# only so many transactions score at once across every visitor: past that the answer is a
# polite 503 rather than a queue that grows until the box swaps
_score_slots = threading.BoundedSemaphore(public.SCORE_CONCURRENCY)


@router.get("/models", summary="The model generations", tags=["models"])
def get_models():
    """Every generation this server can score with, oldest first: the hardware it trained on,
    how long it took, its test numbers (mean AUROC across the sources, the generic head) and
    one line on what changed from the one before. `loaded` says which are resident now,
    `active` which one answers when a request names no `run`."""
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


@router.post("/score/batch", summary="Score a csv file", tags=["scoring"],
             responses={200: {"content": {"text/csv": {}}, "description": "the input rows with the scores appended"}})
async def post_score_batch(request: Request, file: UploadFile = File(..., description="a csv with one transaction per row; a `source` column overrides the query parameter"),
                           source: str | None = Query(None, description="dataset family of the rows (`sparkov`, `ieee`, ...); generic when absent"),
                           fpr_target: str = Query("fpr_0.01", description="false-positive budget the flag is set at: `fpr_0.001`, `fpr_0.01` or `fpr_0.05`"),
                           label_delay_days: float | None = Query(None, description="how long fraud labels take to arrive in this deployment"),
                           run: str | None = Query(None, description="which generation scores the file (default: the active one)")):
    """Upload a csv and get it back with `p_fraud`, `booster_prob`, `novelty_score`,
    `novelty_raw` and `flag` columns appended; the `X-Summary` response header carries the counts (rows, flagged, fraud rate)
    as json. Columns are matched by name the way `/api/score` matches `features`. On the
    public site a file may have at most 5,000 rows and 3,000 columns, 4 uploads a minute,
    two running at once server-wide."""
    public.limit("batch", request)
    m = _model_named(run)
    raw = await file.read()
    cap = public.BATCH_ROWS if public.PUBLIC else 500_000
    if len(raw) > (cap * 4000):
        raise HTTPException(413, f"file too large for {cap:,} rows")
    # the header decides how wide the frame gets; a file that is all commas must not become a
    # million-column frame before the row cap is even checked
    header = raw.split(b"\n", 1)[0]
    if header.count(b",") + header.count(b";") + header.count(b"\t") > 3000:
        raise HTTPException(413, "too many columns")
    try:
        df = pd.read_csv(io.BytesIO(raw), nrows=cap + 1)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, _error_detail("not a csv that could be read", e))
    if len(df) > cap:
        raise HTTPException(413, f"batch too large, {cap:,} rows max" + (" on the public site" if public.PUBLIC else " through the ui"))
    if len(df.columns) > 3000:
        raise HTTPException(413, "too many columns")
    if source and "source" not in df:
        df["source"] = source
    if public.PUBLIC and not public.batch_slots.acquire(blocking=False):
        raise HTTPException(429, "the batch scorer is busy, try again in a minute", headers={"Retry-After": "60"})
    try:
        scored = m.score_frame(df, fpr_target=fpr_target, label_delay_days=label_delay_days)
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, _error_detail("could not score the file", e))
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


@router.get("/runs", summary="The run directories", tags=["models"])
def get_runs():
    """The raw list behind `/api/models`: every run directory with its status, its evaluation
    numbers and what it has (a model, a calibration, a booster, the v5 modules). `/api/models`
    is the readable version; this one is for inspecting a run."""
    m = inference.active()
    return _clean({"runs": [public.scrub_run(r) for r in inference.list_runs()], "active": m.name if m else None})


@router.post("/runs/{name}/load")
def post_load(name: str):
    public.forbid("changing the default model (pick a model per request instead)")
    _run_dir(name)
    try:
        m = inference.load_run(name)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except inference.IncompatibleRun as e:
        raise HTTPException(409, str(e))
    return {"active": m.name, "score_type": m.score_type, "has_calibration": m.cal is not None,
            "has_booster": m.booster is not None, "has_blend": m.has_blend}


@router.get("/runs/{name}/metrics", summary="Everything recorded about a run", tags=["models"])
def get_metrics(name: str):
    """The training curve (`metrics`: one row per epoch), the run's settings and status, the
    evaluation on the test split (`eval`, and `eval_delay` under delayed labels), and the
    report of each side model: calibration, booster, sequence head, specialists, entity graph,
    text heads, behaviour verifiers. A few hundred kilobytes; cached for ten minutes."""
    d = _run_dir(name)
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
    # where a run was fetched from is the owner's business, not a visitor's
    return _clean(public.scrub_run(out))


@router.get("/runs/{name}/log", summary="A run's training log", tags=["models"])
def get_log(name: str, tail: int = Query(300, description="how many lines from the end, at most 5,000")):
    """The last lines of the training log as it was written, so the numbers in the story
    pages can be checked against where they came from."""
    d = _run_dir(name)
    tail = max(1, min(int(tail), 5000))
    for fname in ("train.log", "launcher.log"):
        p = d / fname
        if p.exists():
            lines = public.scrub_log_lines(p.read_text(errors="replace").splitlines())
            return {"lines": lines[-tail:], "n": len(lines)}
    return {"lines": [], "n": 0}


@router.get("/runs/{name}/latent", summary="A run's latent map", tags=["models"])
def get_latent(name: str):
    """A two-dimensional projection of the test split through the autoencoder: `xy` the
    coordinates, `y` the fraud label, `src` the source index, `score` the novelty and `blend`
    the final probability of each point. About three megabytes, what the scatter plot on the
    Score tab draws; cached for ten minutes."""
    p = _run_dir(name) / "latent_map.npz"
    if not p.exists():
        raise HTTPException(404, "no latent map for this run")
    # three megabytes of json that never change for a run: serialised once per file version,
    # not once per visitor (a scanner asking for it 300 times a minute would otherwise own a core)
    key = (name, p.stat().st_mtime_ns)
    with _latent_lock:
        hit = _latent_cache.get(key)
    if hit is None:
        z = np.load(p)
        out = {"xy": z["xy"].round(3).tolist(), "y": z["y"].tolist(), "src": z["src"].tolist(), "score": z["score"].round(3).tolist()}
        if "blend" in z:
            out["blend"] = z["blend"].round(4).tolist()
        hit = json.dumps(out, separators=(",", ":")).encode()
        with _latent_lock:
            for k in [k for k in _latent_cache if k[0] == name]:
                _latent_cache.pop(k, None)
            if len(_latent_cache) >= 12:
                _latent_cache.pop(next(iter(_latent_cache)))
            _latent_cache[key] = hit
    return Response(content=hit, media_type="application/json")


_latent_cache: dict[tuple[str, int], bytes] = {}
_latent_lock = threading.Lock()


@router.get("/configs", summary="The training configurations", tags=["models"])
def get_configs():
    """The yaml of every training configuration that ships with the code (`cpu_small`,
    `gpu_full`, the v5 ones), as text, and the prepared sets a workbench has on disk."""
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
    if not _RUN_NAME.match(name) or ".." in name or not re.match(r"^[A-Za-z0-9_-]{1,40}$", req.config) or not re.match(r"^[A-Za-z0-9_-]{1,40}$", req.prepared):
        raise HTTPException(400, "run, config and prepared-set names are plain names")
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
    if public.PUBLIC:
        return []
    return tasks.list_tasks()


@router.get("/tasks/{task_id}")
def get_task(task_id: str, tail: int = 400):
    public.forbid("background tasks")
    tail = max(1, min(int(tail), 5000))
    t = tasks.get(task_id)
    if not t:
        raise HTTPException(404, "no such task")
    return _clean(t.to_dict(tail=tail))
