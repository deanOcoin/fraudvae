"""FastAPI app. `fraudvae serve` or `uvicorn fraudvae.server.app:app`."""
from __future__ import annotations

import faulthandler
import os
import platform
import socket
import threading
import time
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .. import __version__
from ..paths import DATA_DIR, RUNS_DIR, SERVER_LOG, WEB_DIR, ensure_dirs
from ..data import schema
from . import api_data, api_model, api_remote, inference, public

_started = time.time()


def _log(msg: str):
    """terminal + the server log file. the file is what's left when the process dies
    somewhere the terminal scrollback doesn't explain."""
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    try:
        with open(SERVER_LOG, "a") as f:
            f.write(line + "\n")
    except OSError:
        pass


try:
    SERVER_LOG.parent.mkdir(parents=True, exist_ok=True)
    _crash_log = open(SERVER_LOG, "a")   # noqa: SIM115 - must outlive the process
    # a segfault / abort in a native library (openmp runtimes fighting, a torch op)
    # leaves no python traceback. this writes the python stack of every thread to
    # the log at that moment, which turns "the server vanished" into a file:line.
    faulthandler.enable(file=_crash_log, all_threads=True)
except OSError:
    pass


def _preload():
    """warm the other generations in the background so the first visitor's comparison does
    not pay six model loads: FRAUDVAE_PRELOAD=milestones (the default on a public site),
    all, or none"""
    want = os.environ.get("FRAUDVAE_PRELOAD", "milestones" if public.PUBLIC else "none").lower()
    if want in ("", "none", "0", "no"):
        return
    from ..milestones import MILESTONES

    names = [r["name"] for r in inference.list_runs() if r["has_model"] and r["has_calibration"] and r["compatible"]
             and (want == "all" or r["name"] in MILESTONES)]

    def go():
        for n in names:
            try:
                inference.get_model(n)
                _log(f"preloaded run '{n}'")
            except Exception as e:  # noqa: BLE001
                _log(f"could not preload run {n}: {e}")

    threading.Thread(target=go, name="fraudvae-preload", daemon=True).start()


@asynccontextmanager
async def _lifespan(app):
    ensure_dirs()
    _log(f"fraudvae {__version__} starting: python {platform.python_version()}, torch {torch.__version__}, "
         f"{platform.platform()}, pid {os.getpid()}")
    try:
        from ..model.booster import _lgb
        _log(f"lightgbm {_lgb().__version__}")
    except Exception as e:  # noqa: BLE001
        _log(f"lightgbm unavailable, boosters will not load: {e}")
    name = os.environ.get("FRAUDVAE_RUN") or inference.pick_default_run()
    if name:
        _log(f"loading run '{name}'")
        try:
            m = inference.load_run(name)
            _log(f"loaded run '{m.name}' ({m.score_type} score, booster {'yes' if m.booster is not None else 'no'}, "
                 f"blend {'yes' if m.has_blend else 'no'})" + (f"; booster error: {m.booster_error}" if m.booster_error else ""))
        except Exception as e:  # noqa: BLE001
            _log(f"could not load run {name}: {e}")
    else:
        _log("no trained run found yet, inference disabled until one exists")
    _preload()
    _log("ready")
    yield
    _log("shutting down")


_DESCRIPTION = """
The scoring API behind [finfraud.dev](https://www.finfraud.dev). One call does the work:
`POST /api/score` takes a transaction and answers with the chance that it is fraud, whether
it would be flagged at a given false-positive budget, how unusual it looks, and what each
part of the model contributed. `POST /api/score/batch` does the same for a csv.

Everything else is reference material: the feature schema, the training corpus, the model
generations and their records, example transactions to try. There is no key and no account;
the limits are per address (60 scores a minute, 4 batch uploads) and a request past them
gets a 429 with a `Retry-After`. The code, the pretrained models and how to run the same
API on your own machine are in the
[repository](https://github.com/deanOcoin/fraudvae).
"""

app = FastAPI(title="FinFraud", version=__version__, description=_DESCRIPTION, docs_url=None, redoc_url=None,
              openapi_url="/api/openapi.json", lifespan=_lifespan,
              openapi_tags=[{"name": "scoring", "description": "score transactions"},
                            {"name": "data", "description": "what the models were trained on and what they take"},
                            {"name": "models", "description": "the generations and their records"}])
app.include_router(api_data.router)
app.include_router(api_model.router)
if not public.PUBLIC:
    # the remote-training api provisions and drives machines over ssh with the app's own key:
    # a workbench feature, never a public one
    app.include_router(api_remote.router)

# what the public site answers with 403 is not part of its api either: the reference lists only
# what a visitor can call
_HIDDEN = {"/api/train/local", "/api/data/fetch", "/api/data/build", "/api/data/prepare", "/api/tasks",
           "/api/tasks/{task_id}", "/api/runs/{name}/load", "/api/history/{source}/{customer_id}"}
_openapi_full = app.openapi


def _openapi():
    schema = _openapi_full()
    if public.PUBLIC:
        return {**schema, "paths": {k: v for k, v in schema["paths"].items() if k not in _HIDDEN}}
    return schema


app.openapi = _openapi


@app.get("/api/docs", include_in_schema=False)
def api_docs():
    """the interactive api reference. swagger-ui is served from this origin (fraudvae/web/vendor), and
    the page has no inline script, so the site's content security policy can be script-src 'self'"""
    return FileResponse(WEB_DIR / "apidocs.html")


@app.get("/api/meta", summary="Server status", tags=["models"])
def meta():
    """Version, device, uptime, which generations are resident and what the active one can
    do, and the public-mode limits. Answers only once a model is loaded, which makes it the
    address to point an uptime monitor at."""
    m = inference.active()
    dev = "cuda" if torch.cuda.is_available() else ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")
    return {
        "version": __version__, "host": None if public.PUBLIC else socket.gethostname(),
        # the kernel and libc versions are of no use to a visitor and of some to an attacker
        "platform": platform.system() if public.PUBLIC else platform.platform(),
        "python": platform.python_version(), "torch": torch.__version__, "device": dev, "uptime": time.time() - _started,
        "data_dir": None if public.PUBLIC else str(DATA_DIR), "runs_dir": None if public.PUBLIC else str(RUNS_DIR),
        "server_log": None if public.PUBLIC else str(SERVER_LOG),
        "schema_version": schema.SCHEMA_VERSION,
        "public": public.limits(),
        "loaded_runs": inference.loaded_models(),
        "active_run": None if m is None else {
            "name": m.name, "score_type": m.score_type, "n_samples": m.n_samples,
            "sources": list(m.meta.get("source_index", {})), "params": m.run.get("params"),
            "device_name": m.run.get("device_name"), "mean_test_auroc": (m.eval or {}).get("mean_auroc"),
            "mean": (m.eval or {}).get("mean"), "generic_test_auroc": ((m.eval or {}).get("generic") or {}).get("auroc"),
            "has_calibration": m.cal is not None, "has_booster": m.booster is not None, "has_blend": m.has_blend,
            "booster_trees": (m.booster.info.get("best_iteration") if m.booster is not None else None),
            "booster_error": m.booster_error,
            "schema_version": m.schema_version,
            # v5: what the run can take next to the transaction
            "history_window": m.window, "has_seqhead": m.seqhead is not None,
            "heads": {name: {"label": info.get("label", name), "fields": (info.get("numeric") or []) + (info.get("categorical") or []),
                             "val_auroc": info.get("val_auroc")} for name, info in m.head_info.items()},
            "graph": m.graph.describe() if m.graph is not None else None,
            "specialists": m.specialists.describe() if m.specialists is not None else None,
            "text": m.text.describe() if m.text is not None else None,
            "behaviour": m.behaviour.describe() if m.behaviour is not None else None,
        },
    }


@app.get("/")
def index():
    return FileResponse(WEB_DIR / "index.html")


app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")
