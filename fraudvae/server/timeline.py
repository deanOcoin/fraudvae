"""The model timeline: every generation of the model that can be loaded, in order, with what
it was trained on, how long it took, what it scored and what changed. Read by the model
page and the score tab's model picker; the curated notes (fraudvae/milestones.py) are the
short version of docs/TRAINING_NOTES.md, and the numbers come from each run's own files.

A run not in MILESTONES still appears (sorted after the known ones), with the numbers and
none of the commentary: a run someone trained themselves, say."""
from __future__ import annotations

import json
from pathlib import Path

from ..milestones import MILESTONES
from .inference import list_runs, run_schema_version

def _train_log_minutes(run_dir: Path) -> dict:
    """wall times from the run's own log: the sum of the epoch lines is the VAE's training
    time, the last post-training line the booster side (a run can have several post passes,
    the last one is what ships), 'done. total' the whole train command"""
    out = {}
    p = run_dir / "train.log"
    if not p.exists():
        return out
    epochs, seen = 0.0, False
    try:
        for line in p.read_text(errors="replace").splitlines():
            if " done in " in line and "] epoch " in line:
                epochs += float(line.split(" done in ", 1)[1].split(" min")[0])
                seen = True
            elif "post-training pass done in" in line:
                out["post_minutes"] = float(line.rsplit("in ", 1)[1].split(" min")[0])
            elif "done. total" in line:
                out["total_minutes"] = float(line.rsplit("total ", 1)[1].split(" min")[0])
    except (ValueError, IndexError):
        pass
    if seen:
        out["epoch_minutes"] = round(epochs, 1)
    # a train command that ran its post stage inline may not have logged the pass's own time
    if "post_minutes" not in out and out.get("total_minutes") and seen and out["total_minutes"] > epochs:
        out["post_minutes"] = round(out["total_minutes"] - epochs, 1)
    return out


def _metrics_minutes(run_dir: Path) -> float | None:
    """the VAE's training time as the sum of the per-epoch records (metrics.jsonl travels with
    a copied checkpoint, so a booster-side refit of another run's VAE reports that VAE's time)"""
    p = run_dir / "metrics.jsonl"
    if not p.exists():
        return None
    total, seen = 0.0, False
    for line in p.read_text(errors="replace").splitlines():
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("epoch_seconds") is not None:
            total += float(rec["epoch_seconds"])
            seen = True
    return round(total / 60, 1) if seen else None


def timeline() -> list[dict]:
    runs = {r["name"]: r for r in list_runs() if r["has_model"]}
    out = []
    from ..paths import RUNS_DIR

    for name, r in runs.items():
        d = RUNS_DIR / name
        cur = MILESTONES.get(name, {})
        st = r.get("status") or {}
        run = r.get("run") or {}
        log_t = _train_log_minutes(d)
        ev = None
        if r.get("has_eval"):
            try:
                ev = json.loads((d / "eval.json").read_text())
            except json.JSONDecodeError:
                ev = None
        ev30 = None
        for f in sorted(d.glob("eval_delay*.json")):
            try:
                ev30 = json.loads(f.read_text())
            except json.JSONDecodeError:
                pass
        sv = run_schema_version(d)
        item = {
            "name": name,
            "version": cur.get("version") or (f"v{sv}" if sv else "v3"),
            "label": cur.get("label") or name,
            "order": cur.get("order", 100),
            "milestone": name in MILESTONES,
            "schema_version": sv or 3,
            "hardware": cur.get("hardware") or run.get("device_name") or "",
            "corpus": cur.get("corpus") or "",
            "what": cur.get("what") or "",
            "changed": cur.get("changed") or "",
            "params": run.get("params"),
            "config": r.get("config_name"),
            "epochs_run": st.get("epoch"),
            "epochs_planned": r.get("epochs"),
            "train_minutes": None,
            "post_minutes": log_t.get("post_minutes"),
            "vae_from": cur.get("vae_from"),
            "has_booster": r.get("has_booster"),
            "has_blend": bool(ev and ev.get("with_booster")),
            "mean_test_auroc": r.get("mean_test_auroc"),
            "mean": r.get("mean"),
            "generic_test_auroc": r.get("generic_test_auroc"),
            "sources": sorted((ev or {}).get("sources", {}).keys()) if ev else list((run.get("prepared_meta") or {}).get("sources", [])),
            "test_sources": r.get("test_sources"),
            "mean_test_auroc_slow_labels": (ev30 or {}).get("mean_auroc"),
            "loaded": False,
        }
        # the VAE's training time: the epoch lines of the log; a run that only refit the
        # booster side on another run's checkpoint (f, h) inherits that run's time, and the
        # status file's elapsed is the fallback when it is the train command's own
        metrics_minutes = _metrics_minutes(d)
        if metrics_minutes is not None:
            item["train_minutes"] = metrics_minutes
        elif log_t.get("epoch_minutes") is not None:
            item["train_minutes"] = log_t["epoch_minutes"]
        elif cur.get("vae_from"):
            base = RUNS_DIR / cur["vae_from"]
            item["train_minutes"] = _train_log_minutes(base).get("epoch_minutes") if base.exists() else None
        elif st.get("elapsed") and st.get("stage") != "post":
            item["train_minutes"] = round(st["elapsed"] / 60, 1)
        # the nine sources every generation since v4 shares, for a like-for-like number
        shared = ["fdh", "capone", "tabformer", "ulb", "paysim", "sparkov", "banksim", "ieee", "amlworld"]
        ts = r.get("test_sources") or {}
        if all(s in ts for s in shared):
            item["mean_shared9"] = sum(ts[s] for s in shared) / 9
        out.append(item)
    out.sort(key=lambda x: (x["order"], x["name"]))
    return out
