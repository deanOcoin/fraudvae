"""Should this run locally or on a rented GPU? Benchmarks the actual model for a
few hundred steps on this machine, extrapolates, and compares against rough
throughput numbers for the GPUs you'd typically rent on vast.ai.

The GPU numbers are ballpark figures for *this* model shape (a ~1M param MLP
VAE, batch 4096-16384). They're meant to pick a tier, not to be precise.
"""
from __future__ import annotations

import json
import math
import platform
import time

import torch

from ..config import Config, load_config
from ..model.vae import build_model
from ..paths import PREPARED_DIR, PROCESSED_DIR

# rows/sec for one epoch of training (fwd+bwd), observed/estimated for this model
# the 4090 row is measured (434k rows/s on vast with the v4 schema, batch 8192 + amp; 516k on v3); the rest are scaled guesses
GPU_TIERS = [
    {"name": "RTX 3090 / 4070 Ti (24/12 GB)", "rows_per_sec": 280_000, "vram_gb": 24, "typical_usd_hr": 0.20},
    {"name": "RTX 4090 (24 GB)", "rows_per_sec": 430_000, "vram_gb": 24, "typical_usd_hr": 0.35},
    {"name": "A100 40/80 GB", "rows_per_sec": 650_000, "vram_gb": 40, "typical_usd_hr": 0.90},
    {"name": "H100 80 GB", "rows_per_sec": 1_000_000, "vram_gb": 80, "typical_usd_hr": 1.90},
    {"name": "A10 / L4 (24 GB)", "rows_per_sec": 300_000, "vram_gb": 24, "typical_usd_hr": 0.25},
    {"name": "T4 (16 GB)", "rows_per_sec": 140_000, "vram_gb": 16, "typical_usd_hr": 0.15},
]

LOCAL_THRESHOLD_HOURS = 1.5     # above this, recommend going remote


def corpus_summary() -> dict:
    """what we'd train on if everything that's built is used"""
    stats_p = PROCESSED_DIR / "stats.json"
    out = {"sources": {}, "rows": 0, "train_rows": 0, "fraud": 0, "parquet_bytes": 0}
    if not stats_p.exists():
        return out
    stats = json.loads(stats_p.read_text())
    for k, s in stats.get("sources", {}).items():
        p = PROCESSED_DIR / s.get("parquet", f"{k}.parquet")
        out["sources"][k] = {"rows": s["rows"], "train_rows": s["splits"]["train"]["rows"], "fraud": s["fraud"],
                             "parquet_bytes": p.stat().st_size if p.exists() else 0}
        out["rows"] += s["rows"]
        out["train_rows"] += s["splits"]["train"]["rows"] - s["splits"]["train"]["fraud"]
        out["fraud"] += s["fraud"]
        out["parquet_bytes"] += out["sources"][k]["parquet_bytes"]
    return out


def prepared_summary(name="default") -> dict | None:
    p = PREPARED_DIR / name / "meta.json"
    if not p.exists():
        return None
    meta = json.loads(p.read_text())
    size = sum(f.stat().st_size for f in (PREPARED_DIR / name).glob("*.npy"))
    return {"name": name, "counts": meta["counts"], "fraud_counts": meta["fraud_counts"], "bytes": size,
            "sources": meta["sources"], "max_rows_per_source": meta.get("max_rows_per_source")}


def benchmark_local(cfg: Config, n_num=57, cat_sizes=None, steps=12, device=None) -> dict:
    """time fwd+bwd of the real model on random data on this machine"""
    cat_sizes = cat_sizes or [5, 9, 18, 6, 4, 3, 3, 9, 3, 3, 3, 6, 6, 25, 8]
    if device is None:
        device = "cuda" if torch.cuda.is_available() else ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")
    dev = torch.device(device)
    meta = {"numeric": [{"name": f"f{i}"} for i in range(n_num)], "categorical": [{"size": k} for k in cat_sizes], "source_index": {"x": 8}}
    model = build_model(meta, cfg.model).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    bs = cfg.train.batch_size
    if dev.type == "cpu":
        bs = min(bs, 2048)   # cpu throughput is ~linear in batch anyway, no point waiting a minute
    num = torch.randn(bs, n_num, device=dev)
    mask = (torch.rand(bs, n_num, device=dev) > 0.4).float()
    cat = torch.stack([torch.randint(0, k, (bs,), device=dev) for k in cat_sizes], dim=1)
    src = torch.randint(0, 9, (bs,), device=dev)
    model.train()
    # warmup
    for _ in range(3):
        out = model(num, mask, cat, src)
        a, b = model.nll_terms(out, num, mask, cat)
        loss = (a.sum(1) + b.sum(1) + model.kl_terms(out["mu"], out["logvar"]).sum(1)).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    if dev.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(steps):
        out = model(num, mask, cat, src)
        a, b = model.nll_terms(out, num, mask, cat)
        loss = (a.sum(1) + b.sum(1) + model.kl_terms(out["mu"], out["logvar"]).sum(1)).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    if dev.type == "cuda":
        torch.cuda.synchronize()
    dt = time.time() - t0
    rows_per_sec = steps * bs / dt
    return {"device": str(dev), "device_name": torch.cuda.get_device_name(0) if dev.type == "cuda" else platform.processor() or platform.machine(),
            "batch_size": bs, "rows_per_sec": rows_per_sec, "params": sum(p.numel() for p in model.parameters())}


def advise(config: str | None = None, prepared: str | None = "default", full_corpus=True, epochs=None) -> dict:
    cfg = load_config(config)
    if epochs:
        cfg.train.epochs = epochs
    corpus = corpus_summary()
    prep = prepared_summary(prepared) if prepared else None
    rows = corpus["train_rows"] if full_corpus or not prep else (prep["counts"]["train"] - prep["fraud_counts"]["train"])
    bench = benchmark_local(cfg)
    # eval + calibration passes are roughly 0.6 epoch equivalents, be a bit pessimistic
    work_rows = rows * (cfg.train.epochs + 0.6 * math.ceil(cfg.train.epochs / cfg.train.eval_every) + 1)
    local_hours = work_rows / max(bench["rows_per_sec"], 1) / 3600
    tiers = []
    for t in GPU_TIERS:
        hours = work_rows / t["rows_per_sec"] / 3600
        tiers.append({**t, "est_hours": hours, "est_cost_usd": hours * t["typical_usd_hr"]})
    data_gb = corpus["parquet_bytes"] / 1e9
    prepared_gb_est = rows * (57 * 4 + 57 + 15 * 2 + 10) / 1e9 * 1.35   # train split + val/test
    rec = {
        "config": cfg.name, "epochs": cfg.train.epochs, "train_rows": rows, "work_rows": work_rows,
        "local": {**bench, "est_hours": local_hours},
        "recommend_remote": local_hours > LOCAL_THRESHOLD_HOURS,
        "threshold_hours": LOCAL_THRESHOLD_HOURS,
        "gpu_tiers": tiers,
        "corpus": corpus, "prepared": prep,
        "requirements": {
            "gpu": "any CUDA GPU with >= 16 GB; a single RTX 4090 or A100 is the sweet spot for this model",
            "vram_gb_min": 16,
            "ram_gb_min": int(max(32, math.ceil(prepared_gb_est * 1.5 + 8))),
            "disk_gb_min": int(math.ceil(data_gb * 2.5 + prepared_gb_est * 1.2 + 15)),
            "upload_gb": round(data_gb, 2),
            "image": "a recent PyTorch image, e.g. pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime (vast.ai 'PyTorch' template works)",
            "cuda": ">= 12.1 driver",
            "ssh": "direct ssh enabled (vast.ai: 'Direct SSH' toggle on the instance), user root",
            "ports": "only ssh; the app pulls results back over sftp",
        },
        "vast_ai_notes": [
            "add the public key below under Account > SSH Keys *before* renting so the instance picks it up",
            "rent with the 'PyTorch' template, pick 'ssh' as launch mode, disk >= the number above",
            "copy the ssh line from the instance card: `ssh -p <PORT> root@<IP>` -> that's host, port, user",
            "interruptible instances are cheaper but can vanish mid-run; the app saves a checkpoint every epoch and can resume",
        ],
    }
    return rec
