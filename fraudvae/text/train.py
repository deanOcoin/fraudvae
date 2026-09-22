"""Fit the text heads on the public text-fraud sets and calibrate them. Each head is trained
on its own stratified 80/10/10 split (temperature fitted on the 10% val, numbers reported on
the 10% test) and written to runs/<run>/text/<name>.pt with a text.json summary.

  malurl      "is this url malicious"           651k urls, 34% malicious (phishing / defacement / malware)
  fakejob     "is this posting / long text a scam"  17.9k job postings, 4.8% fraudulent
  twitterbot  "is this profile text a bot's"    37k twitter profiles, 33% bots
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ..model.scoring import auprc, auroc
from ..paths import RAW_DIR
from .features import encode_batch
from .model import TextHead, fit_platt

TEXT_SETS = {
    "malurl": {"label": "URL", "payload": "url", "files": ("malicious_phish.csv",)},
    "fakejob": {"label": "Free text", "payload": "description", "files": ("fake_job_postings.csv",)},
    "twitterbot": {"label": "Profile text", "payload": "profile", "files": ("twitter_human_bots_dataset.csv",)},
}


def _find(raw: Path, key: str, name: str) -> Path | None:
    from ..data.aux_registry import find_file

    return find_file(Path(raw), key, name)


def load_malurl(raw=RAW_DIR):
    import pandas as pd

    p = _find(raw, "malurl", "malicious_phish.csv")
    if p is None:
        return None
    df = pd.read_csv(p)
    return df["url"].astype(str).tolist(), (df["type"].astype(str) != "benign").to_numpy().astype(np.float32)


def load_fakejob(raw=RAW_DIR):
    import pandas as pd

    p = _find(raw, "fakejob", "fake_job_postings.csv")
    if p is None:
        return None
    df = pd.read_csv(p)
    cols = ["title", "company_profile", "description", "requirements", "benefits"]
    text = df[cols].fillna("").astype(str).agg(" | ".join, axis=1).tolist()
    return text, df["fraudulent"].to_numpy().astype(np.float32)


def load_twitterbot(raw=RAW_DIR):
    import pandas as pd

    p = _find(raw, "twitterbot", "twitter_human_bots_dataset.csv")
    if p is None:
        return None
    df = pd.read_csv(p)
    text = (df["screen_name"].fillna("").astype(str) + " | " + df["description"].fillna("").astype(str)
            + " | " + df["location"].fillna("").astype(str)).tolist()
    return text, (df["account_type"].astype(str) == "bot").to_numpy().astype(np.float32)


LOADERS = {"malurl": load_malurl, "fakejob": load_fakejob, "twitterbot": load_twitterbot}


def _split(y: np.ndarray, rng) -> dict:
    parts = {"train": [], "val": [], "test": []}
    for cls in (0, 1):
        idx = rng.permutation(np.flatnonzero(y == cls))
        a, b = int(0.8 * len(idx)), int(0.9 * len(idx))
        parts["train"].append(idx[:a]); parts["val"].append(idx[a:b]); parts["test"].append(idx[b:])
    return {k: np.sort(np.concatenate(v)) for k, v in parts.items()}


def _encode_all(texts: list[str], buckets: int, chunk=20_000):
    """hash every text once; returns a list of (ids array) per text, memory-light"""
    out = []
    for s in range(0, len(texts), chunk):
        ids, offsets = encode_batch(texts[s : s + chunk], buckets)
        ends = np.append(offsets[1:], len(ids))
        out.extend(ids[a:b] for a, b in zip(offsets, ends))
    return out


def _batch(enc, idx, device):
    ids = [enc[i] for i in idx]
    offsets = np.cumsum([0] + [len(x) for x in ids[:-1]])
    return (torch.from_numpy(np.concatenate(ids)).to(device), torch.from_numpy(offsets.astype(np.int64)).to(device))


def _logits(head, enc, idx, device, bs=4096):
    head.eval()
    out = np.empty(len(idx), dtype=np.float32)
    with torch.no_grad():
        for s in range(0, len(idx), bs):
            b = idx[s : s + bs]
            ids, off = _batch(enc, b, device)
            out[s : s + len(b)] = head(ids, off).float().cpu().numpy()
    return out


def train_text_head(name: str, texts: list[str], y: np.ndarray, device="cpu", log=print, epochs=4, batch_size=512, lr=3e-3,
                    seed=1337) -> tuple[TextHead, dict]:
    t0 = time.time()
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    sp = _split(y, rng)
    head = TextHead(name).to(device)
    enc = _encode_all(texts, head.buckets)
    log(f"  text {name}: {len(texts):,} texts hashed ({time.time() - t0:.0f}s), {len(sp['train']):,} train / "
        f"{len(sp['val']):,} val / {len(sp['test']):,} test, positive rate {y.mean():.1%}")
    pos_weight = min(float((y[sp["train"]] == 0).sum() / max(1, (y[sp["train"]] == 1).sum())), 20.0)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    yt = torch.from_numpy(y).to(device)
    best, best_state = -1.0, None
    for ep in range(epochs):
        head.train()
        order = rng.permutation(sp["train"])
        tot, n = 0.0, 0
        for s in range(0, len(order), batch_size):
            b = order[s : s + batch_size]
            ids, off = _batch(enc, b, device)
            loss = loss_fn(head(ids, off), yt[b])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += loss.item() * len(b)
            n += len(b)
        va = auroc(y[sp["val"]], _logits(head, enc, sp["val"], device))
        log(f"  text {name} epoch {ep + 1}/{epochs}: loss {tot / max(n, 1):.4f} val auroc {va:.4f}")
        if va > best:
            best, best_state = va, {k: v.detach().clone() for k, v in head.state_dict().items()}
    head.load_state_dict(best_state)
    lv = _logits(head, enc, sp["val"], device)
    T, bias = fit_platt(lv, y[sp["val"]])
    base = float(y[sp["train"]].mean())
    head.temperature.fill_(T)
    head.bias.fill_(bias)
    head.base_logit.fill_(float(np.log(base / (1 - base))))
    lt = _logits(head, enc, sp["test"], device) / T + bias
    info = {"name": name, "label": TEXT_SETS[name]["label"], "payload": TEXT_SETS[name]["payload"], "n": int(len(texts)),
            "positive_rate": base, "temperature": T, "bias": bias, "val_auroc": float(best),
            "test_auroc": auroc(y[sp["test"]], lt), "test_auprc": auprc(y[sp["test"]], lt), "test_n": int(len(sp["test"])),
            "epochs": epochs, "seconds": round(time.time() - t0, 1)}
    pt = 1.0 / (1.0 + np.exp(-lt))
    info["test_mean_prob"] = float(pt.mean())
    info["test_positive_rate"] = float(y[sp["test"]].mean())
    log(f"  text {name}: test auroc {info['test_auroc']:.4f} auprc {info['test_auprc']:.4f}, platt T {T:.2f} b {bias:+.2f}, "
        f"mean p {pt.mean():.3f} vs rate {y[sp['test']].mean():.3f} ({info['seconds']:.0f}s)")
    return head, info


def train_text(run_dir: Path, raw=RAW_DIR, device="cpu", log=print, names=None, epochs=4, seed=1337) -> dict:
    """every text set found under the raw folder; writes runs/<run>/text/<name>.pt + text.json"""
    run_dir = Path(run_dir)
    out = run_dir / "text"
    out.mkdir(parents=True, exist_ok=True)
    summary = {"created": time.strftime("%Y-%m-%dT%H:%M:%S"), "heads": {}}
    for name in names or list(LOADERS):
        data = LOADERS[name](raw)
        if data is None:
            log(f"  text {name}: data not found under {raw}, skipping")
            continue
        texts, y = data
        try:
            head, info = train_text_head(name, texts, y, device=device, log=log, epochs=epochs, seed=seed)
        except Exception as e:  # noqa: BLE001
            log(f"  text {name}: failed ({type(e).__name__}: {e}), skipping")
            continue
        # fp16 on disk: the embedding table is most of the head and half precision loses nothing a hash table had
        state = {k: (v.half() if v.is_floating_point() and v.numel() > 1000 else v) for k, v in head.state_dict().items()}
        torch.save({"state_dict": state, "config": head.config(), "info": info}, out / f"{name}.pt")
        summary["heads"][name] = info
    (run_dir / "text.json").write_text(json.dumps(summary, indent=1))
    return summary
