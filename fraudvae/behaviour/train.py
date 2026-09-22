"""Fit the behaviour verifiers on the public biometric sets and report an equal-error rate on
users the verifier never saw.

Protocol (open set): users are split 70 / 30. The embedding trains on the 70% (user
classification over feature windows). For every held-out user the first half of their
sessions enrols them (mean embedding) and the second half are genuine probes; the other
held-out users' probes are the impostor trials.

Loaders (all optional, each skipped when its files are not under the raw folder):
  balabit   mouse                  csv sessions per user
  bbmas     mouse keystroke touch motion   one big zip, one folder per user
  hmog      touch motion           zip of per-user zips, 24 sessions each
"""
from __future__ import annotations

import io
import json
import time
import zipfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ..paths import RAW_DIR
from .features import MODALITIES, keystroke_features, motion_features, mouse_features, touch_features
from .model import Verifier, equal_error_rate


# ---- loaders: each yields dicts {dataset, user, session, modality, windows [n, D]} -------------

def _parse_ts(s):
    import pandas as pd

    dt = pd.to_datetime(pd.Series(s, dtype="object").astype(str), errors="coerce")
    return dt.astype("int64").to_numpy().astype("float64") / 1e9


def _pseudo_sessions(records, user, modality, dataset, windows, per_session=10):
    for k, s in enumerate(range(0, len(windows), per_session)):
        records.append({"dataset": dataset, "user": user, "session": f"{user}-{modality}-{k}", "modality": modality,
                        "windows": windows[s : s + per_session]})


def load_balabit(raw=RAW_DIR, log=print):
    import pandas as pd

    root = Path(raw) / "balabit"
    if not (root / "training_files").exists():
        return []
    kinds = {"Move": "move", "Drag": "drag"}
    out = []
    for udir in sorted((root / "training_files").glob("user*")):
        for f in sorted(udir.iterdir()):
            d = pd.read_csv(f)
            kind = d["state"].map(kinds).fillna("scroll").to_numpy()
            w, _ = mouse_features(d["client timestamp"].to_numpy(), d["x"].to_numpy(), d["y"].to_numpy(), kind)
            if len(w):
                out.append({"dataset": "balabit", "user": udir.name, "session": f.name, "modality": "mouse", "windows": w})
    log(f"  balabit: {len(out)} sessions")
    return out


def load_bbmas(raw=RAW_DIR, log=print, max_users=None):
    import pandas as pd

    zpath = next(iter(sorted((Path(raw) / "bbmas").rglob("BB-MAS_Dataset.zip"))), None)
    if zpath is None:
        return []
    out = []
    with zipfile.ZipFile(zpath) as z:
        names = [n for n in z.namelist() if "__MACOSX" not in n and n.endswith(".csv")]
        users = sorted({n.split("/")[1] for n in names if n.count("/") >= 2})
        if max_users:
            users = users[:max_users]
        for u in users:
            mine = [n for n in names if n.split("/")[1] == u]

            def read(suffix):
                hit = [n for n in mine if n.split("/")[-1].startswith(f"{u}_{suffix}")]
                return pd.read_csv(io.BytesIO(z.read(hit[0])), low_memory=False) if hit else None

            km = read("Desktop_Keyboard")
            if km is not None and len(km) > 50:
                w, _ = keystroke_features(_parse_ts(km["time"]), km["key"].astype(str).to_numpy(), km["direction"].to_numpy())
                _pseudo_sessions(out, u, "keystroke", "bbmas", w)
            mv = read("Mouse_Move")
            if mv is not None and len(mv) > 200:
                w, _ = mouse_features(_parse_ts(mv["time"]), mv["pX"].to_numpy(), mv["pY"].to_numpy(), np.full(len(mv), "move"))
                _pseudo_sessions(out, u, "mouse", "bbmas", w)
            te = read("HandPhone_TouchEvent")
            if te is not None and len(te) > 200:
                w, _ = touch_features(_parse_ts(te["time"]), te["Xvalue"].to_numpy(), te["Yvalue"].to_numpy(), te["pressure"].to_numpy(),
                                      te["touchMajor"].to_numpy(), te["actionType"].to_numpy())
                _pseudo_sessions(out, u, "touch", "bbmas", w)
            ac = read("HandPhone_Accelerometer")
            if ac is not None and len(ac) > 500:
                w, _ = motion_features(_parse_ts(ac["time"]), ac["Xvalue"].to_numpy(), ac["Yvalue"].to_numpy(), ac["Zvalue"].to_numpy())
                _pseudo_sessions(out, u, "motion", "bbmas", w)
    log(f"  bbmas: {len(out)} pseudo-sessions from {len({r['user'] for r in out})} users")
    return out


def load_hmog(raw=RAW_DIR, log=print, max_users=None):
    import pandas as pd

    zpath = next(iter(sorted((Path(raw) / "hmog").rglob("hmog_dataset.zip"))), None)
    if zpath is None:
        return []
    out = []
    with zipfile.ZipFile(zpath) as outer:
        inner_names = sorted(n for n in outer.namelist() if n.endswith(".zip"))
        if max_users:
            inner_names = inner_names[:max_users]
        for n in inner_names:
            user = Path(n).stem
            with zipfile.ZipFile(io.BytesIO(outer.read(n))) as z:
                sessions = sorted({m.split("/")[1] for m in z.namelist() if m.count("/") >= 2 and "_session_" in m})
                for sess in sessions:
                    def read(name):
                        p = f"{user}/{sess}/{name}"
                        return pd.read_csv(io.BytesIO(z.read(p)), header=None) if p in z.namelist() else None

                    te = read("TouchEvent.csv")
                    if te is not None and len(te) > 100:
                        w, _ = touch_features(te[0].to_numpy(), te[6].to_numpy(), te[7].to_numpy(), te[8].to_numpy(), te[9].to_numpy(),
                                              te[5].to_numpy())
                        out.append({"dataset": "hmog", "user": user, "session": sess, "modality": "touch", "windows": w})
                    ac = read("Accelerometer.csv")
                    if ac is not None and len(ac) > 500:
                        w, _ = motion_features(ac[0].to_numpy(), ac[3].to_numpy(), ac[4].to_numpy(), ac[5].to_numpy())
                        out.append({"dataset": "hmog", "user": user, "session": sess, "modality": "motion", "windows": w})
    log(f"  hmog: {len(out)} sessions from {len({r['user'] for r in out})} users")
    return out


LOADERS = {"balabit": load_balabit, "bbmas": load_bbmas, "hmog": load_hmog}


# ---- training + evaluation ----------------------------------------------------------------------

def _fit_verifier(modality, train_recs, device, log, epochs=25, lr=2e-3, seed=0):
    torch.manual_seed(seed)
    users = sorted({r["user"] for r in train_recs})
    uid = {u: i for i, u in enumerate(users)}
    X = np.concatenate([r["windows"] for r in train_recs]).astype(np.float32)
    Y = np.concatenate([[uid[r["user"]]] * len(r["windows"]) for r in train_recs]).astype(np.int64)
    ver = Verifier(modality, X.shape[1], n_users=len(users)).to(device)
    ver.x_mean.copy_(torch.from_numpy(X.mean(0)))
    ver.x_std.copy_(torch.from_numpy(np.maximum(X.std(0), 1e-6)))
    xt, yt = torch.from_numpy(X).to(device), torch.from_numpy(Y).to(device)
    opt = torch.optim.AdamW(ver.parameters(), lr=lr, weight_decay=1e-4)
    rng = np.random.default_rng(seed)
    for ep in range(epochs):
        ver.train()
        order = rng.permutation(len(X))
        for s in range(0, len(order), 256):
            b = torch.from_numpy(order[s : s + 256]).to(device)
            loss = nn.functional.cross_entropy(ver.logits(xt[b]), yt[b])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        log(f"    {modality} epoch {ep + 1}/{epochs}: loss {loss.item():.3f}")
    ver.eval()
    return ver


def _trials(ver, recs_by_user):
    """enrol on the first half of each user's sessions, probe the rest; genuine and impostor cosines"""
    enrol, probes = {}, {}
    for u, recs in recs_by_user.items():
        recs = sorted(recs, key=lambda r: r["session"])
        if len(recs) < 2:
            continue
        half = len(recs) // 2
        e = np.mean([ver.session_embedding(r["windows"]) for r in recs[:half]], axis=0)
        enrol[u] = e / max(np.linalg.norm(e), 1e-9)
        probes[u] = [ver.session_embedding(r["windows"]) for r in recs[half:]]
    genuine, impostor = [], []
    for u, e in enrol.items():
        genuine.extend(float(e @ p) for p in probes[u])
        for v, ps in probes.items():
            if v != u:
                impostor.extend(float(e @ p) for p in ps)
    return np.asarray(genuine), np.asarray(impostor)


def _platt(genuine, impostor):
    from ..text.model import fit_platt

    s = np.concatenate([genuine, impostor])
    y = np.concatenate([np.ones(len(genuine)), np.zeros(len(impostor))])
    T, b = fit_platt(s, y)
    return 1.0 / T, b


def train_behaviour(run_dir: Path, raw=RAW_DIR, device="cpu", log=print, datasets=None, epochs=25, seed=1337, max_users=None) -> dict:
    """one verifier per modality over every dataset found; writes runs/<run>/behaviour/<modality>.pt + behaviour.json"""
    run_dir = Path(run_dir)
    out_dir = run_dir / "behaviour"
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    recs = []
    for name in datasets or list(LOADERS):
        recs += LOADERS[name](raw, log=log, **({"max_users": max_users} if name != "balabit" and max_users else {}))
    summary = {"created": time.strftime("%Y-%m-%dT%H:%M:%S"), "modalities": {}, "seconds": 0.0}
    all_genuine, all_impostor = [], []
    rng = np.random.default_rng(seed)
    for modality in MODALITIES:
        mine = [r for r in recs if r["modality"] == modality]
        if not mine:
            continue
        users = sorted({r["user"] for r in mine})
        rng.shuffle(users)
        eval_users = set(users[: int(round(0.3 * len(users)))])
        train_recs = [r for r in mine if r["user"] not in eval_users]
        if len({r["user"] for r in train_recs}) < 2:
            continue
        ver = _fit_verifier(modality, train_recs, device, log, epochs=epochs, seed=seed)
        info = {"modality": modality, "train_users": len({r["user"] for r in train_recs}), "train_sessions": len(train_recs)}
        by_user = {}
        for r in mine:
            if r["user"] in eval_users:
                by_user.setdefault(r["user"], []).append(r)
        genuine, impostor = _trials(ver, by_user)
        if len(genuine) > 5 and len(impostor) > 5:
            all_genuine.append(genuine)
            all_impostor.append(impostor)
            a, b = _platt(genuine, impostor)
            ver.platt_a.fill_(a)
            ver.platt_b.fill_(b)
            info["platt"] = {"a": a, "b": b}
        torch.save({"state_dict": ver.state_dict(), "config": ver.config(), "info": info}, out_dir / f"{modality}.pt")
        summary["modalities"][modality] = info
    if all_genuine:
        eer, thr = equal_error_rate(np.concatenate(all_genuine), np.concatenate(all_impostor))
        summary["eer"] = eer
        log(f"  behaviour: pooled open-set EER {eer:.3f}")
    summary["seconds"] = round(time.time() - t0, 1)
    (run_dir / "behaviour.json").write_text(json.dumps(summary, indent=1, default=float))
    return summary
