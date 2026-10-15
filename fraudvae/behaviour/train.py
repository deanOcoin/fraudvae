"""Fit the behaviour verifiers on the public biometric sets and report what a deployment
would get: an equal-error rate on users the verifier never saw.

Protocol (open set): users are split 70 / 30. The embedding trains on the 70% (user
classification over feature windows). For every held-out user the first half of their
sessions enrols them (mean embedding) and the second half are genuine probes; the other
held-out users' probes are the impostor trials. EER and a platt fit on those trials give the
calibrated likelihood ratio. Balabit, with 10 users and labelled illegal test sessions, uses
its own protocol: enrol on each user's training sessions, probe its test sessions, and the
public labels say which were intruders (that is the challenge's AUC).

Loaders (all optional, each skipped when its files are not under the raw folder):
  balabit   mouse                  csv sessions per user
  bbmas     mouse keystroke touch motion   one big zip, one folder per user, one long
                                   recording per device that we cut into pseudo-sessions
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

from ..model.scoring import auroc
from ..paths import RAW_DIR
from .features import MODALITIES, keystroke_features, motion_features, mouse_features, touch_features
from .model import Verifier, equal_error_rate

WINDOWS_PER_SESSION = 12     # pseudo-session size when a recording is one long stream


# ---- loaders: each yields dicts {dataset, user, session, modality, windows [n, D]} -------------

def _parse_ts(s):
    """'2017-04-14 18:09:41.538' / '2017-4-14 18:56:47:185' -> epoch seconds (naive)"""
    import pandas as pd

    s = pd.Series(s, dtype="object").astype(str).str.strip()
    # the touch files write the millisecond separator as ':'
    s = s.str.replace(r":(\d{3})$", r".\1", regex=True)
    dt = pd.to_datetime(s, errors="coerce", format="mixed")
    # not astype(int64) / 1e9: pandas 3 parses to datetime64[us] and that would be 1000x off
    vals = dt.to_numpy().astype("datetime64[ns]").astype("int64").astype("float64") / 1e9
    vals[dt.isna().to_numpy()] = np.nan
    return vals


def _pseudo_sessions(records, user, modality, dataset, windows, per_session=WINDOWS_PER_SESSION):
    for k, s in enumerate(range(0, len(windows), per_session)):
        w = windows[s : s + per_session]
        if len(w) >= max(3, per_session // 3):
            records.append({"dataset": dataset, "user": user, "session": f"{user}-{modality}-{k}", "modality": modality, "windows": w})


def load_balabit(raw=RAW_DIR, log=print):
    import pandas as pd

    root = Path(raw) / "balabit"
    if not (root / "training_files").exists():
        hits = list(root.rglob("training_files"))
        if not hits:
            return []
        root = hits[0].parent
    labels = {}
    lab = root / "public_labels.csv"
    if lab.exists():
        df = pd.read_csv(lab)
        labels = dict(zip(df["filename"].astype(str), df["is_illegal"].astype(int)))
    kinds = {"Move": "move", "Drag": "drag", "Down": "down", "Pressed": "down", "Up": "up", "Released": "up"}
    out = []
    for part in ("training_files", "test_files"):
        for udir in sorted((root / part).glob("user*")):
            for f in sorted(udir.iterdir()):
                if part == "test_files" and f.name not in labels:
                    continue
                d = pd.read_csv(f)
                kind = d["state"].map(kinds).fillna("scroll").to_numpy()
                w, _ = mouse_features(d["client timestamp"].to_numpy(), d["x"].to_numpy(), d["y"].to_numpy(), kind)
                if len(w) == 0:
                    continue
                out.append({"dataset": "balabit", "user": udir.name, "session": f.name, "modality": "mouse", "windows": w,
                            "split": "train" if part == "training_files" else "test", "illegal": labels.get(f.name)})
    log(f"  balabit: {len(out)} sessions with mouse windows from {len({r['user'] for r in out})} users")
    return out


def load_bbmas(raw=RAW_DIR, log=print, max_users=None):
    import pandas as pd

    zpath = next(iter(sorted((Path(raw) / "bbmas").rglob("BB-MAS_Dataset.zip"))), None)
    if zpath is None:
        return []
    out = []
    with zipfile.ZipFile(zpath) as z:
        names = [n for n in z.namelist() if "__MACOSX" not in n and n.endswith(".csv")]
        users = sorted({n.split("/")[1] for n in names if n.count("/") >= 2}, key=lambda u: int(u) if u.isdigit() else 0)
        if max_users:
            users = users[:max_users]
        for u in users:
            mine = [n for n in names if n.split("/")[1] == u]

            def read(suffix):
                hit = [n for n in mine if n.split("/")[-1].startswith(f"{u}_{suffix}")]
                if not hit:
                    return None
                try:
                    return pd.read_csv(io.BytesIO(z.read(hit[0])), low_memory=False)
                except Exception:  # noqa: BLE001
                    return None

            km = read("Desktop_Keyboard")
            if km is not None and len(km) > 50:
                w, _ = keystroke_features(_parse_ts(km["time"]), km["key"].astype(str).to_numpy(), km["direction"].to_numpy())
                _pseudo_sessions(out, u, "keystroke", "bbmas", w)
            mv, bt = read("Mouse_Move"), read("Mouse_Button")
            if mv is not None and len(mv) > 200:
                frames = [pd.DataFrame({"t": _parse_ts(mv["time"]), "x": mv["pX"], "y": mv["pY"], "kind": "move"})]
                if bt is not None and len(bt):
                    frames.append(pd.DataFrame({"t": _parse_ts(bt["time"]), "x": bt["pX"], "y": bt["pY"],
                                                "kind": np.where(bt["state"].to_numpy() == 0, "down", "up")}))
                ev = pd.concat(frames).dropna().sort_values("t")
                w, _ = mouse_features(ev["t"].to_numpy(), ev["x"].to_numpy(), ev["y"].to_numpy(), ev["kind"].to_numpy())
                _pseudo_sessions(out, u, "mouse", "bbmas", w)
            te = read("HandPhone_TouchEvent")
            if te is not None and len(te) > 200:
                te = te[te["time"].astype(str) != "null"]
                w, _ = touch_features(_parse_ts(te["time"]), te["Xvalue"].to_numpy(), te["Yvalue"].to_numpy(), te["pressure"].to_numpy(),
                                      te["touchMajor"].to_numpy(), te["actionType"].to_numpy())
                _pseudo_sessions(out, u, "touch", "bbmas", w)
            ac = read("HandPhone_Accelerometer")
            if ac is not None and len(ac) > 500:
                w, _ = motion_features(_parse_ts(ac["time"]), ac["Xvalue"].to_numpy(), ac["Yvalue"].to_numpy(), ac["Zvalue"].to_numpy())
                _pseudo_sessions(out, u, "motion", "bbmas", w)
    log(f"  bbmas: {len(out)} pseudo-sessions from {len({r['user'] for r in out})} users "
        f"({', '.join(f'{m}={sum(r['modality'] == m for r in out)}' for m in MODALITIES)})")
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
                        if p not in z.namelist():
                            return None
                        raw_bytes = z.read(p)
                        if not raw_bytes.strip():
                            return None
                        return pd.read_csv(io.BytesIO(raw_bytes), header=None)

                    te = read("TouchEvent.csv")
                    if te is not None and len(te) > 100:
                        # systime, eventtime, activity, pointer_count, pointer_id, action, x, y, pressure, size, orientation
                        w, _ = touch_features(te[0].to_numpy() / 1e3, te[6].to_numpy(), te[7].to_numpy(), te[8].to_numpy(), te[9].to_numpy(),
                                              te[5].to_numpy())
                        if len(w):
                            out.append({"dataset": "hmog", "user": user, "session": sess, "modality": "touch", "windows": w})
                    ac = read("Accelerometer.csv")
                    if ac is not None and len(ac) > 500:
                        w, _ = motion_features(ac[0].to_numpy() / 1e3, ac[3].to_numpy(), ac[4].to_numpy(), ac[5].to_numpy())
                        if len(w):
                            out.append({"dataset": "hmog", "user": user, "session": sess, "modality": "motion", "windows": w})
    log(f"  hmog: {len(out)} sessions from {len({r['user'] for r in out})} users "
        f"({', '.join(f'{m}={sum(r['modality'] == m for r in out)}' for m in ('touch', 'motion'))})")
    return out


LOADERS = {"balabit": load_balabit, "bbmas": load_bbmas, "hmog": load_hmog}


# ---- training + evaluation ----------------------------------------------------------------------

def _fit_verifier(modality, train_recs, device, log, epochs=25, lr=2e-3, seed=0):
    torch.manual_seed(seed)
    users = sorted({r["user"] for r in train_recs})
    uid = {u: i for i, u in enumerate(users)}
    X = np.concatenate([r["windows"] for r in train_recs]).astype(np.float32)
    Y = np.concatenate([[uid[r["user"]]] * len(r["windows"]) for r in train_recs]).astype(np.int64)
    d_in = X.shape[1]
    ver = Verifier(modality, d_in, n_users=len(users)).to(device)
    ver.x_mean.copy_(torch.from_numpy(X.mean(0)))
    ver.x_std.copy_(torch.from_numpy(np.maximum(X.std(0), 1e-6)))
    xt, yt = torch.from_numpy(X).to(device), torch.from_numpy(Y).to(device)
    opt = torch.optim.AdamW(ver.parameters(), lr=lr, weight_decay=1e-4)
    rng = np.random.default_rng(seed)
    bs = 256
    for ep in range(epochs):
        ver.train()
        order = rng.permutation(len(X))
        tot, n = 0.0, 0
        for s in range(0, len(order), bs):
            b = torch.from_numpy(order[s : s + bs]).to(device)
            loss = nn.functional.cross_entropy(ver.logits(xt[b]), yt[b])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += loss.item() * len(b)
            n += len(b)
        if ep % 5 == 4 or ep == epochs - 1:
            ver.eval()
            with torch.no_grad():
                acc = (ver.logits(xt).argmax(1) == yt).float().mean().item()
            log(f"    {modality} epoch {ep + 1}/{epochs}: loss {tot / max(n, 1):.3f}, window id-accuracy {acc:.3f} over {len(users)} users")
    ver.eval()
    return ver


def _trials(ver, recs_by_user):
    """enrol on the first half of each user's sessions, probe the rest; genuine and impostor cosines"""
    enrol, probes = {}, {}
    for u, recs in recs_by_user.items():
        recs = sorted(recs, key=lambda r: r["session"])
        if len(recs) < 2:
            continue
        half = max(1, len(recs) // 2)
        e = [ver.session_embedding(r["windows"]) for r in recs[:half]]
        e = [x for x in e if x is not None]
        if not e:
            continue
        enrol[u] = np.mean(e, axis=0)
        enrol[u] /= max(np.linalg.norm(enrol[u]), 1e-9)
        probes[u] = [ver.session_embedding(r["windows"]) for r in recs[half:]]
        probes[u] = [x for x in probes[u] if x is not None]
    genuine, impostor = [], []
    for u, e in enrol.items():
        for p in probes.get(u, []):
            genuine.append(float(e @ p))
        for v, ps in probes.items():
            if v != u:
                impostor.extend(float(e @ p) for p in ps)
    return np.asarray(genuine), np.asarray(impostor), len(enrol)


def _platt(genuine, impostor):
    from ..text.model import fit_platt

    s = np.concatenate([genuine, impostor])
    y = np.concatenate([np.ones(len(genuine)), np.zeros(len(impostor))])
    T, b = fit_platt(s, y)
    # llr(cos) = cos / T + b - logit(prior of genuine among the trials): the trials' mix is an
    # artefact of how many impostors we paired, not a prior anything should inherit
    prior = len(genuine) / max(len(s), 1)
    return 1.0 / T, b - float(np.log(prior / (1 - prior)))


def train_behaviour(run_dir: Path, raw=RAW_DIR, device="cpu", log=print, datasets=None, epochs=25, seed=1337, max_users=None) -> dict:
    """one verifier per modality over every dataset found; writes runs/<run>/behaviour/<modality>.pt + behaviour.json"""
    run_dir = Path(run_dir)
    out_dir = run_dir / "behaviour"
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    recs = []
    for name in datasets or list(LOADERS):
        try:
            recs += LOADERS[name](raw, log=log, **({"max_users": max_users} if name != "balabit" and max_users else {}))
        except Exception as e:  # noqa: BLE001
            log(f"  {name}: failed to load ({type(e).__name__}: {e}), skipping")
    summary = {"created": time.strftime("%Y-%m-%dT%H:%M:%S"), "modalities": {}, "seconds": 0.0}
    rng = np.random.default_rng(seed)
    for modality in MODALITIES:
        mine = [r for r in recs if r["modality"] == modality]
        if not mine:
            continue
        # balabit keeps its own protocol; everything else: users split 70 / 30
        bal = [r for r in mine if r["dataset"] == "balabit"]
        rest = [r for r in mine if r["dataset"] != "balabit"]
        users = sorted({r["user"] for r in rest})
        rng.shuffle(users)
        n_eval = int(round(0.3 * len(users))) if len(users) >= 10 else 0
        eval_users = set(users[:n_eval])
        train_recs = [r for r in rest if r["user"] not in eval_users] + [r for r in bal if r.get("split") == "train"]
        if len({r["user"] for r in train_recs}) < 2:
            log(f"  {modality}: fewer than two users to learn from, skipping")
            continue
        log(f"  {modality}: training on {len({r['user'] for r in train_recs})} users / {len(train_recs)} sessions, "
            f"{sum(len(r['windows']) for r in train_recs):,} windows")
        ver = _fit_verifier(modality, train_recs, device, log, epochs=epochs, seed=seed)
        info = {"modality": modality, "train_users": len({r["user"] for r in train_recs}), "train_sessions": len(train_recs),
                "datasets": sorted({r["dataset"] for r in mine})}
        genuine, impostor = np.zeros(0), np.zeros(0)
        if eval_users:
            by_user = {}
            for r in rest:
                if r["user"] in eval_users:
                    by_user.setdefault(r["user"], []).append(r)
            genuine, impostor, n_enrolled = _trials(ver, by_user)
            if len(genuine) and len(impostor):
                eer, thr = equal_error_rate(genuine, impostor)
                info["open_set"] = {"users": n_enrolled, "genuine_trials": int(len(genuine)), "impostor_trials": int(len(impostor)),
                                    "eer": eer, "threshold": thr, "auroc": auroc(np.r_[np.ones(len(genuine)), np.zeros(len(impostor))],
                                                                                 np.r_[genuine, impostor])}
                log(f"  {modality}: open-set EER {eer:.3f} (AUROC {info['open_set']['auroc']:.3f}) on {n_enrolled} unseen users, "
                    f"{len(genuine)} genuine / {len(impostor)} impostor trials")
        if bal:
            # the challenge protocol: enrol on training sessions, score test sessions, labels say intruders
            by_user = {}
            for r in bal:
                by_user.setdefault(r["user"], []).append(r)
            scores, labels = [], []
            for u, rs in by_user.items():
                e = [ver.session_embedding(r["windows"]) for r in rs if r.get("split") == "train"]
                e = [x for x in e if x is not None]
                if not e:
                    continue
                enrol = np.mean(e, axis=0)
                enrol /= max(np.linalg.norm(enrol), 1e-9)
                for r in rs:
                    if r.get("split") == "test" and r.get("illegal") is not None:
                        p = ver.session_embedding(r["windows"])
                        if p is not None:
                            scores.append(float(enrol @ p))
                            labels.append(int(r["illegal"]))
            scores, labels = np.asarray(scores), np.asarray(labels)
            if len(scores) and labels.min() != labels.max():
                g, i = scores[labels == 0], scores[labels == 1]
                eer, thr = equal_error_rate(g, i)
                info["balabit"] = {"sessions": int(len(scores)), "illegal": int(labels.sum()), "auroc": auroc(1 - labels, scores), "eer": eer}
                log(f"  {modality}: balabit test sessions AUROC {info['balabit']['auroc']:.3f}, EER {eer:.3f} ({len(scores)} sessions, {int(labels.sum())} intruders)")
                if not len(genuine):
                    genuine, impostor = g, i
        if len(genuine) > 5 and len(impostor) > 5:
            a, b = _platt(genuine, impostor)
            ver.platt_a.fill_(a)
            ver.platt_b.fill_(b)
            ver.eer.fill_(float(info.get("open_set", info.get("balabit", {})).get("eer", 0.0)))
            info["platt"] = {"a": a, "b": b}
        torch.save({"state_dict": ver.state_dict(), "config": ver.config(), "info": info}, out_dir / f"{modality}.pt")
        summary["modalities"][modality] = info
    summary["seconds"] = round(time.time() - t0, 1)
    (run_dir / "behaviour.json").write_text(json.dumps(summary, indent=1, default=float))
    return summary
