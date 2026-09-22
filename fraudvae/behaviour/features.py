"""Raw interaction events -> fixed-length feature windows, one function per modality. Each
takes plain arrays (what a json payload decodes to) and returns a [n_windows, D] float32
matrix plus the feature names.

  mouse      events (t, x, y, kind): strokes between pauses and their kinematics
  keystroke  events (t, key, direction) with direction 0 = down, 1 = up: hold times, latencies
  touch      points (t, x, y, pressure): swipes, speed and pressure
  motion     samples (t, x, y, z) from an accelerometer or gyroscope
"""
from __future__ import annotations

import numpy as np

STROKES_PER_WINDOW = 30
KEYS_PER_WINDOW = 40
SWIPES_PER_WINDOW = 15
MOTION_WINDOW = 250

_STATS = ("mean", "std", "p50")


def _stats(v) -> list[float]:
    v = np.asarray(v, dtype=np.float64)
    if len(v) == 0:
        return [0.0] * len(_STATS)
    return [float(v.mean()), float(v.std()), float(np.median(v))]


def _stat_names(prefix: str) -> list[str]:
    return [f"{prefix}_{s}" for s in _STATS]


# ---- mouse -----------------------------------------------------------------------------------

MOUSE_SCALARS = ["duration", "path", "displacement", "straightness", "v_mean", "v_max"]
MOUSE_FEATURES = [n for s in MOUSE_SCALARS for n in _stat_names(s)]


def mouse_features(t, x, y, kind, strokes_per_window: int = STROKES_PER_WINDOW) -> tuple[np.ndarray, list[str]]:
    t = np.asarray(t, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    cuts = np.flatnonzero(np.diff(t) > 0.5) + 1
    feats = []
    for a, b in zip(np.r_[0, cuts], np.r_[cuts, len(t)]):
        if b - a < 4:
            continue
        tt, xx, yy = t[a:b], x[a:b], y[a:b]
        dt = np.diff(tt)
        dt = np.where(dt <= 0, 1e-3, dt)
        step = np.hypot(np.diff(xx), np.diff(yy))
        v = step / dt
        path = float(step.sum())
        disp = float(np.hypot(xx[-1] - xx[0], yy[-1] - yy[0]))
        feats.append({"duration": float(tt[-1] - tt[0]), "path": path, "displacement": disp,
                      "straightness": disp / max(path, 1e-6), "v_mean": float(v.mean()), "v_max": float(v.max())})
    windows = []
    for s in range(0, len(feats), strokes_per_window):
        chunk = feats[s : s + strokes_per_window]
        if len(chunk) < strokes_per_window:
            continue
        row = []
        for name in MOUSE_SCALARS:
            row += _stats([f[name] for f in chunk])
        windows.append(row)
    return np.asarray(windows, dtype=np.float32), MOUSE_FEATURES


# ---- keystrokes -----------------------------------------------------------------------------

KEY_FEATURES = _stat_names("hold") + _stat_names("dd") + ["keys_per_s"]


def keystroke_features(t, key, direction, keys_per_window: int = KEYS_PER_WINDOW) -> tuple[np.ndarray, list[str]]:
    t = np.asarray(t, dtype=np.float64)
    d = np.asarray([int(v) for v in direction])
    presses = []
    open_keys = {}
    for i in range(len(t)):
        if d[i] == 0:
            open_keys[key[i]] = t[i]
        elif key[i] in open_keys:
            presses.append((open_keys.pop(key[i]), t[i]))
    windows = []
    for s in range(0, len(presses), keys_per_window):
        chunk = presses[s : s + keys_per_window]
        if len(chunk) < keys_per_window:
            continue
        dn = np.asarray([a for a, _ in chunk])
        up = np.asarray([b for _, b in chunk])
        span = max(float(up[-1] - dn[0]), 1e-3)
        windows.append(_stats(up - dn) + _stats(np.diff(dn)) + [len(chunk) / span])
    return np.asarray(windows, dtype=np.float32), KEY_FEATURES


# ---- touch ----------------------------------------------------------------------------------

SWIPE_SCALARS = ["duration", "length", "v_mean", "pressure_mean"]
TOUCH_FEATURES = [n for s in SWIPE_SCALARS for n in _stat_names(s)]


def touch_features(t, x, y, pressure=None, swipes_per_window: int = SWIPES_PER_WINDOW):
    t = np.asarray(t, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    pressure = np.ones(len(t)) if pressure is None else np.asarray(pressure, dtype=np.float64)
    cuts = np.flatnonzero(np.diff(t) > 0.3) + 1
    feats = []
    for a, b in zip(np.r_[0, cuts], np.r_[cuts, len(t)]):
        if b - a < 2:
            continue
        dt = np.diff(t[a:b])
        dt = np.where(dt <= 0, 1e-3, dt)
        step = np.hypot(np.diff(x[a:b]), np.diff(y[a:b]))
        feats.append({"duration": float(t[b - 1] - t[a]), "length": float(step.sum()), "v_mean": float((step / dt).mean()),
                      "pressure_mean": float(pressure[a:b].mean())})
    windows = []
    for s in range(0, len(feats), swipes_per_window):
        chunk = feats[s : s + swipes_per_window]
        if len(chunk) < swipes_per_window:
            continue
        row = []
        for name in SWIPE_SCALARS:
            row += _stats([f[name] for f in chunk])
        windows.append(row)
    return np.asarray(windows, dtype=np.float32), TOUCH_FEATURES


# ---- motion ---------------------------------------------------------------------------------

MOTION_FEATURES = [f"{ax}_{s}" for ax in ("x", "y", "z", "mag") for s in ("mean", "std", "min", "max")]


def motion_features(t, x, y, z, window: int = MOTION_WINDOW) -> tuple[np.ndarray, list[str]]:
    xyz = np.stack([np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64), np.asarray(z, dtype=np.float64)], axis=1)
    mag = np.linalg.norm(xyz, axis=1)
    cols = [xyz[:, 0], xyz[:, 1], xyz[:, 2], mag]
    windows = []
    for s in range(0, len(mag) - window + 1, window):
        row = []
        for c in cols:
            v = c[s : s + window]
            row += [float(v.mean()), float(v.std()), float(v.min()), float(v.max())]
        windows.append(row)
    return np.asarray(windows, dtype=np.float32), MOTION_FEATURES


MODALITIES = {"mouse": (mouse_features, MOUSE_FEATURES), "keystroke": (keystroke_features, KEY_FEATURES),
              "touch": (touch_features, TOUCH_FEATURES), "motion": (motion_features, MOTION_FEATURES)}
