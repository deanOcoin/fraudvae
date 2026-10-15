"""Raw interaction events -> fixed-length feature windows, one function per modality. Each
takes plain arrays (what a json payload decodes to) and returns a [n_windows, D] float32
matrix plus the feature names, so the same code serves the datasets and the API.

  mouse      events (t, x, y, kind) with kind in {move, down, up, drag, scroll}: strokes
             between pauses / clicks, their kinematics (velocity, acceleration, jerk,
             curvature, straightness, direction) and click / pause timing
  keystroke  events (t, key, direction) with direction 0 = down, 1 = up: hold times,
             down-down / up-down latencies, speed, correction rate
  touch      points (t, x, y, pressure, size, action) with action 0 down / 2 move / 1 up:
             swipes and their geometry, speed, pressure and size
  motion     samples (t, x, y, z) from an accelerometer or gyroscope: per-window statistics
             of each axis and of the magnitude

Windows: a fixed number of strokes / keys / swipes, or a fixed number of seconds for motion,
so a window is a few seconds to a minute of behaviour; a session yields several.
"""
from __future__ import annotations

import numpy as np

STROKES_PER_WINDOW = 30
KEYS_PER_WINDOW = 40
SWIPES_PER_WINDOW = 15
MOTION_WINDOW_S = 5.0

_STATS = ("mean", "std", "p10", "p50", "p90")


def _stats(v: np.ndarray) -> list[float]:
    v = np.asarray(v, dtype=np.float64)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return [0.0] * len(_STATS)
    return [float(v.mean()), float(v.std()), float(np.percentile(v, 10)), float(np.median(v)), float(np.percentile(v, 90))]


def _stat_names(prefix: str) -> list[str]:
    return [f"{prefix}_{s}" for s in _STATS]


# ---- mouse -----------------------------------------------------------------------------------

MOUSE_KINDS = {"move": 0, "drag": 1, "down": 2, "up": 3, "scroll": 4, "pressed": 2, "released": 3}


def _mouse_strokes(t, x, y, kind, pause=0.5):
    """split the pointer path into strokes at pauses longer than `pause` seconds and at clicks"""
    strokes, cur = [], [0]
    for i in range(1, len(t)):
        gap = t[i] - t[i - 1]
        if gap > pause or kind[i] in (2, 3) or kind[i - 1] in (2, 3):
            if len(cur) >= 4:
                strokes.append(np.asarray(cur))
            cur = [i]
        else:
            cur.append(i)
    if len(cur) >= 4:
        strokes.append(np.asarray(cur))
    return strokes


def _stroke_features(t, x, y, idx):
    tt, xx, yy = t[idx], x[idx], y[idx]
    dt = np.diff(tt)
    dt = np.where(dt <= 0, 1e-3, dt)
    dx, dy = np.diff(xx), np.diff(yy)
    step = np.hypot(dx, dy)
    v = step / dt
    a = np.diff(v) / dt[1:] if len(v) > 1 else np.zeros(1)
    j = np.diff(a) / dt[2:] if len(a) > 1 else np.zeros(1)
    ang = np.arctan2(dy, dx)
    dang = np.diff(ang)
    dang = (dang + np.pi) % (2 * np.pi) - np.pi
    w = dang / dt[1:] if len(dang) else np.zeros(1)
    path = float(step.sum())
    disp = float(np.hypot(xx[-1] - xx[0], yy[-1] - yy[0]))
    dur = float(tt[-1] - tt[0])
    return {
        "duration": dur, "path": path, "displacement": disp, "straightness": disp / max(path, 1e-6),
        "v_mean": float(v.mean()), "v_max": float(v.max()), "v_std": float(v.std()),
        "a_mean": float(np.abs(a).mean()), "a_max": float(np.abs(a).max()), "jerk": float(np.abs(j).mean()),
        "curvature": float(np.abs(dang).sum() / max(path, 1e-6)), "angular_v": float(np.abs(w).mean()),
        "direction": float(np.arctan2(yy[-1] - yy[0], xx[-1] - xx[0])), "points": float(len(idx)),
        "pause_before": 0.0,
    }


MOUSE_SCALARS = ["duration", "path", "displacement", "straightness", "v_mean", "v_max", "v_std", "a_mean", "a_max", "jerk",
                 "curvature", "angular_v", "points", "pause_before"]
MOUSE_FEATURES = ([n for s in MOUSE_SCALARS for n in _stat_names(s)] + [f"dir_{k}" for k in range(8)]
                  + ["click_hold_mean", "click_hold_std", "clicks_per_stroke", "drag_share", "scroll_share"])


def mouse_features(t, x, y, kind, strokes_per_window: int = STROKES_PER_WINDOW) -> tuple[np.ndarray, list[str]]:
    t = np.asarray(t, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    kind = np.asarray([MOUSE_KINDS.get(str(k).lower(), 0) if not isinstance(k, (int, np.integer)) else int(k) for k in kind])
    order = np.argsort(t, kind="stable")
    t, x, y, kind = t[order], x[order], y[order], kind[order]
    strokes = _mouse_strokes(t, x, y, kind)
    if not strokes:
        return np.zeros((0, len(MOUSE_FEATURES)), dtype=np.float32), MOUSE_FEATURES
    feats = []
    prev_end = None
    for idx in strokes:
        f = _stroke_features(t, x, y, idx)
        f["pause_before"] = float(t[idx[0]] - prev_end) if prev_end is not None else 0.0
        prev_end = t[idx[-1]]
        feats.append(f)
    # click hold times: down -> next up
    downs = np.flatnonzero(kind == 2)
    ups = np.flatnonzero(kind == 3)
    holds = []
    for d in downs:
        nxt = ups[ups > d]
        if len(nxt) and t[nxt[0]] - t[d] < 3.0:
            holds.append(t[nxt[0]] - t[d])
    holds = np.asarray(holds) if holds else np.zeros(0)
    windows = []
    for s in range(0, len(feats), strokes_per_window):
        chunk = feats[s : s + strokes_per_window]
        if len(chunk) < max(5, strokes_per_window // 3):
            continue
        row = []
        for name in MOUSE_SCALARS:
            row += _stats([f[name] for f in chunk])
        dirs = np.asarray([f["direction"] for f in chunk])
        hist = np.histogram(dirs, bins=8, range=(-np.pi, np.pi))[0] / max(len(dirs), 1)
        row += hist.tolist()
        span = (t >= t[strokes[s][0]]) & (t <= t[strokes[min(s + len(chunk) - 1, len(strokes) - 1)][-1]])
        row += [float(holds.mean()) if len(holds) else 0.0, float(holds.std()) if len(holds) else 0.0,
                float((kind[span] == 2).sum() / len(chunk)), float((kind[span] == 1).mean()), float((kind[span] == 4).mean())]
        windows.append(row)
    return np.asarray(windows, dtype=np.float32).reshape(-1, len(MOUSE_FEATURES)), MOUSE_FEATURES


# ---- keystrokes -----------------------------------------------------------------------------

KEY_FEATURES = (_stat_names("hold") + _stat_names("dd") + _stat_names("ud") + _stat_names("uu")
                + ["keys_per_s", "backspace_share", "space_share", "shift_share", "n_keys"])


def keystroke_features(t, key, direction, keys_per_window: int = KEYS_PER_WINDOW) -> tuple[np.ndarray, list[str]]:
    """direction: 0 / 'down' = press, 1 / 'up' = release"""
    t = np.asarray(t, dtype=np.float64)
    key = np.asarray([str(k).lower() for k in key], dtype=object)
    d = np.asarray([0 if str(v).lower() in ("0", "down", "press", "p") else 1 for v in direction])
    order = np.argsort(t, kind="stable")
    t, key, d = t[order], key[order], d[order]
    # pair each press with its release
    presses = []
    open_keys = {}
    for i in range(len(t)):
        if d[i] == 0:
            open_keys[key[i]] = i
        elif key[i] in open_keys:
            presses.append((open_keys.pop(key[i]), i))
    presses.sort()
    if len(presses) < 5:
        return np.zeros((0, len(KEY_FEATURES)), dtype=np.float32), KEY_FEATURES
    windows = []
    for s in range(0, len(presses), keys_per_window):
        chunk = presses[s : s + keys_per_window]
        if len(chunk) < max(5, keys_per_window // 3):
            continue
        dn = np.asarray([t[a] for a, _ in chunk])
        up = np.asarray([t[b] for _, b in chunk])
        hold = up - dn
        dd = np.diff(dn)
        ud = dn[1:] - up[:-1]
        uu = np.diff(up)
        keys = [key[a] for a, _ in chunk]
        span = max(float(up[-1] - dn[0]), 1e-3)
        row = (_stats(hold) + _stats(dd) + _stats(ud) + _stats(uu)
               + [len(chunk) / span, float(np.mean([k in ("backspace", "delete", "\x08") for k in keys])),
                  float(np.mean([k in (" ", "space") for k in keys])), float(np.mean(["shift" in k for k in keys])), float(len(chunk))])
        windows.append(row)
    return np.asarray(windows, dtype=np.float32).reshape(-1, len(KEY_FEATURES)), KEY_FEATURES


# ---- touch ----------------------------------------------------------------------------------

SWIPE_SCALARS = ["duration", "length", "displacement", "straightness", "v_mean", "v_max", "pressure_mean", "pressure_max",
                 "size_mean", "start_x", "start_y", "end_x", "end_y", "direction", "points"]
TOUCH_FEATURES = [n for s in SWIPE_SCALARS for n in _stat_names(s)] + [f"dir_{k}" for k in range(8)] + ["taps_share", "swipes"]


def touch_features(t, x, y, pressure=None, size=None, action=None, swipes_per_window: int = SWIPES_PER_WINDOW):
    """action: 0 down, 2 move, 1 up (android convention); without actions a gap of >0.3s starts a new swipe"""
    t = np.asarray(t, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = len(t)
    pressure = np.ones(n) if pressure is None else np.nan_to_num(np.asarray(pressure, dtype=np.float64), nan=0.0)
    size = np.zeros(n) if size is None else np.nan_to_num(np.asarray(size, dtype=np.float64), nan=0.0)
    order = np.argsort(t, kind="stable")
    t, x, y, pressure, size = t[order], x[order], y[order], pressure[order], size[order]
    if action is not None:
        act = np.asarray(action, dtype=object)[order]
        act = np.asarray([0 if str(a) in ("0", "0.0", "down") else (1 if str(a) in ("1", "1.0", "up") else 2) for a in act])
        starts = np.flatnonzero(act == 0)
        ends = np.flatnonzero(act == 1)
        swipes = []
        for s in starts:
            e = ends[ends > s]
            if len(e):
                swipes.append(np.arange(s, e[0] + 1))
    else:
        cuts = np.flatnonzero(np.diff(t) > 0.3) + 1
        swipes = [np.arange(a, b) for a, b in zip(np.r_[0, cuts], np.r_[cuts, n])]
    feats = []
    taps = 0
    for idx in swipes:
        if len(idx) < 2:
            taps += 1
            continue
        tt, xx, yy = t[idx], x[idx], y[idx]
        dt = np.diff(tt)
        dt = np.where(dt <= 0, 1e-3, dt)
        step = np.hypot(np.diff(xx), np.diff(yy))
        v = step / dt
        length = float(step.sum())
        disp = float(np.hypot(xx[-1] - xx[0], yy[-1] - yy[0]))
        if length < 5:
            taps += 1
            continue
        feats.append({"duration": float(tt[-1] - tt[0]), "length": length, "displacement": disp, "straightness": disp / max(length, 1e-6),
                      "v_mean": float(v.mean()), "v_max": float(v.max()), "pressure_mean": float(pressure[idx].mean()),
                      "pressure_max": float(pressure[idx].max()), "size_mean": float(size[idx].mean()),
                      "start_x": float(xx[0]), "start_y": float(yy[0]), "end_x": float(xx[-1]), "end_y": float(yy[-1]),
                      "direction": float(np.arctan2(yy[-1] - yy[0], xx[-1] - xx[0])), "points": float(len(idx))})
    if not feats:
        return np.zeros((0, len(TOUCH_FEATURES)), dtype=np.float32), TOUCH_FEATURES
    windows = []
    for s in range(0, len(feats), swipes_per_window):
        chunk = feats[s : s + swipes_per_window]
        if len(chunk) < max(3, swipes_per_window // 3):
            continue
        row = []
        for name in SWIPE_SCALARS:
            row += _stats([f[name] for f in chunk])
        dirs = np.asarray([f["direction"] for f in chunk])
        row += (np.histogram(dirs, bins=8, range=(-np.pi, np.pi))[0] / max(len(dirs), 1)).tolist()
        row += [taps / max(taps + len(feats), 1), float(len(chunk))]
        windows.append(row)
    return np.asarray(windows, dtype=np.float32).reshape(-1, len(TOUCH_FEATURES)), TOUCH_FEATURES


# ---- motion ---------------------------------------------------------------------------------

MOTION_FEATURES = [f"{ax}_{s}" for ax in ("x", "y", "z", "mag") for s in ("mean", "std", "min", "max", "rms", "jerk")] + ["samples_per_s"]


def motion_features(t, x, y, z, window_s: float = MOTION_WINDOW_S) -> tuple[np.ndarray, list[str]]:
    t = np.asarray(t, dtype=np.float64)
    xyz = np.stack([np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64), np.asarray(z, dtype=np.float64)], axis=1)
    order = np.argsort(t, kind="stable")
    t, xyz = t[order], xyz[order]
    if len(t) < 20:
        return np.zeros((0, len(MOTION_FEATURES)), dtype=np.float32), MOTION_FEATURES
    # timestamps in ms / ns come up; normalise to seconds by the median step
    step = np.median(np.diff(t))
    if step > 1.0:
        t = t / (1e3 if step < 1e5 else 1e9)
    mag = np.linalg.norm(xyz, axis=1)
    cols = [xyz[:, 0], xyz[:, 1], xyz[:, 2], mag]
    windows = []
    t0 = t[0]
    while t0 < t[-1]:
        m = (t >= t0) & (t < t0 + window_s)
        t0 += window_s
        if m.sum() < 20:
            continue
        row = []
        for c in cols:
            v = c[m]
            row += [float(v.mean()), float(v.std()), float(v.min()), float(v.max()), float(np.sqrt((v ** 2).mean())),
                    float(np.abs(np.diff(v)).mean())]
        row.append(float(m.sum() / window_s))
        windows.append(row)
    return np.asarray(windows, dtype=np.float32).reshape(-1, len(MOTION_FEATURES)), MOTION_FEATURES


MODALITIES = {"mouse": (mouse_features, MOUSE_FEATURES), "keystroke": (keystroke_features, KEY_FEATURES),
              "touch": (touch_features, TOUCH_FEATURES), "motion": (motion_features, MOTION_FEATURES)}
