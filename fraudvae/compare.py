"""`fraudvae compare A B` - two runs scored on the same test rows, judged paired.

    fraudvae compare v5-h200-h v7-a                 # both label settings when both runs have them
    fraudvae compare v5-h200-h v7-a --delay 30      # the 30-day-label scores only
    fraudvae compare v5-h200-h v7-a --signal booster --fpr 0.01 0.005 --boot 1000 --json out.json

Reads runs/<A>/scores_test.npz and runs/<B>/scores_test.npz (`fraudvae evaluate` writes them),
checks that they hold the same rows, and reports per source and for the mean over sources the
AUROC, the recall at each FPR target and the precision at the first one, for B against A, with
a 95% interval on every difference from a paired bootstrap and a verdict - better, worse or
undecided. The receipt's summary numbers cannot decide a difference of a few thousandths; the
same rows scored twice can, because whatever the rows themselves add to the variance cancels.

The resampling is a Poisson bootstrap: every row draws a weight from Poisson(1), the same
weight for both runs, so a replicate is one re-drawing of the test set applied to both. Metrics
are computed from weighted counts over the sorted scores, which makes a thousand replicates of
an eight-million-row split a matter of minutes rather than hours. The threshold behind the
recall and precision at an FPR target is the lowest score at which the (weighted) share of
legitimate rows at or below it reaches 1 - target; a row is flagged when strictly above it. With
unit weights that is the empirical quantile, a hair from eval.json's interpolated one.

The verdict on a source reads the recall at the first target: better when its interval is
above zero, or when the AUROC's is and the recall's does not fall below; worse by the mirror
rule; undecided otherwise. That rule is the arbiter the v7 plan names for keeping a change.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from .evaluate import read_scores, scores_name
from .paths import RUNS_DIR

DEFAULT_FPRS = (0.01, 0.005)
CI = (2.5, 97.5)


class Ranked:
    """one run's scores on one source, grouped by distinct value, so a weighted AUROC, recall
    and precision are a bincount and a cumulative sum away"""

    def __init__(self, scores, y):
        s = np.asarray(scores, dtype=np.float64)
        ok = np.isfinite(s)
        if not ok.all():
            # a non-finite score ranks as "most anomalous", as scoring.auroc has it
            s = np.where(ok, s, (np.nanmax(s[ok]) + 1.0) if ok.any() else 0.0)
        vals, inv = np.unique(s, return_inverse=True)
        self.inv = inv.reshape(-1).astype(np.int64)
        self.n_vals = len(vals)
        self.pos = np.asarray(y) == 1
        self.neg = ~self.pos

    def metrics(self, w: np.ndarray, fprs=DEFAULT_FPRS) -> dict:
        wp = np.bincount(self.inv, weights=w * self.pos, minlength=self.n_vals)
        wn = np.bincount(self.inv, weights=w * self.neg, minlength=self.n_vals)
        Wp, Wn = float(wp.sum()), float(wn.sum())
        keys = metric_keys(fprs)
        if Wp <= 0 or Wn <= 0:
            return {k: float("nan") for k in keys}
        cum_n = np.cumsum(wn)                       # legit weight at or below each value
        out = {"auroc": float(((cum_n - wn + 0.5 * wn) * wp).sum() / (Wp * Wn))}
        for j, f in enumerate(fprs):
            k = min(int(np.searchsorted(cum_n, (1.0 - f) * Wn, side="left")), self.n_vals - 1)
            tp, fp = float(wp[k + 1:].sum()), float(wn[k + 1:].sum())
            out[f"recall@{f:g}"] = tp / Wp
            if j == 0:
                out[f"precision@{f:g}"] = tp / (tp + fp) if tp + fp > 0 else float("nan")
        return out


def metric_keys(fprs=DEFAULT_FPRS) -> list[str]:
    return ["auroc"] + [f"recall@{f:g}" for f in fprs] + [f"precision@{fprs[0]:g}"]


def _interval(d: np.ndarray) -> tuple[float, float]:
    d = d[np.isfinite(d)]
    if len(d) == 0:
        return float("nan"), float("nan")
    lo, hi = np.percentile(d, CI)
    return float(lo), float(hi)


def _call(lo: float, hi: float) -> str:
    if lo != lo or hi != hi:
        return "undecided"
    return "better" if lo > 0 else ("worse" if hi < 0 else "undecided")


def _source_verdict(cells: dict, fprs) -> str:
    rec, auc = cells[f"recall@{fprs[0]:g}"], cells["auroc"]
    if rec["call"] == "better" or (auc["call"] == "better" and rec["hi"] >= 0):
        return "better"
    if rec["call"] == "worse" or (auc["call"] == "worse" and rec["lo"] <= 0):
        return "worse"
    return "undecided"


def compare_scores(a: dict, b: dict, signal="blend", fprs=DEFAULT_FPRS, boot=1000, seed=0, log=print) -> dict:
    """a, b: read_scores() dicts of two runs on the same prepared set. returns the comparison
    (per source and the mean over sources: point values for A and B, the difference, its
    interval, a call per metric and a verdict per source)"""
    fprs = tuple(float(f) for f in fprs)
    if len(a["row"]) != len(b["row"]) or not (np.array_equal(a["row"], b["row"]) and np.array_equal(a["src"], b["src"])):
        raise SystemExit("the two runs were not scored on the same rows (different prepared sets?); a paired comparison needs the same rows")
    if not np.array_equal(a["y"], b["y"]):
        raise SystemExit("the two score files disagree on the labels of the same rows")
    for name, d in (("A", a), ("B", b)):
        if signal not in d:
            raise SystemExit(f"run {name} has no '{signal}' scores (it has {', '.join(d['meta']['signals'])})")
    if (a["meta"].get("label_delay_days") or None) != (b["meta"].get("label_delay_days") or None):
        raise SystemExit("one file holds delayed-label scores and the other does not")
    names = a["meta"].get("source_names", {})
    y, src = a["y"], a["src"]
    keys = metric_keys(fprs)
    rng = np.random.default_rng(seed)
    t0 = time.time()
    out = {"signal": signal, "label_delay_days": a["meta"].get("label_delay_days"), "fprs": list(fprs), "boot": int(boot),
           "seed": int(seed), "rows": int(len(y)), "fraud": int(y.sum()), "ci": list(CI),
           "a": {k: a["meta"].get(k) for k in ("created", "prepared")}, "b": {k: b["meta"].get(k) for k in ("created", "prepared")},
           "sources": {}, "mean": {}}
    codes = [int(c) for c in np.unique(src)]
    # per source: point values and the replicate values, kept so the mean over sources pairs too
    reps_a = {k: np.full((len(codes), boot), np.nan) for k in keys}
    reps_b = {k: np.full((len(codes), boot), np.nan) for k in keys}
    for i, code in enumerate(codes):
        m = src == code
        name = names.get(str(code), str(code))
        ra, rb = Ranked(a[signal][m], y[m]), Ranked(b[signal][m], y[m])
        ones = np.ones(int(m.sum()), dtype=np.float64)
        pa, pb = ra.metrics(ones, fprs), rb.metrics(ones, fprs)
        n = int(m.sum())
        for r in range(boot):
            w = rng.poisson(1.0, n).astype(np.float64)
            ma, mb = ra.metrics(w, fprs), rb.metrics(w, fprs)
            for k in keys:
                reps_a[k][i, r], reps_b[k][i, r] = ma[k], mb[k]
        cells = {}
        for k in keys:
            lo, hi = _interval(reps_b[k][i] - reps_a[k][i])
            cells[k] = {"a": pa[k], "b": pb[k], "delta": pb[k] - pa[k], "lo": lo, "hi": hi, "call": _call(lo, hi)}
        out["sources"][name] = {"n": n, "fraud": int(y[m].sum()), "metrics": cells, "verdict": _source_verdict(cells, fprs)}
        log(f"  {name}: {n:,} rows, {int(y[m].sum()):,} fraud; {keys[0]} {pa[keys[0]]:.4f} -> {pb[keys[0]]:.4f} "
            f"[{cells[keys[0]]['lo']:+.4f}, {cells[keys[0]]['hi']:+.4f}], {keys[1]} {pa[keys[1]]:.3f} -> {pb[keys[1]]:.3f} "
            f"[{cells[keys[1]]['lo']:+.3f}, {cells[keys[1]]['hi']:+.3f}] -> {out['sources'][name]['verdict']} ({time.time() - t0:.0f}s)")
    cells = {}
    for k in keys:
        pa = float(np.nanmean([s_["metrics"][k]["a"] for s_ in out["sources"].values()]))
        pb = float(np.nanmean([s_["metrics"][k]["b"] for s_ in out["sources"].values()]))
        d = np.nanmean(reps_b[k], axis=0) - np.nanmean(reps_a[k], axis=0)
        lo, hi = _interval(d)
        cells[k] = {"a": pa, "b": pb, "delta": pb - pa, "lo": lo, "hi": hi, "call": _call(lo, hi)}
    out["mean"] = {"n_sources": len(codes), "metrics": cells, "verdict": _source_verdict(cells, fprs)}
    out["seconds"] = round(time.time() - t0, 1)
    return out


def _pct(f: float) -> str:
    return f"{100 * f:g}%"


def _label(k: str) -> str:
    if "@" in k:
        m, f = k.split("@")
        return f"{m}@{_pct(float(f))}"
    return k


def format_table(res: dict, a_name="A", b_name="B") -> str:
    """the comparison as a markdown table, one row per source and one for the mean"""
    keys = metric_keys(res["fprs"])
    tag = f"{res['label_delay_days']:g}-day labels" if res.get("label_delay_days") else "labels as published"
    lines = [f"### {res['signal']}, {tag}: {b_name} against {a_name} on {res['rows']:,} rows ({res['fraud']:,} fraud), "
             f"{res['boot']} replicates, {res['ci'][1] - res['ci'][0]:g}% intervals", ""]
    head = ["source", "n", "fraud"]
    for k in keys:
        head += [f"{_label(k)} {a_name}", b_name, "delta [interval]"]
    head.append("verdict")
    lines.append("| " + " | ".join(head) + " |")
    lines.append("|" + "---|" * len(head))

    def row(name, n, fraud, cells, verdict):
        parts = [name, f"{n:,}" if n is not None else "", f"{fraud:,}" if fraud is not None else ""]
        for k in keys:
            c = cells[k]
            digits = 4 if k == "auroc" else 3
            mark = {"better": " +", "worse": " -", "undecided": ""}[c["call"]]
            parts += [f"{c['a']:.{digits}f}", f"{c['b']:.{digits}f}",
                      f"{c['delta']:+.{digits}f} [{c['lo']:+.{digits}f}, {c['hi']:+.{digits}f}]{mark}"]
        parts.append(verdict)
        return "| " + " | ".join(parts) + " |"

    for name, s_ in res["sources"].items():
        lines.append(row(name, s_["n"], s_["fraud"], s_["metrics"], s_["verdict"]))
    lines.append(row(f"mean of {res['mean']['n_sources']}", None, None, res["mean"]["metrics"], res["mean"]["verdict"]))
    lines.append("")
    lines.append("a + or - after an interval marks one that excludes zero; the verdict reads the recall at the first target, "
                 "then the auroc (see `fraudvae compare --help`)")
    return "\n".join(lines)


def _run_dir(run: str) -> Path:
    p = Path(run)
    return p if p.exists() else RUNS_DIR / run


def compare_runs(run_a: str, run_b: str, delay=None, signal="blend", fprs=DEFAULT_FPRS, boot=1000, seed=0, log=print) -> list[dict]:
    """every label setting both runs were scored under (or the one asked for), compared"""
    da, db = _run_dir(run_a), _run_dir(run_b)
    settings = [delay] if delay is not None else [None] + sorted(
        {float(p.name[len("scores_test_delay"):-len(".npz")]) for p in da.glob("scores_test_delay*.npz")}
        & {float(p.name[len("scores_test_delay"):-len(".npz")]) for p in db.glob("scores_test_delay*.npz")})
    out = []
    for d in settings:
        fa, fb = da / scores_name(d), db / scores_name(d)
        for f, run in ((fa, run_a), (fb, run_b)):
            if not f.exists():
                raise SystemExit(f"{f} is missing: run `fraudvae evaluate {run}`" + (f" --label-delay {d:g}" if d else "") + " first")
        log(f"comparing {db.name} against {da.name} ({'labels as published' if not d else f'{d:g}-day labels'}, signal {signal}) ...")
        res = compare_scores(read_scores(fa), read_scores(fb), signal=signal, fprs=fprs, boot=boot, seed=seed, log=log)
        res["a"]["run"], res["b"]["run"] = da.name, db.name
        out.append(res)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(prog="fraudvae compare", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_a", help="the baseline")
    ap.add_argument("run_b", help="the run judged against it")
    ap.add_argument("--delay", type=float, default=None, help="compare the scores made under this label latency only (e.g. 30)")
    ap.add_argument("--signal", default="blend", help="which score to compare: blend (default), booster, vae, local, seq, ...")
    ap.add_argument("--fpr", type=float, nargs="+", default=list(DEFAULT_FPRS), help="FPR targets for the recall (the first also gives the precision)")
    ap.add_argument("--boot", type=int, default=1000, help="bootstrap replicates (1000; 200 for a quick look)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", default=None, help="also write the full comparison here")
    ap.add_argument("--quiet", action="store_true", help="only the tables")
    args = ap.parse_args(argv)
    log = (lambda m: None) if args.quiet else print
    results = compare_runs(args.run_a, args.run_b, delay=args.delay, signal=args.signal, fprs=tuple(args.fpr), boot=args.boot,
                           seed=args.seed, log=log)
    for res in results:
        print()
        print(format_table(res, res["a"]["run"], res["b"]["run"]))
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=1))
        print(f"\nwritten to {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
