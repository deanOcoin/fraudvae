"""v7: a fast reimplementation of the Fraud Detection Handbook simulator's shape, at volume.

    fraudvae simulate --rows 50000000 --out data/raw/fdh_sim/sim365 [--days 365] [--seed 1337]

Writes one pickle per day in the handbook's format (TRANSACTION_ID, TX_DATETIME, CUSTOMER_ID,
TERMINAL_ID, TX_AMOUNT, TX_TIME_SECONDS, TX_TIME_DAYS, TX_FRAUD, TX_FRAUD_SCENARIO), so the fdh_sim
adapter builds the output unchanged. The handbook's generator is pure Python per transaction and
cannot produce 50M rows in a session; this one is vectorised numpy, a day at a time, and does more
than ten million rows a minute on one core. Customers have more habits than the original's -
three preferred terminals, an hour of day, a weekend rate - because per-customer regularities
are what a sequence encoder learns from. The fraud is the handbook's three scenarios: any amount
above 220; terminals compromised for 28 days (every transaction fraud); customers compromised for
14 days (a third of their transactions fraud at five times the amount). The daily compromise
counts are the handbook's two terminals and three customers scaled to the population (it ran
5k customers and 10k terminals), so the fraud share stays near the published set's at any size.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

COLUMNS = ["TRANSACTION_ID", "TX_DATETIME", "CUSTOMER_ID", "TERMINAL_ID", "TX_AMOUNT", "TX_TIME_SECONDS", "TX_TIME_DAYS",
           "TX_FRAUD", "TX_FRAUD_SCENARIO"]
GRID = 100.0                 # the handbook's 100 x 100 grid
RADIUS = 5.0                 # a customer pays at terminals within this distance
PREFERRED, PREFERRED_SHARE = 3, 0.8
AMOUNT_FRAUD = 220.0
TERMINAL_DAYS, CUSTOMER_DAYS = 28, 14
TERMINALS_PER_DAY, CUSTOMERS_PER_DAY = 2 / 10_000, 3 / 5_000   # the handbook's counts per terminal and per customer
WEEKEND_RATE = 1.3


def _customers(n: int, rng):
    """locations, amount habits, rate, hour of day; the terminals (as many as customers) and
    each customer's three preferred terminals within the radius (the nearest ones when the
    population is too thin for three inside it)"""
    from scipy.spatial import cKDTree

    cx, cy = rng.uniform(0, GRID, n), rng.uniform(0, GRID, n)
    mean_amt = rng.lognormal(np.log(40.0), 0.5, n)          # about 45 on average, a long tail
    rate = rng.uniform(0, 4, n)                             # transactions a day
    hour = rng.uniform(8, 20, n)                            # the centre of the customer's day
    tree = cKDTree(np.column_stack([rng.uniform(0, GRID, n), rng.uniform(0, GRID, n)]))
    k = min(32, n)
    d, i = tree.query(np.column_stack([cx, cy]), k=k)
    d, i = np.atleast_2d(d).reshape(n, k), np.atleast_2d(i).reshape(n, k)
    keys = rng.random((n, k))
    keys[d > RADIUS] += 1.0                                 # inside the radius first, at random among them
    pref = i[np.arange(n)[:, None], np.argsort(keys, axis=1)[:, :PREFERRED]]
    if k < PREFERRED:
        pref = np.tile(i[:, :1], (1, PREFERRED))
    return cx, cy, mean_amt, mean_amt / 2, rate, hour, tree, pref


def _day(day: int, c0: int, c1: int, counts, cust_info, until_t, until_c, rng) -> dict:
    """one day's transactions of customers c0..c1 (vectorised), with the fraud scenarios applied"""
    cx, cy, mean_amt, sd_amt, rate, hour, tree, pref = cust_info
    cust = np.repeat(np.arange(c0, c1), counts[c0:c1])
    m = len(cust)
    amt = rng.normal(mean_amt[cust], sd_amt[cust])
    bad = amt <= 0
    if bad.any():
        amt[bad] = rng.uniform(0.01, 2 * mean_amt[cust[bad]])
    sec = np.clip(rng.normal(hour[cust] * 3600, 3 * 3600), 0, 86399).astype(np.int64)
    term = pref[cust, rng.integers(0, PREFERRED, m)]
    other = rng.random(m) >= PREFERRED_SHARE
    if other.any():
        mo = int(other.sum())
        r, th = RADIUS * np.sqrt(rng.random(mo)), rng.uniform(0, 2 * np.pi, mo)
        _, ti = tree.query(np.column_stack([cx[cust[other]] + r * np.cos(th), cy[cust[other]] + r * np.sin(th)]))
        term[other] = ti
    scen = np.zeros(m, dtype=np.int64)
    scen[amt > AMOUNT_FRAUD] = 1
    scen[until_t[term] > day] = 2
    hit = (until_c[cust] > day) & (rng.random(m) < 1 / 3)
    amt[hit] *= 5
    scen[hit] = 3
    return {"CUSTOMER_ID": cust, "TERMINAL_ID": term, "TX_AMOUNT": np.maximum(np.round(amt, 2), 0.01), "sec": sec, "TX_FRAUD_SCENARIO": scen}


def simulate(rows: int, out, days: int = 365, seed: int = 1337, start: str = "2024-01-01", log=print, chunk_rows: int = 5_000_000) -> dict:
    """about `rows` transactions over `days` days, one pickle a day under `out`; returns the summary
    also written as out/simulate.json"""
    t0 = time.time()
    rng = np.random.default_rng(seed)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    days = max(1, int(days))
    n = max(3, int(round(rows / (days * 2.0))))              # a mean of two transactions a customer a day
    info = _customers(n, rng)
    rate = info[4]
    until_t, until_c = np.full(n, -1, dtype=np.int64), np.full(n, -1, dtype=np.int64)
    acc_t = acc_c = 0.0
    start_ts = pd.Timestamp(start)
    per_chunk = max(1, chunk_rows // 8)                      # customers per block: at most about 5.2 rows a day each
    tid, written, fraud = 0, 0, 0
    per_scenario = {1: 0, 2: 0, 3: 0}
    episodes = {"terminals": 0, "customers": 0}
    log(f"simulate: {n:,} customers and terminals, {days} days from {start_ts.date()}, seed {seed} -> {out}")
    for day in range(days):
        date = start_ts + pd.Timedelta(days=day)
        acc_t += TERMINALS_PER_DAY * n
        k = int(acc_t)
        acc_t -= k
        if k:
            t_ = rng.choice(n, min(k, n), replace=False)
            until_t[t_] = np.maximum(until_t[t_], day + TERMINAL_DAYS)
            episodes["terminals"] += int(len(t_))
        acc_c += CUSTOMERS_PER_DAY * n
        k = int(acc_c)
        acc_c -= k
        if k:
            c_ = rng.choice(n, min(k, n), replace=False)
            until_c[c_] = np.maximum(until_c[c_], day + CUSTOMER_DAYS)
            episodes["customers"] += int(len(c_))
        counts = rng.poisson(rate * (WEEKEND_RATE if date.dayofweek in (4, 5) else 1.0))
        parts = [_day(day, c0, min(n, c0 + per_chunk), counts, info, until_t, until_c, rng) for c0 in range(0, n, per_chunk)]
        block = {k_: np.concatenate([p[k_] for p in parts]) for k_ in parts[0]}
        order = np.argsort(block["sec"], kind="stable")
        m = len(order)
        secs = day * 86400 + block["sec"][order]
        scen = block["TX_FRAUD_SCENARIO"][order]
        df = pd.DataFrame({"TRANSACTION_ID": np.arange(tid, tid + m, dtype=np.int64), "TX_DATETIME": start_ts + pd.to_timedelta(secs, unit="s"),
                           "CUSTOMER_ID": block["CUSTOMER_ID"][order], "TERMINAL_ID": block["TERMINAL_ID"][order],
                           "TX_AMOUNT": block["TX_AMOUNT"][order], "TX_TIME_SECONDS": secs, "TX_TIME_DAYS": np.full(m, day, dtype=np.int64),
                           "TX_FRAUD": (scen > 0).astype(np.int64), "TX_FRAUD_SCENARIO": scen})[COLUMNS]
        df.to_pickle(out / f"{date.date()}.pkl")
        tid += m
        written += m
        fraud += int((scen > 0).sum())
        for s_ in (1, 2, 3):
            per_scenario[s_] += int((scen == s_).sum())
        if (day + 1) % 50 == 0 or day + 1 == days:
            log(f"  day {day + 1}/{days}: {written:,} rows so far, {fraud:,} fraud ({fraud / max(1, written):.2%}), {time.time() - t0:.0f}s")
    summary = {"created": time.strftime("%Y-%m-%dT%H:%M:%S"), "rows": written, "fraud": fraud, "fraud_rate": fraud / max(1, written),
               "per_scenario": {str(k_): v for k_, v in per_scenario.items()}, "customers": n, "terminals": n, "days": days,
               "start": str(start_ts.date()), "seed": seed, "episodes": episodes, "rows_requested": int(rows),
               "seconds": round(time.time() - t0, 1), "rows_per_minute": round(written / max(1e-9, time.time() - t0) * 60)}
    (out / "simulate.json").write_text(json.dumps(summary, indent=1))
    log(f"simulate: {written:,} rows, {fraud:,} fraud ({summary['fraud_rate']:.2%}; scenarios {per_scenario}), "
        f"{summary['rows_per_minute']:,} rows/min, {days} pickles under {out}")
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(prog="fraudvae simulate", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", type=int, required=True, help="about this many transactions in all")
    ap.add_argument("--out", required=True, help="the directory for the daily pickles (under data/raw/fdh_sim/ to build it as fdh_sim)")
    ap.add_argument("--days", type=int, default=365)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--start", default="2024-01-01", help="the first day")
    args = ap.parse_args(argv)
    simulate(args.rows, args.out, days=args.days, seed=args.seed, start=args.start)
    return 0


if __name__ == "__main__":
    sys.exit(main())
