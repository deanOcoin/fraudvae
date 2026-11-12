"""Example rows that may travel with the code.

The build writes `data/processed/samples.parquet`, a few hundred labelled rows per source for
the score tab's example picker. That file is a derivative of every dataset, including the
ones whose licences do not allow redistribution, so it never leaves the machine that built
it. This module writes the part that may: for each source whose registry entry says
`redistributable`, the same kind of sample (held-out test rows, fraud oversampled, with the
source's wide block and entity codes) goes to `data/examples/<source>.parquet`, next to a
README that carries each dataset's licence and citation. That directory is committed, so
the public site - which has no processed data at all - can offer example rows and say
whether the model was right about them, and so can anyone who cloned the repository
without downloading a dataset.

    fraudvae examples               # write data/examples/ from the build's samples
    fraudvae examples --n 200       # fewer rows per source

The server reads the examples directory when it has no samples file, and only the examples
directory when it runs in public mode, whatever else is on the disk.

A row comes with its past. Where a dataset has customers, each example row carries the
payments its customer made before it (`history`, the last HISTORY of them, in the shape
`/api/score` takes), because that is how every generation since v5 was trained and measured:
the v6 encoder reads a payment against the ones before it, and a payment from a customer with
a past that is scored alone gets judged as a new customer's (on TabFormer's rows that turned
1 false alarm in 150 into 43 for v6-base-e). The build's sampler attaches them; for a samples
file from an older build, `fraudvae examples` does, from the processed tables beside it."""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from ..paths import DATA_DIR, PROCESSED_DIR
from . import schema
from .registry import SOURCES

EXAMPLES_DIR = DATA_DIR / "examples"
SAMPLES = PROCESSED_DIR / "samples.parquet"
# earlier payments kept beside each example row: enough for the history models to judge the row
# as they were measured (8 already bring v6 and v7 to their test behaviour on TabFormer's rows),
# and what the public site lets the transaction encoder read without queueing
HISTORY = 8

README_HEAD = """# Example rows

A few hundred labelled transactions per source, drawn from the held-out *test* split of the
build (the newest 20% of each dataset, which no generation of the model was trained on),
with fraud oversampled to roughly half the rows. They exist so the score tab can offer
real rows to score and say whether the model was right, on a machine that has none of the
datasets - the public demonstration site included.

Only sources whose licence allows redistribution are here. The rows are in the project's
unified schema (`docs/SCHEMA.md`), plus the source's own raw columns in the `wide` column
(json) and the entity codes the graph module uses; `is_fraud` is the dataset's label,
`split` is always `test`. Where the dataset has customers, `history` (json) holds the eight
payments the row's customer made before it, oldest first, in the shape `/api/score` takes:
the models read a payment against its customer's past, so a row is scored the way it was
measured. Those earlier payments carry no label and may come from any split.

Each file keeps the licence of the dataset it was drawn from, listed below - in particular
the Bank Account Fraud rows are **non-commercial** (CC BY-NC-SA 4.0) and the PaySim /
AMLworld rows are share-alike. Nothing here is a substitute for the datasets: download them
from the sources below for anything beyond trying the model.

The sources that are *not* here (Fraud Detection Handbook, Capital One, BankSim, IEEE-CIS,
fraud-ecommerce) either state no licence or forbid redistribution; `fraudvae fetch` gets
them from their own locations and `fraudvae examples` then writes rows for them too, on
your machine only.

"""


def redistributable_keys() -> list[str]:
    return [k for k, s in SOURCES.items() if s.redistributable]


def row_features(row) -> dict:
    """unified-schema parquet row -> the raw-units dict the score form and /api/score use"""
    feats = {}
    for f in schema.NUMERIC:
        v = row.get(f.name)
        if v is not None and not (isinstance(v, float) and math.isnan(v)):
            feats[f.name] = round(float(v), 4)
    for c in schema.CATEGORICAL:
        code = row.get(c.name, 0)
        code = int(code) if code is not None and code == code else 0
        if code:
            feats[c.name] = c.values[code]
    return feats


def attach_history(sample: pd.DataFrame, table: Path, k: int = HISTORY) -> pd.DataFrame:
    """the sample with a `history` column: for each row, the k payments its customer made before
    it in the processed table, oldest first, as the json /api/score takes for `history`
    ([{"features": {...}, "secs_before": seconds}]); None for a row without a customer or
    without an earlier payment. The order is the prepared set's sequence index: by customer,
    then time, then the table's own row order, whatever split the earlier rows fell into."""
    out = sample.copy()
    out["history"] = None
    table = Path(table)
    if k <= 0 or not len(out) or not table.exists():
        return out
    pf = pq.ParquetFile(table)
    if not {"customer", "ts", "txn_id"} <= set(pf.schema_arrow.names):
        return out
    keys = pf.read(columns=["customer", "ts", "txn_id"])
    cust = keys.column("customer").to_numpy(zero_copy_only=False)
    cust = np.where(cust == cust, cust, -1).astype(np.int64)
    ts = keys.column("ts").to_numpy(zero_copy_only=False).astype(np.float64)
    ids = keys.column("txn_id")
    hit = pc.is_in(ids, value_set=pa.array(out["txn_id"].astype(str).tolist(), type=ids.type))
    ex_rows = np.flatnonzero(hit.to_numpy(zero_copy_only=False))
    row_of = dict(zip(ids.take(pa.array(ex_rows)).to_pylist(), ex_rows.tolist()))
    del keys, ids, hit
    custs = np.unique(cust[ex_rows])
    custs = custs[custs >= 0]
    if not len(custs):
        return out
    cand = np.flatnonzero(np.isin(cust, custs))
    order = cand[np.lexsort((cand, ts[cand], cust[cand]))]        # customer, then time, then row
    wanted = set(ex_rows.tolist())
    at = {r: i for i, r in enumerate(order.tolist()) if r in wanted}
    before = {}                                                    # example's table row -> its k predecessors, oldest first
    for r in ex_rows.tolist():
        i = at.get(int(r))
        if i is None:
            continue
        j = i
        while j > 0 and i - j < k and cust[order[j - 1]] == cust[r]:
            j -= 1
        if j < i:
            before[int(r)] = order[j:i].tolist()
    need = np.unique(np.fromiter((x for v in before.values() for x in v), dtype=np.int64))
    feats, start = {}, 0
    for rg in range(pf.metadata.num_row_groups):
        n = pf.metadata.row_group(rg).num_rows
        lo, hi = np.searchsorted(need, [start, start + n])
        if hi > lo:
            frame = pf.read_row_group(rg).take(pa.array(need[lo:hi] - start)).to_pandas()
            for row_id, (_, row) in zip(need[lo:hi].tolist(), frame.iterrows()):
                feats[int(row_id)] = row_features(row)
        start += n
    hist = []
    for tid in out["txn_id"].astype(str):
        r = row_of.get(tid)
        past = before.get(int(r)) if r is not None else None
        hist.append(json.dumps([{"features": feats[p], "secs_before": round(float(ts[r] - ts[p]), 3)} for p in past]) if past else None)
    out["history"] = hist
    return out


def write_examples(out_dir: Path = EXAMPLES_DIR, n: int | None = None, samples: Path = SAMPLES, keys=None,
                   include_restricted=False, history: int = HISTORY, processed_dir: Path | None = None) -> list[str]:
    """samples.parquet -> one parquet per redistributable source (+ README.md); returns the keys
    written. history: the earlier payments kept beside each row; a samples file that has none
    (a build from before they existed) gets them from the processed table of its source, looked
    for in processed_dir (default: the directory the samples file is in)"""
    processed_dir = Path(processed_dir) if processed_dir else Path(samples).parent
    if not samples.exists():
        raise FileNotFoundError(f"{samples} does not exist; run `fraudvae build` first")
    df = pd.read_parquet(samples)
    df = df[df["split"].astype(str) == "test"]
    want = list(keys or (list(SOURCES) if include_restricted else redistributable_keys()))
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for key in want:
        part = df[df["source"].astype(str) == key]
        if part.empty:
            print(f"[{key}] no test rows in {samples.name}, skipping")
            continue
        if n:
            # keep the fraud / legit balance the build chose, just fewer of each
            fraud = part[part["is_fraud"] == 1].head(n // 2)
            legit = part[part["is_fraud"] == 0].head(n - len(fraud))
            part = pd.concat([legit, fraud]).sort_index()
        if not SOURCES[key].redistributable and not include_restricted:
            continue
        if history and ("history" not in part or part["history"].isna().all()):
            part = attach_history(part.drop(columns=["history"], errors="ignore"), processed_dir / f"{key}.parquet", history)
        elif not history and "history" in part:
            part = part.drop(columns=["history"])
        part.to_parquet(out_dir / f"{key}.parquet", index=False)
        with_past = int(part["history"].notna().sum()) if "history" in part else 0
        print(f"[{key}] {len(part)} rows ({int(part['is_fraud'].sum())} fraud, {with_past} with their customer's earlier payments) -> {out_dir / (key + '.parquet')}"
              + ("" if SOURCES[key].redistributable else "  (NOT redistributable: keep this file local)"))
        written.append(key)
    _write_readme(out_dir, written)
    return written


def _write_readme(out_dir: Path, keys: list[str]):
    lines = [README_HEAD, "| source | dataset | licence | rows here |", "|---|---|---|---:|"]
    for key in keys:
        s = SOURCES[key]
        lic = f"[{s.license}]({s.license_url})" if s.license_url else s.license
        n = len(pd.read_parquet(out_dir / f"{key}.parquet", columns=["is_fraud"]))
        lines.append(f"| {key} | {s.name} | {lic} | {n:,} |")
    lines.append("")
    for key in keys:
        s = SOURCES[key]
        lines += [f"## {key}: {s.name}", "", f"- source: <{s.url}>", f"- citation: {s.citation}",
                  f"- licence: {s.license}" + (f" (<{s.license_url}>)" if s.license_url else ""), ""]
    lines.append(f"Written by `fraudvae examples` on {time.strftime('%Y-%m-%d')}; regenerate rather than edit.\n")
    (out_dir / "README.md").write_text("\n".join(lines))


def load_examples(out_dir: Path = EXAMPLES_DIR) -> pd.DataFrame | None:
    """every example file as one frame, or None when the directory is empty"""
    files = sorted(Path(out_dir).glob("*.parquet")) if Path(out_dir).exists() else []
    if not files:
        return None
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="fraudvae examples", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=None, help="rows per source (default: everything the build sampled, ~300)")
    ap.add_argument("--out", default=str(EXAMPLES_DIR))
    ap.add_argument("--samples", default=str(SAMPLES))
    ap.add_argument("--source", action="append", help="limit to these sources (repeatable)")
    ap.add_argument("--include-restricted", action="store_true",
                    help="also write the sources whose licence does not allow redistribution (for your own machine)")
    ap.add_argument("--history", type=int, default=HISTORY,
                    help=f"earlier payments of the same customer kept beside each row (default {HISTORY}; 0 for none)")
    args = ap.parse_args(argv)
    written = write_examples(Path(args.out), args.n, Path(args.samples), args.source, args.include_restricted, args.history)
    print(f"\n{len(written)} sources written to {args.out}")
    return 0 if written else 2


if __name__ == "__main__":
    sys.exit(main())
