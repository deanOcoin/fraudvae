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
directory when it runs in public mode, whatever else is on the disk."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

from ..paths import DATA_DIR, PROCESSED_DIR
from .registry import SOURCES

EXAMPLES_DIR = DATA_DIR / "examples"
SAMPLES = PROCESSED_DIR / "samples.parquet"

README_HEAD = """# Example rows

A few hundred labelled transactions per source, drawn from the held-out *test* split of the
build (the newest 20% of each dataset, which no generation of the model was trained on),
with fraud oversampled to roughly half the rows. They exist so the score tab can offer
real rows to score and say whether the model was right, on a machine that has none of the
datasets - the public demonstration site included.

Only sources whose licence allows redistribution are here. The rows are in the project's
unified schema (`docs/SCHEMA.md`), plus the source's own raw columns in the `wide` column
(json) and the entity codes the graph module uses; `is_fraud` is the dataset's label,
`split` is always `test`. Each file keeps the licence of the dataset it was drawn from,
listed below - in particular the Bank Account Fraud rows are **non-commercial** (CC BY-NC-SA
4.0) and the PaySim / AMLworld rows are share-alike. Nothing here is a substitute for the
datasets: download them from the sources below for anything beyond trying the model.

The sources that are *not* here (Fraud Detection Handbook, Capital One, BankSim, IEEE-CIS,
fraud-ecommerce) either state no licence or forbid redistribution; `fraudvae fetch` gets
them from their own locations and `fraudvae examples` then writes rows for them too, on
your machine only.

"""


def redistributable_keys() -> list[str]:
    return [k for k, s in SOURCES.items() if s.redistributable]


def write_examples(out_dir: Path = EXAMPLES_DIR, n: int | None = None, samples: Path = SAMPLES, keys=None,
                   include_restricted=False) -> list[str]:
    """samples.parquet -> one parquet per redistributable source (+ README.md); returns the keys written"""
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
        part.to_parquet(out_dir / f"{key}.parquet", index=False)
        print(f"[{key}] {len(part)} rows ({int(part['is_fraud'].sum())} fraud) -> {out_dir / (key + '.parquet')}"
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
    args = ap.parse_args(argv)
    written = write_examples(Path(args.out), args.n, Path(args.samples), args.source, args.include_restricted)
    print(f"\n{len(written)} sources written to {args.out}")
    return 0 if written else 2


if __name__ == "__main__":
    sys.exit(main())
