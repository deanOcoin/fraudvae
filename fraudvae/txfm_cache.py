"""v6: the transaction encoder's block pass over every row of a prepared set, cached next to the
set so the booster, calibration and evaluation read one pass per split instead of encoding their
own rows again (TxfmScorer.features_split).

    fraudvae txfm-cache RUN --prepared full                       # every split, one process
    fraudvae txfm-cache RUN --prepared full --split train --part 0 --parts 4 --device cuda:0
    ...                                                           # one part per gpu, then
    fraudvae txfm-cache RUN --prepared full --merge

RUN is a run holding txfm.pt (a pretraining run, or a post-stage run the checkpoint was copied
into). The cache lives at data/prepared/<set>/txfm_cache/<sha of txfm.pt>/txfm_<split>.npy.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from .model.dataset import PreparedSet
from .model.txfm import TxfmScorer
from .paths import RUNS_DIR

SPLITS = ("train", "val", "test")


def ensure_cache(scorer: TxfmScorer, prepared: str, splits=SPLITS, device=None, log=print, batch_size=256):
    """write the cache of every listed split that does not exist yet, in this process"""
    scorer.with_cache(prepared)
    missing = [s for s in splits if not (scorer.cache_dir / f"txfm_{s}.npy").exists()]
    if not missing:
        return scorer.cache_dir
    data = PreparedSet(prepared, splits=tuple(missing), in_memory=False)
    for s in missing:
        t0 = time.time()
        scorer.cache_split(data[s], data.seq, s, scorer.cache_dir, device=device, batch_size=batch_size, log=log)
        TxfmScorer.merge_cache(scorer.cache_dir, s, len(data[s]))
        log(f"  txfm cache: {s} done, {len(data[s]):,} rows in {time.time() - t0:.0f}s -> {scorer.cache_dir / f'txfm_{s}.npy'}")
    return scorer.cache_dir


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run")
    ap.add_argument("--prepared", default="default")
    ap.add_argument("--split", action="append", help="train / val / test (default: all three)")
    ap.add_argument("--part", type=int, default=None, help="this process does the part-th of --parts slices of the split")
    ap.add_argument("--parts", type=int, default=1)
    ap.add_argument("--merge", action="store_true", help="assemble the parts written so far into txfm_<split>.npy")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch-size", type=int, default=256)
    args = ap.parse_args(argv)
    p = Path(args.run)
    run_dir = p if p.exists() else RUNS_DIR / args.run
    scorer = TxfmScorer.load(run_dir, device=args.device)
    if scorer is None:
        raise SystemExit(f"{run_dir} has no txfm.pt")
    scorer.with_cache(args.prepared)
    splits = tuple(args.split or SPLITS)
    data = PreparedSet(args.prepared, splits=splits, in_memory=False)
    if args.merge:
        for s in splits:
            print(f"merged {TxfmScorer.merge_cache(scorer.cache_dir, s, len(data[s]))}")
        return 0
    if args.part is None:
        ensure_cache(scorer, args.prepared, splits, device=args.device, batch_size=args.batch_size)
        return 0
    for s in splits:
        scorer.cache_split(data[s], data.seq, s, scorer.cache_dir, part=args.part, parts=args.parts, device=args.device,
                           batch_size=args.batch_size)
    return 0


if __name__ == "__main__":
    sys.exit(main())
