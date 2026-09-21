"""`fraudvae <command>` - one entry point for the whole pipeline.

  fraudvae fetch            download / locate raw datasets
  fraudvae build            raw -> unified parquet
  fraudvae prepare          parquet -> model arrays
  fraudvae train            train the VAE (+ booster, calibration, evaluation)
  fraudvae boost RUN        (re)train the supervised booster for a run
  fraudvae post RUN         booster + calibration + evaluation for an existing run (new --config ok)
  fraudvae calibrate RUN    (re)compute thresholds for a run
  fraudvae evaluate RUN     test metrics for a run
  fraudvae serve            web ui
  fraudvae key              print the app's ssh public key
  fraudvae advise           local vs remote compute recommendation
  fraudvae remote ...       probe / provision / launch / status / fetch / run on a gpu box over ssh
"""
from __future__ import annotations

import os
import sys


def _serve(argv):
    import argparse

    import uvicorn

    ap = argparse.ArgumentParser(prog="fraudvae serve")
    ap.add_argument("--host", default=os.environ.get("FRAUDVAE_HOST", "127.0.0.1"))
    ap.add_argument("--port", type=int, default=int(os.environ.get("FRAUDVAE_PORT", "8765")))
    ap.add_argument("--run", default=None, help="run to load (default: best by test auroc)")
    ap.add_argument("--reload", action="store_true")
    a = ap.parse_args(argv)
    if a.run:
        os.environ["FRAUDVAE_RUN"] = a.run
    print(f"fraudvae ui -> http://{a.host}:{a.port}")
    uvicorn.run("fraudvae.server.app:app", host=a.host, port=a.port, reload=a.reload, log_level="info")


def _key(argv):
    from .remote import keys

    print(keys.public_key_text())
    print(f"# {keys.fingerprint()}  ({keys.PUBLIC})", file=sys.stderr)


def _advise(argv):
    import argparse
    import json

    from .remote.advisor import advise

    ap = argparse.ArgumentParser(prog="fraudvae advise")
    ap.add_argument("--config", default="gpu_full")
    ap.add_argument("--prepared", default="default")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    rec = advise(a.config, a.prepared)
    if a.json:
        print(json.dumps(rec, indent=1))
        return
    loc = rec["local"]
    print(f"train rows: {rec['train_rows']:,}  epochs: {rec['epochs']}")
    print(f"local ({loc['device']}, {loc['device_name']}): {loc['rows_per_sec']:,.0f} rows/s -> ~{loc['est_hours']:.1f} h")
    print("recommendation:", "RENT A GPU" if rec["recommend_remote"] else "train locally")
    for t in rec["gpu_tiers"]:
        print(f"  {t['name']:<32} ~{t['est_hours']:.2f} h  ~${t['est_cost_usd']:.2f}")
    print("requirements:", json.dumps(rec["requirements"], indent=1))


COMMANDS = {
    "fetch": lambda argv: __import__("fraudvae.data.fetch", fromlist=["main"]).main(argv),
    "build": lambda argv: __import__("fraudvae.data.build", fromlist=["main"]).main(argv),
    "prepare": lambda argv: __import__("fraudvae.data.prepare", fromlist=["main"]).main(argv),
    "train": lambda argv: __import__("fraudvae.train", fromlist=["main"]).main(argv),
    "boost": lambda argv: __import__("fraudvae.model.booster", fromlist=["main"]).main(argv),
    "post": lambda argv: __import__("fraudvae.train", fromlist=["post_main"]).post_main(argv),
    "graph-bench": lambda argv: __import__("fraudvae.graph.bench", fromlist=["main"]).main(argv),
    "calibrate": lambda argv: __import__("fraudvae.calibrate", fromlist=["main"]).main(argv),
    "evaluate": lambda argv: __import__("fraudvae.evaluate", fromlist=["main"]).main(argv),
    "serve": _serve,
    "key": _key,
    "advise": _advise,
    "remote": lambda argv: __import__("fraudvae.remote.cli", fromlist=["main"]).main(argv),
}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(__doc__)
        return 0
    cmd, rest = argv[0], argv[1:]
    if cmd not in COMMANDS:
        print(f"unknown command {cmd!r}\n{__doc__}", file=sys.stderr)
        return 2
    rc = COMMANDS[cmd](rest)
    return rc or 0


if __name__ == "__main__":
    sys.exit(main())
