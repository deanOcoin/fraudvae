"""`fraudvae <command>` - one entry point for the whole pipeline.

  fraudvae fetch            download / locate raw datasets
  fraudvae build            raw -> unified parquet
  fraudvae prepare          parquet -> model arrays
  fraudvae train            train the VAE (+ booster, calibration, evaluation)
  fraudvae boost RUN        (re)train the supervised booster for a run
  fraudvae calibrate RUN    (re)compute thresholds for a run
  fraudvae evaluate RUN     test metrics for a run
  fraudvae serve            web ui
"""
from __future__ import annotations

import sys


def _serve(argv):
    import argparse

    import uvicorn

    ap = argparse.ArgumentParser(prog="fraudvae serve")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--reload", action="store_true")
    a = ap.parse_args(argv)
    print(f"fraudvae ui -> http://{a.host}:{a.port}")
    uvicorn.run("fraudvae.server.app:app", host=a.host, port=a.port, reload=a.reload)


COMMANDS = {
    "fetch": lambda argv: __import__("fraudvae.data.fetch", fromlist=["main"]).main(argv),
    "build": lambda argv: __import__("fraudvae.data.build", fromlist=["main"]).main(argv),
    "prepare": lambda argv: __import__("fraudvae.data.prepare", fromlist=["main"]).main(argv),
    "train": lambda argv: __import__("fraudvae.train", fromlist=["main"]).main(argv),
    "boost": lambda argv: __import__("fraudvae.model.booster", fromlist=["main"]).main(argv),
    "calibrate": lambda argv: __import__("fraudvae.calibrate", fromlist=["main"]).main(argv),
    "evaluate": lambda argv: __import__("fraudvae.evaluate", fromlist=["main"]).main(argv),
    "serve": _serve,
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
