"""`fraudvae <command>` - one entry point for the whole pipeline.

  fraudvae fetch            download / locate raw datasets
  fraudvae build            raw -> unified parquet
  fraudvae prepare          parquet -> model arrays
  fraudvae train            train the VAE
"""
from __future__ import annotations

import sys


COMMANDS = {
    "fetch": lambda argv: __import__("fraudvae.data.fetch", fromlist=["main"]).main(argv),
    "build": lambda argv: __import__("fraudvae.data.build", fromlist=["main"]).main(argv),
    "prepare": lambda argv: __import__("fraudvae.data.prepare", fromlist=["main"]).main(argv),
    "train": lambda argv: __import__("fraudvae.train", fromlist=["main"]).main(argv),
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
