"""`fraudvae <command>` - one entry point for the whole pipeline.

  fraudvae fetch            download / locate raw datasets
  fraudvae build            raw -> unified parquet
"""
from __future__ import annotations

import sys


COMMANDS = {
    "fetch": lambda argv: __import__("fraudvae.data.fetch", fromlist=["main"]).main(argv),
    "build": lambda argv: __import__("fraudvae.data.build", fromlist=["main"]).main(argv),
}


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    cmd = argv[0]
    if cmd not in COMMANDS:
        print(f"unknown command {cmd!r}\n{__doc__}", file=sys.stderr)
        return 2
    rc = COMMANDS[cmd](argv[1:])
    return rc or 0


if __name__ == "__main__":
    sys.exit(main())
