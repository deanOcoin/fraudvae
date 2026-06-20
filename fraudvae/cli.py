"""`fraudvae <command>` - one entry point for the whole pipeline.

  fraudvae paths     show where data, raw files and runs live
  fraudvae fetch     download / locate raw datasets
"""
from __future__ import annotations

import sys

from . import paths


def _paths(argv):
    for name in ("ROOT", "DATA", "RAW", "RUNS"):
        print(f"{name:<6} {getattr(paths, name)}")


def _fetch(argv):
    from .data import fetch

    return fetch.main(argv)


COMMANDS = {
    "paths": _paths,
    "fetch": _fetch,
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
