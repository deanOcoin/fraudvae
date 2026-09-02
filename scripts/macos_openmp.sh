#!/usr/bin/env bash
# macOS only. The PyTorch wheel ships its own copy of the LLVM OpenMP runtime
# (torch/lib/libomp.dylib) and the LightGBM wheel links the homebrew one
# (brew install libomp). Loading both into one process - which is what the
# server and `fraudvae train` do - crashes it the first time LightGBM runs.
#
# This points torch at the homebrew copy, so there is exactly one runtime.
# The original is kept next to it as libomp.dylib.orig; `--undo` puts it back.
# Run it inside the virtualenv you use for fraudvae (it asks python where torch is).
set -euo pipefail

PY=${PYTHON:-python3}
if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "not macOS, nothing to do"; exit 0
fi
TORCH_LIB=$($PY -c "import os, torch; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))")
TARGET="$TORCH_LIB/libomp.dylib"

if [[ "${1:-}" == "--undo" ]]; then
  if [[ -f "$TARGET.orig" ]]; then
    rm -f "$TARGET" && mv "$TARGET.orig" "$TARGET" && echo "restored $TARGET"
  else
    echo "nothing to undo ($TARGET.orig not found)"
  fi
  exit 0
fi

BREW_OMP="$(brew --prefix libomp 2>/dev/null || true)/lib/libomp.dylib"
if [[ ! -f "$BREW_OMP" ]]; then
  echo "homebrew libomp not found - run: brew install libomp" >&2; exit 1
fi
if [[ -L "$TARGET" && "$(readlink "$TARGET")" == "$BREW_OMP" ]]; then
  echo "already done: $TARGET -> $BREW_OMP"; exit 0
fi
if [[ -f "$TARGET" && ! -L "$TARGET" ]]; then
  mv "$TARGET" "$TARGET.orig"
fi
ln -sf "$BREW_OMP" "$TARGET"
echo "$TARGET -> $BREW_OMP"
echo "restart anything that had torch imported (the server, a training run)"
