#!/usr/bin/env bash
# What the app does when it provisions a fresh gpu box, as a script, for when
# you'd rather do it by hand (or the box has no python and you need to fix
# that first). Run on the server:
#   bash bootstrap_remote.sh [remote_dir]
set -euo pipefail

RD="${1:-$HOME/fraudvae}"
mkdir -p "$RD/data/processed" "$RD/runs"
cd "$RD"

PY=""
for cand in python3 python /opt/conda/bin/python; do
  if command -v "$cand" >/dev/null 2>&1 && "$cand" -c 'import torch; assert torch.cuda.is_available()' >/dev/null 2>&1; then
    PY="$(command -v "$cand")"; break
  fi
done
if [ -z "$PY" ]; then
  PY="$(command -v python3)"
  echo "no cuda torch found, installing (big download)"
  "$PY" -m pip install --quiet torch
fi
echo "using $PY"
"$PY" -c 'import torch; print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")'

if [ -f code.tar.gz ]; then
  rm -rf fraudvae configs scripts
  tar xzf code.tar.gz && rm code.tar.gz
fi
if [ -f requirements.txt ]; then
  grep -v -E '^(torch|pytest|asyncssh|#)' requirements.txt | grep -v '^$' > /tmp/req.txt
  "$PY" -m pip install --quiet -r /tmp/req.txt
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo "ready. upload data/processed/*.parquet then: $PY -m fraudvae.data.prepare --name full && $PY -m fraudvae.train --config gpu_full --prepared full"
