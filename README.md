# fraudvae

Fraud screening for payment transactions, trained on a merged corpus of public
transaction datasets, with a web UI to score new transactions, browse the data,
train locally, or push a training run to a rented GPU over ssh.

The model is a **variational autoencoder** trained on legitimate traffic only.
It scores a transaction by how badly it reconstructs it, field by field, so
every score comes with a receipt of *which* fields looked unusual and what the
model expected instead. It never sees a fraud label.

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

fraudvae fetch                 # clones the github hosted sources
fraudvae build                 # raw -> data/processed/<source>.parquet
fraudvae prepare --max-rows-per-source 1500000     # parquet -> model arrays, train capped per source
fraudvae train --config cpu_small
fraudvae serve                 # http://127.0.0.1:8765
```

`make data build prepare train serve` does the same. `make test` runs the suite
(synthetic data, a 2 epoch model, an in-process ssh server).

Python 3.10+, PyTorch 2.1+. CPU is fine for the capped set, CUDA for the full
corpus.

## Paths

Everything lives under the repo root unless overridden:

- `data/raw/<key>/` - downloads, one folder per source
- `data/processed/<key>.parquet` - the unified table per source
- `data/arrays/` - model-ready arrays written by `prepare`
- `runs/<name>/` - checkpoints, metrics and the config of each training run

Set `FRAUDVAE_DATA` and `FRAUDVAE_RUNS` to move them elsewhere. `prepare`
estimates the size of the arrays it is about to write and checks the free disk
under `data/` first; if there isn't room it stops before writing anything and
says how much it needs (lower `--max-rows-per-source` or move `FRAUDVAE_DATA`).

## Remote training

The full corpus is far more than a laptop wants to chew on; a rented GPU does it
in a few hours. The *Compute* tab benchmarks the real model on your machine,
estimates the run, and if it's too long walks through renting a box:

1. the app generates its own ed25519 key (`~/.fraudvae/ssh/`); add the public
   key to the server (vast.ai: Account → SSH keys, then rent with the PyTorch
   template and direct ssh on)
2. paste the instance's `ssh -p PORT root@IP` line, *Connect & inspect* shows
   GPU/RAM/disk/torch
3. *Provision* checks the box has the disk for it, uploads code + processed
   parquet, installs deps, runs `prepare` on the box
4. *Launch* starts `fraudvae train` under nohup; *Refresh status* tails the
   log and per-epoch metrics; *Fetch* pulls the run back and loads it

Everything in that flow is also plain CLI: `fraudvae key`, `fraudvae advise`,
`fraudvae remote run --host IP --port PORT` (provision + launch + poll + fetch
in one command), `scripts/bootstrap_remote.sh`. See `docs/REMOTE_TRAINING.md`.

## Layout

```
fraudvae/
  data/        registry (sources, licenses), schema, adapters/<source>.py,
               fetch / build / prepare
  model/       vae.py, scoring.py, dataset.py
  train.py  config.py  paths.py
  remote/      keys, ssh (paramiko), provision, jobs, advisor
  server/      fastapi app
  web/         the ui: index.html, css/, js/
configs/       base, cpu_small, gpu_full, smoke
tests/         pytest, no real data needed
docs/
```

## License

MIT for the code. The datasets keep their own licenses (listed per source in
`fraudvae/data/registry.py`).
