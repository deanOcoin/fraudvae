# fraudvae

Fraud screening for payment transactions, trained on a merged corpus of public
transaction datasets, with a web UI to score new transactions and browse the
data.

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
fraudvae prepare               # parquet -> model arrays
fraudvae train --config cpu_small
fraudvae serve                 # http://127.0.0.1:8765
```

`make test` runs the suite (synthetic data, no downloads).

Python 3.10+, PyTorch 2.1+. CPU is fine for a small run.

## Paths

Everything lives under the repo root unless overridden:

- `data/raw/<key>/` - downloads, one folder per source
- `data/processed/<key>.parquet` - the unified table per source
- `data/arrays/` - model-ready arrays written by `prepare`
- `runs/<name>/` - checkpoints, metrics and the config of each training run

Set `FRAUDVAE_DATA` and `FRAUDVAE_RUNS` to move them elsewhere.

## Layout

```
fraudvae/
  data/        registry (sources, licenses), schema, adapters/<source>.py,
               fetch / build / prepare
  model/       vae.py, scoring.py, dataset.py
  train.py  config.py  paths.py
  server/      fastapi app
  web/         the ui: index.html, css/, js/
configs/       base, cpu_small, smoke
tests/         pytest, no real data needed
```

## License

MIT for the code. The datasets keep their own licenses (listed per source in
`fraudvae/data/registry.py`).
