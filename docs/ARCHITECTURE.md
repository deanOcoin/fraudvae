# Architecture notes

## Pipeline

```
raw downloads ──fetch──▶ data/raw/<source>/            (never in git)
   │
   └─build──▶ data/processed/<source>.parquet         unified schema, zstd
                 + stats.json (per-source distributions for the ui)
   │
   └─prepare──▶ data/prepared/<name>/{split}_*.npy     standardised float32 numerics, uint8 masks,
                 + meta.json                            int16 categorical codes, source idx, label
   │
   └─train──▶ runs/<name>/model.pt + metrics.jsonl + status.json + train.log
   └─calibrate──▶ runs/<name>/calibration.json + latent_map.npz                  (thresholds on val)
   └─evaluate──▶ runs/<name>/eval.json                                           (test split)
   │
   └─serve──▶ fastapi + the static ui, loads the newest compatible run
```

Adapters return a DataFrame, or a generator of DataFrames for the 24M-row
tabformer file, which is written in 2M row chunks so the build fits in a few
GB of RAM. Stats are computed per source on a 1M row sample of the written
parquet.

`prepare` reads one source at a time, concatenates, splits by time and
standardises with train-legit means/stds. `--max-rows-per-source` caps rows
per source (legit subsampled, every fraud row kept) so a laptop can train.

## Model

`fraudvae/model/vae.py`

- inputs: `num ⊙ mask`, `mask`, one embedding per categorical (index 0 =
  unknown is itself an embedding, so "we don't know the card brand" is a
  signal), source embedding (index 0 = generic)
- encoder MLP (LayerNorm + SiLU) → `μ, log σ²` of a diagonal Gaussian
- decoder MLP on `[z, source embedding]` → `(μ_f, log σ²_f)` per numeric
  and logits per categorical
- loss per row: `Σ_f m_f · GaussNLL_f + Σ_g m_g · CE_g + β · KL`, averaged over
  the batch; β ramps linearly over the first `beta_warmup_epochs`
- Adam, cosine schedule; grad clipping at 5
- early stopping / checkpoint selection on validation loss of the legit rows

## Scoring and calibration

`fraudvae/model/scoring.py`, `fraudvae/calibrate.py`

Two scores from one forward pass with K = 8 posterior samples:

- `raw = Σ nll_f + KL` (Monte Carlo −ELBO over observed fields)
- `norm = Σ_f obs_f · (nll_f − μ_f^(s)) / σ_f^(s) / √n_obs + (KL − μ_KL^(s)) / σ_KL^(s)`
  with per-source baselines fitted on ≤100k legit validation rows (pass 1),
  plus a "generic" baseline from rows scored with the source token zeroed

Pass 2 scores the whole validation split and records per source and for the
generic context: legit quantiles, thresholds at FPR targets (0.1/0.5/1/2/5%)
with recall/precision, best-F1 threshold. A 2-d PCA of posterior means (fit
on legit rows) and a background sample go to `latent_map.npz` for the UI
scatter.

`evaluate.py` applies the frozen validation thresholds to the test split and
reports, per source, realised FPR / recall / precision alongside AUROC,
AUPRC and recall@FPR for both scores.

Runs record the schema version they were trained on; the server refuses to
load one from another version.

## Server

`fraudvae/server/`

- `inference.py`: `ActiveModel` wraps a run (VAE checkpoint, prepared meta
  for transforms/standardisation, calibration). `score()` encodes a raw dict
  (missing → unobserved), runs K samples and returns both scores, z,
  percentile, flag and thresholds, plus the per-field surprises and the
  latent 2-d point. `score_frame()` is the vectorised version for CSV
  uploads.
- `api_data.py`: schema, dataset registry + stats, fetch / build / prepare
  as subprocess tasks
- `api_model.py`: score, batch score, runs, metrics, latent map, local training
  as a subprocess task, task polling
- `api_remote.py`: ssh key, remembered hosts, probe, launch, status, stop,
  fetch
- `tasks.py`: tiny in-memory task registry with log buffers; the UI polls

The UI (`fraudvae/web/`) is plain ES modules with Chart.js vendored so it
runs on an offline box. One tab per concern.

## Remote training

`fraudvae/remote/` — see `docs/REMOTE_TRAINING.md`. paramiko sessions are
opened per operation; host keys are trust-on-first-use into
`~/.fraudvae/ssh/known_hosts`; the app's own ed25519 key is generated with
`cryptography` and never leaves the machine.
