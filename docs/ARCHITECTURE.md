# Architecture notes

## Pipeline

```
raw downloads ──fetch──▶ data/raw/<source>/            (never in git)
   │
   └─build──▶ data/processed/<source>.parquet         unified schema, zstd, row groups of 500k
                 + stats.json (per-source distributions for the ui) + samples.parquet
   │
   └─prepare──▶ data/prepared/<name>/{split}_*.npy     standardised float32 numerics, uint8 masks,
                 + meta.json                            int16 categorical codes, source idx, label, row id
   │
   └─train──▶ runs/<name>/model.pt + metrics.jsonl + status.json + train.log      (the VAE)
   └─boost──▶ runs/<name>/booster.txt + booster.json                             (LightGBM)
   └─calibrate──▶ runs/<name>/calibration.json + latent_map.npz                  (blender + thresholds on val)
   └─evaluate──▶ runs/<name>/eval.json                                           (VAE / booster / blend on test)
   │
   └─serve──▶ fastapi + the static ui, loads the best compatible run by test (blend) auroc

`fraudvae train` runs the last four in sequence; each is also its own command,
and `fraudvae post RUN` redoes the last three on an existing run (a booster
re-tune is minutes, the VAE is the expensive half).
```

Adapters return a DataFrame, or a generator of DataFrames for the 24M-row
tabformer file, which is processed on compact numpy arrays and written in 2M
row chunks so the build fits in a few GB of RAM. Stats are computed from the
written parquet with a small column subset; coverage comes from parquet
null-count metadata.

`prepare` streams the parquet in 1M row batches into pre-sized memmaps, then
standardises in place with train-legit means/stds. `--max-rows-per-source`
caps *train* rows per source (legit subsampled, every fraud row kept) and never
touches val/test, so evaluation stays honest while a laptop can train.

The context features (`fraudvae/data/velocity.py`) are computed at build time
per source: sort by (customer, ts), build a key that is monotone within a
customer's block and jumps by more than any window between blocks, then one
`searchsorted` per window gives every row's window start. First-seen flags are
"first row of the (customer, merchant) pair in time order". The same is done
sorted by merchant for the merchant block. Customers are processed in groups
of ~3M rows so the 24M-row tabformer file peaks around 3 GB.

## Model

`fraudvae/model/vae.py`

- inputs: `num ⊙ mask`, `mask`, one embedding per categorical (index 0 =
  unknown is itself an embedding, so "we don't know the card brand" is a
  signal), source embedding (index 0 = generic)
- encoder MLP (LayerNorm + SiLU, dropout 0.05) → `μ, log σ²` of a diagonal
  Gaussian, `log σ²` clamped to [−10, 6]
- decoder MLP on `[z, source embedding]` → `(μ_f, log σ²_f)` per numeric
  (clamped to [−7, 4]) and logits per categorical
- loss per row: `Σ_f m_f · GaussNLL_f + Σ_g m_g · CE_g + β · KL`, averaged over
  the batch; β ramps linearly over the first `beta_warmup_epochs`; optional
  free bits; grad clipping at 5
- AdamW, linear warmup then cosine to 5% of the base lr; bf16 autocast on CUDA
- source dropout: 10% of rows get the generic token during training
- early stopping / checkpoint selection on mean per-source validation AUROC of
  the raw score, computed on a ≤300k-rows-per-source val subsample each epoch

## Booster

`fraudvae/model/booster.py`

LightGBM binary classifier on `[standardised numerics with NaN restored where
unobserved, categorical codes, source]` (the same arrays the VAE reads), fit on
the labelled train split: every fraud row plus up to `max_legit_per_source`
legit rows per source, legit rows weighted back to the base rate. A
`generic_fraction` (25%) slice is duplicated with source = 0 so the model has
a generic mode. Early stopping watches two validation sets - the (legit-subsampled)
val rows as they are and the same rows with the source blanked - and stops on
the mean of the two AUCs, so the generic mode is never quietly traded for
named-bank fit. `num_leaves` 127, lr 0.05, feature/bagging fraction 0.8, L2 1,
categorical smoothing. It never sees a VAE output, so stacking it with the VAE
later is clean (the VAE trained on the same legit rows; feeding its in-sample
surprises to the booster would shift the train/val distribution).

The receipt's "pushes" are `predict(pred_contrib=True)`: SHAP values in
log-odds, one per feature plus the bias.

A run whose `booster.txt` can't be opened (lightgbm not installed) still
serves as a novelty-only model, with the reason in `/api/meta` →
`booster_error`.

## Scoring, blending and calibration

`fraudvae/model/scoring.py`, `fraudvae/calibrate.py`

Two VAE scores from one forward pass with K posterior samples (K = 8 cpu / 16 gpu):

- `raw = Σ nll_f + KL` (Monte Carlo −ELBO over observed fields)
- `norm = Σ_f obs_f · (nll_f − μ_f^(s)) / σ_f^(s) / √n_obs + (KL − μ_KL^(s)) / σ_KL^(s)`
  with per-source baselines fitted on ≤100k legit validation rows (pass 1),
  plus a "generic" baseline from rows scored with the source token zeroed

Pass 2 takes a labelled val subsample (≤100k legit per source + every fraud
row, plus a copy scored with the source token blanked), picks the VAE score
type with the higher mean per-source AUROC, standardises it per source, and
fits the **blender**: one logistic regression (C = 0.5, balanced) on
`[booster logit, VAE score z, KL z, source one-hot, and each of the three ×
one-hot]`, the three signals clipped to fixed ranges (`Scorer.BLEND_CLIP`).
The interaction columns give each bank its own weighting of booster vs
novelty; the generic one-hot gets its own. Half the rows fit, the other half
report a held-out AUROC. The standardisation is folded into the stored
weights, so inference is one dot product on the raw features. The per-field
surprises and the latent mean are deliberately not inputs: unbounded and
time-drifting, they pulled the blend below both of its components on
tabformer's newest months (see TRAINING_NOTES); they remain the receipt's
explanation of the novelty score.

Pass 3 scores the whole validation split once (VAE, booster, blend) and
records per source and for the generic context, for each of the three
scores: legit quantiles, thresholds at FPR targets (0.1/0.5/1/2/5%) with
recall/precision, best-F1 threshold. A 2-d PCA of posterior means (fit on
legit rows) and a background sample (≤2500 legit + ≤1200 fraud per source)
go to `latent_map.npz` for the UI scatter.

`evaluate.py` applies the frozen validation thresholds to the test split and
reports, per source and for a generic subsample, realised FPR / recall /
precision alongside AUROC, AUPRC, recall@FPR for the VAE, the booster and
the blend. `mean_auroc` (what runs are ranked by) is the blend's.

Runs record the schema version they were trained on; the server refuses to
load one from another version and the UI greys it out.

## Server

`fraudvae/server/`

- `inference.py`: `ActiveModel` wraps a run (VAE checkpoint, booster, prepared
  meta for transforms/standardisation, calibration with the blender). `score()`
  encodes a raw dict (missing → unobserved), runs K samples, the booster and the
  blender, and returns three blocks: `blend` (p(fraud), percentile, flag at the
  chosen operating point - the verdict), `supervised` (booster probability, its
  own flag, SHAP pushes with observed values) and `novelty` (both VAE scores,
  z, percentile, flag, thresholds), plus the per-field surprises with decoder
  expectations in original units and the latent 2-d point. `score_frame()` is
  the vectorised version for CSV uploads.
- `api_data.py`: schema, dataset registry + stats, held-out examples, fetch /
  build / prepare as subprocess tasks
- `api_model.py`: score, batch score, runs, metrics, latent map, local training
  as a subprocess task, task polling
- `api_remote.py`: ssh key, remembered hosts (+ `~/.ssh/config` autofill),
  probe, provision (thread task), launch, status, stop, fetch
- `tasks.py`: tiny in-memory task registry with log buffers; the UI polls

The UI (`fraudvae/web/`) is plain ES modules, Chart.js and KaTeX vendored so it
runs on an offline box. One tab per concern; `explain.html` is the long-form
"how it works".

## Remote training

`fraudvae/remote/` — see `docs/REMOTE_TRAINING.md`. paramiko sessions are
opened per operation; host keys are trust-on-first-use into
`~/.fraudvae/ssh/known_hosts`; the app's own ed25519 key is generated with
`cryptography` and never leaves the machine.
