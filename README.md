# fraudvae

Fraud screening for payment transactions, trained on a merged corpus of nine
public transaction datasets (~42M rows), with a web UI to score new
transactions, browse the data, train locally, or push a training run to a
rented GPU over ssh.

Two models look at every transaction and a small per-bank blender combines
them into one probability:

- a **variational autoencoder** trained on legitimate traffic only. It scores a
  transaction by how badly it reconstructs it, field by field, so every score
  comes with a receipt of *which* fields looked unusual and what the model
  expected instead. This is the novelty detector: it never saw a fraud label.
- a **gradient-boosted booster** (LightGBM) trained on the labelled fraud, over
  the same fields plus a block of causal customer / merchant context (bursts,
  the customer's own amount z-score, tenure, first-time-at-this-merchant...).
  It finds the fraud that looks ordinary, and explains itself with per-feature
  SHAP pushes.
- the **blend**: a logistic layer fit on labelled validation rows with a
  weight per bank for each model, which is what the app flags on. Every bank
  also has a "generic" mode for transactions from an institution the model
  never trained on.

![score tab](docs/img/score.png)

The other tabs: [data explorer](docs/img/data.png), [model & runs](docs/img/model.png),
[compute / remote training](docs/img/compute.png), [how it works](docs/img/explain.png).

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

fraudvae fetch                 # clones the github hosted sources (fdh, capone, tabformer)
#   kaggle ones: see scripts/kaggle_manual_download.md, drop the zips in data/raw/<key>/
fraudvae build                 # raw -> data/processed/<source>.parquet  (unified schema)
fraudvae prepare --max-rows-per-source 1500000     # parquet -> model arrays, train capped per source
fraudvae train --config cpu_small                   # VAE + booster + calibration + eval, ~1h on a laptop cpu
fraudvae serve                 # http://127.0.0.1:8765
```

`make data build prepare train serve` does the same. `make test` runs the suite
(synthetic data, a 2 epoch model, the API, an in-process ssh server).

Python 3.10+, PyTorch 2.1+, LightGBM 4+. CPU is fine for the capped set, MPS
works on Apple silicon, CUDA for the full corpus.

**macOS:** LightGBM needs `brew install libomp`, and then `make fix-omp` once
per virtualenv. The PyTorch wheel carries its own copy of the OpenMP runtime
and LightGBM links the homebrew one; two copies in one process segfault it the
first time the booster runs. The script points torch at the homebrew copy
(reversible with `scripts/macos_openmp.sh --undo`). If you skip it the app
still starts, novelty-only, and the sidebar tells you why.

## What's in the corpus

| key | dataset | rows | fraud | access |
|---|---|---:|---:|---|
| fdh | Fraud Detection Handbook simulated card transactions | 1.75M | 0.84% | github |
| capone | Capital One synthetic card authorisations | 786k | 1.58% | github |
| tabformer | IBM TabFormer synthetic credit card transactions | 24.4M | 0.12% | github (lfs) / kaggle |
| ulb | ULB / Worldline real European card transactions | 285k | 0.17% | kaggle |
| paysim | PaySim mobile money simulation | 6.36M | 0.13% | kaggle |
| sparkov | Sparkov simulated US card purchases | 1.85M | 0.58% | kaggle |
| banksim | BankSim agent based bank payments | 595k | 1.21% | kaggle |
| ieee | IEEE-CIS / Vesta real e-commerce transactions | 590k | 3.5% | kaggle |
| amlworld | IBM AMLworld synthetic inter-bank transfers (HI-Small), label = laundering | 5.08M | 0.10% | kaggle |

Every source is mapped onto one schema (`fraudvae/data/schema.py`): numeric
fields (amount, hour, the customer's history in the last hour / day / week and
their own amount z-score, tenure, the merchant's volume and typical amount,
balances, limits, distances), categoricals (channel, transaction type, an
18-way merchant category taxonomy, card brand/kind, auth errors, first-time
flags, payment format...), plus a source id and a customer key that never
enters a model. Fields a source can't fill are *unobserved* and masked out of
the likelihood, never imputed. All context features are computed causally per
source. Splits are by time within each source: oldest 70% train, 10%
validation (calibration), newest 20% test. Details, licenses and per-source
quirks: `docs/DATA_SOURCES.md` (generated from the registry), `docs/SCHEMA.md`.

## The models

**Novelty (VAE).** A heterogeneous-likelihood VAE (HI-VAE style): MLP encoder
over `[standardised numerics, mask bits, embedded categoricals, embedded
source]` into a 16-24-d Gaussian posterior, mirrored decoder with a Gaussian
head per numeric field and a softmax head per categorical. Trained on
legitimate rows only with β warm-up and source-token dropout (so a "generic"
bank context exists). Scoring is a Monte Carlo −ELBO plus a per-field
normalised variant; calibration picks the better one on validation.

**Supervised (booster).** A LightGBM classifier on the same fields (NaN where
unobserved) plus the source token, trained on the labelled train split with
legit rows subsampled per source and re-weighted (capped, so no bank outweighs
the others), early-stopped on validation AUC, with a 25% source-blanked slice
for a generic mode. It never sees a VAE output, which keeps the stacking
honest. If LightGBM can't be loaded the run is saved novelty-only and the app
scores with the VAE alone.

**Blend.** One logistic regression on bounded signals - booster logit,
VAE score z, KL z - with a source one-hot and the signals × one-hot, fit on
labelled validation rows. The interaction terms give every bank its
own booster-vs-novelty weighting. Calibration then scores the whole
validation split once and records, per bank and for the generic context,
operating thresholds at 0.1/0.5/1/2/5% false positive rate for every
score. The app flags on the blend; the receipt shows them all.

`docs/ARCHITECTURE.md` has the longer version, the UI's "How it works" tab
the friendly one.

## Results

Held-out test split: the newest 20% of every source, thresholds frozen on
validation. `gpu_full-4090-v4` is the full corpus trained on a rented RTX 4090.
`runs/` is not in git - train your own with the commands above or pull a run
back from a GPU box through the UI.

| source | **blend** | r@1% FPR |
|---|---|---|
| fdh | **0.672** | 34% |
| capone | **0.765** | 8% |
| tabformer | **0.981** | 73% |
| ulb | **0.982** | 84% |
| paysim | **1.000** | 100% |
| sparkov | **0.999** | 99% |
| banksim | **0.999** | 97% |
| ieee | **0.858** | 35% |
| amlworld | **0.990** | 84% |
| **mean (9)** | **0.916** | |
| generic (bank unknown) | **0.956** | 66% |

AUROC; r@1% FPR is the share of fraud caught at a 1% false-positive budget.
The simulators (PaySim, Sparkov, BankSim) saturate because their fraud rules
are learnable; fdh and Capital One are the hard ones, their fraud looks like
the customer's ordinary traffic.

## Remote training

The full corpus is ~29M legitimate rows per epoch; a laptop does it in days,
an RTX 4090 in a few minutes per epoch. The *Compute* tab benchmarks the real model on your machine,
estimates the run, and if it's too long walks through renting a box:

1. the app generates its own ed25519 key (`~/.fraudvae/ssh/`); add the public
   key to the server (vast.ai: Account → SSH keys, then rent with the PyTorch
   template and direct ssh on)
2. paste the instance's `ssh -p PORT root@IP` line, *Connect & inspect* shows
   GPU/RAM/disk/torch
3. *Provision* uploads code + processed parquet, installs deps, runs `prepare`
   on the box
4. *Launch* starts `fraudvae train` under nohup; *Refresh status* tails the
   log and per-epoch metrics; *Fetch* pulls the run back and loads it

Everything in that flow is also plain CLI: `fraudvae key`, `fraudvae advise`,
`fraudvae remote run --host IP --port PORT` (provision + launch + poll + fetch
in one command), `scripts/bootstrap_remote.sh`. See `docs/REMOTE_TRAINING.md`.

## Layout

```
fraudvae/
  data/        registry (sources, licenses), schema, taxonomy, velocity features,
               adapters/<source>.py, fetch / build / prepare, docs generator
  model/       vae.py, scoring.py, booster.py (lazy lightgbm import), dataset.py
  train.py  calibrate.py  evaluate.py  config.py
  remote/      keys, ssh (paramiko), provision, jobs, advisor
  server/      fastapi app + api modules, inference (active model, novelty-only fallback)
  web/         the ui: index.html, css/, js/, explain.html, vendored libs
configs/       base, cpu_small, gpu_full, smoke
scripts/       macos_openmp.sh, bootstrap_remote.sh, kaggle_manual_download.md
tests/         pytest, no real data needed
docs/
```

## License

MIT for the code. The datasets keep their own licenses (listed per source in
`fraudvae/data/registry.py` and `docs/DATA_SOURCES.md`); IEEE-CIS in particular
is non-commercial.
