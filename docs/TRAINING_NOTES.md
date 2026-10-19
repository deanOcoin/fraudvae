# Training notes

Running log of what was tried, what it did, and the numbers for the runs that
ship. Newest at the bottom. Metrics are AUROC on the held-out (newest 20%)
test split of each source unless stated otherwise; "r@1%" is recall at 1% false
positive rate on the same split, with the threshold frozen from validation.

## Lessons from the first runs (smoke + quality probes, fdh + capone, 200k train rows / source)

- The raw −ELBO is dominated by the categorical hour/dow terms and by whatever
  numeric the decoder is least sure about. On fdh the amount term alone is a
  better detector than the sum. Hence the per-field normalised score; on the
  probe it was within ±0.01 AUROC of raw on these two sources, but the field
  level z's are what make the receipt readable either way. Calibration now
  picks whichever wins on validation.
- β warm-up matters: without it the first epoch collapses to KL≈0 and the
  decoder ignores z (train nll flat, val auroc 0.5). 3–4 epochs of ramp fixed
  it.
- Capital One is hard for anything unsupervised (~0.54 on the probe). The
  fraud there is mostly ordinary-looking purchases; the only strong signal is
  the CVV mismatch which is rare (0.9% of rows) and not fraud-exclusive.
- fdh scenario 3 (compromised card, amounts ×5) is caught by
  `amount_over_cust_mean`; scenario 1 (amount > 220) by `amount`; scenario 2
  (compromised terminal) is invisible without terminal features, which is
  by design here (unsupervised, no terminal risk scores).
- 24M tabformer rows blew the 6 GB build container twice before the adapter
  moved to compact arrays + chunked parquet writes. pandas' `to_datetime` on a
  24M row frame of int64s was the single worst offender.
- pandas 3 parses datetimes to microsecond resolution; `astype(int64)//1e9`
  silently gave timestamps 1000× too small for capone until the velocity
  features looked absurd. Now converted through `datetime64[s]`.

## v4: context features, the booster and the blend

Everything below "Runs" up to the v4 section is from the v3 schema and the
VAE alone. What changed and why, in order of how much it mattered:

- The two v3 runs made the case: 7× the rows and 3× the parameters moved the
  mean AUROC by 0.004, and the validation curve was flat from epoch 9. More of
  the same was not going to help. The ceiling was (a) no labels and (b) no
  customer context beyond the four velocity windows.
- **Customer / merchant context** (schema v4): 1h burst window, the customer's
  own amount z-score over 30 days, tenure (prior count, days since first
  seen), first-time flags (merchant, category, cross-border), merchant volume
  and amount-vs-merchant-mean. All causal, all computed at build time. This
  is where fraud actually lives: a strange payment *for this customer*.
- **The booster**: LightGBM on the observable fields + context + bank id,
  trained on the labelled train split (legit subsampled per source and
  re-weighted, every fraud row kept, a source-blanked slice for the generic
  mode: 15% in the baseline run below, 25% since the GPU run), early stopped on
  the mean of the named-bank and generic val AUCs. It never sees a VAE output,
  so stacking stays honest.
- **The blend** replaces the old logistic "head": one logistic regression on
  [booster logit, VAE score z, KL z, bank one-hot and the three signals ×
  one-hot], fit on half of a labelled val subsample, checked on the other
  half, thresholds from the full val split. The interaction columns are the
  point: each bank gets its own booster vs novelty weighting without separate
  models.
- **A second lesson, from the first v4 baseline.** The blender originally
  also took the 81 per-field surprise z's and the latent mean. Validation
  loved it (tabformer blend 0.996) and the test split hated it (0.839, below
  both the booster's 0.979 and the VAE's 0.901). The surprises are unbounded
  and the newest 20% of a 30-year file sits in a different regime: on test
  fraud rows `amount_over_limit` reached z≈1000, and with a slightly negative
  pooled weight that one column outvoted the booster. Diagnosed by scoring
  the blend block by block (the field-z block alone: AUROC 0.60 on val, 0.02
  on test). Now the blender sees only the three calibrated signals, each
  clipped, and the surprises stay on the receipt as explanation.
- **A bug, found on the way.** The old head's generic mode scored 0.50 and I
  had explained it as "no baseline without a bank one-hot". The real cause:
  `PreparedSplit.batch()` sorted the row index for the memmap read and
  returned rows in sorted order, while the caller had passed a *shuffled*
  index (`rng.choice` without a sort) and matched labels to the shuffled
  order. Every generic-mode head feature was paired with the wrong label.
  `batch()` now un-permutes after the read, so any index order is safe.
  Nothing else was affected (every other caller passed sorted indices).
- **amlworld** (IBM AMLworld HI-Small, 5.1M inter-bank transfers, label =
  laundering) joined as a ninth source for diversity in the generic mode.
  Its per-row signal is weak by construction (laundering legs look ordinary);
  it is there so "a bank the model never saw" has seen more than one kind of
  institution.

## Runs

### cpu_small-8src (baseline, 2 cpu cores, capped corpus)

`fraudvae prepare --max-rows-per-source 500000` → 3.46M legit train rows (full val/test),
`cpu_small` config (latent 16, 384-384-192, 0.64M params), 12 epochs planned, early
stopped after 10 (best epoch 6), ~3.2 min/epoch at 16k rows/s. Calibration picked the
normalised score (val mean AUROC 0.8225 vs 0.8214 raw). Generic context val AUROC 0.761.

| source | test AUROC | AUPRC | r@1% FPR | head AUROC |
|---|---|---|---|---|
| fdh | 0.657 | 0.084 | 19.7% | 0.663 |
| capone | 0.516 | 0.016 | 1.0% | 0.604 |
| tabformer | 0.906 | 0.008 | 11.9% | 0.914 |
| ulb | 0.950 | 0.281 | 80.0% | 0.924 |
| paysim | 0.937 | 0.181 | 68.8% | 0.949 |
| sparkov | 0.939 | 0.251 | 58.0% | 0.965 |
| banksim | 0.977 | 0.485 | 61.9% | 0.990 |
| ieee | 0.751 | 0.094 | 0.1% | 0.745 |
| **mean** | **0.829** | | | **0.844** |

Reads as expected: the simulators with structural fraud (BankSim, PaySim, Sparkov) and
ULB's PCA space are easy for a density model; the two e-commerce / card-not-present
real-ish sets (IEEE, TabFormer online rows) are middling; FDH and Capital One are the
"ordinary looking fraud" cases the limitations section warns about. The supervised head
barely helps where the VAE is already strong and only lifts Capital One (0.52 → 0.60),
so the surprises the VAE produces really are most of the usable signal. The head's
*generic* mode is useless (0.50) — with the source one-hot zeroed it has no baseline to
compare against, so the UI only trusts it in a named context.

### gpu_full-4090 (full corpus, rented RTX 4090 via the Compute tab)

Whole flow driven from the UI on a laptop: probe → provision (code tarball + 1.0 GB of
processed parquet at ~3.8 MB/s, then `prepare` on the box in 71 s: 25,628,695 train /
3,661,242 val / 7,322,485 test rows) → launch `gpu_full` (latent 24, 768-768-384,
2.19M params, batch 8192, AMP, 16 score samples) → fetch. Torch 2.14.1+cu132.

~516k rows/s, 0.8 min/epoch. 30 epochs planned, stopped at 23 (patience 6), best epoch
17 at val mean AUROC 0.8337. Scoring the 7.3M test rows took 79 s. 21.7 min end to end;
the instance cost less than the coffee. Calibration picked the **raw** −ELBO this time
(val 0.8337 vs 0.8285 norm) - with 16 samples the raw score is far less noisy than it was
on the CPU run with 4, and the per-field normalisation stops paying for itself. Generic
context val AUROC 0.736.

| source | test AUROC | AUPRC | r@1% FPR | head AUROC | Δ vs cpu_small |
|---|---|---|---|---|---|
| fdh | 0.659 | 0.128 | 20.6% | 0.666 | +0.002 |
| capone | 0.557 | 0.017 | 1.0% | 0.603 | +0.041 |
| tabformer | 0.898 | 0.008 | 11.6% | 0.648 | −0.008 |
| ulb | 0.931 | 0.185 | 78.7% | 0.976 | −0.019 |
| paysim | 0.944 | 0.183 | 66.0% | 0.963 | +0.007 |
| sparkov | 0.955 | 0.330 | 67.5% | 0.969 | +0.016 |
| banksim | 0.976 | 0.425 | 49.5% | 0.990 | −0.001 |
| ieee | 0.746 | 0.099 | 0.1% | 0.767 | −0.005 |
| **mean** | **0.833** | | | 0.823 | +0.004 |

Things worth saying out loud:

- 7× the training data and 3.4× the parameters moved the mean by 0.004. The curve
  flattened by epoch 9 and the last 14 epochs were noise around 0.832. Whatever is left
  on fdh / capone / ieee is not going to come from a wider decoder; it needs features
  the public data doesn't carry (device, terminal, account age).
- ULB *dropped* (0.950 → 0.931). The ULB test split has 75 fraud rows, so one or two
  rows reorder and the AUROC swings by 0.02; the 0.1/0.5% FPR recalls are identical to
  the CPU run. Don't read anything into it.
- TabFormer's head AUROC fell off a cliff (0.914 → 0.648) while the VAE score held.
  The head is a tiny logistic model on z-scores; on 4.9M test rows with 0.1% fraud it
  is now mostly fitting noise in the latent means. Left as is - the UI only shows the
  head as a second opinion and the receipt says which one it is quoting.
- The trained model is 8.8 MB. It scores ~90k rows/s on the laptop CPU, which is more
  than enough for the UI and would be enough for a modest card issuer's live stream.

### cpu_small-9src-v4 (v4 baseline, 2 cpu cores, capped corpus, 9 sources)

`fraudvae prepare --max-rows-per-source 500000` → 4.03M train rows (3.96M legit), full
val 4.17M / test 8.34M. `cpu_small` (latent 16, 384-384-192, 0.66M params): 9 epochs at
3.7 min each, early stopped (best epoch 5, val mean VAE AUROC 0.807, norm score chosen).
Booster: 1.42M labelled rows (74k fraud), 794 trees in 141 s, pooled val AUC 0.961.
Blender held-out half 0.964. Calibration + evaluation ~25 min on cpu (the VAE scoring of
12.5M rows is the slow part, not the booster).

| source | VAE | booster | blend | blend AUPRC | r@1% FPR | r@5% FPR |
|---|---|---|---|---|---|---|
| fdh | 0.652 | 0.679 | **0.675** | 0.341 | 34.1% | 38.1% |
| capone | 0.502 | 0.765 | **0.765** | 0.052 | 6.8% | 24.9% |
| tabformer | 0.907 | 0.980 | **0.978** | 0.233 | 71.2% | 91.4% |
| ulb | 0.956 | 0.985 | **0.954** | 0.716 | 82.7% | 85.3% |
| paysim | 0.936 | 1.000 | **1.000** | 0.957 | 100.0% | 100.0% |
| sparkov | 0.925 | 0.999 | **0.999** | 0.962 | 98.7% | 99.6% |
| banksim | 0.986 | 0.999 | **0.999** | 0.934 | 96.2% | 100.0% |
| ieee | 0.738 | 0.863 | **0.861** | 0.405 | 36.5% | 52.3% |
| amlworld | 0.863 | 0.990 | **0.989** | 0.625 | 83.6% | 91.8% |
| **mean** | **0.829** | **0.918** | **0.913** | | | |
| generic (bank unknown) | 0.691 | 0.957 | **0.957** | 0.748 | 66.0% | 77.9% |

Reading it against the v3 runs:

- The VAE alone is exactly where it was (0.829 vs 0.829/0.833): the context features
  did not move the density model, which says the per-field Gaussian surprise was never the
  bottleneck for it.
- The booster is the step change, and it is the context features that make it one: the
  top gain features are the customer amount z-score, the 1h / 24h bursts, amount over the
  customer's mean, and first-time-at-merchant. IEEE 0.75 → 0.86, TabFormer 0.91 → 0.98,
  Capital One 0.52 → 0.77, and recall at a 1% alert budget goes from "useless" to usable
  on the real-ish sources (IEEE 0.1% → 37%, TabFormer 12% → 71%).
- The simulators saturate (PaySim / Sparkov / BankSim ≥ 0.999). That is the simulators'
  fraud rules being learnable, not a claim about real fraud.
- fdh stays at 0.68 for both models: a third of its fraud is "compromised terminal",
  which has no per-transaction signal without a terminal-level fraud history (a
  label-derived feature this project deliberately does not build).
- Generic (bank blanked): 0.74 → 0.957. This is the number for "a transaction from a
  bank we never trained on", and it is now within 0.04 of the named-bank mean.
- ULB blend 0.954 vs booster 0.985: 75 fraud rows in test, ~60 in val, so its per-bank
  blend weights are barely determined and the difference is inside the noise. Left alone.

### gpu_full-4090-v4 (the run that ships: full corpus, rented RTX 4090, v4 schema)

Same flow as the v3 GPU run, from the Compute tab: provision (code + 1.4 GB of parquet,
`prepare --name full` on the box: 29,183,537 train / 4,169,076 val / 8,338,154 test rows,
9 sources) → launch `gpu_full` (latent 24, 768-768-384, 2.25M params, batch 8192, AMP,
16 score samples) → fetch. 434k rows/s, 1.1 min per epoch. 30 epochs planned, stopped at
16 (patience 6), best epoch 10 at val mean VAE AUROC 0.810; the curve was flat from
epoch 7 on, as in v3. Calibration picked the normalised score (val 0.826 vs 0.807 raw).
The VAE never changed after that; everything below is the `post` stage (booster +
calibration + evaluation on the frozen VAE, 10 minutes on the box), which exists because
the booster took three passes to get right.

**Pass 1 - the config as first written** (600k legit rows per source, lr 0.08): 4.6M
labelled rows, 604 trees in 23 min on the box's CPUs, pooled val AUC 0.960 - and on the
newest test months it did *worse* than the 2-core baseline's booster (tabformer 0.944 vs
0.980, ieee 0.848 vs 0.863, mean 0.909 vs 0.918). Three times the legit rows per bank bought
a sharper fit of time-local patterns (`cust_gender` had become the third feature by gain),
not generalisation. 200k rows per source and lr 0.05 fixed the named banks.

**Pass 2 - same rows, worse generic mode.** With 200k rows/source the box booster matched
the sandbox one on every named bank (0.916 vs 0.918 mean) but not in the source-blanked
generic mode: 0.936 against 0.955, *on identical test rows*. It was not the number of
trees (the 1000-tree box model scored 0.935 when cut to its first 575) and it was not the
sample (both are uniform draws of 200k legit rows per source). It was the re-weighting.
Subsampled legit rows were weighted back to each bank's true base rate, and on the full
corpus that weight is 85 for tabformer (17M legit train rows behind 200k) against 1 for
ulb - so in the blanked mode one bank's traffic *was* the definition of normal. On the
500k-rows-per-source sandbox set the same formula gives at most 2.4, which is why the
sandbox booster never had the problem. The booster is a ranker; the blender and the
per-bank thresholds do the calibrating. `legit_weight_cap: 3` now bounds the weight, and
the box's validation curve became the sandbox's (0.957 at round 50 instead of 0.943).

**Pass 3 - what ships.** Capped weights, early stopping on the mean of two validation
AUCs (the val rows as they are, and the same rows source-blanked, so the generic mode is
a first-class stopping criterion instead of a 15% minority of a pooled set), and the
source-blanked training slice raised from 15% to 25% (picked on validation: mean per-bank
val AUROC 0.922 vs 0.920, generic 0.9633 vs 0.9627; on test it is a wash, 0.916 vs 0.918
mean, generic identical). 1.87M labelled rows, stopped at 555 trees in 4 min, blender
held-out half 0.964.

| source | VAE | booster | blend | blend AUPRC | r@1% FPR | r@5% FPR |
|---|---|---|---|---|---|---|
| fdh | 0.654 | 0.672 | **0.672** | 0.342 | 34.2% | 37.5% |
| capone | 0.508 | 0.765 | **0.765** | 0.052 | 7.6% | 24.6% |
| tabformer | 0.941 | 0.980 | **0.981** | 0.275 | 73.3% | 92.1% |
| ulb | 0.947 | 0.987 | **0.982** | 0.785 | 84.0% | 90.7% |
| paysim | 0.937 | 1.000 | **1.000** | 0.958 | 99.8% | 100.0% |
| sparkov | 0.911 | 0.999 | **0.999** | 0.965 | 99.0% | 99.6% |
| banksim | 0.985 | 0.999 | **0.999** | 0.944 | 97.1% | 99.9% |
| ieee | 0.727 | 0.862 | **0.858** | 0.406 | 35.1% | 52.6% |
| amlworld | 0.824 | 0.990 | **0.990** | 0.649 | 84.5% | 92.7% |
| **mean** | **0.826** | **0.917** | **0.916** | | | |
| generic (bank unknown) | 0.712 | 0.957 | **0.956** | 0.749 | 66.0% | 78.1% |

What to take from it, against the capped-corpus v4 baseline above:

- The VAE half is the same model it was on 4M rows (0.826 vs 0.829 mean), source by
  source. For the third run in a row, 7× the legit rows did not change what the density
  model sees; its ceiling here is the feature set, not the fit. Where it earns its place
  is the blend on tabformer (above the booster alone) and in the receipt.
- The booster is where the labels pay, and it is *not* where the row count pays: its
  training set is 200k legit rows per bank plus every fraud row whichever prepared set
  it is drawn from, so the full-corpus booster lands within noise of the 2-core one. The
  hybrid's quality was set by the context features and the labels, and a laptop gets the
  same model; the GPU bought the VAE epochs and 16 scoring samples, nothing else.
- At a 1% false-positive budget, with thresholds frozen on validation, the blend catches
  73% of tabformer's fraud, 35% of IEEE's, 84% of ULB's, 85% of the
  laundering legs in amlworld and 66% in the generic mode. fdh and capone stay where
  every model has left them (0.67 / 0.77): their fraud is ordinary-looking by
  construction and the public columns do not carry the terminal / device / account-age
  signal that would separate it.
- The blend tracks the booster within a few thousandths everywhere, beats it where the
  VAE adds information (tabformer) and trails it slightly on ulb (75 test fraud rows,
  ~60 in val; its per-bank weights are barely determined). It is still what the app flags
  on: it is the one score with a calibrated probability.
- Generic mode (bank blanked) at 0.956: the number for a transaction from an
  institution the model never saw. It sits above the named-bank *mean* only because fdh
  and capone drag that mean down; bank by bank it costs 0.01-0.03 against the named
  score, the price of not knowing whose "normal" to compare against. It was 0.939 one
  pass earlier - the weighting story above - and 0.74 in v3.

Three things the run taught about the plumbing, all fixed in this version: LightGBM with
`num_threads=0` on a 256-thread rented box ran 5× slower than on a 2-core sandbox (one
thread per hardware thread fighting over 2M rows; the booster now caps itself at the
cores it is actually allowed, 32 at most - the same booster went from 21 min to 4);
`fraudvae post` on a `cuda` config died in `torch.load` on a machine without a GPU
(device choice now falls back); and on macOS the PyTorch and LightGBM wheels bring two
OpenMP runtimes that segfault the server on the first booster call (`make fix-omp`, see
the README; the server now logs to `~/.fraudvae/server.log` with `faulthandler` armed, and
refuses the booster with a message instead of crashing when it sees the two runtimes).

## v5 development runs (2-core sandbox, capped sets)

The v5 work was built against capped prepared sets on the sandbox while the box was
being rented; these runs size the pieces, they are not the numbers that ship.

### IEEE: the wide block and the graph block (booster only, `v5test` = ieee + ulb, 150k legit / source)

| IEEE test | AUROC | recall @1% FPR |
|---|---|---|
| v4 fields (gpu_full-4090-v4, full corpus) | 0.862 | 35% |
| v5 fields: graph block, virtual customer id, device block | 0.926 | 52% |
| + the wide block (431 raw IEEE columns) | **0.937** | **56%** |

The booster's top gains are `ieee.V258`, `cust_fraud_share_delayed`, `ieee.C5`,
`ieee.DeviceInfo`, `ieee.id_31`: two from the wide block, one from the graph block,
two from the device block. The delayed-label entity features are the single biggest
new idea per column; the wide block is the biggest in total. Both are specific to the
bank that has them, which is the point: a source's own columns stay in a side table
the VAE never sees, and the booster alone reads them.

### History conditioning on vs off (`v5seq` = fdh + sparkov + banksim + capone, 250k rows / source, 4 epochs, window 16, d_model 64)

| test | VAE | VAE r@1% | sequence head | booster | blend | blend r@1% |
|---|---|---|---|---|---|---|
| history off, mean | 0.777 | - | - | 0.875 | 0.8755 | - |
| history on, mean | **0.793** | - | 0.865 | 0.875 | 0.8760 | - |
| sparkov, off → on | 0.869 → **0.938** | 51% → **60%** | 0.996 | 0.9998 | 0.9998 | 99.6% |
| fdh, off → on | 0.613 → 0.624 | 2% → 6% | 0.660 | 0.667 | 0.672 | 34% |
| capone, off → on | 0.627 → 0.613 | 1% → 3% | 0.808 | 0.832 | 0.833 | 19% |
| banksim | 0.997 | 92-94% | 0.998 | 0.999 | 0.999 | 98% |

Reading: conditioning on the customer's own sequence makes the novelty model itself
noticeably better where there is a sequence to read (sparkov: +0.07 AUROC, +9 points
of recall at 1% FPR from the VAE alone), and the sequence head is a respectable third
signal (0.865 mean, within 0.01 of the booster), but on these four sources the blend
does not move: the booster already has the hand-made velocity block, and the
simulators' fraud is the kind it describes. The history model costs 2.2× the epoch
time at this size. Whether it pays on the sources with long real sequences
(tabformer's 30 years, IEEE's reconstructed customers, amlworld's accounts) is a
question for the full-scale run on the H200; the module stays on because the VAE is
the part that must not depend on labels, and "unusual for this customer" is the
right definition of novelty whatever the blend does with it.

## v5 on the H200: what the box taught in the first hours

- **bf16 autocast and the history transformer do not mix.** The v4 model trained in bf16
  at lr 2e-3 for 30 epochs without a hiccup. With the history encoder the gradient norm
  overflowed around step 700 of epoch 1 whatever the learning rate, with every
  individual gradient still finite (the sum of squares passes fp32's range) and the loss
  still ~40; a few steps later the embeddings were NaN. fp32 ran the same 700 steps clean
  and was only 10% slower, because the window gather on the cpu is the bottleneck, not
  the matmuls. `gpu_full` is fp32 now, inputs and targets are bounded at ±20σ
  (`TabularVAE.NUM_CLIP`: a 150σ cell's Gaussian NLL used to own a batch's gradient),
  the history tokens get a LayerNorm before the attention, and the training loop skips
  a non-finite step instead of letting it reach the weights (200 in a row aborts the
  run with a message).
- **The box is fast where it matters.** Building all eleven sources from raw (42.8M rows,
  the wide blocks, the graph block over every entity) took 7 minutes on 64 cores and
  `prepare` of the full corpus with the sequence index 95 seconds; the v5 history model
  does an epoch of 30M rows in 5.8 minutes at 85k rows/s.

### The modules' own numbers (trained on the box, `runs/aux-modules`)

| module | data | protocol | result |
|---|---|---|---|
| text: url head | malurl, 651k urls, 34% malicious | stratified 80/10/10, platt on val | test AUROC **0.994**, AUPRC 0.991 |
| text: free-text head | fakejob, 17.9k postings, 4.8% fraudulent | same | test AUROC **0.983**, AUPRC 0.866 |
| text: profile head | twitterbot, 37k profiles, 33% bots (bio + name + location only) | same | test AUROC 0.778, AUPRC 0.654 |
| behaviour: mouse | Balabit 10 users + BB-MAS 117 users | Balabit challenge: enrol on training sessions, score the 816 labelled test sessions | AUROC **0.836**, EER 0.237 |
| behaviour: keystroke | BB-MAS desktop keyboard | open set, 35 unseen users, 228 genuine / 7,752 impostor trials | EER **0.110**, AUROC 0.963 |
| behaviour: touch | BB-MAS phone + HMOG, 151 users | open set, 34 unseen users | EER 0.172, AUROC 0.908 |
| behaviour: motion | BB-MAS + HMOG accelerometer, 151 users | open set, 65 unseen users, 1,028 / 65,792 trials | EER **0.121**, AUROC 0.956 |
| graph: public benchmarks | DGraph-Fin, Elliptic, YelpChi, Amazon | each dataset's standard split | 0.784 / 0.880 / 0.923 / 0.978 AUROC (`docs/GRAPH_BENCHMARKS.md`) |

The text and behaviour numbers are what a deployer gets from those payloads; they never
touch the held-out transaction tables, which have no text and no sessions. The
Balabit mouse number is well short of the challenge's best (0.9+ with heavy stroke
engineering and session-level models); the keystroke and motion verifiers are in the
range the literature reports for feature-MLP verifiers on these sets.

### The booster on the full v5 corpus: a sweep and an ablation (test AUROC / recall @1% FPR)

Trained on the box's full prepared set (200k legit rows per source + every fraud row,
11 sources, 589 features), scored on the 8.6M-row test split. The v4 numbers are the
shipped `gpu_full-4090-v4` booster for reference.

| variant | mean | fdh | capone | tabformer | ulb | ieee | amlworld | baf | fraudecom |
|---|---|---|---|---|---|---|---|---|---|
| v4 booster (9 sources, v4 fields) | – | .672/34% | .765/8% | **.980/73%** | .987/84% | .862/35% | .990/85% | – | – |
| v5 base (lr .05, 1000 rounds) | .898 | .664/34% | .833/19% | .928/16% | .983/87% | .940/55% | .990/85% | .892/26% | .644/10% |
| 255 leaves | .899 | .664 | .837 | .930 | .988 | .939 | .990 | .891 | .653 |
| lr .03, 3000 rounds | .900 | .671 | .834 | .941/19% | .986 | .940 | .990 | .893 | .650 |
| feature_fraction .5 | .900 | .670 | .834 | .939 | .985 | .939 | .990 | .892 | .650 |
| − wide blocks (469 cols) | .893 | .683 | .833 | .936 | .987 | .927/51% | .990 | .833/17% | .638 |
| − delayed-label features (12) | .887 | .673 | .775/8% | .895/7% | .987 | .907/43% | .990 | .891 | .647 |
| − whole graph block (29) | .883 | .674 | .765/9% | **.984/75%** | .988 | .905/42% | .990 | .891 | .515/1% |
| − lifetime counts and ages (12) | **.904** | .680 | .834 | **.977/69%** | .985 | .939 | .989 | .891 | .651 |

Reading, in order of what it changed:
- The **graph block is worth +0.07 on capone and +0.035 on IEEE**, and the delayed-label
  features inside it are most of that (removing only them costs capone 0.06 and IEEE
  0.03). fraudecom without the graph block is a coin flip: its fraud *is* the shared
  device.
- The same block **cost tabformer 0.05 AUROC and 57 points of recall**, and the
  ablation pinned it on the twelve lifetime features - distinct customers ever seen
  with an entity, days since an entity was first seen, the customer's distinct
  counterparties / devices ever. On a 30-year file those grow with the calendar; the
  booster read them as the era and the newest months are an era it never saw. The
  windowed count (`txn_count_30d`) and the delayed fraud share do not have that
  problem. The schema now carries windowed and capped versions instead
  (`*_new_cust_30d`, `*_age_days` capped at a year, `cust_new_*_30d`): tabformer back
  at 0.977 / 69%, nothing else lost, best mean of the table.
- The wide blocks are +0.013 / +4 points of recall on IEEE and +0.06 on baf, as before.
- Booster hyperparameters are a rounding error next to features: lr 0.03 with 3000
  rounds is the one change worth taking (+0.003 mean, tabformer +0.01).

### v5-h200-e (the run that ships: full v5 corpus, 11 sources, rented H200 NVL)

The whole thing ran with the laptop off, through the relay (`docs/REMOTE_TRAINING.md`):
raw files pushed once (18 GB), `build` of all eleven sources in 7 minutes, `prepare --name
full` in 95 seconds (29,989,316 train / 4,284,187 val / 8,568,376 test rows, 93k / 13k /
27k fraud, 589 features with the wide blocks), then `gpu_full` as written for v5: latent 24,
768-768-384, history encoder with a 32-transaction window, d_model 128, 2 layers, 4 heads
(3.01M parameters), batch 8192, lr 1.5e-3, fp32, 30 epochs, checkpoint and stopping on the
held-out −ELBO with patience 8, 16 scoring samples.

**Training.** 5.8 minutes per epoch at 85k rows/s (the window gather on the cpu is the
bottleneck, the GPU sat at 75%), 30 epochs in 2.9 hours, no skipped steps. The held-out
−ELBO fell from 21.6 to −2.8 (best epoch 28) with two excursions on the way (epoch 10,
epoch 15) that recovered within two epochs each; the train loss ended at −5.1. The
per-epoch VAE AUROC - the number v4 stopped on - oscillated between 0.72 and 0.78 with no
trend after epoch 3, and single sources swung by 0.3 between neighbouring epochs (paysim
0.34 → 0.67 → 0.76), because the per-epoch check scores with one posterior sample and the
fraud sets are small. Two earlier launches of this config (c, d) stopped on that number and
would have kept an epoch-2 checkpoint; e selects on the ELBO, which is what the model is
trained to lower and moves smoothly. The post stage took 29 minutes: booster 1,435 rounds
in 285 s (val AUC 0.965 pooled; v4's was 0.950), eleven specialists, the sequence head
(70 s), the device head (13 s), nine entity graphs (paysim's capped at 4M nodes, 3 minutes
for all), the three text heads, the four behaviour verifiers, calibration and evaluation.

**The blender needed one fix.** The blend is a logistic regression over every signal with
per-bank interaction terms, fit on half of a validation subsample and checked on the other
half. On the held-out half ULB's blend scored 0.915 against 0.953 for the booster on the
same rows: the fit half holds 21 ULB fraud rows and the unregularised ULB interactions had
learnt them by heart (`kl_z x ulb` had a weight of −1.1). Lowering C across the board did
nothing until it hurt the big banks; dropping the interactions altogether cost fdh 0.02.
What works is hierarchical shrinkage: each bank's interaction columns are scaled by
sqrt(fraud rows / 10,000) before the fit, so a bank with few labelled fraud rows pays more
penalty for leaving the shared weights and ULB (factor 0.05) is pulled onto them while
tabformer (0.45) and fdh (0.28) keep theirs. The scaling folds back into the stored
weights, so inference is unchanged. Held-out half: ULB 0.953, every other bank identical to
the third decimal; on test ULB went from 0.971 to 0.984 and the mean blend edged above the
booster (0.9056 vs 0.9052) instead of below it. Re-running calibration and evaluation on
the frozen models took 10 minutes on the box.

**Results** (test split, newest 20% of every source, 8.57M rows, 27,303 fraud; thresholds
frozen on validation; AUROC unless marked):

| source | VAE | booster | seq | local | device | graph | **blend** | r@0.1% | r@1% | r@5% | v4 blend | v4 r@1% |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fdh | 0.624 | 0.688 | 0.656 | 0.674 | – | 0.501 | **0.691** | 32% | 34% | 38% | 0.672 | 34% |
| capone | 0.612 | 0.834 | 0.815 | 0.836 | – | 0.767 | **0.838** | 11% | 21% | 41% | 0.765 | 8% |
| tabformer | 0.940 | 0.980 | 0.985 | 0.988 | – | 0.815 | **0.983** | 43% | 71% | 93% | 0.981 | 73% |
| ulb | 0.963 | 0.987 | 0.987 | 0.985 | – | – | **0.984** | 80% | 89% | 92% | 0.982 | 84% |
| paysim | 0.797 | 1.000 | 0.991 | 1.000 | – | 0.554 | **1.000** | 95% | 100% | 100% | 1.000 | 100% |
| sparkov | 0.862 | 1.000 | 0.996 | 1.000 | – | 0.953 | **1.000** | 99% | 100% | 100% | 0.999 | 99% |
| banksim | 0.990 | 0.999 | 0.999 | 1.000 | – | 0.998 | **1.000** | 89% | 99% | 100% | 0.999 | 97% |
| ieee | 0.795 | 0.939 | 0.904 | 0.939 | 0.733 | 0.825 | **0.940** | 33% | 56% | 73% | 0.858 | 35% |
| amlworld | 0.911 | 0.990 | 0.984 | 0.989 | – | 0.846 | **0.990** | 62% | 86% | 93% | 0.990 | 84% |
| baf | 0.657 | 0.891 | 0.812 | 0.890 | 0.768 | – | **0.892** | 7% | 26% | 55% | – | – |
| fraudecom | 0.641 | 0.649 | 0.644 | 0.644 | 0.639 | 0.516 | **0.643** | 0% | 7% | 32% | – | – |
| **mean, 11 sources** | 0.799 | 0.905 | 0.889 | 0.904 | 0.714 | 0.753 | **0.906** | | | | | |
| **mean, v4's 9 sources** | 0.833 | 0.935 | | | | | **0.936** | | | | **0.916** | |
| generic (bank unknown) | 0.625 | 0.952 | 0.930 | – | 0.735 | – | **0.953** | 46% | 62% | 75% | 0.956 | 66% |

`seq` is the sequence head on the history read-out, `local` the bank's own specialist
booster, `device` the device-block head (only the three sources with a device block),
`graph` the entity-graph head (NaN-free rows only; paysim's covers 48% of its test rows,
fraudecom's 5%). The modules that never touch the transaction tables keep the numbers in
the table above: text 0.994 / 0.972 / 0.783, behaviour EER 0.24 / 0.11 / 0.17 / 0.12.

What to take from it:

- **On the nine sources v4 had, the blend went from 0.916 to 0.936 mean AUROC**, and the
  gains are where the new columns are: Capital One 0.765 → 0.838 (recall at 1% FPR 8% →
  21%), IEEE 0.858 → 0.940 (35% → 56%), fdh 0.672 → 0.691, ULB 84% → 89% recall. The
  ablation on the previous page says which columns: the graph block with its delayed labels
  for capone and IEEE, the wide blocks for IEEE. Tabformer is unchanged (0.981 → 0.983,
  73% → 71% recall: inside the month-to-month noise of its newest months, and the
  lifetime-count regression of the first v5 booster is gone). The simulators stay saturated.
- **The specialist and the sequence head are second opinions, not new information.** `local`
  tracks the pooled booster within 0.005 everywhere except fdh (0.674 vs 0.688: its
  specialist stopped after 15 rounds) and tabformer (0.988 vs 0.980, 75% vs 66% recall),
  where the per-bank model sees the newest months better than the blend does;
  `seq` is 0.01-0.08 below the booster on every bank but beats it on nothing, and on
  tabformer (0.985) it is the strongest single neural signal the project has produced.
  The blender's coefficients say the same: the VAE score's weight is ~0, the booster and
  the specialist carry the sum. The blend is still the right thing to serve - it is the one
  calibrated probability, and it now matches or beats the best single signal on every bank
  but tabformer and fraudecom.
- **The history-conditioned VAE is a better novelty model on the banks with customer
  sequences and a worse one on the simulators**: capone 0.51 → 0.61, IEEE 0.73 → 0.79,
  amlworld 0.82 → 0.91, ULB 0.95 → 0.96; paysim 0.94 → 0.80, sparkov 0.91 → 0.86, fdh
  0.65 → 0.62; 0.826 → 0.833 mean over the nine. PaySim's fraud is a balance drained in one
  transfer, which the v4 model saw as an outlier of the row and the history model, having
  read the account's previous rows, half expects; the ±20σ bound on the Gaussian terms
  (`NUM_CLIP`, needed for the fp32 run) takes the rest. Nothing in the blend cares, but
  the receipt's "surprise" column is a different animal on PaySim now.
- **Generic mode (bank blanked)** is 0.953 against 0.956, on a pool that now includes the two
  hardest new sources (baf at 0.89 and fraudecom at 0.64 contribute 15% of the generic
  pool's fraud rows). Recall at 1% FPR 62% vs 66% for the same reason. On the shared banks it is the
  same model.
- **BAF and fraudecom are honest numbers for honest reasons.** BAF is account applications
  with a device block and no customer: 0.892 AUROC and 55% of the fraud at 5% FPR is the
  range the dataset's own paper reports for gradient-boosted trees on its Base variant
  (about half the fraud at 5% FPR), and the device head alone gets 0.77 from os + session
  + validity flags. fraudecom's fraud *is* the shared device (the graph head's
  AUPRC 0.24 on the 5% of test rows whose device the graph has seen is the best signal
  there), but 95% of its test rows are a device and an ip seen for the first time, and on
  those nothing beats the base rate by much: every signal sits at 0.64.

### v5-h200-f (the same VAE, the booster side refit with a one-week label delay)

The table above left fdh where every model had left it, 0.69, and the reason is
mechanical rather than statistical. The handbook simulator's second scenario compromises
a terminal for 28 days; the schema's fraud-history features count labels that are at
least 30 days old (`ENTITY_LABEL_DELAY_DAYS`), so by the time a terminal's fraud is
"known" the terminal is clean again, and the feature is a perfect description of last
month. The handbook's own baseline uses a one-week delay for exactly these
terminal-risk features and gets ~0.87 AUROC with XGBoost. Whether a week is honest is a
statement about a deployment's label latency, not about the model, so v5 carries both:
the 30-day pair stays in the schema (the VAE and the booster see it), and the same pair at
7 days (`ENTITY_LABEL_DELAY_FAST_DAYS`) rides in the wide block for the booster only, for
every source with entities (`wide__<entity>_fraud_{count,share}_fast`). Nothing about the
VAE changed, so this was a `build` (7.5 minutes for all eleven sources, the entity pass
run twice), a `prepare` and a `post` on a copy of the e checkpoint: 30 minutes on the box.

| source | VAE | booster | seq | local | device | graph | **blend** | r@0.1% | r@1% | r@5% | v5-h200-e blend | v4 blend |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fdh | 0.624 | 0.891 | 0.656 | 0.892 | – | 0.501 | **0.893** | 67% | 75% | 79% | 0.691 | 0.672 |
| capone | 0.613 | 0.838 | 0.815 | 0.839 | – | 0.767 | **0.841** | 11% | 21% | 42% | 0.838 | 0.765 |
| tabformer | 0.940 | 0.980 | 0.985 | 0.989 | – | 0.815 | **0.986** | 50% | 74% | 94% | 0.983 | 0.981 |
| ulb | 0.962 | 0.984 | 0.987 | 0.985 | – | – | **0.983** | 81% | 88% | 91% | 0.984 | 0.982 |
| paysim | 0.797 | 1.000 | 0.991 | 1.000 | – | 0.554 | **1.000** | 95% | 100% | 100% | 1.000 | 1.000 |
| sparkov | 0.863 | 1.000 | 0.996 | 1.000 | – | 0.953 | **1.000** | 99% | 100% | 100% | 1.000 | 0.999 |
| banksim | 0.990 | 0.999 | 0.999 | 1.000 | – | 0.998 | **1.000** | 89% | 99% | 100% | 1.000 | 0.999 |
| ieee | 0.795 | 0.958 | 0.904 | 0.954 | 0.733 | 0.825 | **0.958** | 41% | 63% | 79% | 0.940 | 0.858 |
| amlworld | 0.912 | 0.990 | 0.984 | 0.989 | – | 0.846 | **0.990** | 62% | 86% | 93% | 0.990 | 0.990 |
| baf | 0.658 | 0.891 | 0.812 | 0.890 | 0.768 | – | **0.892** | 7% | 26% | 55% | 0.892 | – |
| fraudecom | 0.641 | 0.646 | 0.644 | 0.643 | 0.639 | 0.516 | **0.645** | 1% | 8% | 31% | 0.643 | – |
| **mean, 11 sources** | 0.799 | 0.925 | 0.889 | 0.926 | 0.714 | 0.753 | **0.926** | | | | 0.906 | |
| **mean, v4's 9 sources** | 0.833 | 0.960 | | | | | **0.961** | | | | 0.936 | **0.916** |
| generic (bank unknown) | 0.625 | 0.966 | 0.930 | – | 0.735 | – | **0.966** | 52% | 69% | 81% | 0.953 | 0.956 |

- **fdh 0.691 → 0.893, 34% → 75% of its fraud at 1% FPR, 67% at 0.1%.** That is the
  handbook benchmark's range on the handbook's data, reached with the handbook's
  assumption. The terminal's week-old fraud share is the feature; the customer's own
  (scenario 3, a compromised card for 14 days) is the rest.
- **IEEE 0.940 → 0.958 (56% → 63%)** and **tabformer 0.983 → 0.986 (71% → 74%)**: the
  card and the counterparty with a fresher label. Capital One +0.003, the rest within
  ±0.001 - the simulators were saturated and the two new sources have no label history
  that a shorter delay changes.
- **Generic mode 0.953 → 0.966, recall 62% → 69%**: now above v4's 0.956 on a pool that
  includes the two hardest sources, because the delayed-label features are the same
  columns for every bank and the booster can use them without knowing whose they are.
- Validation agrees with test for once (held-out half: fdh 0.904, IEEE 0.972, pooled
  0.974), which is what a feature that fixes a mechanism rather than a fit looks like.
- The receipt names which delay fired: a `wide` push on `cpty_fraud_share_fast` is the
  one-week label, a push on `cpty_fraud_share_delayed` the 30-day one. A deployment whose
  labels take a month should leave the `wide__*_fast` columns out of its requests (NaN,
  which the booster learnt as "no labelled history at that delay yet", since every
  entity's first week looks like that in training) rather than feed it labels it would
  not have had.

### Round 3 (v5-h200-h, what ships): label latency as a first-class thing, and the booster reads the transformer

Three changes on the booster side, each its own `post` on the e checkpoint (the VAE never
moved), compared on the same test rows as f:

1. **Label-latency dropout** (`boost.fast_dropout` 0.3). The one-week block is only honest
   for a deployment whose labels arrive within a week, and a deployment whose labels take
   a month had, with f, two bad options: feed the model labels it would not have had, or
   send NaN in columns the booster had only ever seen NaN on an entity's first week. Now a
   random 30% of the booster's (and each specialist's) training rows have the whole block
   blanked, the score API takes `label_delay_days` and blanks it past a week (the receipt
   says "used", "blanked" or "not sent"), and the post stage evaluates the test split a
   second time with the block blanked (`eval_delay30.json`): every bank gets a one-week
   number and a one-month number, on the model page and below.
2. **Recent-fraud windows** (`wide__*_fraud_{count,share}_fast_w7` / `_w30`): the same
   count and share over the entity's rows in the 7 and 30 days before the one-week label
   horizon, so a terminal compromised this month is not diluted by a clean history.
3. **The history encoder's read-out as booster input** (`boost.history_features`, run i):
   the sequence head beat the booster on tabformer (0.985 vs 0.980) from that 128-d
   vector alone, and feeding it to the trees is the one stacking that cannot leak - the
   encoder was trained on legitimate rows with no label in sight. The booster and the
   specialists get it as `hist_0..127`, computed with the bank blanked for the generic
   slice, and the receipt shows the block as one line.

**Results of round 3** (same test rows; "30-day labels" is the second evaluation, with the
one-week block blanked; the last two columns are the v4 release):

| source | vae | booster | seq | local | device | graph | **blend** | r@0.1% | r@1% | r@5% | 30-day labels: blend / r@1% | v4 blend | v4 r@1% |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fdh | 0.624 | 0.903 | 0.656 | 0.905 | – | 0.501 | **0.905** | 72% | 78% | 81% | 0.679 / 34% | 0.672 | 34% |
| capone | 0.613 | 0.837 | 0.815 | 0.839 | – | 0.767 | **0.841** | 11% | 21% | 41% | 0.836 / 19% | 0.765 | 8% |
| tabformer | 0.940 | 0.991 | 0.985 | 0.995 | – | 0.815 | **0.994** | 68% | 86% | 98% | 0.986 / 73% | 0.981 | 73% |
| ulb | 0.962 | 0.986 | 0.987 | 0.985 | – | – | **0.984** | 80% | 89% | 92% | 0.984 / 89% | 0.982 | 84% |
| paysim | 0.797 | 1.000 | 0.991 | 1.000 | – | 0.554 | **1.000** | 95% | 100% | 100% | 1.000 / 100% | 1.000 | 100% |
| sparkov | 0.863 | 1.000 | 0.996 | 1.000 | – | 0.953 | **1.000** | 99% | 100% | 100% | 1.000 / 100% | 0.999 | 99% |
| banksim | 0.990 | 0.999 | 0.999 | 1.000 | – | 0.998 | **1.000** | 88% | 99% | 100% | 0.999 / 99% | 0.999 | 97% |
| ieee | 0.795 | 0.958 | 0.904 | 0.955 | 0.733 | 0.825 | **0.958** | 41% | 63% | 79% | 0.940 / 56% | 0.858 | 35% |
| amlworld | 0.912 | 0.990 | 0.984 | 0.990 | – | 0.846 | **0.990** | 62% | 85% | 93% | 0.990 / 85% | 0.990 | 84% |
| baf | 0.658 | 0.893 | 0.812 | 0.891 | 0.768 | – | **0.893** | 7% | 25% | 55% | 0.893 / 25% | – | – |
| fraudecom | 0.641 | 0.645 | 0.644 | 0.639 | 0.639 | 0.516 | **0.641** | 1% | 10% | 32% | 0.641 / 9% | – | – |
| **mean, 11 sources** | 0.799 | 0.927 | 0.889 | 0.927 | 0.714 | 0.753 | **0.928** | | | | 0.904 | | |
| **mean, v4's 9 sources** | 0.833 | 0.963 | | | | | **0.963** | | | | 0.935 | **0.916** | |
| generic (bank unknown) | 0.625 | 0.969 | 0.930 | – | 0.735 | – | **0.969** | 56% | 70% | 82% | 0.952 / 62% | 0.956 | 66% |

- **Run h (1 + 2) is what ships: 0.963 mean over v4's nine banks, 0.928 over all eleven,
  generic 0.969.** The recent-fraud windows did more than expected on tabformer, 0.986 →
  0.994 and 74% → 86% of its fraud at 1% FPR (98% at 5%): card fraud on that file comes in
  bursts, and a lifetime share buried this month's burst under years of clean history.
  fdh went 0.893 → 0.905 (78% at 1%, 72% at 0.1%). The latency dropout cost nothing on
  the one-week numbers (booster val AUC 0.9764 against f's 0.9762) and bought the second
  column: scored as a deployment whose labels take a month, the same model lands within
  0.001 of run e on every bank (0.935 mean, generic 0.952), i.e. exactly the 30-day model,
  instead of somewhere undefined. Those are the two numbers a deployer should read, and the
  model page shows both.
- **Run i (the encoder's read-out as booster input) is a wash**: 0.9627 against h's 0.9634
  over the nine, tabformer AUPRC +0.02, amlworld −0.003, everything else within 0.001. The
  booster stopped earlier (937 rounds against 1,387) at the same validation AUC, so the
  128 columns are not useless - they let the trees get there faster - but the velocity
  and entity blocks already say what the summary says. The option stays in the code
  (`boost.history_features`, `configs/gpu_hist.yaml`) as the documented negative result;
  it is off in `gpu_full`.
- A longer VAE (`gpu_long`: window 48, 40 epochs) was started alongside and stopped at
  epoch 7 once h's numbers were in: the blender gives the VAE score a weight of about zero,
  so a better novelty model would have changed the receipt and nothing the app flags on.
  The config is kept for whoever wants the novelty score itself to be better.
