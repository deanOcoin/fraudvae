# Architecture notes

## Pipeline

```
raw downloads ──fetch──▶ data/raw/<source>/            (never in git)
   │
   └─build──▶ data/processed/<source>.parquet         unified schema, zstd, row groups of 500k
                 + <source>.wide.parquet (+ .json)      v5: the source's own raw columns, row-aligned
                 + stats.json (per-source distributions for the ui) + samples.parquet
   │
   └─prepare──▶ data/prepared/<name>/{split}_*.npy     standardised float32 numerics, uint8 masks,
                 + {split}_wide_<source>.npy            int16 categorical codes, source idx, label, row id,
                 + seq_order / seq_pos / seq_start      v5: customer keys + timestamps, wide blocks, and the
                 + meta.json                            (customer, time) sequence index over every split
   │
   └─train──▶ runs/<name>/model.pt + metrics.jsonl + status.json + train.log      (the VAE + history encoder)
   └─boost──▶ runs/<name>/booster.txt + booster.json                             (LightGBM on [common | wide])
   └─seqhead─▶ runs/<name>/seqhead.pt + seqhead.json                              (v5: classifier on the history read-out)
   └─calibrate──▶ runs/<name>/calibration.json + latent_map.npz                  (blender + thresholds on val)
   └─evaluate──▶ runs/<name>/eval.json                                           (VAE / booster / seq / blend on test)
   │
   └─serve──▶ fastapi + the static ui, loads the best compatible run by test (blend) auroc

`fraudvae train` runs the last four in sequence; each is also its own command,
and `fraudvae post RUN [--config X]` redoes the last three on an existing run
(a booster re-tune is minutes, the VAE is the expensive half). `prepare
--stats-from-run RUN` builds a set standardised exactly like the one a run was
trained on, so a GPU-trained model can be calibrated / evaluated on a
laptop-sized set.
```

Adapters return a DataFrame, or a generator of DataFrames for the 24M-row
tabformer file, which is processed on compact numpy arrays and written in 2M
row chunks so the build fits in a few GB of RAM. Stats are computed from the
written parquet with a small column subset; coverage comes from parquet
null-count metadata.

`prepare` streams the parquet in 1M row batches into pre-sized memmaps, then
standardises in place with train-legit means/stds. `--max-rows-per-source`
caps *train* rows per source (legit subsampled, every fraud row kept) and never
touches val/test, so evaluation stays honest while a laptop can train. A field
added to the schema after a source was built is read as unobserved for it.

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

## v5: entity keys, the graph block and the wide block

`fraudvae/data/schema.py`, `fraudvae/data/entities.py`, `fraudvae/data/build.py`

Every adapter now emits six entity key columns (`customer`, `counterparty`,
`device`, `email`, `address`, `card`; dense codes per source, −1 = the source has
no such thing; see SCHEMA.md for what each source maps). They are never model
inputs. Two things are built from them:

- **The graph block** (29 numerics, group `graph`): for each entity the row
  touches, its 30-day volume, the number of distinct customers seen with it
  before, its age, and its confirmed-fraud count and share among rows older
  than `ENTITY_LABEL_DELAY_DAYS` (30); for the customer, the number of distinct
  counterparties / devices used before and the same delayed fraud history.
  Same machinery as the velocity features (sort by entity and time, one
  monotone key, searchsorted), computed in groups of entities so the 24M-row
  file fits. The label delay is the honest version of the "negative file"
  every bank keeps: a terminal that was compromised last month is known to be,
  yesterday's label is not. Rows with no delayed history get NaN (unobserved),
  not 0. These were the biggest single lever on IEEE-CIS (0.86 → 0.93 AUROC
  before the wide block was even on). The same count / share pair at a
  one-week delay (`ENTITY_LABEL_DELAY_FAST_DAYS`, the lag the Fraud Detection
  Handbook assumes) rides in the wide block as `wide__<entity>_fraud_*_fast`,
  for the booster only: on the handbook simulator a terminal stays compromised
  for 28 days, which a 30-day-old label never catches and a week-old one
  catches for three weeks, plus the same pair over the 7 and 30 days before
  that horizon. Which delay is honest depends on how fast a deployment's labels
  arrive, so the booster is trained for both regimes: on a random 30% of its
  training rows the one-week block is blanked (`boost.fast_dropout`), the score
  API takes `label_delay_days` and blanks it for any deployment slower than a
  week, the receipt says whether the block was used, blanked or not sent, and
  the post stage evaluates the test split a second time with the block blanked
  (`eval_delay30.json`, the model page's last column), so every bank has a
  one-week number and a one-month number.
- **The sequence index**: `prepare` writes every row of every split in
  (customer, time) order plus, per row, its position and the start of its
  customer's block. A row's history window is the slice just before it,
  whatever split those older rows fell into (they are older, which is all that
  matters: a val row's history is train rows, a test row's is train and val).

The **wide block**: an adapter may hand `build` its raw columns as `wide__<col>`
(IEEE: all 431 of them, categoricals as frequency-ranked codes with a vocab in
`<source>.wide.json`). They go to a row-aligned `<source>.wide.parquet`, then
to `{split}_wide_<source>.npy` in `prepare`, NaN for every other source, and
the booster sees them as `<source>.<col>` features after the common ones. The
VAE never does: the common schema is what transfers across banks, the wide
block is what a bank's own model should not leave on the table.

## Booster

`fraudvae/model/booster.py`

LightGBM binary classifier on `[standardised numerics with NaN restored where
unobserved, categorical codes, source, the source's wide block]` (the same
arrays the VAE reads, plus the raw columns and the one-week label block it does
not), and optionally (`boost.history_features`) the frozen history encoder's
read-out of the customer's past as `hist_*` columns - the one stacking that is
leak-free, since the encoder never saw a label - fit on
the labelled train split: every fraud row plus up to `max_legit_per_source`
legit rows per source, legit rows weighted back towards the base rate but by at
most `legit_weight_cap` (3×): the natural weights on the full corpus run from 1
(ulb) to 85 (tabformer) and with those one bank's traffic defines "normal" in
the source-blanked mode - same config, same test rows, generic AUROC 0.936
against 0.955 with the cap. The booster ranks; the blender and the per-bank
thresholds calibrate. A
`generic_fraction` (25%) slice is duplicated with source = 0 so the model has
a generic mode. Early stopping watches two validation sets - the (legit-subsampled)
val rows as they are and the same rows with the source blanked - and stops on
the mean of the two AUCs, so the generic mode is never quietly traded for
named-bank fit (lightgbm's own rule, "stop when any one stalls", was not
the problem; a single pooled val set was: it let the first full run keep
boosting to the 1000-round cap while the generic mode got worse).
`num_leaves` 127, lr 0.05, feature/bagging fraction 0.8, L2 1, categorical
smoothing. It never sees a VAE output, so stacking it with the VAE later is
clean (the VAE trained on the same legit rows; feeding its in-sample surprises
to the booster would shift the train/val distribution).

The receipt's "pushes" are `predict(pred_contrib=True)`: SHAP values in
log-odds, one per feature plus the bias.

Loading is tolerant: a run whose `booster.txt` can't be opened (lightgbm not
installed, or its native library refusing to load) still serves as a
novelty-only model, with the reason in `/api/meta` → `booster_error` and in
the sidebar. macOS is the case that matters: the LightGBM wheel links the
homebrew `libomp`, the PyTorch wheel ships its own, and with both mapped the
first LightGBM call segfaults the process (`KMP_DUPLICATE_LIB_OK`, which the
package sets at import, only stops the runtime's own abort, not the crash
after it). `_lgb()` therefore walks dyld's image list after importing
lightgbm and raises `BoosterUnavailable` if it sees two distinct `libomp`
files, before anything can crash; `scripts/macos_openmp.sh` (`make fix-omp`)
makes torch use the homebrew copy so there is one runtime. The server also
keeps a log at `$FRAUDVAE_HOME/server.log` (`FRAUDVAE_SERVER_LOG` to move it)
with `faulthandler` armed on it, so a native crash leaves a Python stack
behind instead of a vanished process.

## v5: the history encoder and the sequence head

`fraudvae/model/vae.py` (HistoryEncoder), `fraudvae/model/dataset.py`
(SequenceIndex), `fraudvae/model/seqhead.py`

The VAE is conditioned on the customer's previous `window` transactions (32 in
`gpu_full`, 16 in `cpu_small`). Each past row is embedded with the VAE's own
field embeddings - shared weights, so "this amount" means the same thing in
the history and in the row being scored - plus an embedding of how long ago
it was and a recency position, prepended with a learned "no history" token,
run through a 2-layer transformer encoder, and summarised by one attention
read-out whose query is the current row's embedding: the history is read for
this transaction. The summary h enters both the encoder, q(z | x, h, s), and
the decoder, p(x | z, h, s), so the reconstruction and every per-field
surprise are relative to this customer's past. Training is unchanged (ELBO on
legit rows); `history_dropout` hides the window for 10% of rows so the
no-history read-out is a trained state and a customer's first transaction (or
an id-less bank like ULB) is well defined. A model with `history.enabled:
false` is exactly the v4 model; v4 checkpoints load as before.

Batching: the sequence iterator visits rows in customer order and shuffles the
order of the batches, so neighbouring rows share most of their window and the
gather from the memmaps stays local. `score_split` does the same and re-sorts
its outputs into the caller's order.

The **sequence head** is a small MLP on [h, the VAE's input embedding of the
row], trained in the post stage on the same labelled subsample as the booster
(legit subsampled per source, class-weighted BCE, 3 epochs) with the history
encoder frozen. Its logit is the "sequence" signal: the one place the
unsupervised history summary meets the fraud labels, kept separate from the
booster so the receipt can say "the history says". It is calibrated and
evaluated like the booster (per bank and generic).

## v5: device telemetry - the two new sources and the field heads

`fraudvae/data/adapters/baf.py`, `fraudvae/data/adapters/fraudecom.py`,
`fraudvae/model/heads.py`, `fraudvae/server/payloads.py`

Two sources carry the device block for real. **baf** (Feedzai's Bank Account
Fraud, 1M account-opening applications) has os, session length, keep-alive,
phone and email validity, name/email similarity and the generator's own
device velocity (`device_distinct_emails_8w`, mapped onto
`device_new_cust_30d`); it has no amount in the card sense (the
proposed credit limit stands in), no customer id and no clock finer than the
month, so hour / dow stay unobserved and the row order inside a month is the
time order. **fraudecom** (151k first purchases with a raw device id shared
across accounts) is the one public file where entity sharing is observable:
a device seen with other accounts before carries a 62% fraud rate against
4% elsewhere, and the graph block computes exactly that, causally. Both keep
every raw column in a wide block. Both are sources 10 and 11 in the schema,
which means `n_sources` grew: the scorer sizes its tables from whichever is
larger, the schema or the run's baselines, and a blender fit with fewer
sources maps unknown ones to the generic row, so v4 runs still load.

A **field head** (`FieldHead`) is a small supervised MLP over one named block
of the common schema, with the source embedding (blanked for 25% of training
rows, like the booster's generic slice). The device head reads the device
block, the device / email / address graph columns and `device_type` -
those decide *presence* - plus account age, payer email bucket and the
cross-border flag as context. It is trained in the post stage on the labelled
rows that carry the block, across every source that has one (IEEE identity
rows, every baf and fraudecom row), legit subsampled per source, 4 epochs of
class-weighted BCE. Rows without the block get NaN, which the blender sees as
"absent" through the presence bit, so a bank that sends no telemetry is
scored exactly as before and one that does gets a signal that was learned
from three sources' worth of devices, not from its own labels. The receipt
shows it as its own section with the fields it saw. `HEAD_SPECS` is where
the next heads (graph, text, behaviour) will be declared; everything
downstream - `score_split`, calibration, evaluation, the API, the model page
- iterates over whatever heads a run has.

At the API, a `device` payload (user-agent style strings and numbers: `os`,
`browser`, `screen`, `session_length_min`, `keep_alive`, `email_domain`...)
is mapped onto the block by `payloads.device_fields` using the same taxonomy
the adapters use, with fields set directly on the transaction taking
precedence. The batch csv takes the block's columns by name.

## v5: the entity graph

`fraudvae/graph/` (build.py, model.py, train.py, scorer.py, bench.py)

The graph block above is one hop of hand-made structure. The graph module learns the
rest: per source, the entities (customers, counterparties, devices, emails, addresses,
cards) are nodes, every transaction connects the entities it touched (weighted
co-occurrence edges, both directions), each node carries a feature block computed
from its own rows up to the snapshot (volume, delayed confirmed fraud, amount scale,
age, distinct partners, kind one-hot), and a two-layer GraphSAGE - plain sparse
matrix products, no graph library - turns that into a 32-d embedding per node. A
transaction head reads the embeddings of a row's six entity slots (zeros plus a
"known" bit where the slot is empty or the entity is new) into the "graph" logit.

Time is handled with three snapshots so that nothing leaks: A, the train period minus
its last fifth, is the graph the encoder and head are fitted on, using the rows of
that last fifth as the labelled examples (a row's own label never reaches its own
entities' features); B, the whole train period, scores the validation rows that fit
the blender; C, train plus validation, scores the test rows and serves. The encoder is
inductive, so B and C are forward passes over bigger graphs. The embeddings of B and C
are stored with the run (fp16) and looked up by entity code; the API takes `entities`
(the codes the build assigned) and the receipt names which of them were known.
Entities seen fewer than `min_degree` times, and everything past `max_nodes` per
source, are "unknown" to the graph, which is what a brand-new card is to a bank. The
same encoder, with one weight per relation, runs the public fraud-graph benchmarks
(`fraudvae graph-bench`: DGraph-Fin, Elliptic, YelpChi, Amazon) that validate the
code against the literature; see `docs/GRAPH_BENCHMARKS.md`.

## v5: text and behaviour - the external modules and the fusion step

`fraudvae/text/`, `fraudvae/behaviour/`, `ActiveModel._fuse` in `fraudvae/server/inference.py`

Two modules have no counterpart in the transaction corpus. The **text heads** are
fastText-shaped classifiers (word unigrams and bigrams and character trigrams hashed
into a 2^20-row embedding table, a mean, a small MLP) trained on three public labelled
text-fraud sets: malicious URLs, fraudulent job postings and bot profiles. Each is
platt-scaled on its own validation split and knows its own base rate, so what it
returns for a piece of text is a calibrated probability *and* a likelihood ratio
(log-odds against that base rate). The **behaviour verifiers** take raw interaction
events - mouse strokes, keystrokes, touch swipes, accelerometer samples - through
hand-made dynamics features per window (kinematics, timing, geometry; one function per
modality in `behaviour/features.py`), an MLP embedding trained to tell apart the users
of the public biometric sets (Balabit, BB-MAS, HMOG), and a cosine similarity between
a session's embedding and the account holder's enrolled one, platt-scaled on genuine
and impostor trials of users the verifier never trained on (the equal-error rate on
those trials is the module's number). The entity store keeps one enrolment per
customer, a running mean of the sessions the server has scored for them; the first
session only enrols.

Neither signal can be blended, because no validation row ever carried text or a
session, so the blender would learn a zero weight for them. They are **fused**
instead: each ratio is clipped to ±6 and added, with a fixed weight (0.5 for text,
whose training data is a different domain; 1.0 for behaviour, whose ratio is about
this very account), to the blend's log-odds. The blend probability stays on the receipt
as the data-calibrated verdict; `p_fraud` becomes the fused one when a module spoke
and the flag is the blend's threshold applied to it. A deployment without these
payloads never sees the difference; one with them gets evidence the labelled data
could not have taught.

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
one-hot]`, the three signals clipped to fixed ranges (`Scorer.BLEND_CLIP`),
and (v5) per extra signal `[value × present, present, value × one-hot]`. The
extra signals today: `seq` (the sequence head), `local` (the bank's specialist
booster), `device` (the device-block head) and `graph` (the entity-graph head),
each a logit. The presence column is
what makes the later modules pluggable: a device-risk, text-risk or
behaviour LLR that is NaN for a row simply contributes its "absent" weight,
so a deployment that sends none of the new payloads is scored exactly like v4
and one that sends them all uses them all (`blend.signals` in calibration.json
lists what the run was fit with).
The interaction columns give each bank its own weighting of booster vs
novelty; the generic one-hot gets its own. Half the rows fit, the other half
report a held-out AUROC. A bank's interaction columns are shrunk toward the
shared weights by `sqrt(its fraud rows in the fit half / 10,000)`
(`calibrate.SHRINK_REF`): a bank with a few dozen labelled fraud rows cannot
earn weights of its own, and before the shrink ULB's 21 rows had been learnt by
heart (held-out 0.915 against the booster's 0.953). The shrink and the
standardisation are folded into the stored weights, so inference is one dot
product on the raw features. The per-field surprises and the latent mean are
deliberately not inputs: unbounded and time-drifting, they pulled the blend
below both of its components on tabformer's newest months (see
TRAINING_NOTES); they remain the receipt's explanation of the novelty score.

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

v5 additions. `POST /api/score` takes, besides `features` and `source`:
`wide` (the source's raw columns by raw name, for the booster), `history` (the
customer's previous transactions, oldest first, each `{features, ts |
secs_before}`), `customer_id` (the server's **entity store** - SQLite in the
app home, one table per run - supplies the last `window` transactions it has
scored for that id and remembers this one; `remember: false` to only read),
and `ts` (this transaction's epoch seconds, default now). The response gains a
`sequence` block (history count and origin, the sequence head's probability
with its own calibration) and the booster's pushes may include `wide`
features. `DELETE /api/history/{source}/{customer_id}` forgets a customer. A
CSV through `/api/score/batch` carries no history: every row gets the
no-history context (a trained state) and a `sequence_prob` column.

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
