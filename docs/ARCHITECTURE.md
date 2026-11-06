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
   └─evaluate──▶ runs/<name>/eval.json + scores_test.npz                        (VAE / booster / seq / blend on test;
                                                                                   every row's scores, for `compare`)
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

**The label horizon** (v7, `boost.label_horizon_days`). A deployment trained at a
boundary did not yet have the labels of the last weeks before it: a chargeback
takes a month to arrive. With a horizon of N days, the train rows within N days
of their source's validation boundary leave the supervised fits (the booster,
the specialists, the encoder head) and the validation rows within N days of the
test boundary leave the blender's fit; `booster.json`, `specialists.json`,
`txfmhead.json` and `calibration.json` record the horizon and how many rows it
cost. The thresholds still read every validation row, and the delayed-label
evaluation (`eval_delay30.json`) remains where late labels are judged: the
horizon is about what the models were allowed to learn from, the evaluation
about what they are shown. 0 keeps every labelled row, as every published
generation did; 30 is what a card deployment lives with.

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

## v6: the transaction encoder

`fraudvae/model/txfm.py` (Tokeniser, TransactionEncoder, TxfmScorer),
`fraudvae/pretrain.py`, `fraudvae/configs/txfm_small.yaml`, `txfm_base.yaml`, `v6_full.yaml`

The v5 history encoder is 0.1M parameters trained as a side effect of the ELBO on
legit rows, and the blender gives the novelty score it feeds almost no weight. v6
adds the thing every published payments foundation model has converged on
(Revolut's PRAGMA, Visa's TransactionGPT, Sardine's issuer model): a
self-supervised **transaction-sequence encoder**, pretrained on every row of every
source with no label in sight, whose output is read by the booster.

**Tokens.** A row becomes a bag of (field, value) tokens. Each numeric field owns
`n_bins` quantile buckets (edges fitted on a sample of train rows, standardised
units) plus an "unobserved" id; each categorical its own value index (0 =
unobserved); then the source and the time since the customer's previous event in
log buckets (0 = no previous event). Every field owns a slice of one vocabulary
(~3-4k ids for the v5 schema), so one embedding table and one output layer serve
all of them, and a prediction is always a softmax *restricted to the field's own
slice*. A field embedding is added so a masked token still knows which field it
is.

**Model.** An event encoder (a small bidirectional transformer over a row's
tokens with an [EVT] read-out; `event_layers: 0` swaps in a mean-pool + MLP)
turns each event into one vector; a causal history encoder reads a customer's
events oldest to newest with a learned position and a log-time term, after a
learned BOS, so `states[:, t]` has seen events strictly before t and
`states[:, t + 1]` has seen event t. Two heads train it:

- *next-event*: from `states[:, t]`, the token of every observed field of event t
  (cross-entropy within the field's slice). Its per-field negative log likelihood
  is the **surprise** - how unexpected this amount / hour / merchant category is
  for this customer, given their past alone.
- *masked-field*: PRAGMA's three masking levels (15% of tokens, 10% of whole
  events, 10% of whole fields across the window; source and dt are never masked)
  and the hidden tokens are predicted from their token output plus the state
  after the event.

**Pretraining** (`fraudvae pretrain`, or `torchrun --nproc_per_node N -m
fraudvae.pretrain`) is step-based: a sample is a window of `window` consecutive
events of one customer ending on a train-split row, and every position in it is
a target. The rows a process needs are read from the prepared memmaps once,
tokenised to int16 ids and kept on the device; a window is then an on-device
gather through the sequence index, which is what removes the cpu gather that
made the v5 history model data-bound. Under torchrun each process takes a
disjoint share of the customer blocks (so its windows never leave its own rows)
and gradients are averaged once per step. Validation is the next-event nll of
the last event of windows ending on val rows; the best checkpoint is `txfm.pt`
(weights + tokeniser + prepared meta, read with the weights-only loader).

Four things about the loop that the first node run taught (`docs/TRAINING_NOTES.md`, v6):

- *Packing.* A source carries 18-57 of the schema's 121 fields; the rest would be
  "unobserved" tokens on every row, three quarters of the event encoder's work for
  nothing, and a trivial target for the masked-field loss. The tokeniser keeps the
  fields each source ever observes (`source_fields`, scanned from the train split
  at fit time and saved in `txfm.pt`) and the event encoder reads only those
  (`Tokeniser.fields_for`); the losses index the same packed layout. A row's own
  missing values inside that list stay as tokens - a blank e-mail on a source that
  usually has one is signal. A batch that mixes sources reads the union; a blanked
  source reads every field.
- *One source per step.* A step's windows come from one source, drawn with
  probability proportional to its rows to the power `source_alpha` (1 = by size, as
  drawing rows at random would; 0.5 = square root, what `txfm_base` uses so the 431M
  rows of one synthetic generator, 92% of the corpus, do not set the whole curriculum). The batch then
  runs at that source's packed width. Validation windows are the same number from
  every source, and `metrics.jsonl` keeps the next-event nll per source.
- *The state before a first event.* `states[:, t]` for a customer's first event is
  read off the slot before it, which in a training window holds another customer's
  row, at serving time a padding row and in the block pass (below) the previous
  customer's last event: three different inputs for one quantity. The history
  encoder therefore replaces every invalid slot's state with the start token's, so
  the prior for a first transaction is one trained vector, the same everywhere.
- *Resume.* `state.pt` (weights, optimiser, step, best) is written at every eval;
  `--resume` with the same `--name` carries on, so a run can be gated at 10k steps,
  judged with the booster, and continued rather than restarted. Two additions from the
  node run (v7): `save_every` writes a full `txfm_step<N>.pt` on a step cadence whatever the
  validation says (the 32k snapshot had to be rebuilt from `state.pt`, because `txfm.pt` only
  moves on a new best), and `probe: tabformer` reads, at every eval, a logistic regression on
  the frozen state of that source's val windows, fraud against legit, into `metrics.jsonl` as
  `probe_auroc` - the downstream curve a pretraining is gated on, since the next-event nll
  kept improving from 32k to 60k steps while the detector built on the state did not move.
- *Only where there is a history.* With `txfm.gate_first` the block is NaN for a
  customer's first event: there the state is the start token's and the surprise is the
  marginal of the row's own values, an echo of fields the trees already see, and on a
  source with no sequences at all (ulb) the gate round's booster overfitted that echo.
  The booster records the choice in `booster.json` (`txfm_gate_first`) so calibration,
  evaluation and the server blank the same rows; `v6_full_gated` turns it on.
- *Only where it earns its place* (v7). `txfm.sources: [tabformer, fdh]` routes the block to
  the named sources and hands the booster NaN for every row of the others, on the block pass,
  the cache and the server's window path alike, recorded in `booster.json` (`txfm_sources`)
  like the gate. The node run's gain was one source's (tabformer, the long card histories the
  encoder was trained on) while 888 mostly uninformative columns cost capone and ieee a few
  splits' worth of accuracy in every round; routing is the cheapest answer to that, and the
  round that settles whether it was dilution is in `docs/TRAINING_NOTES.md` (v7).
- *Fewer columns* (v7). `boost.txfm_pca: 64` hands the trees 64 principal components of the
  block plus the mean surprise instead of its 888 columns. The shared booster fits the
  compression on a sample of its train rows (`txfm_pca.npz` in the run), the specialists read
  the same file, and a booster given the raw block - by `score_split` or the server - compresses
  it itself (`Booster._x`), so the scorer, the cache and the receipt's surprise block stay as
  they are. A row whose block is NaN throughout stays NaN. The other answer to dilution; the
  node run's booster took a few hundred splits from the block, which 65 columns can carry.
- *The control* (v7). `txfm.permute_block: true` hands the trees the block with its rows
  shuffled within each source (a fixed seed, before the gate and the routing, on the cache
  and the on-the-fly pass alike): the same marginals, no information about the row. A round
  built on it is the dilution control for the losses the node run saw on capone and ieee -
  if they lose the same amount with a block that carries nothing, the loss is the columns'
  weight on the trees, not the encoder's reading of those sources. Recorded in `booster.json`
  (`txfm_permuted`); the server's one-row path does not shuffle, and such a run is never served.

**Where it reaches the score.** `boost.txfm_features: true` with `txfm.run:
<pretraining run>` makes the post stage copy `txfm.pt` into the run and append
one block to the booster's (and each specialist's) features, after the v5
`hist_*` block: the state after the current event (`txfm_h*`, d_model columns),
the per-field surprise (`txfm_s_<field>`, NaN where the field is unobserved) and
the mean surprise (`txfm_surprise`). The block is computed with the bank blanked
for the generic slice like everything else; a caller without the encoder sends
NaN, which the trees treat as "no summary". The server computes the block for
one transaction from the entity store at the encoder's own window (`v6_full`
uses 128 against the VAE's 32). A whole split goes through the **block pass**
instead (`TxfmScorer.features_split`): consecutive events of the sequence index
in segments of `window` events, several customers per segment behind a block
mask (an event attends to earlier events of its own customer and to the start
token), every row read off the position it sits at. A segment starts
`window - stride` events before the first row it serves, so a row has at least
three quarters of a window of history (all of it when the customer has less),
and each event is encoded about four times instead of once per row. With stride
1 it is the server's window path to the last decimal; `tests/test_txfm.py` checks
that. `fraudvae txfm-cache` writes that pass once per (prepared set, encoder)
under `data/prepared/<set>/txfm_cache/<sha of txfm.pt>/`, one part per gpu if
wanted, and the booster, calibration and evaluation read it; the post stage
writes it itself when nothing has. The receipt carries
one `transaction_encoder` line in the booster's pushes and a `sequence_model`
block with the mean surprise and the most surprising fields - the explanation
Sardine's write-up says embeddings lack.

Why features into the trees and not a fine-tuned head first: it is the lowest-risk
of the three routes the literature reports (Sardine: +68% AUC-PR from exactly
this), it keeps the booster's SHAP receipt, and the v5 result that the history
read-out was "a wash" as booster input was for a 0.1M-parameter encoder trained
on another objective; whether a 100M-parameter next-event model changes that is
the first v6 experiment (`docs/TRAINING_NOTES.md`, v6). A fine-tuned head as a
blend signal and a TabICLv2 in-context signal per bank are the next two.

**The labels reach the state** (v7, `fraudvae/model/txfmhead.py`). The node run
answered the first experiment - a measured gain on the one source with long
labelled histories, nothing elsewhere - and showed that more pretraining does
not move it: the booster takes a few hundred splits from the block, and the
fraud labels never touch the encoder's state. `boost.txfm_head: true` (or
`fraudvae txfm-head RUN`) trains, per source with at least
`specialist_min_fraud` labelled fraud rows, a small MLP on the standardised
block - the same labelled subsample the booster uses, early stopped on that
source's validation AUROC - and its logit is the `txsup` blend signal, present
on the rows of a source that has a head and NaN elsewhere, exactly as the
specialists' `local`. It reads the block through the scorer (so the cache
serves it, and the gate and the routing apply), the encoder stays frozen, and
the receipt gets a `sequence_supervised` block with the signal's own operating
point. The fine-tune of the encoder's top layers is the step after this one,
taken only if this one earns it.

**Fine-tuning on the labels** (v7, `fraudvae txfm-finetune RUN --prepared SET --sources a,b,c
--name NAME`). Until here the labels never touched the encoder's weights: pretraining is
self-supervised and the supervised head reads a frozen state. The fine-tune unfreezes the last
two history layers and their norm, puts a small head (a learned source embedding beside the
state after the last event, one hidden layer) on labelled windows of the named sources - every
fraud train row plus legit rows at 20 to one, capped per source - and makes one pass with
class-weighted cross-entropy, the encoder layers at 3e-5 and the head at 1e-3. The checkpoint
with the best mean validation recall at 1% FPR is written as `runs/NAME/txfm.pt` in the
pretraining format, so `TxfmScorer`, `txfm-cache` and the post stage read it like any other
encoder (`extra` records the base checkpoint's sha256, the sources and the validation numbers;
`txfmft.json` keeps them beside it). The fine-tune head itself goes no further: the post stage's
booster and `txsup` head read the exported encoder's block, and the paired comparison against
the run built on the pretrained encoder is what decides whether the fine-tune earned its place.

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

**Without the VAE** (v7). `boost.blend_without_vae: true` fits the blender on the booster
logit, the one-hot, booster × one-hot and the extra signals alone, the VAE's two columns and
their interactions left out; `calibration.json` records it (`blend.without_vae`) and
`Scorer.set_blend` builds the same layout at serving time, `fraudvae txfm-head RUN` keeps the
run's choice when it recalibrates, and `fraudvae calibrate RUN --without-vae` is the same by
hand. It is the ablation that says what the novelty side is worth at the operating point
once the booster carries the encoder's block and the supervised head.

Pass 3 scores the whole validation split once (VAE, booster, blend) and
records per source and for the generic context, for each of the three
scores: legit quantiles, thresholds at FPR targets (0.1/0.5/1/2/5%) with
recall/precision, best-F1 threshold. A 2-d PCA of posterior means (fit on
legit rows) and a background sample (≤2500 legit + ≤1200 fraud per source)
go to `latent_map.npz` for the UI scatter.

`evaluate.py` applies the frozen validation thresholds to the test split and
reports, per source and for a generic subsample, realised FPR / recall /
precision alongside AUROC, AUPRC, recall@FPR for the VAE, the booster and
the blend. `mean_auroc` (what runs are ranked by) is the blend's. It also
keeps every test row's scores (`scores_test.npz`, `scores_test_delay30.npz`
for the delayed pass), and **`fraudvae compare A B`** reads two of them: the
same rows scored by two runs, judged paired with a Poisson bootstrap over the
rows (the same weights for both runs), per source and for the mean over
sources, on AUROC, recall at the FPR targets and precision at the first. Each
difference comes with a 95% interval and a call; a source's verdict reads the
recall at the first target, then the AUROC. The summary numbers of two
receipts cannot settle a difference of a few thousandths - the v6 node run
spent a night finding that out - but the rows can, because whatever the rows
themselves add to the variance cancels when they are the same rows.

Runs record the schema version they were trained on; the server refuses to
load one from another version and the UI greys it out.

**What a run is made of** (v7, `fraudvae/manifest.py`). The stages that leave
artefacts behind - pretraining, the post stage (also at the end of `train`), a
standalone `boost` or `evaluate` - write `manifest.json`: the code (version, git
commit, whether the tree was clean), the configuration and its hash, the
prepared set with the hash of its `meta.json`, and the sha256 of every model
file in the run, with earlier records kept under `history`. The post stage
also removes any evaluation the run already holds (`eval*.json`,
`scores_test*.npz`) before it writes its own. Both come from one night: a
checkpoint copied into a new run brought its old `eval.json` along, and a
round was read against it until the file's time stamp gave it away. A
comparison should be between things whose provenance is written down.

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
