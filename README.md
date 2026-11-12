# FinFraud

[![tests](https://github.com/deanOcoin/fraudvae/actions/workflows/tests.yml/badge.svg)](https://github.com/deanOcoin/fraudvae/actions/workflows/tests.yml)
[![pretrained bundles](https://github.com/deanOcoin/fraudvae/actions/workflows/pretrained.yml/badge.svg)](https://github.com/deanOcoin/fraudvae/actions/workflows/pretrained.yml)
[![uptime](https://github.com/deanOcoin/fraudvae/actions/workflows/uptime.yml/badge.svg)](https://github.com/deanOcoin/fraudvae/actions/workflows/uptime.yml)
[![live site](https://img.shields.io/badge/live-finfraud.dev-2563eb)](https://www.finfraud.dev)
[![pretrained models](https://img.shields.io/badge/pretrained-8%20generations-10b981)](https://github.com/deanOcoin/fraudvae/releases)
[![python](https://img.shields.io/badge/python-3.10%2B-3776ab)](pyproject.toml)
[![pytorch](https://img.shields.io/badge/pytorch-2.4%2B-ee4c2c)](https://pytorch.org)
[![lightgbm](https://img.shields.io/badge/lightgbm-4%2B-5e8b2f)](https://lightgbm.readthedocs.io)
[![license: MIT](https://img.shields.io/badge/license-MIT-lightgrey)](LICENSE)

Fraud screening for payment transactions: a variational autoencoder that learnt what
normal payments look like, a boosted model that learnt what known fraud looks like,
and a per-bank blend of the two, trained on a merged corpus of eleven public
transaction datasets (~43M rows); since v6 the boosted model also reads a
transformer pretrained on unlabelled payment histories. With a web UI to score new transactions, browse
the data, train locally, or push a training run to a rented GPU over ssh. The
Python package and its command are called `fraudvae`.

**Try it without installing anything:** [finfraud.dev](https://www.finfraud.dev)
runs this repository in its public mode, with every generation of the model
loaded - score a transaction on any of them ([score.finfraud.dev](https://score.finfraud.dev)),
compare them side by side, read how they were built
([story.finfraud.dev](https://story.finfraud.dev), also as a [thirteen-page PDF](docs/FinFraud-development-process.pdf)) and how they work
([define.finfraud.dev](https://define.finfraud.dev)). **Every pretrained generation is on the
[Releases](https://github.com/deanOcoin/fraudvae/releases) page**, and
`fraudvae models pull v7-f` drops the shipped one into `runs/` so the app
below scores with it in minutes; training it yourself is optional and
documented, not required.

Two models look at every transaction and a small per-bank blender combines
them - and, since v5, the signals of five more modules - into one probability:

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
- **v5**: the VAE reads the customer's previous transactions through a small
  transformer, so "unusual" means unusual *for this customer*; a per-bank
  specialist booster, a device head and an entity-graph GNN add their own
  signals to the blend; text heads (urls, memos) and behavioural-biometrics
  verifiers (mouse, keystrokes, touch, motion) trained on public data are fused
  into the verdict when a request carries those payloads.

![score tab](docs/img/score.png)

<table>
  <tr>
    <td align="center"><a href="docs/img/data.png"><img src="docs/img/data.png" width="290" alt="data explorer"></a><br><sub><b>Data</b> · browse the corpus, per-source</sub></td>
    <td align="center"><a href="docs/img/model.png"><img src="docs/img/model.png" width="290" alt="model and runs"></a><br><sub><b>Model</b> · generations, timeline, results</sub></td>
    <td align="center"><a href="docs/img/explain.png"><img src="docs/img/explain.png" width="290" alt="how it works"></a><br><sub><b>How it works</b> · the models, explained</sub></td>
  </tr>
  <tr>
    <td align="center"><a href="docs/img/compute.png"><img src="docs/img/compute.png" width="290" alt="compute and remote training"></a><br><sub><b>Compute</b> · benchmark, rent a GPU, train there</sub></td>
    <td align="center"><a href="docs/img/story.png"><img src="docs/img/story.png" width="290" alt="the story"></a><br><sub><b>Story</b> · how it was built, step by step</sub></td>
    <td align="center"><a href="docs/img/model_results.png"><img src="docs/img/model_results.png" width="290" alt="per-source results table"></a><br><sub><b>Results</b> · every generation, every bank</sub></td>
  </tr>
</table>

## Contents

[Quick start](#quick-start) · [What's in the corpus](#whats-in-the-corpus) · [The models](#the-models) · [Results](#results) · [The generations](#the-generations) · [Hosting it](#hosting-it) · [Remote training](#remote-training) · [Layout](#layout) · [Credits](#credits) · [License](#license)

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# score with the pretrained generations (no data needed; the example rows ship with the repo)
fraudvae models pull v7-f          # the shipped generation, ~845 MB (513 MB of it the pretrained encoder); `--all` for every one on the timeline
fraudvae serve                     # http://127.0.0.1:8765

# or train your own
fraudvae fetch                 # clones the github hosted sources (fdh, capone, tabformer)
#   kaggle ones: see scripts/kaggle_manual_download.md, drop the zips in data/raw/<key>/
fraudvae build                 # raw -> data/processed/<source>.parquet  (unified schema)
fraudvae prepare --max-rows-per-source 1500000     # parquet -> model arrays, train capped per source
fraudvae train --config cpu_small                   # VAE + booster + calibration + eval, ~1h on a laptop cpu
```

`make data build prepare train serve` does the same. `make test` runs the suite
(synthetic data, a 2 epoch model, the API, an in-process ssh server).

A faster machine gives you the same model sooner, not a better one: the
configuration that ships (`gpu_full`) is fixed, and a CPU gets there in days
where the rented H200 took three hours. `cpu_small` is a different, smaller
configuration (0.6M parameters on a capped corpus) that a laptop finishes in
half an hour - within 0.003 mean AUROC of the full run for v3 and v4, and
clearly below it for v5, whose history encoder needs the full corpus's long
customer sequences. `docs/TRAINING_NOTES.md` has every run's hardware and time;
the site's story page has the same as a table.

Python 3.10+, PyTorch 2.4+, LightGBM 4+. CPU is fine for the capped set, MPS
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
| baf | Bank Account Fraud (Feedzai) account applications with device telemetry (v5) | 1.0M | 1.10% | kaggle |
| fraudecom | e-commerce first purchases with shared device ids (v5) | 151k | 9.4% | kaggle |
| amlworld_li_small, _hi_medium, _li_medium, _hi_large, _li_large | the other five AMLworld variants (v6, pretraining corpus; not in a shipped generation) | 6.9M / 31.9M / 31.3M / 180M / 176M | 0.05-0.12% | the same kaggle zip |
| fdh_sim, sparkov_sim | the two open-source generators behind fdh and sparkov, run locally (v7, pretraining only; never in the evaluation frame; `fraudvae simulate` writes fdh_sim's rows at ten million a minute) | as generated | as generated | local |

No dataset is redistributed here: each keeps its own licence (`docs/DATA_SOURCES.md`
has the links). All sixteen are public (the five v6 variants come out of the zip the ninth already needs). The GitHub-hosted ones need nothing; the Kaggle
ones need a free Kaggle account (IEEE-CIS additionally means joining its competition
and accepting the rules, and is non-commercial), and Kaggle refuses some VPN exit
addresses. `fraudvae fetch` downloads what it can, says what to fetch by hand and where
to put it (`scripts/kaggle_manual_download.md`), and then checks every file against
`fraudvae/data/expected.json` - the size and sha256 of each raw file the shipped
generations were built from (`fraudvae fetch --verify` does only the check). That
check is the part that matters: datasets get re-uploaded and re-cut, and a file that
differs will build and train but will not reproduce the numbers in the docs. The six
sources whose licences allow it ship a few hundred labelled rows from the held-out
test split in `data/examples/` (`fraudvae examples` regenerates them from a build),
which is what the score tab's example picker uses on a machine without the data - the
demonstration site included.

A source that cannot be public - a partner's transactions - is an adapter module in a
directory named by `FRAUDVAE_ADAPTERS` (v7, `fraudvae/data/private.py`): it joins the
source list after the public ones and goes through `fetch`, `build` and `prepare` like
any other, and nothing of it enters this repository.

Every source is mapped onto one schema (`fraudvae/data/schema.py`, v5): 95
numeric fields (amount, hour, the customer's history in the last hour / day /
week / month and their own amount z-score, tenure, the merchant's volume and
typical amount, balances, limits, distances, and the v5 *graph block*: each
entity's 30-day volume, distinct customers seen with it, age and
30-day-delayed confirmed fraud for the counterparty, device, e-mail domain,
address and card), 24 categoricals (channel, transaction type, an 18-way
merchant category taxonomy, card brand/kind, auth errors, first-time flags,
payment format, OS and browser family...), plus a source id and six entity
keys (customer, counterparty, card, device, email, address) that drive the
sequence index and the graph block but never enter a model. Each source may
also carry a *wide block* of its own raw columns (IEEE-CIS: 431 of them)
stored beside the common table and read only by the booster. Fields a source
can't fill are *unobserved* and masked out of the likelihood, never imputed.
All context features are computed causally per source. Splits are by time
within each source: oldest 70% train, 10% validation (calibration), newest
20% test. Details, licenses and per-source quirks: `docs/DATA_SOURCES.md`,
`docs/SCHEMA.md`; the v5 plan and data assessment: `docs/ROADMAP_V5.md`.

## The models

**Novelty (VAE).** A heterogeneous-likelihood VAE (HI-VAE style): MLP encoder
over `[standardised numerics, mask bits, embedded categoricals, embedded
source]` into a 16-24-d Gaussian posterior, mirrored decoder with a Gaussian
head per numeric field and a softmax head per categorical. Trained on
legitimate rows only with β warm-up and source-token dropout (so a "generic"
bank context exists). Scoring is a Monte Carlo −ELBO plus a per-field
normalised variant; calibration picks the better one on validation.

**Customer history (v5).** A small transformer, the history encoder, reads
the customer's previous 16-32 transactions (same field encoding, plus the
log time gap to each) into a vector `h` that conditions both the encoder
`q(z|x,h,s)` and the decoder `p(x|z,h,s)`: "unusual" becomes unusual *for
this customer*. 10% history dropout keeps a learned no-history token
trained for first transactions and callers that send none. A sequence head
(MLP on `[h, embedding of x]`, trained on the labelled subsample) turns the
same read-out into a fourth supervised signal. The server keeps a per-customer
store of what it has scored, so the API needs only a `customer_id` (or an
explicit `history` list) to score with context.

**Transaction encoder (v6, v7).** A self-supervised
transaction-sequence model in the shape of the published payments foundation
models: each row becomes (field, value) tokens with learned quantile buckets, an
event encoder turns a row into a vector, a causal history encoder reads the
customer's last 64-128 events, and two label-free objectives train it - predict
every field of the next event from the past (its per-field negative log
likelihood is the *surprise*), and reconstruct masked tokens, events and fields.
`fraudvae pretrain` runs it on every row of the corpus, on one process or under
`torchrun` across a node, keeping each process's rows on its GPU as token ids.
The booster can read its block (state, per-field surprise, mean surprise) as
features; the post stage computes it for every row of a prepared set in one pass
over whole sequences (`fraudvae txfm-cache`) and the receipt shows the most
surprising fields for this customer.
Two generations carry it. `v6-base-e` hands the block of a 128M-parameter
encoder (60,000 steps on sixteen sources, no labels) to the booster for every
bank: tabformer gains a great deal, three banks lose a little. `v7-f` routes the
block to the two banks where it earns its place (`txfm.sources`), adds a
supervised head on it as one more blend signal (`txsup`), and fits the blend
without the VAE's columns (`boost.blend_without_vae`); it is what ships. The
encoder reads a payment against the customer's earlier ones, so those have to
come with it: the example rows carry their customer's last eight, and a payment
whose own fields count earlier payments but which arrives alone is scored by
`v7-f` without the block rather than as a first payment.
`docs/ARCHITECTURE.md` (v6) has the design and `docs/TRAINING_NOTES.md` the
rounds, the controls and what serving it took.

**Supervised (booster).** A LightGBM classifier on the same fields (NaN where
unobserved) plus the source token and the source's wide block, trained on the
labelled train split with legit rows subsampled per source and re-weighted
(capped, so no bank outweighs the others), early-stopped on the mean of the
named-bank and source-blanked validation AUCs, with a 25% source-blanked slice
for a generic mode. It never sees a VAE output, which keeps the stacking
honest.

**Device head and entity graph (v5).** A small MLP over the device block alone
("what a risky device looks like", learned across every source that has one) and,
per source, a GraphSAGE over the entity graph - customers, counterparties, devices,
emails, addresses, cards as nodes, transactions as edges, three time snapshots so a
row never sees its own label - whose transaction head reads a row's entities'
embeddings. Each is one more calibrated signal with a presence bit.

**Text and behaviour (v5).** fastText-style heads trained on public text-fraud
sets (malicious URLs, scam job postings, bot profiles) read a `text` payload;
behaviour verifiers trained on public biometric sets (Balabit, BB-MAS, HMOG)
compare a `session` of raw mouse / keystroke / touch / motion events with the
customer's enrolled profile. Both return calibrated likelihood ratios that are
*fused* into the verdict (added to the blend's log-odds with a fixed weight),
because no transaction row ever carried text or a session.

**Blend.** One logistic regression on bounded signals - booster logit,
VAE score z, KL z and (v5) the sequence, device and graph logits, each with a
presence bit - with a source one-hot and the signals × one-hot, fit on labelled
validation rows. From v7 the two VAE columns are left out (the VAE explains a
payment on the receipt and does not vote) and the encoder's supervised head
joins the signals. The interaction terms give every bank its
own booster-vs-novelty weighting. Calibration then scores the whole
validation split once and records, per bank and for the generic context,
operating thresholds at 0.1/0.5/1/2/5% false positive rate for every
score. The app flags on the blend; the receipt shows them all. The recall and
precision a receipt quotes for a threshold are what that threshold did on the
held-out test split (`eval.json`, `at_val_thresholds`); the figures measured on
the validation rows the threshold was set on are kept as `expected_*` and
labelled in-sample, because at 0.1% FPR a threshold is the n-th highest of a
few thousand legitimate scores and fitted to noise. Every test row's scores are
kept beside the receipt (`scores_test.npz`, and `scores_test_delay30.npz` for the
30-day pass), so two runs on the same rows can be compared row by row rather
than by their summary numbers: `fraudvae compare A B` gives, per source and for
the mean, the change in AUROC, recall and precision at the alert budget with a
paired-bootstrap interval and a verdict.

`docs/ARCHITECTURE.md` has the longer version, the UI's "How it works" tab
the friendly one.

## Results

Held-out test split: the newest 20% of every source, 8.57M rows, 27k fraud,
thresholds frozen on validation. `v7-f` is the full 30M-row v5 corpus trained on
a rented H200 through the relay (30 epochs of the history-conditioned VAE in
2.9 h, run `v5-h200-e`) with the booster side refit on a rented 2× H200 box in
91 minutes: the booster reading the pretrained encoder's block for tabformer and
fdh, the specialists, the sequence / device / graph heads and the encoder's
supervised head, the blend without the VAE's columns, calibration and evaluation. Two blend columns: as trained, for a deployment whose fraud labels
arrive within a week, and the same model scored as one whose labels take a month
(the one-week fraud-history block blanked - `label_delay_days` on the API, the
"labels arrive within" control in the UI). The last two columns are `v5-h200-h`,
the generation that shipped before it, on the same rows. `runs/` is not
in git: `fraudvae models pull --all` fetches every generation on this timeline from
the Releases page (`fraudvae/pretrained.json` is the manifest it verifies against),
or train your own with the commands above.

| source | VAE | booster | seq | local | history head | device | graph | **blend** | r@1% FPR | 30-day labels: blend / r@1% | h blend | h r@1% |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fdh | 0.624 | 0.903 | 0.656 | 0.907 | 0.693 | – | 0.501 | **0.907** | 79% | 0.686 / 35% | 0.905 | 78% |
| capone | 0.613 | 0.837 | 0.815 | 0.839 | – | – | 0.767 | **0.841** | 21% | 0.838 / 20% | 0.841 | 21% |
| tabformer | 0.940 | 0.996 | 0.985 | 0.998 | 0.974 | – | 0.815 | **0.998** | 95% | 0.996 / 91% | 0.994 | 86% |
| ulb | 0.962 | 0.986 | 0.987 | 0.985 | – | – | – | **0.985** | 89% | 0.985 / 89% | 0.984 | 89% |
| paysim | 0.798 | 1.000 | 0.991 | 1.000 | – | – | 0.554 | **1.000** | 100% | 1.000 / 100% | 1.000 | 100% |
| sparkov | 0.862 | 1.000 | 0.996 | 1.000 | – | – | 0.953 | **1.000** | 100% | 1.000 / 100% | 1.000 | 100% |
| banksim | 0.990 | 1.000 | 0.999 | 1.000 | – | – | 0.998 | **1.000** | 99% | 0.999 / 99% | 1.000 | 99% |
| ieee | 0.795 | 0.957 | 0.904 | 0.955 | – | 0.733 | 0.825 | **0.958** | 63% | 0.940 / 56% | 0.958 | 63% |
| amlworld | 0.912 | 0.990 | 0.984 | 0.990 | – | – | 0.846 | **0.990** | 86% | 0.990 / 86% | 0.990 | 85% |
| baf | 0.658 | 0.892 | 0.812 | 0.891 | – | 0.768 | – | **0.892** | 26% | 0.892 / 26% | 0.893 | 25% |
| fraudecom | 0.643 | 0.643 | 0.644 | 0.639 | – | 0.639 | 0.516 | **0.641** | 10% | 0.641 / 9% | 0.641 | 10% |
| **mean (11)** | 0.800 | 0.928 | 0.889 | 0.928 | 0.833 | 0.714 | 0.753 | **0.928** | | 0.906 | 0.928 | |
| **mean (v4's 9)** | 0.833 | 0.963 | | | | | | **0.964** | | 0.937 | 0.963 | |
| generic (bank unknown) | 0.625 | 0.970 | 0.930 | – | – | 0.737 | – | **0.971** | 72% | 0.955 / 64% | 0.969 | 70% |

AUROC unless marked; r@1% FPR is the share of fraud caught at a 1% false-positive
budget; `seq` is the sequence head on the history encoder, `local` the bank's own
specialist booster, `history head` the supervised head on the pretrained
encoder's block (the two banks it is routed to), `device` and `graph` the
device-block and entity-graph heads (only where a source carries those columns).
What v7 changed against `v5-h200-h`, each difference from a paired bootstrap over
the same test rows (`fraudvae compare`): the mean share of fraud caught at 1% FPR
is up 0.011 [0.009, 0.013], and 0.018 [0.016, 0.021] when labels take a month;
tabformer goes from 86% to 95% (73% to 91% with month-old labels, because the
encoder reads payments, not labels), fdh from 78% to 79%, amlworld from 85% to
86%, and no bank is worse beyond a few millionths of AUROC on paysim at its
ceiling. The mean over eleven hardly moves (0.9277 to 0.9283): eight of them were
already where they are. `v6-base-e`, the block for every bank, is the trade that
came first: tabformer 94%, and capone, ieee and fraudecom a little worse by an
interval. Since v4 (0.916 over its nine sources) the honest reading has been that
the new columns are the step change - the entity-graph block with its delayed fraud labels lifted
Capital One from 0.77 to 0.84 (8% → 21% of its fraud at 1% FPR); the same labels
at a one-week delay with 7 / 30-day windows (the Fraud Detection Handbook's
assumption about label latency, booster-only, `docs/SCHEMA.md`) took fdh from
0.67 to 0.91 (34% → 78%) and tabformer from 73% to 86% of its fraud; the graph
block plus the wide blocks took IEEE from 0.86 to 0.96 (35% → 63%) - while the
extra models are second opinions that the blend uses where they help and ignores
where they do not. The month-latency column is the same model with the one-week
block blanked, which the booster was trained to expect: it lands where the model
stood before that block existed. The simulators (PaySim, Sparkov, BankSim)
saturate because their fraud rules are learnable; fraudecom's fraud is a shared
device, and 95% of its test rows are a device seen for the first time. The
modules trained on their own public data keep their own numbers
(`docs/TRAINING_NOTES.md`): the URL head 0.994 AUROC, the free-text head 0.972,
the profile head 0.783; the behaviour verifiers 0.11 (keystroke), 0.12 (motion),
0.17 (touch) and 0.24 (mouse) equal error rate; the graph encoder 0.78 / 0.88 /
0.92 / 0.98 on DGraph-Fin / Elliptic / YelpChi / Amazon. `docs/TRAINING_NOTES.md`
has the per-run commentary, including the features that made tabformer worse
before they made everything else better, the blender that memorised ULB's 21
validation fraud rows, why a 30-day-old label can never catch a terminal that is
compromised for 28, and the two things that did not help (the transformer's
summary as booster input, a longer VAE).

## The generations

Eight trained generations are kept, so the model's progress is something you can score
a transaction on rather than read about: the score tab's **Model generation** picker
takes any of them and **Compare every version** scores one transaction on all
eight. The UI's development-process tab (and `docs/TRAINING_NOTES.md`) tells what changed at each step
and what did not work; the model tab's timeline shows each one's hardware, training
time and numbers.

| generation | schema | trained on | time | mean AUROC, 9 shared banks |
|---|---|---|---|---|
| `cpu_small-8src` | v3 | 2 CPU cores, 8 sources capped at 1.5M rows | 33 min | 0.829 (its 8) |
| `gpu_full-4090` | v3 | 1× RTX 4090, 8 sources, every row | 19 min | 0.833 (its 8) |
| `gpu_full-4090-v4` | v4 | 1× RTX 4090, 9 sources, every row | 18 min + 8 min post | 0.916 |
| `v5-h200-e` | v5 | 1× H200 NVL, 11 sources, every row | 2.9 h + 40 min post | 0.936 |
| `v5-h200-f` | v5 | e's VAE, booster side refit (one-week labels) | + 30 min post | 0.961 |
| `v5-h200-h` | v5 | e's VAE, booster side refit (recent-fraud windows, latency dropout) | + 41 min post | 0.963 |
| `v6-base-e` | v5 | e's VAE; a 128M-parameter transaction encoder pretrained on 8× B200 (16 sources, no labels), its block for every bank | 4.5 h pretraining + 72 min post | 0.963 |
| `v7-f` | v5 | e's VAE and v6's encoder; the block for tabformer and fdh, a supervised head on it, the blend without the VAE (2× H200 NVL) | + 91 min post | **0.964** |

## Hosting it

`docs/DEPLOY.md`: `FRAUDVAE_PUBLIC=1` turns the app into the demonstration site ([finfraud.dev](https://www.finfraud.dev)) -
no training, no data build, no remote routes, no visitor state, rate limits per
address - and `deploy/` has the install script, the systemd unit and the nginx
configuration the site runs on. `SECURITY.md` says what the site exposes, what it
does not, and how the server is held in place; it also covers the one thing worth
knowing on a laptop, which is that a run directory is read with torch's
`weights_only` loader and the pretrained bundles are checksummed before they are
unpacked.

## The workbench on a network

Workbench mode has no login: it is built to be reached from the browser on the same machine,
and `fraudvae serve` binds to 127.0.0.1. Three things stand in for a login, and it is worth
knowing what each one does before putting the server anywhere else:

- The server answers only to its own names (`localhost`, `127.0.0.1`, the address it was
  bound to, and whatever `FRAUDVAE_ALLOWED_HOSTS` adds). A request for any other host name
  is a 400, which is what stops a hostile page from reaching it through DNS rebinding.
- The API refuses requests the browser marks as coming from another site (the
  `Sec-Fetch-Site` and `Origin` headers), so an image tag or a form on some other page cannot
  make your browser train a model, upload a file or touch a remote box here.
- `/api/remote/*` - the Compute tab's backend, which runs commands on a rented machine as
  root with the app's own key - answers only to connections from the local machine. To use it
  from another computer, start the server with `FRAUDVAE_TOKEN=<a long random string>` and the
  UI will ask for the token once per browser session; programs send it as `X-Fraudvae-Token`.
  Without the token the rest of the workbench works over a LAN; the remote routes give a 403.

One more thing a workbench does that the public site does not: when a request names a
`customer_id`, the scored transaction's model-space encoding is kept in
`~/.fraudvae/entity_store.sqlite` (the newest 1,000 per customer, `FRAUDVAE_HISTORY_MAX_ROWS`)
so the next request for that customer is scored with history. `remember: false` reads without
writing, and `DELETE /api/history/{source}/{customer_id}` forgets a customer.

The public site runs with `FRAUDVAE_PUBLIC=1`, which removes the remote routes and every
other mutating endpoint altogether (`SECURITY.md`).

## Remote training

The full corpus is ~29M legitimate rows per epoch; a laptop does it in days,
an RTX 4090 in about a minute per epoch. The *Compute* tab benchmarks the real model on your machine,
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
  data/        registry (sources, licenses), aux_registry (the text / behaviour / graph sets), schema,
               taxonomy, velocity features, entities.py (graph block), adapters/<source>.py,
               fetch / build / prepare, examples.py (the redistributable example rows),
               expected.json (the raw files' checksums), corpus_stats.json
  models.py    pretrained bundles: pack / pull against pretrained.json (github releases)
  model/       vae.py (model + history encoder), scoring.py, booster.py,
               seqhead.py (sequence head), heads.py (device head), dataset.py (+ the sequence index),
               txfm.py (v6 transaction encoder: tokeniser, model, scorer), txfmhead.py (v7: supervised head on its block), txfmft.py (v7: the encoder fine-tuned on the labels)
  graph/       entity-graph snapshots, GraphSAGE, per-source training, scorer, public benchmarks
  text/        hashed-bag text heads (url / free text / profile), training, scorer
  behaviour/   mouse / keystroke / touch / motion features, verifiers, training, scorer
  train.py  pretrain.py (v6)  txfm_cache.py (v6: the encoder's features for a prepared set, once)  calibrate.py  evaluate.py
  compare.py (two runs on the same rows, paired)  manifest.py (what a run is made of, hashed)  config.py
  remote/      keys, ssh (paramiko), provision (incl. raw-data push + box-side pipeline),
               jobs, advisor, relay (a git repo as the laptop-off control channel)
  server/      fastapi app + api modules, inference (the loaded generations + fusion), timeline,
               public (the demonstration-site mode), entity_store, payloads
  web/         the ui: index.html, css/, js/, explain.html, story.html + story_plain.html, vendored libs
  configs/     base, cpu_small, gpu_full, gpu_hist, gpu_long, smoke (yaml; `train --config NAME`);
               v6: txfm_small, txfm_base (`pretrain --config NAME`), v6_full; v7: v7a-v7d (the Session A variants, `post --config NAME`)
data/examples/ labelled example rows for the six sources whose licences allow it
deploy/        install.sh, update.sh, the systemd unit, nginx, env.example, relay_bootstrap.sh
scripts/       kaggle_manual_download.md, macos_openmp.sh, make_data_docs.py, story_pdf.py, bootstrap_remote.sh, relay_agent.sh
tests/         pytest, no real data needed; `make test-fast` skips the ones that train (CONTRIBUTING.md)
docs/
```

## Credits

The ideas are not new - heterogeneous-likelihood VAEs, gradient boosting on
causal velocity features, per-segment calibration, delayed-label features -
and `docs/TRAINING_NOTES.md` says where each came from; what is this project's
is the combination, the eleven-way schema, the latency treatment and the
receipts. A good part of the code was written with the help of an Anthropic
Claude model, the front end more than the training side; the design of the
models and the data, and every decision about what to build, were not the
model's.

## License

MIT for the code. The datasets keep their own licenses (listed per source in
`fraudvae/data/registry.py` and `docs/DATA_SOURCES.md`); IEEE-CIS in particular
is non-commercial.
