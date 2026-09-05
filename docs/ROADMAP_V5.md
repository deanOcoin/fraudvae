# v5 roadmap: from a tabular hybrid to a multi-signal fraud system

Status: planning document, written before any v5 code. It is the honest
assessment asked for in October 2026: what the current model is limited by,
what each of the five proposed additions needs, which public data exists for
it, what our nine sources can and cannot contribute, and what compute and
storage the build needs. Newest decisions at the top of each section; the
TRAINING_NOTES file will keep the results as they come in.

## 1. Where v4 stands and what limits it

v4 (`gpu_full-4090-v4`): blend AUROC 0.916 mean over nine banks, 0.956 in the
generic mode, recall at a 1% false-positive budget of 73% on TabFormer, 84% on
ULB, 85% on AMLworld, 35% on IEEE-CIS, 34% on fdh, 8% on Capital One.

The three weak banks are weak for different reasons, and only one of them is
fixable with what we already have on disk:

- **IEEE-CIS (35%).** Card-not-present e-commerce, criminals and customers buy
  the same things. The raw files have 394 transaction columns and 41 identity
  columns; the v4 adapter maps **16 + 1** of them onto the common schema and
  discards the rest, including the whole device-telemetry block (`DeviceInfo`,
  OS `id_30`, browser `id_31`, screen `id_33`, the `id_*` network and
  identity flags), the 339 Vesta engineered `V` columns, the 14 `C` counts, 15
  `D` timedeltas, 9 `M` match flags, the second address, the recipient email
  domain and card2/3/5. The competition's top solutions reach ~0.945 AUROC on
  that file with all columns plus a "virtual customer id" (card1 + addr1 +
  day − D1); we are at 0.858 with 17 columns. **This is the single biggest
  known gap and needs no new data.**
- **fdh (34%).** A third of its fraud is "compromised terminal": invisible
  per transaction, visible only as a terminal whose fraud rate jumped. v4
  deliberately computed no label-derived entity features. v5 will, with a
  label delay (see graph features below), which is what a real bank has.
- **Capital One (8%).** Ordinary-looking purchases; the only strong column
  is the CVV mismatch flag. More context (customer sequence, merchant graph)
  will move it some; there is no device, text or session information in that
  file and the synthetic generator did not plant any. This one will stay
  the laggard until a richer source replaces it.

The general point for the data question: **columns beat rows here**. Three
GPU runs in a row showed 7× the rows did not move the density model and the
booster trains on 200k legit rows per bank regardless. What moved it was the
customer-context block (v4). Every item below is more columns of a new kind,
not more rows of the same kind.

## 2. The five additions, assessed

Ordered as proposed (most to least helpful). For each: what it is in a fraud
system, what data carries it, whether our sources carry it, and how it fuses
with the VAE core. "LLR fusion" means: the module is trained and calibrated on
its own data, emits a log-likelihood ratio, and the blend adds it to the logit
with a bank-specific weight when the signal is present and ignores it when
absent. It is the standard way to combine independent evidence when no single
dataset carries all signals, and it is the only honest option for the modules
whose public data has no transaction labels next to it.

### 2.1 Advanced device telemetry  — data in hand, highest leverage

What: OS, browser, device model, screen, language/timezone, user-agent
velocity (how many cards or emails seen on this device), IP-derived country
and ASN, proxy/VPN/datacenter flags, email domain class (free, disposable),
name–email similarity, account age, session length, keep-alive.

Data:
- **IEEE-CIS identity block** (ours; 41 columns, 144k of the 590k rows have
  it): DeviceType, DeviceInfo, id_30 OS, id_31 browser, id_33 screen, id_12–
  id_38 flags, id_19/id_20 (network-ish), plus P/R email domains, addr1/2,
  dist1/2 in the transaction file.
- **Bank Account Fraud (BAF, Feedzai, NeurIPS 2022)**: six variants × 1M
  synthetic-but-realistic account-opening applications with `device_os`,
  `session_length_in_minutes`, `keep_alive_session`,
  `device_distinct_emails_8w`, `device_fraud_count`, `source`
  (INTERNET/TELEAPP), `velocity_6h/24h/4w`, `zip_count_4w`,
  `email_is_free`, `phone_home_valid`, `phone_mobile_valid`,
  `name_email_similarity`, `foreign_request`, address tenure, credit score.
  Fraud ≈ 1%. Kaggle, CC BY-NC-SA.
- **fraudecom** (Kaggle "fraud-ecommerce"): 151k e-commerce purchases with
  `device_id`, `browser`, `ip_address` (+ an IP-to-country table),
  `signup_time`, `purchase_time`, age, sex, source. Fraud 10.6%. Small but
  it is the only public file with raw device ids shared across accounts.
- **ipblock** (CINS Army IP blocklist, free): real malicious IPs, for a
  proxy/datacenter/abuse reputation feature.
- Optional: **TalkingData AdTracking** (Kaggle, 7.5 GB, 184M clicks with ip,
  app, device, os, channel): click fraud, not payments, but the richest public
  device/IP velocity stream; useful for pre-training the device-risk model's
  velocity features. Skip unless disk is plentiful.

Fit with our sources: IEEE is the bridge: it has both the telemetry and
transaction labels, so the device module can be validated end to end on it.
BAF and fraudecom become sources 10 and 11 (application and e-commerce fraud)
and train the device-risk sub-model; their rows also widen what "generic"
means. The v5 schema gets a device block (≈20 fields) that is unobserved for
the seven sources without it, exactly like any other missing field today.

### 2.2 Sequential deep learning — data in hand, second-highest leverage

What: the customer's own history as a sequence, not as hand-made window
counts. A transformer over the previous L transactions of the same customer
(TabFormer's own paper was written on our tabformer file for exactly this),
producing a context vector that (a) **conditions the VAE** - encoder
q(z | x, h, s) and decoder p(x | z, h, s), so the novelty score becomes
"surprise given this customer's past", which is what fraud is - and (b) feeds
a supervised sequence head and the booster. The VAE stays the core; it gains
memory.

Data: seven of nine sources carry customer ids with real sequences: tabformer
(2,000 users, 30 years), fdh, sparkov (1,000 customers), banksim, capone,
amlworld (accounts) and paysim (weak: most senders appear once). ULB has no
customer id; IEEE gets the virtual id above (card1 + addr1 + D1), which is
how the competition solutions built sequences for it. No new data needed; the
cost is a per-customer sorted index in the prepared set and ~5-10× the VAE's
training compute.

### 2.3 Behavioral biometrics — public data exists for the module, not for the fusion

What: mouse dynamics, keystroke timing, touch/swipe geometry, device motion
during a session; the question is "is the person at the keyboard the account
holder". Account-takeover fraud is where it bites.

Data (all public, all authentication datasets, none attached to payments):
- **Balabit Mouse Dynamics Challenge** (GitHub, free): 10 users, remote
  desktop sessions, 816 labelled test sessions of which 405 are intruders.
  Raw events: time, button, state, x, y. This is the one dataset with explicit
  impostor labels, i.e. account takeover in miniature.
- **SU-AIS BB-MAS** (IEEE DataPort, free account): 117 users, keystrokes,
  mouse, swipes and gait on desktop, tablet and phone, same person across
  devices.
- **HMOG** (William & Mary, request form) and **BrainRun** (Zenodo): touch
  gestures plus accelerometer/gyroscope for 100 / ~2,000 smartphone users.
- **CMU keystroke benchmark** (free): 51 users typing one password 400 times.

Fit: none of our transaction sources has a behavioral stream and no public
dataset has both, so the module is trained as a verifier (genuine vs impostor,
reported as equal-error rate on its own data), exposed in the API as a session
payload, and fused as an LLR with a weight that is set from its own
calibration, not learned jointly. It will not change any number in the
held-out tables because the held-out data has no sessions; it is there for a
deployment that does. That is the honest limit and it is a hard one: there is
no public dataset of payments with keystrokes.

### 2.4 Deep text analytics — some data in hand, modest expected gain on payments

What: merchant descriptors, memos, product titles, email addresses, URLs,
support messages; and more broadly "sequences of user actions as a language"
(FraudNLP-style), which overlaps with 2.2.

Data:
- Ours: sparkov merchant names and job titles, capone merchant names,
  tabformer merchant city/state (the merchant name is already hashed),
  IEEE email domains, banksim categories. Thin, synthetic, but enough to
  learn a merchant-string encoder that transfers across banks.
- **malurl** (Kaggle, 651k URLs, 34% malicious), **fakejob** (Kaggle, 17.9k
  postings with description text, 4.7% fraudulent), **twitterbot** (Kaggle,
  37k profiles with description text): three supervised text-risk tasks from
  the Amazon Fraud Dataset Benchmark that give the text head real labels.
- A small pretrained sentence encoder (MiniLM / e5-small class, 22M params,
  runs on CPU) embeds any free-text field a deployer sends; the embedding
  (reduced to 16-32 dims) goes to the booster, the supervised heads give LLRs.

Fit: on our nine sources text will add little because there is little text;
it is built for the deployer that has memos and descriptors. The merchant
encoder is the part that pays on our data (generic mode: a merchant name the
model never saw).

### 2.5 Graph network — strong public data, directly relevant to AMLworld and IEEE

What: entities shared across transactions (card, device, email, address, IP,
merchant, counterparty account) as a heterogeneous graph; fraud rings show up
as structure (one device, many cards; fan-in/fan-out cycles; an address that
three confirmed frauds used). Two levels:
- **Causal graph features at build time** (no GNN): for every entity column,
  time-respecting degree and velocity (distinct cards per device in 7 days,
  distinct devices per card, distinct counterparties, cycle membership for
  transfers) and **delayed-label risk** (share of this entity's transactions
  older than a 30-day label delay that were fraud). v4 refused label-derived
  features; v5 allows them with the delay, which is what a bank actually has.
- **A GNN** (GraphSAGE / relational, PyTorch Geometric) over the entity
  graph, trained on the sources that have labels and structure, emitting a
  node-risk embedding and a per-transaction graph-risk logit.

Data:
- Ours: AMLworld is a transaction graph by construction (the laundering
  patterns are subgraph shapes); IEEE has card/addr/email/device entities;
  paysim has sender→receiver; fdh customer–terminal; tabformer, sparkov,
  banksim, capone customer–merchant.
- **DGraph-Fin** (registration at dgraph.xinye.com, non-commercial): 3.7M
  real users, 4.3M "emergency contact" edges, 15.5k fraudsters, 17 node
  features, timestamps. The largest real financial fraud graph in the open.
- **Elliptic** (Kaggle) and **Elliptic++** (GitHub): Bitcoin transaction
  graph with illicit labels, 203k nodes / 234k edges (Elliptic++ adds actor
  nodes and features).
- **YelpChi / Amazon fraud graphs** (DGL, data.dgl.ai): review–user–product
  multi-relation graphs with fraud labels; the standard GNN fraud benchmarks
  (e-commerce skew, as asked).

Fit: the causal graph features apply to every one of our sources and are
expected to move fdh (terminal risk), IEEE (device/email/address sharing) and
AMLworld (cycles). The GNN is validated on AMLworld, IEEE, DGraph, Elliptic
and Yelp, and fused as an LLR plus an embedding.

### 2.6 What is ruled out

Nothing on the list. Two things are ruled *down*: TalkingData is optional
(size vs relevance), and a joint evaluation of behavioral biometrics against
payment labels is impossible on public data, so that module's held-out number
will be an impostor-detection EER, not a fraud AUROC.

## 3. The v5 architecture

```
                 ┌────────────────────────────────────────────────────────────┐
  transaction ──▶│ common schema v5 (v4 fields + device block + entity ids)    │
                 │ + source-native wide block (all raw columns, NaN elsewhere) │
                 └──────┬───────────────┬───────────────────┬─────────────────┘
                        │               │                   │
   customer history ──▶ history transformer ──▶ h ──┐       │
                        │                           ▼       ▼
                        │                  conditional VAE  LightGBM booster
                        │                  q(z|x,h,s) p(x|z,h,s)   [common | wide | h | graph | text | device]
                        │                           │       │
   entity graph ──────▶ causal graph features ─────┼───────┤      GNN ──▶ graph LLR
   free text ─────────▶ sentence encoder ──────────┼───────┤      text heads ──▶ text LLR
   device payload ────▶ device block ──────────────┼───────┤      device head ──▶ device LLR
   session events ────▶ behavior verifier ─────────────────────────────────────▶ behavior LLR
                                                    │       │
                                                    ▼       ▼
                        blend v2: logistic over [booster, VAE, KL, seq, graph, device, text, behavior]
                        × bank, each signal with a presence indicator, LLR terms added when present
                                                    │
                                                    ▼
                                      p(fraud), receipt per signal
```

Decisions:
- The VAE stays the core and becomes history-conditioned. Novelty is defined
  relative to the customer, not only to the bank.
- The booster consumes everything observable. Missing stays missing.
- Every new module produces one calibrated signal for the blend and, where it
  has one, an embedding for the booster. A deployment with none of the new
  payloads gets exactly the v4 behaviour; one with all of them gets all of it.
- The server grows an **entity store** (SQLite) so history, device velocity
  and graph degrees can be computed online from the transactions it has
  already scored; the batch pipeline computes the same features offline.

## 4. Compute and storage

- **Disk.** New raw data ≈ 12 GB without TalkingData (BAF 2.5 GB csv, DGraph
  ~0.5, Elliptic/++ ~1.5, Yelp/Amazon 0.3, BB-MAS ~3, HMOG/BrainRun ~3-8,
  Balabit 0.3, fraudecom/malurl/fakejob/twitterbot < 0.3) and ≈ 20 GB with
  it. Processed and prepared outputs roughly double that; the sequence index
  for 42M rows adds ~6 GB; the IEEE wide block 1 GB. The Mac's 100 GB is
  enough for raw + processed if TalkingData stays out. The rented box needs
  **≥ 500 GB of disk** - the current 4090 instance has 59 GB free and will
  not do for v5.
- **GPU.** The history transformer over 29M legit rows with a 32-step window
  is the expensive part: on the order of 1 billion token-steps per epoch. A
  single **A100 80 GB or H100** does an epoch in roughly 10-25 minutes with
  the model sizes planned; a 4090 is 2-3× slower and its 24 GB forces smaller
  batches. GNN training on DGraph/IEEE/AMLworld is minutes to an hour per
  model. Text and behaviour modules are small. **Recommendation: one box with
  1× A100 80 GB or H100, ≥ 64 GB RAM, ≥ 16 cores, ≥ 500 GB disk**, kept for
  the whole v5 build; the Compute tab drives it as before.
- **CPU-side builds** (wide blocks, graph features over 42M rows) run on the
  same box, not the 2-core sandbox.

## 5. Phases

0. This document; data requests sent.
1. **No new data needed, start now:** IEEE wide block and virtual customer
   id; per-customer sequence index in `prepare`; history transformer +
   conditional VAE + sequence head; causal graph features with a 30-day
   label delay for every entity column; blend v2 with presence indicators;
   entity store and history/entities in the score API. Expected: IEEE recall
   at 1% FPR from 35% to the 55-70% band, fdh from 34% toward 45-50% via
   terminal risk, gains on every sequence-rich source.
2. **Device telemetry:** BAF + fraudecom + ipblock as new sources; device
   block in the schema; device head; API device payload.
3. **Graph:** DGraph-Fin, Elliptic/++, YelpChi/Amazon; GNN on AMLworld, IEEE
   and the new graphs; graph LLR into the blend.
4. **Text:** merchant encoder; sentence encoder for free text; malurl /
   fakejob / twitterbot heads; text LLR.
5. **Behaviour:** Balabit, BB-MAS, HMOG/BrainRun; session verifier; session
   payload in the API; behaviour LLR.
6. UI made workable for the new payloads, docs and How-it-works updated,
   full retrain on the big box.

Each phase ends with a GPU run and an entry in `docs/TRAINING_NOTES.md`. If
phase 1 does not land IEEE inside the expected band, the plan is revised
before any new data is downloaded.
