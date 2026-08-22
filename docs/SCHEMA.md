# Unified transaction schema

Defined in `fraudvae/data/schema.py` (version 5; v4 added the customer-context
block, the merchant block, `payment_format` and the amlworld source; v5 added
the entity keys, the graph block, the device block and the wide block). Four
kinds of columns: meta (never fed to the model), numeric (float32, NaN =
unobserved), categorical (int8 codes, 0 = unknown/unobserved) and, per
source, an optional wide block of its own raw columns (only the booster reads
it; see ARCHITECTURE.md). Adapters under `fraudvae/data/adapters/`
fill what their source has and call `schema.finalize()`, which fixes dtypes
and range-checks codes. `hour` and `dow` are stored as numbers but become
categoricals (24 and 7 classes) in `prepare`.

## Meta

| column | type | notes |
|---|---|---|
| source | category | one of fdh, capone, tabformer, ulb, paysim, sparkov, banksim, ieee, amlworld |
| txn_id | string | `<source>-<original id or row>` |
| ts | float64 | seconds. real epoch where the source has dates, simulation seconds otherwise |
| is_fraud | int8 | the label |
| split | category | train / val / test, assigned by time within a source (70/10/20) |
| fraud_scenario | int8 | fdh (1 amount>220, 2 compromised terminal, 3 compromised card); amlworld (laundering pattern: 1 fan-out, 2 fan-in, 3 gather-scatter, 4 scatter-gather, 5 cycle, 6 bipartite, 7 stack, 8 random) |
| customer, counterparty, device, email, address, card | int64 | v5 entity keys: dense codes within the source, −1 = the source has no such thing. Drive the sequence index and the graph block, never model inputs. customer: fdh `CUSTOMER_ID`, capone `accountNumber`, tabformer `User`, sparkov `cc_num`, banksim `customer`, paysim `nameOrig`, ieee `card1 + addr1 + (day − D1)`, amlworld payer account, ulb none. counterparty: fdh terminal, capone / sparkov / banksim merchant, tabformer merchant name, paysim `nameDest`, amlworld payee account. card: capone account + last 4 digits, tabformer `User*100+Card`, sparkov `cc_num`, ieee `card1-2-3-5`. device: ieee identity fingerprint (DeviceInfo, OS, browser, screen, id_19/20) where the identity table has a row. email: ieee `P_emaildomain`. address: ieee `addr1_addr2`. |

## Numeric fields

| name | transform | group | meaning |
|---|---|---|---|
| amount | log1p | core | transaction amount in source currency units |
| hour | – (→ categorical) | core | hour of day 0–23 |
| dow | – (→ categorical) | core | day of week, 0 = Monday |
| secs_since_prev_txn | log1p | velocity | seconds since the customer's previous transaction |
| cust_txn_count_24h | log1p | velocity | customer transactions in the previous 24 h |
| cust_amount_sum_24h | log1p | velocity | customer spend in the previous 24 h |
| cust_txn_count_7d | log1p | velocity | customer transactions in the previous 7 days |
| cust_amount_mean_30d | log1p | velocity | mean amount over the previous 30 days (NaN if no history) |
| amount_over_cust_mean | log1p | velocity | amount / that mean |
| cust_txn_count_1h | log1p | velocity | customer transactions in the previous hour (burst) |
| cust_amount_sum_1h | log1p | velocity | customer spend in the previous hour |
| cust_amount_z_30d | – | velocity | (amount − 30d mean) / 30d std of the customer's own history; std floored at 5% of the mean; needs ≥ 2 prior rows; clipped to [−20, 50] |
| cust_txn_count_prior | log1p | velocity | how many transactions we have seen from this customer before (tenure) |
| cust_days_since_first | log1p | velocity | days since the customer's first transaction in the source |
| merch_txn_count_24h | log1p | merchant | merchant / terminal / payee transactions in the previous 24 h |
| merch_amount_ratio | log1p | merchant | amount / the merchant's mean amount over everything before this row (NaN on its first row) |
| cust_age | – | account | years |
| credit_limit | log1p | account | |
| amount_over_limit | – | account | amount / credit limit |
| balance_before | signed_log1p | account | available balance / funds before the transaction |
| balance_after | signed_log1p | account | after |
| account_age_days | log1p | account | days since account/card open (ieee: D1, days since card first seen) |
| dist_km | log1p | account | customer home to merchant, haversine (sparkov) |
| city_pop | log1p | ext | sparkov customer city population |
| days_since_addr_change | log1p | ext | capone |
| dest_balance_before / dest_balance_after | signed_log1p | ext | paysim recipient balances |
| yearly_income, total_debt, fico_score, num_cards | log1p / log1p / – / – | ext | tabformer users table (kaggle copy only) |
| ieee_dist1, ieee_c1, ieee_c13, ieee_c14, ieee_d15 | log1p… | ext | the few anonymised ieee columns with a known meaning |
| screen_w, screen_h | log1p | device | v5 device block: screen size in px (ieee id_33) |
| session_length_min, email_name_similarity | log1p / – | device | v5 device block, for the application-fraud sources (BAF); unobserved today |
| {cpty,device,email,addr,card}_txn_count_30d | log1p | graph | v5: that entity's transactions in the previous 30 days |
| {…}_new_cust_30d | log1p | graph | customers first seen with the entity in the last 30 days (a device that acquired five cards this month). Windowed, not lifetime: lifetime counts and ages grow with the calendar and the booster learned the era from them |
| {…}_age_days | log1p | graph | the entity's age in the source, capped at a year |
| {…}_fraud_count_delayed, {…}_fraud_share_delayed | log1p / – | graph | confirmed fraud among the entity's rows older than 30 days (`ENTITY_LABEL_DELAY_DAYS`); NaN with no such rows |
| cust_new_cpty_30d, cust_new_device_30d | log1p | graph | counterparties / devices the customer used for the first time in the last 30 days |
| cust_fraud_count_delayed, cust_fraud_share_delayed | log1p / – | graph | the customer's own confirmed fraud, 30 days delayed |
| wide__{cpty,device,email,addr,card,cust}_fraud_count_fast, …_share_fast | – | wide | the same delayed-label pair at `ENTITY_LABEL_DELAY_FAST_DAYS` (7 days, the Fraud Detection Handbook's lag). Wide, so the booster reads it and the VAE does not; every source with entities gets it, next to its own raw columns if it has any |
| wide__{…}_fraud_count_fast_w7, …_share_fast_w7, …_w30 | – | wide | the same pair over the entity's rows in the 7 / 30 days before the one-week label horizon: a terminal compromised this month against one compromised two years ago. NaN with no labelled rows in the window |
| ulb_v1 … ulb_v28 | – | ulb_pca | ULB PCA components, only this source |

"velocity" and "merchant" features are computed per source from its own
customer / merchant keys with `fraudvae/data/velocity.py`, strictly on past
rows (ties at the same timestamp count as past, which only matters for
banksim's day resolution). The customer key: fdh `CUSTOMER_ID`, capone
`accountNumber`, tabformer `User*100+Card`, sparkov `cc_num`, banksim
`customer`, paysim `nameOrig` (almost unique, so mostly zeros), ieee
`card1 + addr1 + (day − D1)`, amlworld `From Bank-Account`, ulb none. The
merchant key: fdh `TERMINAL_ID`, capone `merchantName`, tabformer
`Merchant Name`, sparkov `merchant`, banksim `merchant`, paysim `nameDest`,
amlworld `To Bank-Account`; ieee and ulb have none, so their merchant block
is unobserved.

## Categorical fields

| name | values (0 = unknown) | group |
|---|---|---|
| channel | card_present, card_not_present, online, mobile_money | core |
| txn_type | purchase, cash_in, cash_out, debit, payment, transfer, reversal, address_verification | core |
| merchant_category | grocery, retail, online_retail, restaurants, entertainment, travel, transport, fuel, health, personal_care, home, utilities, financial, digital, kids_pets, hotel, misc | core |
| card_brand | visa, mastercard, amex, discover, other | core |
| device_os | windows, macos, ios, android, linux, other | device (v5; ieee id_30) |
| browser | chrome, safari, firefox, edge, ie, samsung, opera, other | device (v5; ieee id_31) |
| email_bucket_recipient | gmail, yahoo, microsoft, other_free, other | device (v5; ieee R_emaildomain) |
| phone_valid, keep_alive_session | no, yes | device (v5; application-fraud sources, unobserved today) |
| card_kind | credit, debit, prepaid | core |
| is_foreign | no, yes | core |
| cust_gender | F, M | account |
| cust_new_merchant | no, yes | velocity | first time this customer is seen at this merchant (yes on their very first row too) |
| cust_new_category | no, yes | velocity | first time in this merchant category |
| cust_first_foreign | no, yes | velocity | this is cross-border and the customer had no earlier cross-border row |
| auth_error | none, insufficient_funds, bad_pin, bad_cvv, bad_card_number, bad_expiration, bad_zipcode, technical_glitch | core |
| device_type | desktop, mobile | ext |
| is_recurring | no, yes | ext |
| dest_is_merchant | no, yes | ext |
| product_code | W, C, R, H, S | ext |
| email_bucket | gmail, yahoo, microsoft, other_free, other | ext |
| payment_format | ach, wire, cheque, cash, credit_card, reinvestment, crypto | ext (amlworld) |

Merchant categories: `fraudvae/data/taxonomy.py` has the explicit maps for
sparkov (14 categories), banksim (15), capone (19) and the MCC table in
`taxonomy/mcc_map.json` (271 codes + range fallbacks) for tabformer.

## Per-source decisions worth knowing

- **fdh**: terminals are POS, so channel = card_present. No merchant category.
  Scenario label kept in `fraud_scenario` for analysis, not as a feature.
- **capone**: `posEntryMode 09` → online, otherwise `cardPresent` decides
  card_present vs card_not_present. CVV mismatch → `auth_error = bad_cvv`.
  `is_foreign` = merchant country ≠ acquirer country (unknown when either is
  blank). `recurringAuthInd` and `posOnPremises` are empty in the whole file.
- **tabformer**: `Use Chip` → channel. `Merchant State` that isn't a US state →
  foreign (online rows have no state → unknown). `Errors?` can list two
  errors; the first wins. Demographics and card brand/kind/limit only when the
  Kaggle `sd254_users.csv` / `sd254_cards.csv` tables are present.
- **ulb**: `Time` is seconds from the first transaction; hour assumes the window
  started at midnight. V1–V28 live in their own block.
- **paysim**: step is an hour; hour = step mod 24, day of week unknown. Both
  parties' balances are features; `isFlaggedFraud` is a rule output and dropped.
- **sparkov**: `*_net` categories → online, `*_pos` → card_present. Card brand
  from the IIN prefix. Age from date of birth at transaction time.
- **banksim**: day resolution only, hour and dow unobserved. Age buckets mapped
  to midpoints; gender E (enterprise) and U → unknown.
- **ieee**: all online. `TransactionDT` origin assumed 2017-12-01 for hour /
  dow. Card brand/kind from card4/card6, email bucket from `P_emaildomain`,
  device type from the identity table. The V columns are not used.
- **amlworld**: HI-Small only (the Kaggle zip holds six variants, 8 GB; the
  build extracts just the two files it needs). Payer = customer, payee =
  merchant. Amounts converted to USD with fixed rough 2022 rates so one column
  makes sense across 13 currencies; `is_foreign` = payment currency ≠
  receiving currency; `Payment Format` → `payment_format` and a coarse
  `txn_type`. No merchant category, no channel. The label is laundering, not
  card fraud; the pattern a flagged row belongs to goes to `fraud_scenario`
  from `HI-Small_Patterns.txt` (a few flagged rows are in no pattern → 0).

## Prepared arrays (v5 additions)

`fraudvae prepare` writes, next to the v4 arrays, `{split}_cust.npy` (global customer
key, `source << 40 | code`), `{split}_ts.npy`, `{split}_ent.npy` (the six entity codes
per row, int32, −1 none; the graph module looks embeddings up by them), the per-source
wide blocks `{split}_wide_<source>.npy` and the sequence index (`seq_order.npy`,
`seq_pos.npy`, `seq_start.npy`: every row of every split sorted by (customer, time),
so a row's previous L transactions are a contiguous slice).
