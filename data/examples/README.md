# Example rows

A few hundred labelled transactions per source, drawn from the held-out *test* split of the
build (the newest 20% of each dataset, which no generation of the model was trained on),
with fraud oversampled to roughly half the rows. They exist so the score tab can offer
real rows to score and say whether the model was right, on a machine that has none of the
datasets - the public demonstration site included.

Only sources whose licence allows redistribution are here. The rows are in the project's
unified schema (`docs/SCHEMA.md`), plus the source's own raw columns in the `wide` column
(json) and the entity codes the graph module uses; `is_fraud` is the dataset's label,
`split` is always `test`. Each file keeps the licence of the dataset it was drawn from,
listed below - in particular the Bank Account Fraud rows are **non-commercial** (CC BY-NC-SA
4.0) and the PaySim / AMLworld rows are share-alike. Nothing here is a substitute for the
datasets: download them from the sources below for anything beyond trying the model.

The sources that are *not* here (Fraud Detection Handbook, Capital One, BankSim, IEEE-CIS,
fraud-ecommerce) either state no licence or forbid redistribution; `fraudvae fetch` gets
them from their own locations and `fraudvae examples` then writes rows for them too, on
your machine only.


| source | dataset | licence | rows here |
|---|---|---|---:|
| tabformer | IBM TabFormer synthetic credit card transactions | [Apache-2.0 (repo)](https://github.com/IBM/TabFormer/blob/main/LICENSE) | 300 |
| ulb | ULB / Worldline credit card fraud | [DbCL v1.0](https://opendatacommons.org/licenses/dbcl/1-0/) | 179 |
| paysim | PaySim mobile money simulation | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) | 300 |
| sparkov | Sparkov simulated credit card transactions | [CC0 (kaggle listing)](https://creativecommons.org/publicdomain/zero/1.0/) | 300 |
| amlworld | IBM AMLworld synthetic bank transfers (HI-Small) | [CDLA-Sharing-1.0 (kaggle listing)](https://cdla.dev/sharing-1-0/) | 300 |
| baf | Bank Account Fraud (Feedzai, NeurIPS 2022) account applications | [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) | 300 |

## tabformer: IBM TabFormer synthetic credit card transactions

- source: <https://github.com/IBM/TabFormer>
- citation: Padhi et al. Tabular Transformers for Modeling Multivariate Time Series, ICASSP 2021
- licence: Apache-2.0 (repo) (<https://github.com/IBM/TabFormer/blob/main/LICENSE>)

## ulb: ULB / Worldline credit card fraud

- source: <https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud>
- citation: Dal Pozzolo et al. Calibrating Probability with Undersampling for Unbalanced Classification, 2015
- licence: DbCL v1.0 (<https://opendatacommons.org/licenses/dbcl/1-0/>)

## paysim: PaySim mobile money simulation

- source: <https://www.kaggle.com/datasets/ealaxi/paysim1>
- citation: Lopez-Rojas, Elmir, Axelsson. PaySim: A financial mobile money simulator for fraud detection, EMSS 2016
- licence: CC BY-SA 4.0 (<https://creativecommons.org/licenses/by-sa/4.0/>)

## sparkov: Sparkov simulated credit card transactions

- source: <https://www.kaggle.com/datasets/kartik2112/fraud-detection>
- citation: Generated with Sparkov_Data_Generation (Brandon Harris), published by Kartik Shenoy on Kaggle
- licence: CC0 (kaggle listing) (<https://creativecommons.org/publicdomain/zero/1.0/>)

## amlworld: IBM AMLworld synthetic bank transfers (HI-Small)

- source: <https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml>
- citation: Altman, Blanuša, von Niederhäusern, Egressy, Anghel, Atasu. Realistic Synthetic Financial Transactions for Anti-Money Laundering Models, NeurIPS 2023
- licence: CDLA-Sharing-1.0 (kaggle listing) (<https://cdla.dev/sharing-1-0/>)

## baf: Bank Account Fraud (Feedzai, NeurIPS 2022) account applications

- source: <https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022>
- citation: Jesus, Pombal, Alves, Cruz, Saleiro, Ribeiro, Gama, Bizarro. Turning the Tables: Biased, Imbalanced, Dynamic Tabular Datasets for ML Evaluation, NeurIPS 2022
- licence: CC BY-NC-SA 4.0 (<https://creativecommons.org/licenses/by-nc-sa/4.0/>)

Written by `fraudvae examples` on 2026-10-20; regenerate rather than edit.
