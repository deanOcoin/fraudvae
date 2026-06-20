# Getting the Kaggle datasets by hand

`fraudvae fetch` pulls the GitHub-hosted sources itself. The Kaggle ones need a
login. Either configure the Kaggle API (`pip install kaggle`, token in
`~/.kaggle/kaggle.json`) and run `fraudvae fetch` again, or click through:

| key | page | what to download | put it in |
|---|---|---|---|
| ulb | https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud | the zip (creditcard.csv inside) | `data/raw/ulb/` |
| paysim | https://www.kaggle.com/datasets/ealaxi/paysim1 | the zip | `data/raw/paysim/` |
| sparkov | https://www.kaggle.com/datasets/kartik2112/fraud-detection | the zip (fraudTrain.csv + fraudTest.csv) | `data/raw/sparkov/` |
| banksim | https://www.kaggle.com/datasets/ealaxi/banksim1 | the zip | `data/raw/banksim/` |
| ieee | https://www.kaggle.com/competitions/ieee-fraud-detection/data | join the competition, then the zip | `data/raw/ieee/` |
| tabformer (optional) | https://www.kaggle.com/datasets/ealtman2019/credit-card-transactions | the zip, for the users/cards tables | `data/raw/tabformer/` |
| amlworld (optional) | https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml | the zip (8 GB, six variants; only `HI-Small_Trans.csv` + `HI-Small_Patterns.txt` get extracted) | `data/raw/amlworld/` |

Zips can stay zipped; `build` extracts what it needs. Dropping everything into
`data/raw/` directly also works, the archive contents are matched by file name.
