"""Where each dataset comes from and what we expect to find on disk.

Two kinds of access:
  github  - cloned / fetched automatically by `fraudvae fetch`
  kaggle  - needs a kaggle login. either drop the zip into data/raw/<key>/ by
            hand or configure the kaggle api (KAGGLE_USERNAME / KAGGLE_KEY)
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Source:
    key: str
    name: str
    access: str                      # github | github_lfs | kaggle
    url: str
    license: str
    citation: str
    context: str                     # what kind of payments these are
    label_rate_hint: float           # approx fraud share, for the docs/ui only
    approx_rows: int
    files: tuple[str, ...]           # files we look for under data/raw/<key>/
    kaggle_ref: str = ""             # datasets/<owner>/<slug> or competitions/<slug>
    kaggle_zip_names: tuple[str, ...] = ()   # what the kaggle download is usually called
    git_url: str = ""
    lfs_urls: tuple[tuple[str, str], ...] = ()   # (relative path, download url)
    extract_members: tuple[str, ...] = ()        # only pull these out of the archive (amlworld's zip is 8 GB)
    notes: str = ""
    # whether the licence lets a few hundred labelled rows ship with the repository and the
    # public site (data/examples/). False means "not stated" as much as "no": a source
    # whose listing names no licence is not redistributed, whatever everyone else does
    redistributable: bool = False
    license_url: str = ""


SOURCES: dict[str, Source] = {
    "fdh": Source(
        key="fdh",
        name="Fraud Detection Handbook simulated card transactions",
        access="github",
        url="https://github.com/Fraud-Detection-Handbook/simulated-data-raw",
        license="GPL-3.0 (handbook code) / see repo",
        citation="Le Borgne, Siblini, Lebichot, Bontempi. Reproducible Machine Learning for Credit Card Fraud Detection - Practical Handbook, 2022",
        context="simulated point of sale card payments, 5k customers x 10k terminals, 183 days, 3 fraud scenarios",
        label_rate_hint=0.0084,
        approx_rows=1_754_155,
        files=("simulated-data-raw/data/2018-04-01.pkl",),
        git_url="https://github.com/Fraud-Detection-Handbook/simulated-data-raw",
        notes="one pickle per day. scenario 1 = any amount > 220, 2 = compromised terminal, 3 = compromised card (amounts x5)",
    ),
    "capone": Source(
        key="capone",
        name="Capital One synthetic card transactions",
        access="github",
        url="https://github.com/CapitalOneRecruiting/DS",
        license="unspecified, public recruiting dataset",
        citation="Capital One Data Science Challenge dataset (CapitalOneRecruiting/DS)",
        context="synthetic US credit card authorisations for 5k accounts over 2016, with reversals and address verifications",
        label_rate_hint=0.0158,
        approx_rows=786_363,
        files=("DS/transactions.zip",),
        git_url="https://github.com/CapitalOneRecruiting/DS",
        notes="json lines inside transactions.zip. cvv entered vs on card, pos entry mode, credit limit / available money",
    ),
    "tabformer": Source(
        key="tabformer",
        name="IBM TabFormer synthetic credit card transactions",
        access="github_lfs",
        url="https://github.com/IBM/TabFormer",
        license="Apache-2.0 (repo)",
        citation="Padhi et al. Tabular Transformers for Modeling Multivariate Time Series, ICASSP 2021",
        context="synthetic US consumer card transactions for 2k users / 6k cards over ~30 years, chip/swipe/online, MCC, auth errors",
        label_rate_hint=0.0012,
        approx_rows=24_386_900,
        files=("card_transaction.v1.csv|credit_card_transactions-ibm_v2.csv",),
        lfs_urls=(("transactions.tgz", "https://media.githubusercontent.com/media/IBM/TabFormer/main/data/credit_card/transactions.tgz"),),
        kaggle_ref="datasets/ealtman2019/credit-card-transactions",
        kaggle_zip_names=("credit-card-transactions.zip", "archive.zip"),
        notes="same data as kaggle ealtman2019/credit-card-transactions, which also ships the users and cards tables",
        redistributable=True,
        license_url="https://github.com/IBM/TabFormer/blob/main/LICENSE",
    ),
    "ulb": Source(
        key="ulb",
        name="ULB / Worldline credit card fraud",
        access="kaggle",
        url="https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud",
        license="DbCL v1.0",
        citation="Dal Pozzolo et al. Calibrating Probability with Undersampling for Unbalanced Classification, 2015",
        context="real european card transactions, 2 days in sept 2013, features are PCA components except time and amount",
        label_rate_hint=0.00173,
        approx_rows=284_807,
        files=("creditcard.csv",),
        kaggle_ref="datasets/mlg-ulb/creditcardfraud",
        kaggle_zip_names=("creditcardfraud.zip", "archive.zip"),
        notes="only amount + hour map onto the shared schema, V1..V28 live in their own block",
        redistributable=True,
        license_url="https://opendatacommons.org/licenses/dbcl/1-0/",
    ),
    "paysim": Source(
        key="paysim",
        name="PaySim mobile money simulation",
        access="kaggle",
        url="https://www.kaggle.com/datasets/ealaxi/paysim1",
        license="CC BY-SA 4.0",
        citation="Lopez-Rojas, Elmir, Axelsson. PaySim: A financial mobile money simulator for fraud detection, EMSS 2016",
        context="simulated mobile money transfers / cash in / cash out based on an african operator's logs, 1 month in hourly steps",
        label_rate_hint=0.00129,
        approx_rows=6_362_620,
        files=("PS_20174392719_1491204439457_log.csv",),
        kaggle_ref="datasets/ealaxi/paysim1",
        kaggle_zip_names=("paysim1.zip", "archive.zip"),
        notes="balances before/after for both parties. isFlaggedFraud is a rule output and is NOT used as a feature",
        redistributable=True,
        license_url="https://creativecommons.org/licenses/by-sa/4.0/",
    ),
    "sparkov": Source(
        key="sparkov",
        name="Sparkov simulated credit card transactions",
        access="kaggle",
        url="https://www.kaggle.com/datasets/kartik2112/fraud-detection",
        license="CC0 (kaggle listing)",
        citation="Generated with Sparkov_Data_Generation (Brandon Harris), published by Kartik Shenoy on Kaggle",
        context="simulated US card purchases for 1k customers over 2019-2020 with merchant geo, customer demographics and 14 categories",
        label_rate_hint=0.0058,
        approx_rows=1_852_394,
        files=("fraudTrain.csv", "fraudTest.csv"),
        kaggle_ref="datasets/kartik2112/fraud-detection",
        kaggle_zip_names=("fraud-detection.zip", "archive.zip"),
        notes="train+test concatenated then re split by time like everything else",
        redistributable=True,
        license_url="https://creativecommons.org/publicdomain/zero/1.0/",
    ),
    "banksim": Source(
        key="banksim",
        name="BankSim synthetic payments",
        access="kaggle",
        url="https://www.kaggle.com/datasets/ealaxi/banksim1",
        license="see kaggle page",
        citation="Lopez-Rojas, Axelsson. BankSim: A bank payments simulator for fraud detection research, EMSS 2014",
        context="agent based simulation of a spanish bank's card payments, 6 months in daily steps, 15 merchant categories",
        label_rate_hint=0.0121,
        approx_rows=594_643,
        files=("bs140513_032310.csv",),
        kaggle_ref="datasets/ealaxi/banksim1",
        kaggle_zip_names=("banksim1.zip", "archive.zip"),
        notes="only day resolution, so hour is unobserved",
    ),
    "ieee": Source(
        key="ieee",
        name="IEEE-CIS / Vesta e-commerce fraud",
        access="kaggle",
        url="https://www.kaggle.com/competitions/ieee-fraud-detection",
        license="Kaggle competition rules (non commercial)",
        citation="IEEE Computational Intelligence Society & Vesta Corporation, Kaggle 2019",
        context="real card not present e-commerce transactions over ~6 months, heavily anonymised",
        label_rate_hint=0.035,
        approx_rows=590_540,
        files=("train_transaction.csv", "train_identity.csv"),
        kaggle_ref="competitions/ieee-fraud-detection",
        kaggle_zip_names=("ieee-fraud-detection.zip",),
        notes="train only (test has no labels). customer id reconstructed from card1+addr1+D1 the usual way",
    ),
    "amlworld": Source(
        key="amlworld",
        name="IBM AMLworld synthetic bank transfers (HI-Small)",
        access="kaggle",
        url="https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml",
        license="CDLA-Sharing-1.0 (kaggle listing)",
        citation="Altman, Blanuša, von Niederhäusern, Egressy, Anghel, Atasu. Realistic Synthetic Financial Transactions for Anti-Money Laundering Models, NeurIPS 2023",
        context="synthetic account-to-account transfers between banks over 18 days, 13 currencies, 7 payment formats; the label is money laundering, not card fraud",
        label_rate_hint=0.00102,
        approx_rows=5_078_345,
        files=("HI-Small_Trans.csv|HI-Small_Trans.csv.gz",),
        kaggle_ref="datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml",
        kaggle_zip_names=("ibm-transactions-for-anti-money-laundering-aml.zip", "archive.zip"),
        extract_members=("HI-Small_Trans.csv", "HI-Small_Patterns.txt"),
        notes="the kaggle zip holds six variants (8 GB); only HI-Small is used. amounts converted to USD with rough 2022 "
              "rates. fraud_scenario = laundering pattern from HI-Small_Patterns.txt (1 fan-out ... 8 random)",
        redistributable=True,
        license_url="https://cdla.dev/sharing-1-0/",
    ),
    # v5: the two sources with device telemetry. neither is card payments; both widen what
    # "generic" means and both carry the device block the seven older sources cannot fill
    "baf": Source(
        key="baf",
        name="Bank Account Fraud (Feedzai, NeurIPS 2022) account applications",
        access="kaggle",
        url="https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022",
        license="CC BY-NC-SA 4.0",
        citation="Jesus, Pombal, Alves, Cruz, Saleiro, Ribeiro, Gama, Bizarro. Turning the Tables: Biased, Imbalanced, "
                 "Dynamic Tabular Datasets for ML Evaluation, NeurIPS 2022",
        context="synthetic (CTGAN on a real bank's data) online account-opening applications over 8 months with device os, "
                "session length, keep-alive, phone / email validity and bank-wide velocities; label = fraudulent application",
        label_rate_hint=0.011,
        approx_rows=1_000_000,
        files=("Base.csv",),
        kaggle_ref="datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022",
        kaggle_zip_names=("bank-account-fraud-dataset-neurips-2022.zip", "archive.zip"),
        extract_members=("Base.csv",),
        notes="only the Base variant is used (Variants I-V are biased resamples of the same generator). no timestamps "
              "beyond the month, so hour / dow are unobserved; the row order inside a month is kept as the time order. "
              "the paper's protocol (months 0-5 train, 6-7 test) is what the 70/10/20 time split gives here",
        redistributable=True,
        license_url="https://creativecommons.org/licenses/by-nc-sa/4.0/",
    ),
    "fraudecom": Source(
        key="fraudecom",
        name="E-commerce fraud with device ids (fraud-ecommerce)",
        access="kaggle",
        url="https://www.kaggle.com/datasets/vbinh002/fraud-ecommerce",
        license="unspecified (kaggle listing, widely redistributed teaching dataset)",
        citation="Fraud_Data.csv / IpAddress_to_Country.csv, Kaggle 'Fraud Ecommerce' (vbinh002)",
        context="151k first purchases of new e-commerce accounts in 2015 with device id, browser, ip, signup and purchase time; "
                "the only public file where a raw device id is shared across accounts",
        label_rate_hint=0.094,
        approx_rows=151_112,
        files=("Fraud_Data.csv", "IpAddress_to_Country.csv"),
        kaggle_ref="datasets/vbinh002/fraud-ecommerce",
        kaggle_zip_names=("fraud-ecommerce.zip", "archive.zip"),
        notes="one purchase per user, so there is no customer sequence; the device is the entity that repeats (6k devices "
              "shared by several accounts, half of those rows fraud). ip -> country through the lookup table",
    ),
}

ORDER = list(SOURCES)
