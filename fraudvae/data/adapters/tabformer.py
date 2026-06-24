"""IBM TabFormer: 24.4M synthetic card transactions.

If the kaggle copy is present too (sd254_users.csv / sd254_cards.csv) the user
and card tables are joined for age, income, card brand, credit limit... The
github tgz alone only has the transaction table.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .. import velocity
from ..schema import cat_code, time_split_codes
from ..taxonomy import (
    TABFORMER_ERROR,
    US_STATES,
    card_brand_from_name,
    card_kind_from_name,
    category_from_mcc,
)
from .base import _find, require, source_dir

KEY = "tabformer"


def _money(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.replace("$", "", regex=False), errors="coerce")


def _chan(v):
    return cat_code("channel", {"Swipe Transaction": "card_present", "Chip Transaction": "card_present",
                                "Online Transaction": "online"}.get(v, "unknown"))


def _err(v):
    if not isinstance(v, str) or v == "":
        return cat_code("auth_error", "none")
    first = v.split(",")[0].strip()   # a few rows carry two errors
    return cat_code("auth_error", TABFORMER_ERROR.get(first, "unknown"))


def _foreign(v):
    if not isinstance(v, str):
        return 0
    return cat_code("is_foreign", "no" if v in US_STATES else "yes")


def _mcc(code):
    if pd.isna(code) or not 0 < int(code) < 10000:
        return 0
    return cat_code("merchant_category", category_from_mcc(int(code)))


def load(raw_dir: Path, max_rows: int | None = None) -> pd.DataFrame:
    path = require(raw_dir, KEY, "card_transaction.v1.csv|credit_card_transactions-ibm_v2.csv")
    df = pd.read_csv(path, nrows=max_rows)

    hhmm = df["Time"].str.split(":", expand=True).astype(int)
    when = pd.to_datetime(pd.DataFrame({"year": df["Year"], "month": df["Month"], "day": df["Day"],
                                        "hour": hhmm[0], "minute": hhmm[1]}))
    ts = (when.astype("int64") / 1e9).to_numpy(np.float64)
    amount = _money(df["Amount"]).to_numpy(np.float32)
    is_fraud = (df["Is Fraud?"] == "Yes").to_numpy(np.int8)
    user = df["User"].to_numpy(np.int32)
    card = df["Card"].to_numpy(np.int32)
    channel = df["Use Chip"].map(_chan).to_numpy(np.int8)
    auth_error = df["Errors?"].map(_err).to_numpy(np.int8)
    is_foreign = df["Merchant State"].map(_foreign).to_numpy(np.int8)
    mcat = df["MCC"].map(_mcc).to_numpy(np.int8)
    merchant = pd.factorize(df["Merchant Name"])[0].astype(np.int32)

    # velocity per card (a user with 4 cards is 4 behaviours)
    card_key = user * 100 + card
    vel = velocity.rolling_customer_features(card_key, ts, amount, merchant=merchant, category=mcat,
                                             foreign=is_foreign)

    out = pd.DataFrame({
        "ts": ts, "amount": amount, "hour": when.dt.hour.to_numpy(np.float32),
        "dow": when.dt.dayofweek.to_numpy(np.float32),
        "is_fraud": is_fraud, "channel": channel, "txn_type": np.int8(cat_code("txn_type", "purchase")),
        "merchant_category": mcat, "auth_error": auth_error, "is_foreign": is_foreign,
        "split": time_split_codes(ts),
        "customer": user.astype(np.int64), "card": card_key.astype(np.int64), "counterparty": merchant.astype(np.int64),
    })
    for k, v in vel.items():
        out[k] = v
    for k, v in _user_card_columns(raw_dir, df, when, amount).items():
        out[k] = v
    out["source"] = KEY
    return out


def _user_card_columns(raw_dir: Path, df: pd.DataFrame, when: pd.Series, amount: np.ndarray) -> dict:
    """demographics / card attributes from the kaggle users + cards tables"""
    root = source_dir(raw_dir, KEY)
    users_p = _find(root, "sd254_users.csv") if root.exists() else None
    cards_p = _find(root, "sd254_cards.csv") if root.exists() else None
    if users_p is None or cards_p is None:
        return {}

    users = pd.read_csv(users_p)
    users["User"] = np.arange(len(users))
    u = df[["User"]].merge(users, on="User", how="left")
    out = {
        "yearly_income": _money(u["Yearly Income - Person"]).to_numpy(np.float32),
        "total_debt": _money(u["Total Debt"]).to_numpy(np.float32),
        "fico_score": u["FICO Score"].to_numpy(np.float32),
        "num_cards": u["Num Credit Cards"].to_numpy(np.float32),
        "cust_gender": u["Gender"].map({"Female": cat_code("cust_gender", "F"),
                                        "Male": cat_code("cust_gender", "M")}).fillna(0).to_numpy(np.int8),
    }
    # age at transaction time from birth year/month, Current Age is "as of 2020"
    out["cust_age"] = ((when.dt.year - u["Birth Year"]) + (when.dt.month - u["Birth Month"]) / 12.0).to_numpy(np.float32)

    cards = pd.read_csv(cards_p).rename(columns={"CARD INDEX": "Card"})
    c = df[["User", "Card"]].merge(cards, on=["User", "Card"], how="left")
    limit = _money(c["Credit Limit"]).to_numpy(np.float32)
    out["credit_limit"] = limit
    with np.errstate(divide="ignore", invalid="ignore"):
        over = amount / limit
    over[~(limit > 0)] = np.nan
    out["amount_over_limit"] = over
    out["card_brand"] = c["Card Brand"].map(lambda v: cat_code("card_brand", card_brand_from_name(v))).fillna(0).to_numpy(np.int8)
    out["card_kind"] = c["Card Type"].map(lambda v: cat_code("card_kind", card_kind_from_name(v))).fillna(0).to_numpy(np.int8)
    opened = pd.to_datetime(c["Acct Open Date"], format="%m/%Y", errors="coerce")
    acct_age = ((when - opened).dt.days).to_numpy(np.float32)
    acct_age[~(acct_age >= 0)] = np.nan
    out["account_age_days"] = acct_age
    return out
