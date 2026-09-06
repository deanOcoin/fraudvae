"""Bank Account Fraud (Feedzai). One row per account-opening application, not a payment: there
is no amount in the card sense, no customer id (every applicant is new) and no clock finer
than the month. What it does have, and why it is here, is the device block: os, session
length, keep-alive, phone and email validity, name/email similarity, plus the bank-wide
velocities and the applicant's address and bank tenure.

Mapping choices worth knowing:
  amount        = proposed_credit_limit, the exposure the application asks for
  ts            = month * 30 days + the row's position inside its month (the file has no
                  finer time; the original row order is the only ordering we have)
  address       = zip_count_4w is the only trace of an address; it goes to addr_txn_count_30d
                  (4 weeks ~ 30 days) since there is no zip code to key an entity on
  device        = device_distinct_emails_8w -> device_new_cust_30d (the nearest thing the
                  generator defines: distinct emails on the device over 8 weeks).
                  device_fraud_count is identically 0 in Base: dropped
  everything else raw -> the wide block, including the three bank-wide velocities, the
  credit risk score and the categorical codes (payment_type, employment, housing) whose
  meanings the publishers withheld.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..schema import cat_code, empty_frame, finalize
from ..taxonomy import encode_series, os_class
from .base import require, wide_encode, yes_no

KEY = "baf"
_MONTH = 30.0 * 86400.0
_CAT_COLS = ["payment_type", "employment_status", "housing_status", "source", "device_os"]


def _nonneg(s: pd.Series) -> np.ndarray:
    """the file writes -1 for missing"""
    v = pd.to_numeric(s, errors="coerce").to_numpy(dtype=np.float32)
    v[v < 0] = np.nan
    return v


def load(raw_dir: Path) -> pd.DataFrame:
    path = require(raw_dir, KEY, "Base.csv")
    raw = pd.read_csv(path)
    n = len(raw)

    month = raw["month"].to_numpy().astype(np.float64)
    # position inside the month keeps the file order as the time order; ties would otherwise
    # make the split depend on the sort
    pos = raw.groupby("month").cumcount().to_numpy().astype(np.float64)
    per_month = raw["month"].map(raw["month"].value_counts()).to_numpy().astype(np.float64)
    ts = (month + pos / per_month) * _MONTH

    df = empty_frame(n)
    df["txn_id"] = "baf-" + pd.Series(np.arange(n)).astype(str)
    df["ts"] = ts
    df["is_fraud"] = raw["fraud_bool"].astype(np.int8)
    df["amount"] = raw["proposed_credit_limit"].astype(np.float32)
    df["credit_limit"] = raw["proposed_credit_limit"].astype(np.float32)
    df["channel"] = cat_code("channel", "online")
    df["txn_type"] = cat_code("txn_type", "address_verification")   # an application: identity checks, no money moves
    df["cust_age"] = raw["customer_age"].astype(np.float32)
    df["is_foreign"] = yes_no(raw["foreign_request"].to_numpy() == 1)
    df["account_age_days"] = _nonneg(raw["bank_months_count"]) * 30.0
    df["days_since_addr_change"] = _nonneg(raw["current_address_months_count"]) * 30.0
    df["email_bucket"] = encode_series("email_bucket", pd.Series(np.where(raw["email_is_free"].to_numpy() == 1, "other_free", "other")))
    # v5 device block
    df["device_os"] = encode_series("device_os", raw["device_os"].map(os_class))
    df["session_length_min"] = _nonneg(raw["session_length_in_minutes"])
    df["keep_alive_session"] = yes_no(raw["keep_alive_session"].to_numpy() == 1)
    df["phone_valid"] = yes_no((raw["phone_mobile_valid"].to_numpy() == 1) | (raw["phone_home_valid"].to_numpy() == 1))
    df["email_name_similarity"] = raw["name_email_similarity"].astype(np.float32)
    # graph block, from the generator's own pre-aggregated entity counts
    df["device_new_cust_30d"] = _nonneg(raw["device_distinct_emails_8w"])
    df["addr_txn_count_30d"] = raw["zip_count_4w"].astype(np.float32)

    wide, desc = wide_encode(raw.drop(columns=["fraud_bool", "month", "device_fraud_count"]), _CAT_COLS)
    df = pd.concat([df, pd.DataFrame(wide, index=df.index)], axis=1)
    df.attrs["wide_columns"] = desc
    return finalize(df, KEY)
