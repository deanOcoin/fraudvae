"""Kaggle "fraud-ecommerce": 151k first purchases by new accounts, one row per user, with
the one thing no other public file has: a raw device id that repeats across accounts
(6k devices carry several accounts, and half of those rows are fraud). The purchase time
relative to signup is the other famous feature (a purchase one second after signup is a
bot). The ip comes with a range table to a country.

Entities: device = device_id, address = the ip (the only locator in the file; a shared ip
is the same kind of signal as a shared device), customer = user_id so that "distinct
customers on this device" can be counted. Every user appears exactly once, so there is no
customer sequence and the velocity block stays unobserved.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .. import entities
from ..schema import cat_code, empty_frame, entity_codes, finalize
from ..taxonomy import browser_class, encode_series
from .base import epoch_seconds, hour_dow, require, wide_encode

KEY = "fraudecom"
_CAT_COLS = ["source", "browser", "sex", "country", "device_id"]


def ip_country(ip: np.ndarray, table: pd.DataFrame) -> np.ndarray:
    """ip as a number -> country name through the (lower, upper, country) range table,
    None where no range covers it"""
    lo = table["lower_bound_ip_address"].to_numpy(dtype=np.float64)
    hi = table["upper_bound_ip_address"].to_numpy(dtype=np.float64)
    names = table["country"].to_numpy(dtype=object)
    order = np.argsort(lo)
    lo, hi, names = lo[order], hi[order], names[order]
    ipv = np.floor(np.asarray(ip, dtype=np.float64))
    pos = np.searchsorted(lo, ipv, side="right") - 1
    ok = (pos >= 0) & (ipv <= hi[np.clip(pos, 0, len(hi) - 1)])
    out = np.full(len(ipv), None, dtype=object)
    out[ok] = names[pos[ok]]
    return out


def load(raw_dir: Path) -> pd.DataFrame:
    path = require(raw_dir, KEY, "Fraud_Data.csv")
    table = pd.read_csv(require(raw_dir, KEY, "IpAddress_to_Country.csv"))
    raw = pd.read_csv(path)
    n = len(raw)
    purchase = pd.to_datetime(raw["purchase_time"])
    signup = pd.to_datetime(raw["signup_time"])
    age_s = (purchase - signup).dt.total_seconds().to_numpy(dtype=np.float64)
    raw["country"] = ip_country(raw["ip_address"].to_numpy(), table)
    raw["signup_to_purchase_s"] = age_s.astype(np.float32)

    df = empty_frame(n)
    df["txn_id"] = "fraudecom-" + raw["user_id"].astype(str)
    df["ts"] = epoch_seconds(purchase)
    df["hour"], df["dow"] = hour_dow(purchase)
    df["amount"] = raw["purchase_value"].astype(np.float32)
    df["is_fraud"] = raw["class"].astype(np.int8)
    df["channel"] = cat_code("channel", "online")
    df["txn_type"] = cat_code("txn_type", "purchase")
    df["cust_age"] = raw["age"].astype(np.float32)
    df["cust_gender"] = encode_series("cust_gender", raw["sex"].astype(str))
    df["account_age_days"] = (age_s / 86400.0).astype(np.float32)
    df["browser"] = encode_series("browser", raw["browser"].map(browser_class))

    df["customer"] = entity_codes(raw["user_id"].to_numpy())
    df["device"] = entity_codes(raw["device_id"].to_numpy())
    df["address"] = entity_codes(np.floor(raw["ip_address"].to_numpy(dtype=np.float64)).astype(np.int64))
    df = entities.attach(df)

    wide, desc = wide_encode(raw.drop(columns=["user_id", "class", "signup_time", "purchase_time"]), _CAT_COLS)
    df = pd.concat([df, pd.DataFrame(wide, index=df.index)], axis=1)
    df.attrs["wide_columns"] = desc
    return finalize(df, KEY)
