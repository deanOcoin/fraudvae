"""Sparkov (kaggle kartik2112/fraud-detection). fraudTrain + fraudTest glued back
together; the time based split is redone in build like for every other source."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .. import entities, velocity
from ..schema import cat_code, empty_frame, entity_codes, finalize
from ..taxonomy import SPARKOV_CATEGORY, card_brand_from_number, encode_series
from .base import epoch_seconds, hour_dow, require

KEY = "sparkov"

_USE = ["trans_date_trans_time", "cc_num", "merchant", "category", "amt", "gender", "lat", "long",
        "city_pop", "dob", "trans_num", "merch_lat", "merch_long", "is_fraud"]


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def load(raw_dir: Path) -> pd.DataFrame:
    parts = []
    for name in ("fraudTrain.csv", "fraudTest.csv"):
        p = require(raw_dir, KEY, name)
        parts.append(pd.read_csv(p, usecols=_USE, dtype={"cc_num": str}))
    raw = pd.concat(parts, ignore_index=True)
    del parts

    n = len(raw)
    when = pd.to_datetime(raw["trans_date_trans_time"])
    df = empty_frame(n)
    df["txn_id"] = "sparkov-" + raw["trans_num"].astype(str)
    df["ts"] = epoch_seconds(when)
    df["amount"] = raw["amt"].astype(np.float32)
    df["hour"], df["dow"] = hour_dow(when)
    df["is_fraud"] = raw["is_fraud"].astype(np.int8)

    cat = raw["category"].astype(str)
    df["merchant_category"] = encode_series("merchant_category", cat, SPARKOV_CATEGORY)
    chan = np.where(cat.str.endswith("_net"), "online",
                    np.where(cat.str.endswith("_pos"), "card_present", "unknown"))
    df["channel"] = encode_series("channel", pd.Series(chan))
    df["txn_type"] = cat_code("txn_type", "purchase")
    df["card_brand"] = encode_series("card_brand", raw["cc_num"].map(card_brand_from_number))
    df["cust_gender"] = encode_series("cust_gender", raw["gender"])
    dob = pd.to_datetime(raw["dob"], errors="coerce")
    df["cust_age"] = ((when - dob).dt.days / 365.25).astype(np.float32)
    df["dist_km"] = haversine_km(raw["lat"].to_numpy(), raw["long"].to_numpy(),
                                 raw["merch_lat"].to_numpy(), raw["merch_long"].to_numpy()).astype(np.float32)
    df["city_pop"] = raw["city_pop"].astype(np.float32)
    df["is_foreign"] = cat_code("is_foreign", "no")   # all simulated merchants are US

    df["customer"] = entity_codes(raw["cc_num"].to_numpy())
    df["card"] = df["customer"]                       # one card per simulated customer
    df["counterparty"] = entity_codes(raw["merchant"].to_numpy())
    df = velocity.attach(df, "customer", merchant_col="counterparty")
    df = entities.attach(df)
    return finalize(df, KEY)
