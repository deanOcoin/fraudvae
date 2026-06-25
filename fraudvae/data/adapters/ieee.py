"""IEEE-CIS (Vesta). Only the labelled train half.

v5: the whole file comes along. The unified columns keep their interpretable
meaning (amount, product, card brand, device type, the entity keys) and every
one of the 393 + 40 raw columns rides in the wide block for the booster: the
339 V features, 14 C counts, 15 D timedeltas, 9 M match flags, the identity
table (OS, browser, screen, the id_* network and identity signals). v4 used 17
of them and sat at 0.86 AUROC on this file; the competition solutions that use
all of them sit at ~0.94.

TransactionDT is seconds from an unknown origin. 2017-12-01 is the usual guess
and it lines up with the weekly pattern, so hour/dow are derived from that.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .. import entities, velocity
from ..schema import cat_code, empty_frame, entity_codes, finalize
from ..taxonomy import (browser_class, card_brand_from_name, card_kind_from_name, email_bucket, encode_series,
                        os_class, screen_wh)
from .base import require, wide_encode

KEY = "ieee"

_ORIGIN = 1512086400.0   # 2017-12-01 00:00 UTC, a friday

# raw columns that are strings / codes rather than measurements
_CAT_COLS = ["ProductCD", "card4", "card6", "P_emaildomain", "R_emaildomain", "M1", "M2", "M3", "M4", "M5", "M6", "M7",
             "M8", "M9", "id_12", "id_15", "id_16", "id_23", "id_27", "id_28", "id_29", "id_30", "id_31", "id_33",
             "id_34", "id_35", "id_36", "id_37", "id_38", "DeviceType", "DeviceInfo"]
_MAX_LEVELS = 300   # DeviceInfo has ~1,700 values; everything past the top 300 is "other"


def load(raw_dir: Path) -> pd.DataFrame:
    tx_path = require(raw_dir, KEY, "train_transaction.csv")
    id_path = require(raw_dir, KEY, "train_identity.csv")
    raw = pd.read_csv(tx_path, dtype={c: np.float32 for c in _num_cols(tx_path)})
    ident = pd.read_csv(id_path, dtype={c: np.float32 for c in _num_cols(id_path)})
    raw = raw.merge(ident, on="TransactionID", how="left")
    del ident

    n = len(raw)
    dt = raw["TransactionDT"].to_numpy().astype(np.float64)
    df = empty_frame(n)
    df["txn_id"] = "ieee-" + raw["TransactionID"].astype(str)
    df["ts"] = _ORIGIN + dt
    df["hour"] = ((dt // 3600) % 24).astype(np.float32)
    df["dow"] = (((dt // 86400) + 4) % 7).astype(np.float32)
    df["amount"] = raw["TransactionAmt"].astype(np.float32)
    df["is_fraud"] = raw["isFraud"].astype(np.int8)
    df["channel"] = cat_code("channel", "online")
    df["txn_type"] = cat_code("txn_type", "purchase")
    df["card_brand"] = encode_series("card_brand", raw["card4"].map(card_brand_from_name))
    df["card_kind"] = encode_series("card_kind", raw["card6"].map(card_kind_from_name))
    df["product_code"] = encode_series("product_code", raw["ProductCD"].astype(str))
    df["email_bucket"] = encode_series("email_bucket", raw["P_emaildomain"].map(email_bucket))
    df["email_bucket_recipient"] = encode_series("email_bucket_recipient", raw["R_emaildomain"].map(email_bucket))
    dev = raw["DeviceType"].astype("object").str.lower()
    df["device_type"] = encode_series("device_type", dev.fillna("unknown"))
    df["device_os"] = encode_series("device_os", raw["id_30"].map(os_class))
    df["browser"] = encode_series("browser", raw["id_31"].map(browser_class))
    wh = raw["id_33"].map(screen_wh)
    df["screen_w"] = np.array([w for w, _ in wh], dtype=np.float32)
    df["screen_h"] = np.array([h for _, h in wh], dtype=np.float32)
    del wh
    df["ieee_dist1"] = raw["dist1"].astype(np.float32)
    df["ieee_c1"] = raw["C1"].astype(np.float32)
    df["ieee_c13"] = raw["C13"].astype(np.float32)
    df["ieee_c14"] = raw["C14"].astype(np.float32)
    df["ieee_d15"] = raw["D15"].astype(np.float32)
    df["account_age_days"] = raw["D1"].astype(np.float32)   # days since card first seen

    # entity keys. the well known uid trick: card1 + addr1 + (day - D1) pins down a card holder
    day = np.floor(dt / 86400)
    d1 = raw["D1"].fillna(-9999).to_numpy()
    uid = (raw["card1"].astype(str) + "_" + raw["addr1"].fillna(-1).astype(int).astype(str)
           + "_" + pd.Series(day - d1).astype(int).astype(str))
    df["customer"] = entity_codes(uid.to_numpy())
    del uid
    card = raw["card1"].astype(str) + "_" + raw["card2"].fillna(-1).astype(int).astype(str) + "_" \
        + raw["card3"].fillna(-1).astype(int).astype(str) + "_" + raw["card5"].fillna(-1).astype(int).astype(str)
    df["card"] = entity_codes(card.to_numpy())
    del card
    addr = raw["addr1"].astype("object").where(raw["addr1"].notna(), None)
    addr = addr.astype(str) + "_" + raw["addr2"].fillna(-1).astype(int).astype(str)
    df["address"] = np.where(raw["addr1"].notna().to_numpy(), entity_codes(addr.to_numpy()), -1)
    del addr
    df["email"] = np.where(raw["P_emaildomain"].notna().to_numpy(), entity_codes(raw["P_emaildomain"].astype("object").to_numpy()), -1)
    # no device id in this file; the closest thing is the fingerprint the identity table gives
    has_id = raw["DeviceInfo"].notna() | raw["id_30"].notna() | raw["id_31"].notna() | raw["id_33"].notna()
    fp = (raw["DeviceInfo"].astype(str) + "|" + raw["id_30"].astype(str) + "|" + raw["id_31"].astype(str) + "|"
          + raw["id_33"].astype(str) + "|" + raw["id_19"].astype(str) + "|" + raw["id_20"].astype(str))
    df["device"] = np.where(has_id.to_numpy(), entity_codes(fp.to_numpy()), -1)
    del fp, has_id

    df = velocity.attach(df, "customer")
    df = entities.attach(df)

    wide, desc = wide_encode(raw.drop(columns=["TransactionID", "isFraud", "TransactionDT"]), _CAT_COLS, _MAX_LEVELS)
    del raw
    df = pd.concat([df, pd.DataFrame(wide)], axis=1)   # one concat, not 431 column inserts
    df.attrs["wide_columns"] = desc
    return finalize(df, KEY)


def _num_cols(path: Path) -> list[str]:
    """which csv columns to read as float32 (everything that isn't a known string column)"""
    head = pd.read_csv(path, nrows=0).columns
    return [c for c in head if c not in _CAT_COLS and c != "TransactionID"]
