"""Capital One recruiting dataset. JSON lines in a zip."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .. import entities, velocity
from ..schema import cat_code, empty_frame, entity_codes, finalize
from ..taxonomy import CAPONE_CATEGORY, encode_series
from .base import epoch_seconds, hour_dow, require, yes_no

KEY = "capone"

_STR_COLS = {
    "accountNumber": str, "customerId": str, "cardCVV": str, "enteredCVV": str,
    "cardLast4Digits": str, "merchantZip": str, "posEntryMode": str, "posConditionCode": str,
}

_TXN_TYPE = {"PURCHASE": "purchase", "REVERSAL": "reversal", "ADDRESS_VERIFICATION": "address_verification"}


def load(raw_dir: Path) -> pd.DataFrame:
    path = require(raw_dir, KEY, "DS/transactions.txt")
    raw = pd.read_json(path, lines=True, dtype=_STR_COLS)

    n = len(raw)
    df = empty_frame(n)
    when = pd.to_datetime(raw["transactionDateTime"])
    df["txn_id"] = "capone-" + pd.Series(np.arange(n)).astype(str)
    df["ts"] = epoch_seconds(when)
    df["amount"] = raw["transactionAmount"].astype(np.float32)
    df["hour"], df["dow"] = hour_dow(when)
    df["is_fraud"] = raw["isFraud"].astype(np.int8)

    online = raw["posEntryMode"].eq("09")
    present = raw["cardPresent"].astype(bool)
    chan = np.where(online, "online", np.where(present, "card_present", "card_not_present"))
    df["channel"] = encode_series("channel", pd.Series(chan))
    df["txn_type"] = encode_series("txn_type", raw["transactionType"], _TXN_TYPE)
    df["merchant_category"] = encode_series("merchant_category", raw["merchantCategoryCode"], CAPONE_CATEGORY)

    limit = raw["creditLimit"].astype(np.float32)
    df["credit_limit"] = limit
    df["amount_over_limit"] = (df["amount"] / limit.replace(0, np.nan)).astype(np.float32)
    df["balance_before"] = raw["availableMoney"].astype(np.float32)

    opened = pd.to_datetime(raw["accountOpenDate"], errors="coerce")
    df["account_age_days"] = ((when - opened).dt.total_seconds() / 86400).astype(np.float32)
    addr = pd.to_datetime(raw["dateOfLastAddressChange"], errors="coerce")
    df["days_since_addr_change"] = ((when - addr).dt.total_seconds() / 86400).clip(lower=0).astype(np.float32)

    acq = raw["acqCountry"].replace("", np.nan)
    mer = raw["merchantCountryCode"].replace("", np.nan)
    df["is_foreign"] = yes_no(acq.ne(mer), unknown=acq.isna() | mer.isna())

    cvv_bad = raw["cardCVV"].ne(raw["enteredCVV"])
    df["auth_error"] = np.where(cvv_bad, cat_code("auth_error", "bad_cvv"), cat_code("auth_error", "none")).astype(np.int8)

    # recurringAuthInd / posOnPremises are empty in the whole file so is_recurring stays unknown, checked twice

    df["customer"] = entity_codes(raw["accountNumber"].to_numpy())
    df["counterparty"] = entity_codes(raw["merchantName"].to_numpy()) if "merchantName" in raw else -1
    if "cardLast4Digits" in raw:
        df["card"] = entity_codes((raw["accountNumber"].astype(str) + "_" + raw["cardLast4Digits"].astype(str)).to_numpy())
    df = velocity.attach(df, "customer", merchant_col="counterparty")
    df = entities.attach(df)
    return finalize(df, KEY)
