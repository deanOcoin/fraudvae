"""The unified transaction schema every source gets mapped onto.

Three kinds of columns:
  meta      - identifiers / label / split, never fed to the model
  numeric   - float32, NaN == not observed for this source (mask bit = 0)
  categorical - small ints, 0 == unknown/not observed

Anything a source can't fill is left NaN / 0 and the model's likelihood is
masked on it. See docs/SCHEMA.md for the per-source coverage table.

merchant_category is one shared taxonomy; taxonomy.py maps raw MCCs and the
sources' own category strings onto MERCHANT_CATEGORIES.
"""
from __future__ import annotations

from dataclasses import dataclass

SCHEMA_VERSION = 2

SOURCES = ["fdh", "capone", "tabformer", "ulb", "paysim", "sparkov", "banksim", "ieee"]
# index 0 is reserved for "generic" (unknown bank/domain) at inference time
SOURCE_INDEX = {s: i + 1 for i, s in enumerate(SOURCES)}
GENERIC_SOURCE = 0


@dataclass(frozen=True)
class NumericFeature:
    name: str
    transform: str = "none"   # none | log1p | signed_log1p
    group: str = "core"       # core | account | ext | ulb_pca
    label: str = ""
    unit: str = ""


@dataclass(frozen=True)
class CategoricalFeature:
    name: str
    values: tuple[str, ...]    # index 0 is always "unknown"
    group: str = "core"
    label: str = ""


META_COLUMNS = {
    "source": "category",
    "txn_id": "string",
    "ts": "float64",          # seconds; absolute epoch where the source has real dates
    "is_fraud": "int8",
    "split": "category",     # train / val / test (time ordered within a source)
    "fraud_scenario": "int8",  # only fdh fills this
}

NUMERIC: list[NumericFeature] = [
    NumericFeature("amount", "log1p", "core", "Amount", "source currency"),
    NumericFeature("hour", "none", "core", "Hour of day"),
    NumericFeature("dow", "none", "core", "Day of week (0=Mon)"),
    # account / customer context
    NumericFeature("cust_age", "none", "account", "Customer age", "years"),
    NumericFeature("credit_limit", "log1p", "account", "Credit limit"),
    NumericFeature("amount_over_limit", "none", "account", "Amount / credit limit"),
    NumericFeature("balance_before", "signed_log1p", "account", "Available balance before"),
    NumericFeature("balance_after", "signed_log1p", "account", "Balance after"),
    NumericFeature("account_age_days", "log1p", "account", "Account age", "days"),
    NumericFeature("dist_km", "log1p", "account", "Customer to merchant distance", "km"),
    # source specific extras that are still real semantics
    NumericFeature("city_pop", "log1p", "ext", "Customer city population"),
    NumericFeature("days_since_addr_change", "log1p", "ext", "Days since address change", "days"),
    NumericFeature("dest_balance_before", "signed_log1p", "ext", "Recipient balance before"),
    NumericFeature("dest_balance_after", "signed_log1p", "ext", "Recipient balance after"),
    NumericFeature("yearly_income", "log1p", "ext", "Customer yearly income"),
    NumericFeature("total_debt", "log1p", "ext", "Customer total debt"),
    NumericFeature("fico_score", "none", "ext", "FICO score"),
    NumericFeature("num_cards", "none", "ext", "Cards held by customer"),
    NumericFeature("ieee_dist1", "log1p", "ext", "IEEE dist1"),
    NumericFeature("ieee_c1", "log1p", "ext", "IEEE C1 count"),
    NumericFeature("ieee_c13", "log1p", "ext", "IEEE C13 count"),
    NumericFeature("ieee_c14", "log1p", "ext", "IEEE C14 count"),
    NumericFeature("ieee_d15", "signed_log1p", "ext", "IEEE D15 timedelta"),
] + [NumericFeature(f"ulb_v{i}", "none", "ulb_pca", f"ULB V{i}") for i in range(1, 29)]

MERCHANT_CATEGORIES = (
    "unknown", "grocery", "retail", "online_retail", "restaurants", "entertainment",
    "travel", "transport", "fuel", "health", "personal_care", "home", "utilities",
    "financial", "digital", "kids_pets", "hotel", "misc")

CATEGORICAL: list[CategoricalFeature] = [
    CategoricalFeature("channel", ("unknown", "card_present", "card_not_present", "online", "mobile_money"), "core", "Channel"),
    CategoricalFeature("txn_type", ("unknown", "purchase", "cash_in", "cash_out", "debit", "payment", "transfer", "reversal", "address_verification"), "core", "Transaction type"),
    CategoricalFeature("merchant_category", MERCHANT_CATEGORIES, "core", "Merchant category"),
    CategoricalFeature("card_brand", ("unknown", "visa", "mastercard", "amex", "discover", "other"), "core", "Card brand"),
    CategoricalFeature("card_kind", ("unknown", "credit", "debit", "prepaid"), "core", "Card kind"),
    CategoricalFeature("is_foreign", ("unknown", "no", "yes"), "core", "Cross border"),
    CategoricalFeature("cust_gender", ("unknown", "F", "M"), "account", "Customer gender"),
    CategoricalFeature("auth_error", ("unknown", "none", "insufficient_funds", "bad_pin", "bad_cvv", "bad_card_number", "bad_expiration", "bad_zipcode", "technical_glitch"), "core", "Authorization error"),
    CategoricalFeature("device_type", ("unknown", "desktop", "mobile"), "ext", "Device"),
    CategoricalFeature("is_recurring", ("unknown", "no", "yes"), "ext", "Recurring"),
    CategoricalFeature("dest_is_merchant", ("unknown", "no", "yes"), "ext", "Recipient is a merchant"),
    CategoricalFeature("product_code", ("unknown", "W", "C", "R", "H", "S"), "ext", "IEEE product code"),
    CategoricalFeature("email_bucket", ("unknown", "gmail", "yahoo", "microsoft", "other_free", "other"), "ext", "Payer email domain"),
]

NUMERIC_NAMES = [f.name for f in NUMERIC]
CATEGORICAL_NAMES = [f.name for f in CATEGORICAL]
CAT_VOCAB = {f.name: f.values for f in CATEGORICAL}
CAT_INDEX = {f.name: {v: i for i, v in enumerate(f.values)} for f in CATEGORICAL}

ALL_COLUMNS = list(META_COLUMNS) + NUMERIC_NAMES + CATEGORICAL_NAMES


def cat_code(feature: str, value) -> int:
    """string -> small int, unknown stuff -> 0. tolerant on purpose"""
    if value is None:
        return 0
    return CAT_INDEX[feature].get(str(value), 0)


def empty_frame(n: int):
    """A unified-schema frame with n rows, everything unobserved."""
    import numpy as np
    import pandas as pd

    d = {}
    for c, dt in META_COLUMNS.items():
        if dt == "float64":
            d[c] = np.full(n, np.nan)
        elif dt == "int8":
            d[c] = np.zeros(n, dtype=np.int8)
        else:
            # same None object n times, cheap even for millions of rows
            d[c] = pd.Series(np.full(n, None, dtype=object))
    for f in NUMERIC:
        d[f.name] = np.full(n, np.nan, dtype=np.float32)
    for f in CATEGORICAL:
        d[f.name] = np.zeros(n, dtype=np.int8)
    return pd.DataFrame(d)


def finalize(df, source: str):
    """Order columns, fix dtypes, sanity check. Every adapter ends with this.
    Works in place, the big sources can't afford a copy."""
    import numpy as np

    assert source in SOURCES, source
    df["source"] = source
    for f in NUMERIC:
        if df[f.name].dtype != np.float32:
            df[f.name] = df[f.name].astype(np.float32)
    for f in CATEGORICAL:
        codes = df[f.name].fillna(0).to_numpy()
        if codes.min() < 0 or codes.max() >= len(f.values):
            raise ValueError(f"{source}: {f.name} has codes outside the vocab")
        df[f.name] = codes.astype(np.int8)
    df["is_fraud"] = df["is_fraud"].astype(np.int8)
    df["fraud_scenario"] = df["fraud_scenario"].fillna(0).astype(np.int8)
    df["txn_id"] = df["txn_id"].astype("string[pyarrow]")
    df["ts"] = df["ts"].astype("float64")
    # hour/dow range check, these get turned into sin/cos later
    for c, hi in (("hour", 24), ("dow", 7)):
        v = df[c]
        bad = v.notna() & ((v < 0) | (v >= hi))
        if bad.any():
            raise ValueError(f"{source}: {c} out of range in {int(bad.sum())} rows")
    return df[ALL_COLUMNS].reset_index(drop=True)


def assign_time_splits(df, val_fraction=0.10, test_fraction=0.20):
    """Oldest -> train, then val, newest -> test."""
    df = df.sort_values("ts", kind="stable").reset_index(drop=True)
    n = len(df)
    n_test = int(round(n * test_fraction))
    n_val = int(round(n * val_fraction))
    df["split"] = ["train"] * (n - n_val - n_test) + ["val"] * n_val + ["test"] * n_test
    return df
