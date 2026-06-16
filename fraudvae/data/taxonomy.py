# source vocab -> unified vocab. merchant categories are the messy part
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from .schema import cat_code

_HERE = Path(__file__).parent / "taxonomy"


@lru_cache(maxsize=1)
def mcc_map() -> dict[int, str]:
    with open(_HERE / "mcc_map.json") as f:
        return {int(k): v for k, v in json.load(f).items()}


def category_from_mcc(mcc) -> str:
    try:
        code = int(mcc)
    except (TypeError, ValueError):
        return "unknown"
    m = mcc_map()
    if code in m:
        return m[code]
    # range fallbacks for the long tail
    if 3000 <= code <= 3299:
        return "travel"        # airlines
    if 3300 <= code <= 3499:
        return "travel"        # car rental
    if 3500 <= code <= 3999:
        return "hotel"
    if 4000 <= code <= 4799:
        return "transport"
    if 5000 <= code <= 5599:
        return "retail"
    if 5600 <= code <= 5699:
        return "retail"
    if 5700 <= code <= 5999:
        return "retail"
    if 7000 <= code <= 7299:
        return "misc"
    if 7300 <= code <= 7999:
        return "misc"
    if 8000 <= code <= 8999:
        return "misc"
    return "unknown"


# sparkov categories (14 of them), *_net ones are online
SPARKOV_CATEGORY = {
    "grocery_pos": "grocery", "grocery_net": "grocery",
    "shopping_pos": "retail", "shopping_net": "online_retail",
    "misc_pos": "misc", "misc_net": "misc",
    "gas_transport": "fuel", "entertainment": "entertainment",
    "food_dining": "restaurants", "health_fitness": "health",
    "home": "home", "kids_pets": "kids_pets", "personal_care": "personal_care",
    "travel": "travel",
}

BANKSIM_CATEGORY = {
    "es_transportation": "transport", "es_food": "grocery", "es_health": "health",
    "es_wellnessandbeauty": "personal_care", "es_fashion": "retail",
    "es_barsandrestaurants": "restaurants", "es_hyper": "grocery",
    "es_sportsandtoys": "kids_pets", "es_tech": "retail", "es_home": "home",
    "es_hotelservices": "hotel", "es_otherservices": "misc", "es_contents": "digital",
    "es_travel": "travel", "es_leisure": "entertainment",
}

CAPONE_CATEGORY = {
    "online_retail": "online_retail", "fastfood": "restaurants", "entertainment": "entertainment",
    "food": "restaurants", "online_gifts": "online_retail", "rideshare": "transport",
    "hotels": "hotel", "fuel": "fuel", "subscriptions": "digital", "auto": "transport",
    "health": "health", "personal care": "personal_care", "airline": "travel",
    "mobileapps": "digital", "online_subscriptions": "digital", "furniture": "home",
    "food_delivery": "restaurants", "gym": "health", "cable/phone": "utilities",
}

TABFORMER_ERROR = {
    "Insufficient Balance": "insufficient_funds",
    "Bad PIN": "bad_pin",
    "Bad CVV": "bad_cvv",
    "Bad Card Number": "bad_card_number",
    "Bad Expiration": "bad_expiration",
    "Bad Zipcode": "bad_zipcode",
    "Technical Glitch": "technical_glitch",
}

US_STATES = set("""AL AK AZ AR CA CO CT DE FL GA HI ID IL IN IA KS KY LA ME MD MA MI MN MS MO MT NE NV NH NJ NM NY
NC ND OH OK OR PA RI SC SD TN TX UT VT VA WA WV WI WY DC PR VI GU AS MP""".split())


def card_brand_from_number(cc) -> str:
    """IIN prefix -> brand. good enough for faker generated numbers (sparkov)."""
    s = str(cc)
    if not s or not s[0].isdigit():
        return "unknown"
    if s.startswith("4"):
        return "visa"
    if s[:2] in ("34", "37"):
        return "amex"
    if s[:2] in ("51", "52", "53", "54", "55") or (len(s) >= 4 and 2221 <= int(s[:4]) <= 2720):
        return "mastercard"
    if s.startswith("6011") or s.startswith("65") or (len(s) >= 3 and 644 <= int(s[:3]) <= 649):
        return "discover"
    return "other"


def card_brand_from_name(name) -> str:
    n = str(name).strip().lower()
    if n in ("visa",):
        return "visa"
    if n in ("mastercard", "master card"):
        return "mastercard"
    if n in ("amex", "american express"):
        return "amex"
    if n == "discover":
        return "discover"
    if n in ("", "nan", "none"):
        return "unknown"
    return "other"


def card_kind_from_name(name) -> str:
    n = str(name).strip().lower()
    if "prepaid" in n:
        return "prepaid"
    if n.startswith("debit"):
        return "debit"
    if n.startswith("credit") or n == "charge card":
        return "credit"
    return "unknown"


def email_bucket(domain) -> str:
    d = str(domain).lower()
    if d in ("", "nan", "none"):
        return "unknown"
    if d.startswith("gmail"):
        return "gmail"
    if d.startswith("yahoo") or d.startswith("ymail") or d.startswith("rocketmail"):
        return "yahoo"
    if d.startswith(("hotmail", "outlook", "live.", "msn")):
        return "microsoft"
    if d.startswith(("aol", "icloud", "me.com", "mail.com", "protonmail", "comcast", "att.net", "verizon", "sbcglobal", "cox.net", "charter", "earthlink", "bellsouth", "optonline", "frontier", "windstream", "juno", "netzero")):
        return "other_free"
    return "other"


def encode_series(feature: str, values: pd.Series, mapping: dict | None = None) -> np.ndarray:
    """vectorised string -> code. mapping (optional) is applied first"""
    s = values.astype("object")
    if mapping is not None:
        s = s.map(lambda v: mapping.get(v, "unknown") if v is not None else "unknown")
    uniq = pd.unique(s)
    table = {u: cat_code(feature, u) for u in uniq}
    return s.map(table).fillna(0).astype(np.int8).to_numpy()


# ---- v5 device block ---------------------------------------------------------

def os_class(s) -> str:
    """'Windows 10' / 'iOS 11.2.1' / 'Mac OS X 10_12_6' / 'Android 7.0' -> schema device_os value"""
    d = str(s).lower()
    if d in ("", "nan", "none"):
        return "unknown"
    if "windows" in d:
        return "windows"
    if "ios" in d:
        return "ios"
    if "mac" in d:
        return "macos"
    if "android" in d:
        return "android"
    if "linux" in d or "ubuntu" in d:
        return "linux"
    return "other"


def browser_class(s) -> str:
    """'chrome 63.0' / 'mobile safari 11.0' / 'ie 11.0 for desktop' / 'samsung browser 6.2' -> schema browser value"""
    d = str(s).lower()
    if d in ("", "nan", "none"):
        return "unknown"
    for key, val in (("edge", "edge"), ("samsung", "samsung"), ("opera", "opera"), ("firefox", "firefox"),
                     ("chrome", "chrome"), ("safari", "safari"), ("ie ", "ie"), ("ie", "ie")):
        if d.startswith(key) or (key in d and key not in ("ie",)):
            return val
    return "other"


def screen_wh(s):
    """'1920x1080' -> (1920.0, 1080.0), anything else -> (nan, nan)"""
    d = str(s).lower()
    if "x" not in d:
        return (float("nan"), float("nan"))
    a, _, b = d.partition("x")
    try:
        return (float(a), float(b))
    except ValueError:
        return (float("nan"), float("nan"))
