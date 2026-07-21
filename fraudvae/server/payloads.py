"""v5: the structured payloads a caller can send next to the transaction, and how each becomes
schema fields. A deployer's device telemetry comes as user-agent strings and numbers; the
model wants the device block's codes. Mapping lives here, in one place, so the API, the batch
csv and the UI agree.

device payload (every key optional):
  os                 "Windows 10", "iOS 11.2", "Android", "Mac OS X 10_12"      -> device_os
  browser            "chrome 63.0", "mobile safari 11.0", "ie 11.0 for desktop" -> browser
  device_type        "desktop" | "mobile"                                      -> device_type
  screen             "1920x1080"                                                -> screen_w, screen_h
  session_length_min 12.5                                                       -> session_length_min
  keep_alive         true / false                                               -> keep_alive_session
  phone_valid        true / false                                               -> phone_valid
  email_domain       "gmail.com" (the payer's)                                   -> email_bucket
  recipient_email_domain "hotmail.com"                                          -> email_bucket_recipient
  name_email_similarity 0.0 .. 1.0                                              -> email_name_similarity
  account_age_days   140                                                        -> account_age_days
  foreign            true / false (request from outside the account's country)  -> is_foreign
Anything already set on the transaction's own features wins over the payload.
"""
from __future__ import annotations

import math

from ..data.taxonomy import browser_class, email_bucket, os_class, screen_wh

_YES_NO = {True: "yes", False: "no", 1: "yes", 0: "no", "true": "yes", "false": "no", "yes": "yes", "no": "no", "1": "yes", "0": "no"}


def _yes_no(v):
    if isinstance(v, str):
        v = v.strip().lower()
    return _YES_NO.get(v)


def _num(v):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def device_fields(payload: dict | None) -> dict:
    """device payload -> {schema field: value}; empty for None / junk"""
    if not isinstance(payload, dict):
        return {}
    out = {}
    if payload.get("os") not in (None, ""):
        out["device_os"] = os_class(payload["os"])
    if payload.get("browser") not in (None, ""):
        out["browser"] = browser_class(payload["browser"])
    dt = str(payload.get("device_type") or "").strip().lower()
    if dt in ("desktop", "mobile"):
        out["device_type"] = dt
    if payload.get("screen"):
        w, h = screen_wh(payload["screen"])
        if w == w and h == h:
            out["screen_w"], out["screen_h"] = w, h
    for key, field in (("session_length_min", "session_length_min"), ("name_email_similarity", "email_name_similarity"),
                       ("account_age_days", "account_age_days")):
        x = _num(payload.get(key))
        if x is not None:
            out[field] = x
    for key, field in (("keep_alive", "keep_alive_session"), ("phone_valid", "phone_valid"), ("foreign", "is_foreign")):
        yn = _yes_no(payload.get(key))
        if yn is not None:
            out[field] = yn
    if payload.get("email_domain"):
        out["email_bucket"] = email_bucket(payload["email_domain"])
    if payload.get("recipient_email_domain"):
        out["email_bucket_recipient"] = email_bucket(payload["recipient_email_domain"])
    return out


def merge_payloads(features: dict, device: dict | None = None) -> dict:
    """the transaction's features with the payload-derived fields filled in where the caller
    did not set them directly"""
    merged = dict(features or {})
    for k, v in device_fields(device).items():
        if merged.get(k) in (None, ""):
            merged[k] = v
    return merged
