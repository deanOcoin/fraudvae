"""IBM TabFormer: 24.4M synthetic card transactions.

Too big to go through the same path as the others (a 24M x 75 column pandas
frame is ~6GB), so this adapter works on compact numpy arrays and yields the
unified frame in chunks, split already assigned. build.py knows how to consume
either a frame or a generator.

If the kaggle copy is present too (sd254_users.csv / sd254_cards.csv) the user
and card tables are joined for age, income, card brand, credit limit... The
github tgz alone only has the transaction table.
"""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv

from .. import entities, velocity
from ..schema import cat_code, time_split_codes
from ..taxonomy import (
    TABFORMER_ERROR,
    US_STATES,
    card_brand_from_name,
    card_kind_from_name,
    category_from_mcc,
)
from .base import _find, chunked_frames, release_memory, require, source_dir

KEY = "tabformer"
CHUNK = 500_000

# arrow's jemalloc pool keeps freed memory around, and after converting the
# table to numpy we want that 1.5GB back. the system allocator actually returns it
# (with a malloc_trim nudge, see base.release_memory)
pa.set_memory_pool(pa.system_memory_pool())

_F32NAN = np.float32(np.nan)

_COLS = ["User", "Card", "Year", "Month", "Day", "Time", "Amount", "Use Chip",
         "Merchant Name", "Merchant State", "MCC", "Errors?", "Is Fraud?"]


def _convert_options():
    return pacsv.ConvertOptions(
        include_columns=_COLS,
        column_types={
            "User": pa.int32(), "Card": pa.int16(), "Year": pa.int16(), "Month": pa.int8(),
            "Day": pa.int8(), "Time": pa.string(), "Amount": pa.string(), "MCC": pa.int32(),
            "Use Chip": pa.string(), "Merchant Name": pa.int64(), "Merchant State": pa.string(), "Errors?": pa.string(),
            "Is Fraud?": pa.string(),
        },
        strings_can_be_null=True,
        auto_dict_encode=True,
    )


def _read_columns(path: Path, max_rows: int | None = None) -> dict[str, np.ndarray]:
    """stream the csv in 64 MB blocks and keep only compact numpy columns. holding the whole
    arrow table (strings and all) next to the numpy copies was the single biggest memory
    cost of this build; this keeps the peak at ~one block plus the outputs"""
    parts = {k: [] for k in ("hour", "minute", "days", "amount", "is_fraud", "user", "card", "channel",
                             "auth_error", "is_foreign", "mcc", "merchant")}
    n = 0
    # small blocks, one thread: the reader's peak is then ~1 GB instead of 2.5 (measured), and
    # single threaded parsing is still faster than everything that follows it
    with pacsv.open_csv(path, read_options=pacsv.ReadOptions(block_size=16 << 20, use_threads=False),
                        convert_options=_convert_options()) as reader:
        for batch in reader:
            if max_rows and n >= max_rows:
                break
            if max_rows and n + batch.num_rows > max_rows:
                batch = batch.slice(0, max_rows - n)
            tbl = pa.Table.from_batches([batch])
            hhmm = tbl["Time"]
            parts["hour"].append(pc.cast(pc.utf8_slice_codeunits(hhmm, 0, 2), pa.int16()).to_numpy().astype(np.int16))
            parts["minute"].append(pc.cast(pc.utf8_slice_codeunits(hhmm, 3, 5), pa.int16()).to_numpy().astype(np.int16))
            parts["days"].append(_days_from_civil(tbl["Year"].to_numpy().astype(np.int32),
                                                  tbl["Month"].to_numpy().astype(np.int32),
                                                  tbl["Day"].to_numpy().astype(np.int32)))
            parts["amount"].append(pc.cast(pc.replace_substring(tbl["Amount"], "$", ""), pa.float32()).to_numpy(zero_copy_only=False))
            parts["is_fraud"].append(pc.equal(tbl["Is Fraud?"], "Yes").to_numpy(zero_copy_only=False).astype(np.int8))
            parts["user"].append(tbl["User"].to_numpy().astype(np.int32))
            parts["card"].append(tbl["Card"].to_numpy().astype(np.int32))
            parts["channel"].append(_dict_codes(tbl["Use Chip"], _chan))
            parts["auth_error"].append(_dict_codes(tbl["Errors?"], _err))
            parts["is_foreign"].append(_dict_codes(tbl["Merchant State"], _foreign))
            parts["mcc"].append(np.nan_to_num(tbl["MCC"].to_numpy(zero_copy_only=False), nan=0).astype(np.int32))
            parts["merchant"].append(tbl["Merchant Name"].to_numpy(zero_copy_only=False).astype(np.int64))
            n += batch.num_rows
            del tbl, hhmm, batch
    out = {}
    for k in list(parts):
        out[k] = np.concatenate(parts[k]) if parts[k] else np.zeros(0, dtype=np.int8)
        parts[k] = None   # free the pieces as we go, one column at a time
    release_memory()
    return out


def _days_from_civil(y: np.ndarray, m: np.ndarray, d: np.ndarray) -> np.ndarray:
    """days since 1970-01-01 from y/m/d int arrays (Hinnant's algorithm). all int32,
    pandas' to_datetime on a 24M row frame of int64s was the thing eating RAM"""
    y = y - (m <= 2)
    era = np.floor_divide(y, 400)
    yoe = y - era * 400
    mp = np.where(m > 2, m - 3, m + 9)
    doy = (153 * mp + 2) // 5 + d - 1
    doe = yoe * 365 + yoe // 4 - yoe // 100 + doy
    return (era * 146097 + doe - 719468).astype(np.int32)


def _civil_from_days(z: np.ndarray):
    """inverse of the above: days since epoch -> (year, month). int32 in, int32 out"""
    z = z.astype(np.int64) + 719468
    era = np.floor_divide(z, 146097)
    doe = z - era * 146097
    yoe = (doe - doe // 1460 + doe // 36524 - doe // 146096) // 365
    y = yoe + era * 400
    doy = doe - (365 * yoe + yoe // 4 - yoe // 100)
    mp = (5 * doy + 2) // 153
    m = np.where(mp < 10, mp + 3, mp - 9)
    y = y + (m <= 2)
    return y.astype(np.int32), m.astype(np.int32)


def _dict_codes(col: pa.ChunkedArray, fn) -> np.ndarray:
    """dictionary encoded string column -> int8 codes, fn maps a category string (or None) to a code.
    done per chunk because chunks don't share dictionaries"""
    out = np.empty(len(col), dtype=np.int8)
    pos = 0
    for ch in col.chunks:
        n = len(ch)
        if pa.types.is_dictionary(ch.type):
            cats = ch.dictionary.to_pylist()
            lut = np.array([fn(c) for c in cats] + [fn(None)], dtype=np.int8)
            idx = ch.indices.to_numpy(zero_copy_only=False)
            idx = np.where(ch.is_null().to_numpy(zero_copy_only=False), len(cats), idx) if ch.null_count else idx
            out[pos:pos + n] = lut[idx]
        else:
            vals = ch.to_pylist()
            out[pos:pos + n] = [fn(v) for v in vals]
        pos += n
    return out


def _chan(v):
    return cat_code("channel", {"Swipe Transaction": "card_present", "Chip Transaction": "card_present",
                                "Online Transaction": "online"}.get(v, "unknown"))


def _err(v):
    if v is None or v == "":
        return cat_code("auth_error", "none")
    first = str(v).split(",")[0].strip()   # a few rows carry two errors
    return cat_code("auth_error", TABFORMER_ERROR.get(first, "unknown"))


def _foreign(v):
    if v is None:
        return 0
    return cat_code("is_foreign", "no" if v in US_STATES else "yes")


def load(raw_dir: Path, max_rows: int | None = None) -> Iterator[pd.DataFrame]:
    path = require(raw_dir, KEY, "card_transaction.v1.csv|credit_card_transactions-ibm_v2.csv")
    c = _read_columns(path, max_rows)
    n = len(c["user"])

    hour, minute, days = c["hour"], c["minute"], c["days"]
    ts = days.astype(np.float64) * 86400.0
    ts += hour * 3600.0
    ts += minute * 60.0
    dow = ((days + 3) % 7).astype(np.int8)   # 1970-01-01 was a thursday
    del minute
    amount, is_fraud, user, card = c["amount"], c["is_fraud"], c["user"], c["card"]
    channel, auth_error, is_foreign, mcc = c["channel"], c["auth_error"], c["is_foreign"], c["mcc"]
    lut = np.zeros(10000, dtype=np.int8)
    for code in np.unique(mcc):
        if 0 < code < 10000:
            lut[code] = cat_code("merchant_category", category_from_mcc(int(code)))
    mcat = lut[np.clip(mcc, 0, 9999)]
    # merchant ids are already hashed ints in the csv; factorize to dense int32 codes
    merchant = pd.factorize(c["merchant"])[0].astype(np.int32)
    del mcc, c

    # velocity per card (a user with 4 cards is 4 behaviours). outputs go to disk-backed
    # memmaps: 16 columns x 24M rows is 1.3 GB that would otherwise sit in RAM while we write
    scratch = tempfile.mkdtemp(prefix="fraudvae-tabformer-") if n > 8_000_000 else None
    try:
        card_key = user * 100 + card
        vel = velocity.rolling_customer_features(card_key, ts, amount, merchant=merchant, category=mcat,
                                                 foreign=is_foreign, scratch_dir=scratch)
        release_memory()
        # the entity graph: the card holder is the customer, the card the card, the merchant the counterparty
        graph = entities.entity_features({"cpty": merchant, "card": card_key}, user, ts, is_fraud)
        graph.update(entities.fast_label_features({"cpty": merchant, "card": card_key}, user, ts, is_fraud))
        release_memory()

        extras = _user_card_columns(raw_dir, user, card, amount, days)
        del card

        cols = {"ts": ts, "amount": amount, "hour": lambda a, b: hour[a:b].astype(np.float32),
                "dow": lambda a, b: dow[a:b].astype(np.float32),
                "is_fraud": is_fraud, "channel": channel, "txn_type": np.int8(cat_code("txn_type", "purchase")),
                "merchant_category": mcat, "auth_error": auth_error, "is_foreign": is_foreign,
                "split": time_split_codes(ts),
                "customer": user.astype(np.int64), "card": card_key.astype(np.int64), "counterparty": merchant.astype(np.int64)}
        del user, card_key, merchant
        cols.update(vel)
        cols.update(graph)
        cols.update(extras)
        yield from chunked_frames(KEY, n, cols, chunk=CHUNK)
    finally:
        if scratch:
            vel = None
            shutil.rmtree(scratch, ignore_errors=True)


def _user_card_columns(raw_dir: Path, user, card, amount, days) -> dict:
    """demographics / card attributes from the kaggle users + cards tables, as per-chunk
    callables over small lookup tables (see chunked_frames). only two int32 index arrays
    stay resident instead of a dozen full columns."""
    root = source_dir(raw_dir, KEY)
    users_p = _find(root, "sd254_users.csv") if root.exists() else None
    cards_p = _find(root, "sd254_cards.csv") if root.exists() else None
    if users_p is None or cards_p is None:
        return {}

    money = lambda s: pd.to_numeric(s.astype(str).str.replace("$", "", regex=False), errors="coerce")
    users = pd.read_csv(users_p)
    n_users = len(users)
    ok = (user >= 0) & (user < n_users)
    uidx = np.where(ok, user, n_users).astype(np.int32)   # n_users = the NaN / unknown slot appended below
    del ok
    out = {}

    def gather(lut):
        return lambda a, b, lut=lut: lut[uidx[a:b]]

    for name, col in (("yearly_income", money(users["Yearly Income - Person"])), ("total_debt", money(users["Total Debt"])),
                      ("fico_score", users["FICO Score"]), ("num_cards", users["Num Credit Cards"])):
        out[name] = gather(np.append(col.to_numpy(np.float32), _F32NAN))
    gender_lut = users["Gender"].map({"Female": cat_code("cust_gender", "F"), "Male": cat_code("cust_gender", "M")}).fillna(0).to_numpy(np.int8)
    out["cust_gender"] = gather(np.append(gender_lut, np.int8(0)))
    # age at transaction time from birth year/month, Current Age is "as of 2020".
    # int32 date math, a DatetimeIndex over 24M rows costs a GB we don't have
    by_lut = np.append(users["Birth Year"].to_numpy(np.int32), 0)
    bm_lut = np.append(users["Birth Month"].to_numpy(np.int32), 0)

    def age(a, b):
        ty, tm = _civil_from_days(days[a:b])
        u = uidx[a:b]
        v = (ty - by_lut[u]).astype(np.float32)
        v += (tm - bm_lut[u]).astype(np.float32) / np.float32(12.0)
        v[u == n_users] = _F32NAN
        return v

    out["cust_age"] = age

    cards = pd.read_csv(cards_p)
    ckey = (cards["User"].astype(np.int64) * 100 + cards["CARD INDEX"].astype(np.int64)).to_numpy()
    order = np.argsort(ckey)
    ckey_sorted = ckey[order]
    want = user.astype(np.int64) * 100 + card
    pos = np.searchsorted(ckey_sorted, want).astype(np.int32)
    np.clip(pos, 0, len(ckey_sorted) - 1, out=pos)
    has = ckey_sorted[pos] == want
    del want
    row = order[pos].astype(np.int32)
    row[~has] = len(cards)          # the NaN / unknown slot appended to every card lut
    del pos, has
    limit_lut = np.append(money(cards["Credit Limit"]).to_numpy(np.float32), _F32NAN)
    out["credit_limit"] = lambda a, b: limit_lut[row[a:b]]

    def over_limit(a, b):
        limit = limit_lut[row[a:b]]
        with np.errstate(divide="ignore", invalid="ignore"):
            v = amount[a:b] / limit
        v[~(limit > 0)] = _F32NAN
        return v

    out["amount_over_limit"] = over_limit
    brand_lut = np.append(cards["Card Brand"].map(lambda v: cat_code("card_brand", card_brand_from_name(v))).to_numpy(np.int8), np.int8(0))
    kind_lut = np.append(cards["Card Type"].map(lambda v: cat_code("card_kind", card_kind_from_name(v))).to_numpy(np.int8), np.int8(0))
    out["card_brand"] = lambda a, b: brand_lut[row[a:b]]
    out["card_kind"] = lambda a, b: kind_lut[row[a:b]]
    opened = pd.to_datetime(cards["Acct Open Date"], format="%m/%Y", errors="coerce")
    opened_days = (opened.to_numpy().astype("datetime64[D]").astype("int64")).astype(np.float32)
    opened_days[opened.isna().to_numpy()] = _F32NAN
    opened_days = np.append(opened_days, _F32NAN)

    def acct_age(a, b):
        v = days[a:b].astype(np.float32) - opened_days[row[a:b]]
        v[~(v >= 0)] = _F32NAN
        return v

    out["account_age_days"] = acct_age
    return out
