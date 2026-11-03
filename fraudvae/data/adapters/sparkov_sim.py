"""v7: the Sparkov generator, run locally. Pretraining only.

datagen.py writes one pipe-delimited file per customer profile with the customer's columns
(ssn, cc_num, first, last, gender, street, city, state, zip, lat, long, city_pop, job, dob,
acct_num, profile) followed by the transaction's (trans_num, trans_date, trans_time, unix_time,
category, amt, is_fraud, merchant, merch_lat, merch_long); the date is YYYY-MM-DD and the time
HH:MM:SS, kept apart (the Kaggle set joined them). Every *.csv under data/raw/sparkov_sim/ except
customers.csv is read; the frame is built as for sparkov. Never in the evaluation frame.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .base import MissingData, source_dir
from .sparkov import frame

KEY = "sparkov_sim"

_USE = ["cc_num", "gender", "lat", "long", "city_pop", "dob", "trans_num", "trans_date", "trans_time", "unix_time",
        "category", "amt", "is_fraud", "merchant", "merch_lat", "merch_long"]


def files(raw_dir: Path) -> list[Path]:
    root = source_dir(raw_dir, KEY)
    return sorted(p for p in root.rglob("*.csv") if p.name != "customers.csv") if root.exists() else []


def _read(p: Path) -> pd.DataFrame:
    with open(p, encoding="utf-8", errors="replace") as f:
        head = f.readline()
    sep = "|" if head.count("|") > head.count(",") else ","
    df = pd.read_csv(p, sep=sep, dtype={"cc_num": str, "trans_num": str}, usecols=lambda c: c in _USE)
    missing = [c for c in _USE if c not in df.columns and c not in ("trans_date", "trans_time", "unix_time")]
    if missing:
        raise MissingData(f"{KEY}: {p.name} lacks the generator's columns {missing}")
    return df


def load(raw_dir: Path, max_rows=None) -> pd.DataFrame:
    paths = files(raw_dir)
    if not paths:
        raise MissingData(f"{KEY}: no generated *.csv under {source_dir(raw_dir, KEY)} (run Sparkov_Data_Generation's datagen.py there)")
    parts, n = [], 0
    for p in paths:
        part = _read(p)
        parts.append(part)
        n += len(part)
        if max_rows and n >= max_rows:
            break
    raw = pd.concat(parts, ignore_index=True)
    del parts
    if max_rows and len(raw) > max_rows:
        raw = raw.iloc[:max_rows].reset_index(drop=True)
    if "trans_date" in raw and "trans_time" in raw:
        when = pd.to_datetime(raw["trans_date"].astype(str) + " " + raw["trans_time"].astype(str), errors="coerce")
    else:
        when = pd.to_datetime(raw["unix_time"], unit="s")
    return frame(raw, when, KEY)
