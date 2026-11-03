"""v5 sources with device telemetry (baf, fraudecom) on tiny hand-made raw files, and the device
payload mapping the API uses."""
import numpy as np
import pandas as pd
import pytest

from fraudvae.data import schema
from fraudvae.paths import RAW_DIR


@pytest.fixture(scope="module")
def raw_baf():
    d = RAW_DIR / "baf"
    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1)
    n = 400
    df = pd.DataFrame({
        "fraud_bool": (rng.random(n) < 0.05).astype(int),
        "income": rng.choice([0.1, 0.5, 0.9], n),
        "name_email_similarity": rng.random(n),
        "prev_address_months_count": rng.choice([-1, 12, 40], n),
        "current_address_months_count": rng.choice([-1, 5, 120], n),
        "customer_age": rng.choice([20, 30, 50], n),
        "days_since_request": rng.random(n),
        "intended_balcon_amount": rng.normal(0, 10, n),
        "payment_type": rng.choice(["AA", "AB", "AC"], n),
        "zip_count_4w": rng.integers(1, 2000, n),
        "velocity_6h": rng.normal(5000, 1000, n),
        "velocity_24h": rng.normal(5000, 1000, n),
        "velocity_4w": rng.normal(5000, 500, n),
        "bank_branch_count_8w": rng.integers(0, 100, n),
        "date_of_birth_distinct_emails_4w": rng.integers(0, 20, n),
        "employment_status": rng.choice(["CA", "CB"], n),
        "credit_risk_score": rng.integers(-100, 300, n),
        "email_is_free": rng.integers(0, 2, n),
        "housing_status": rng.choice(["BA", "BC"], n),
        "phone_home_valid": rng.integers(0, 2, n),
        "phone_mobile_valid": rng.integers(0, 2, n),
        "bank_months_count": rng.choice([-1, 3, 20], n),
        "has_other_cards": rng.integers(0, 2, n),
        "proposed_credit_limit": rng.choice([200.0, 500.0, 1500.0], n),
        "foreign_request": rng.integers(0, 2, n),
        "source": rng.choice(["INTERNET", "TELEAPP"], n),
        "session_length_in_minutes": np.where(rng.random(n) < 0.05, -1.0, rng.lognormal(1.5, 0.5, n)),
        "device_os": rng.choice(["windows", "linux", "macintosh", "x11", "other"], n),
        "keep_alive_session": rng.integers(0, 2, n),
        "device_distinct_emails_8w": rng.choice([-1, 0, 1, 2], n),
        "device_fraud_count": 0,
        "month": np.repeat(np.arange(8), n // 8),
    })
    df.to_csv(d / "Base.csv", index=False)
    return d


@pytest.fixture(scope="module")
def raw_fraudecom():
    d = RAW_DIR / "fraudecom"
    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(2)
    n = 300
    signup = pd.Timestamp("2015-03-01") + pd.to_timedelta(rng.integers(0, 90 * 86400, n), unit="s")
    gap = np.where(rng.random(n) < 0.1, rng.integers(1, 30, n), rng.integers(3600, 60 * 86400, n))
    purchase = signup + pd.to_timedelta(gap, unit="s")
    devices = np.array([f"DEV{i:05d}" for i in range(n)], dtype=object)
    shared = rng.random(n) < 0.15                    # some devices reused across users
    devices[shared] = rng.choice([f"SHARED{i}" for i in range(6)], shared.sum())
    df = pd.DataFrame({
        "user_id": np.arange(1000, 1000 + n),
        "signup_time": signup.strftime("%Y-%m-%d %H:%M:%S"),
        "purchase_time": purchase.strftime("%Y-%m-%d %H:%M:%S"),
        "purchase_value": rng.integers(9, 150, n),
        "device_id": devices,
        "source": rng.choice(["SEO", "Ads", "Direct"], n),
        "browser": rng.choice(["Chrome", "IE", "Safari", "FireFox", "Opera"], n),
        "sex": rng.choice(["M", "F"], n),
        "age": rng.integers(18, 70, n),
        "ip_address": rng.uniform(16777216, 3758096383, n),
        "class": ((gap < 60) | shared).astype(int),
    })
    df.to_csv(d / "Fraud_Data.csv", index=False)
    pd.DataFrame({"lower_bound_ip_address": [16777216, 16777472, 3000000000], "upper_bound_ip_address": [16777471, 16777727, 3758096383],
                  "country": ["Australia", "China", "United States"]}).to_csv(d / "IpAddress_to_Country.csv", index=False)
    return d


def test_baf_adapter(raw_baf):
    from fraudvae.data.adapters import baf

    df = baf.load(RAW_DIR)
    assert len(df) == 400 and set(df["source"].unique()) == {"baf"}
    assert list(df.columns[: len(schema.ALL_COLUMNS)]) == schema.ALL_COLUMNS
    # time runs with the month and keeps the file order inside it
    assert (np.diff(df["ts"].to_numpy()) > 0).all()
    # device block observed, hour / dow not (no clock in the file), customer absent
    assert df["device_os"].gt(0).all() and df["hour"].isna().all() and (df["customer"] == -1).all()
    assert set(schema.CAT_VOCAB["device_os"][c] for c in df["device_os"].unique()) <= {"windows", "linux", "macos", "other"}
    assert df["session_length_min"].isna().sum() > 0 and df["session_length_min"].min() >= 0   # -1 became unobserved
    assert df["account_age_days"].isna().sum() > 0 and df["device_new_cust_30d"].max() == 2
    assert df["amount"].equals(df["credit_limit"])
    wide = [c for c in df.columns if c.startswith(schema.WIDE_PREFIX)]
    assert len(wide) == 29 and "wide__fraud_bool" not in wide and "wide__device_fraud_count" not in wide
    kinds = {d["name"]: d["kind"] for d in df.attrs["wide_columns"]}
    assert kinds["payment_type"] == "cat" and kinds["credit_risk_score"] == "num"


def test_fraudecom_adapter(raw_fraudecom):
    from fraudvae.data.adapters import fraudecom

    df = fraudecom.load(RAW_DIR)
    assert len(df) == 300 and df["hour"].notna().all() and df["channel"].eq(schema.CAT_INDEX["channel"]["online"]).all()
    assert (df["customer"] >= 0).all() and df["customer"].nunique() == 300 and (df["device"] >= 0).all()
    # a shared device shows up as distinct customers before, causally, and those rows are the fraud
    shared = df["device_new_cust_30d"].fillna(0) > 0
    assert shared.sum() > 10 and df.loc[shared, "is_fraud"].mean() > 0.9
    first = df.sort_values("ts").groupby("device").head(1).index
    assert (df.loc[first, "device_new_cust_30d"].fillna(0) == 0).all()
    assert df["account_age_days"].min() >= 0 and (df["account_age_days"] < 0.001).sum() > 10   # the bots
    wide = {d["name"]: d for d in df.attrs["wide_columns"]}
    assert wide["country"]["kind"] == "cat" and set(wide["country"]["values"]) <= {"Australia", "China", "United States"}
    assert "signup_to_purchase_s" in wide and "class" not in wide and "user_id" not in wide
    assert set(schema.CAT_VOCAB["browser"][c] for c in df["browser"].unique()) == {"chrome", "ie", "safari", "firefox", "opera"}


def test_ip_country_lookup():
    from fraudvae.data.adapters.fraudecom import ip_country

    table = pd.DataFrame({"lower_bound_ip_address": [100, 200, 1000], "upper_bound_ip_address": [199, 299, 1999],
                          "country": ["A", "B", "C"]})
    out = ip_country(np.array([150.7, 199.0, 200.0, 500.0, 1999.9, 50.0, 5000.0]), table)
    assert list(out) == ["A", "A", "B", None, "C", None, None]


def test_build_registers_the_new_sources(raw_baf, raw_fraudecom):
    from fraudvae.data.registry import ORDER, SOURCES
    from fraudvae.data.adapters import get_adapter

    assert ORDER[9:11] == ["baf", "fraudecom"] and schema.SOURCES[9:11] == ["baf", "fraudecom"]   # v6 appended five more
    assert schema.SOURCE_INDEX["baf"] == 10 and schema.SOURCE_INDEX["fraudecom"] == 11
    for key in ("baf", "fraudecom"):
        assert SOURCES[key].access == "kaggle" and get_adapter(key).KEY == key


def test_device_payload_mapping():
    from fraudvae.server.payloads import device_fields, merge_payloads

    assert device_fields(None) == {} and device_fields("junk") == {} and device_fields({}) == {}
    f = device_fields({"os": "iOS 11.2.1", "browser": "mobile safari 11.0", "device_type": "Mobile", "screen": "375x812",
                       "session_length_min": "4.5", "keep_alive": "false", "phone_valid": 1, "email_domain": "yahoo.co.uk",
                       "recipient_email_domain": "outlook.com", "name_email_similarity": 0.42, "account_age_days": 10,
                       "foreign": True})
    assert f == {"device_os": "ios", "browser": "safari", "device_type": "mobile", "screen_w": 375.0, "screen_h": 812.0,
                 "session_length_min": 4.5, "email_name_similarity": 0.42, "account_age_days": 10.0,
                 "keep_alive_session": "no", "phone_valid": "yes", "is_foreign": "yes",
                 "email_bucket": "yahoo", "email_bucket_recipient": "microsoft"}
    # junk values are skipped, bad screens ignored, and the transaction's own fields win
    assert device_fields({"screen": "big", "session_length_min": "lots", "keep_alive": "maybe"}) == {}
    merged = merge_payloads({"amount": 5, "device_os": "android", "browser": ""}, device={"os": "Windows", "browser": "edge 17"})
    assert merged == {"amount": 5, "device_os": "android", "browser": "edge"}


def test_raw_folder_scan(tmp_path):
    """v5: a Downloads-style folder of kaggle zips, named folders and a mixed subfolder is sorted
    into dataset keys by file names and zip contents"""
    import zipfile

    from fraudvae.data.aux_registry import scan

    with zipfile.ZipFile(tmp_path / "archive (9).zip", "w") as z:
        z.writestr("malicious_phish.csv", "url,type\n")
    with zipfile.ZipFile(tmp_path / "archive (6).zip", "w") as z:
        z.writestr("Base.csv", "fraud_bool\n")
    (tmp_path / "ieee-fraud-detection.zip").write_bytes(b"not really a zip")
    reg = tmp_path / "reg-req-data"
    (reg / "DGraphFin").mkdir(parents=True)
    (reg / "DGraphFin" / "dgraphfin.npz").write_bytes(b"x")
    (reg / "DGraphFin" / "Readme.md").write_bytes(b"x")
    (reg / "BB-MAS_Dataset.zip").write_bytes(b"x")
    (reg / "sensors_data.zip").write_bytes(b"x")
    (tmp_path / "twitter_human_bots_dataset.csv").write_bytes(b"x")
    (tmp_path / "_staging").mkdir()
    (tmp_path / "_staging" / "baf_base.csv.zst").write_bytes(b"x")
    (tmp_path / "tabformer_tables").mkdir()
    (tmp_path / "tabformer_tables" / "sd254_users.csv").write_bytes(b"x")
    (tmp_path / "notes.ipynb").write_bytes(b"x")
    found = {k: sorted(p.name for p in v) for k, v in scan(tmp_path).items()}
    assert found == {"baf": ["archive (6).zip"], "malurl": ["archive (9).zip"], "ieee": ["ieee-fraud-detection.zip"],
                     "dgraph": ["Readme.md", "dgraphfin.npz"], "bbmas": ["BB-MAS_Dataset.zip"], "brainrun": ["sensors_data.zip"],
                     "twitterbot": ["twitter_human_bots_dataset.csv"], "tabformer": ["sd254_users.csv"]}


def _generator_raws(n=400, seed=0):
    """tiny files in the two generators' own formats: the handbook's one-pickle-per-day frames and
    datagen.py's pipe-delimited per-profile file (plus a customers.csv that must be skipped)"""
    rng = np.random.default_rng(seed)
    d = RAW_DIR / "fdh_sim" / "out"
    d.mkdir(parents=True, exist_ok=True)
    ts = pd.Timestamp("2024-03-01") + pd.to_timedelta(np.sort(rng.integers(0, 10 * 86400, n)), unit="s")
    df = pd.DataFrame({"TRANSACTION_ID": np.arange(n), "TX_DATETIME": ts, "CUSTOMER_ID": rng.integers(0, 40, n),
                       "TERMINAL_ID": rng.integers(0, 60, n), "TX_AMOUNT": rng.lognormal(3, 0.6, n).round(2),
                       "TX_TIME_SECONDS": (ts - ts[0]).total_seconds().astype(int), "TX_TIME_DAYS": ((ts - ts[0]).days),
                       "TX_FRAUD": (rng.random(n) < 0.05).astype(int), "TX_FRAUD_SCENARIO": 0})
    for day, part in df.groupby(df["TX_DATETIME"].dt.date):
        part.to_pickle(d / f"{day}.pkl")
    d = RAW_DIR / "sparkov_sim"
    d.mkdir(parents=True, exist_ok=True)
    cust = ["ssn", "cc_num", "first", "last", "gender", "street", "city", "state", "zip", "lat", "long", "city_pop", "job", "dob", "acct_num", "profile"]
    tx = ["trans_num", "trans_date", "trans_time", "unix_time", "category", "amt", "is_fraud", "merchant", "merch_lat", "merch_long"]
    when = pd.Timestamp("2024-01-01") + pd.to_timedelta(np.sort(rng.integers(0, 30 * 86400, n)), unit="s")
    cc = rng.choice([f"4{int(x):015d}" for x in rng.integers(0, 10 ** 15, 30)], n)
    rows = pd.DataFrame({"ssn": "000-00-0000", "cc_num": cc, "first": "A", "last": "B", "gender": rng.choice(["F", "M"], n),
                         "street": "1 Main St", "city": "Town", "state": "NC", "zip": "28654", "lat": 36.08, "long": -81.18,
                         "city_pop": 3495, "job": "Analyst", "dob": "1980-05-05", "acct_num": "1", "profile": "adults_2550_female_rural.json",
                         "trans_num": [f"{i:032x}" for i in range(n)], "trans_date": when.strftime("%Y-%m-%d"), "trans_time": when.strftime("%H:%M:%S"),
                         "unix_time": when.astype("int64") // 10 ** 9, "category": rng.choice(["grocery_pos", "misc_net", "gas_transport"], n),
                         "amt": rng.lognormal(3, 0.7, n).round(2), "is_fraud": (rng.random(n) < 0.05).astype(int),
                         "merchant": rng.choice(["fraud_Kub and Mann", "fraud_Lind-Buckridge"], n), "merch_lat": 36.1, "merch_long": -81.2})[cust + tx]
    rows.to_csv(d / "adults_2550_female_rural_0.csv", sep="|", index=False)
    rows.iloc[:0][cust].to_csv(d / "customers.csv", sep="|", index=False)
    return n


def test_generator_sources_are_pretraining_only():
    """v7: the two generators behind fdh and sparkov, run locally, build like any source through the
    published sets' frame-building, carry the pretrain_only flag, and stay out of a prepared set
    unless asked - a simulator a model is tuned on is never in the evaluation frame"""
    from fraudvae.data.adapters import get_adapter
    from fraudvae.data.adapters.base import available
    from fraudvae.data.build import build_source
    from fraudvae.data.prepare import _available_sources
    from fraudvae.data.registry import ORDER, SOURCES
    from fraudvae.paths import PROCESSED_DIR, ensure_dirs

    ensure_dirs()
    assert ORDER[16:18] == ["fdh_sim", "sparkov_sim"] == schema.SOURCES[16:18]
    assert schema.SOURCE_INDEX["fdh_sim"] == 17 and schema.SOURCE_INDEX["sparkov_sim"] == 18
    for key in ("fdh_sim", "sparkov_sim"):
        assert SOURCES[key].pretrain_only and SOURCES[key].access == "local" and get_adapter(key).KEY == key
        assert not available(RAW_DIR, key)
    assert not any(SOURCES[k].pretrain_only for k in ORDER[:16])
    n = _generator_raws()
    for key in ("fdh_sim", "sparkov_sim"):
        assert available(RAW_DIR, key)
        stats, sample = build_source(key, RAW_DIR)
        assert stats["rows"] == n and stats["fraud"] > 0 and stats["schema_version"] == schema.SCHEMA_VERSION
        df = pd.read_parquet(PROCESSED_DIR / f"{key}.parquet", columns=["source", "txn_id", "customer", "ts", "split"])
        assert (df["source"] == key).all() and df["txn_id"].str.startswith(key + "-").all() and (df["customer"] >= 0).all()
    # a capped read stops early, as the other chunked adapters do
    assert len(get_adapter("sparkov_sim").load(RAW_DIR, max_rows=100)) == 100
    assert len(get_adapter("fdh_sim").load(RAW_DIR, max_rows=100)) == 100
    # prepare leaves them out unless asked
    assert not {"fdh_sim", "sparkov_sim"} & set(_available_sources())
    assert {"fdh_sim", "sparkov_sim"} <= set(_available_sources(include_pretrain_only=True))


def test_private_adapter_joins_the_registry(tmp_path):
    """v7: an adapter in FRAUDVAE_ADAPTERS joins the schema's source list after the public sources,
    the registry carries its SOURCE dict, and build reads it like any other - in a fresh
    interpreter, since the source list is fixed when the schema loads"""
    import json
    import os
    import subprocess
    import sys

    from fraudvae.data import private

    d = tmp_path / "adapters"
    d.mkdir()
    (d / "acme.py").write_text(
        "import numpy as np\n"
        "from fraudvae.data.schema import empty_frame, finalize, assign_time_splits\n"
        "SOURCE = {'name': 'Acme Bank card transactions', 'context': 'a partner bank', 'files': ('acme.csv',), 'approx_rows': 300}\n"
        "def load(raw_dir):\n"
        "    rng = np.random.default_rng(0)\n"
        "    n = 300\n"
        "    df = empty_frame(n)\n"
        "    df['txn_id'] = [f'acme-{i}' for i in range(n)]\n"
        "    df['ts'] = np.sort(rng.uniform(1.6e9, 1.6e9 + 30 * 86400, n))\n"
        "    df['amount'] = rng.lognormal(3, 0.5, n).astype(np.float32)\n"
        "    df['hour'] = rng.integers(0, 24, n).astype(np.float32)\n"
        "    df['dow'] = rng.integers(0, 7, n).astype(np.float32)\n"
        "    df['is_fraud'] = (rng.random(n) < 0.05).astype(np.int8)\n"
        "    df['customer'] = rng.integers(0, 20, n).astype(np.int64)\n"
        "    return assign_time_splits(finalize(df, 'acme'))\n")
    (d / "_helper.py").write_text("x = 1\n")             # ignored: a leading underscore
    (d / "Bad-Name.py").write_text("x = 1\n")            # ignored: not a plain key
    (d / "fdh.py").write_text("x = 1\n")                 # ignored: collides with a public source
    code = (
        "import json, sys\n"
        "from fraudvae.data import schema\n"
        "from fraudvae.data.registry import SOURCES, ORDER\n"
        "from fraudvae.data.adapters import get_adapter\n"
        "from fraudvae.data.build import build_source\n"
        "from fraudvae.paths import RAW_DIR, ensure_dirs\n"
        "ensure_dirs()\n"
        "stats, sample = build_source('acme', RAW_DIR)\n"
        "print(json.dumps({'last': schema.SOURCES[-1], 'n': len(schema.SOURCES), 'index': schema.SOURCE_INDEX['acme'],\n"
        "                  'access': SOURCES['acme'].access, 'name': SOURCES['acme'].name, 'order_last': ORDER[-1],\n"
        "                  'loaded': hasattr(get_adapter('acme'), 'load'), 'rows': stats['rows'], 'fraud': stats['fraud']}))\n")
    env = {**os.environ, private.ENV: str(d)}
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=600)
    assert r.returncode == 0, r.stderr[-2000:]
    out = json.loads(r.stdout.strip().splitlines()[-1])
    assert out["last"] == "acme" and out["order_last"] == "acme" and out["n"] == 19 and out["index"] == 19
    assert out["access"] == "local" and out["name"] == "Acme Bank card transactions" and out["loaded"]
    assert out["rows"] == 300 and out["fraud"] > 0
    # the scan itself, in this interpreter: the three decoys are skipped, the key is read from the file name
    os.environ[private.ENV] = str(d)
    try:
        assert private.scan(["fdh"]) == ["acme"]
        assert private.source_fields("acme")["key"] == "acme" and private.source_fields("acme")["license"] == "private"
    finally:
        del os.environ[private.ENV]
    assert private.scan() == [] and private.directory() is None
