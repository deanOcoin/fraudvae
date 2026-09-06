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

    assert ORDER[-2:] == ["baf", "fraudecom"] and schema.SOURCES[-2:] == ["baf", "fraudecom"]
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
