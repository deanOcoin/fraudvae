import numpy as np
import pandas as pd
import pytest

from fraudvae.data import schema, taxonomy


def test_vocab_index_zero_is_unknown():
    for f in schema.CATEGORICAL:
        assert f.values[0] == "unknown"
        assert schema.cat_code(f.name, "unknown") == 0
        assert schema.cat_code(f.name, "definitely not a value") == 0


def test_empty_frame_and_finalize_roundtrip():
    df = schema.empty_frame(5)
    df["txn_id"] = [f"x-{i}" for i in range(5)]
    df["ts"] = np.arange(5, dtype=float)
    df["amount"] = [1.0, 2.0, 3.0, 4.0, 5.0]
    df["hour"] = [0, 5, 23, np.nan, 12]
    df["is_fraud"] = [0, 1, 0, 0, 1]
    df["channel"] = schema.cat_code("channel", "online")
    out = schema.finalize(df, "fdh")
    assert list(out.columns) == schema.ALL_COLUMNS
    assert out["amount"].dtype == np.float32
    assert out["channel"].dtype == np.int8
    assert out["hour"].isna().sum() == 1
    assert (out["source"] == "fdh").all()


def test_finalize_rejects_bad_codes():
    df = schema.empty_frame(2)
    df["txn_id"] = ["a", "b"]
    df["ts"] = [0.0, 1.0]
    df["channel"] = 99
    with pytest.raises(ValueError):
        schema.finalize(df, "fdh")


def test_finalize_rejects_hour_out_of_range():
    df = schema.empty_frame(2)
    df["txn_id"] = ["a", "b"]
    df["ts"] = [0.0, 1.0]
    df["hour"] = [24.0, 1.0]
    with pytest.raises(ValueError):
        schema.finalize(df, "fdh")


def test_time_split_is_ordered():
    df = pd.DataFrame({"ts": np.random.default_rng(0).permutation(1000).astype(float)})
    out = schema.assign_time_splits(df, 0.1, 0.2)
    assert out["split"].value_counts().to_dict() == {"train": 700, "val": 100, "test": 200}
    assert out.loc[out.split == "train", "ts"].max() < out.loc[out.split == "val", "ts"].min()
    assert out.loc[out.split == "val", "ts"].max() < out.loc[out.split == "test", "ts"].min()


@pytest.mark.parametrize("mcc,cat", [(5411, "grocery"), (5812, "restaurants"), (5541, "fuel"), (7011, "hotel"),
                                     (3051, "travel"), (4900, "utilities"), (6011, "financial"), (5945, "kids_pets"),
                                     (9999, "unknown"), ("garbage", "unknown"), (None, "unknown")])
def test_mcc_mapping(mcc, cat):
    assert taxonomy.category_from_mcc(mcc) == cat


@pytest.mark.parametrize("cc,brand", [("4532015112830366", "visa"), ("5425233430109903", "mastercard"),
                                      ("2223000048400011", "mastercard"), ("374245455400126", "amex"),
                                      ("6011000991300009", "discover"), ("3530111333300000", "other"), ("", "unknown")])
def test_card_brand_iin(cc, brand):
    assert taxonomy.card_brand_from_number(cc) == brand


def test_email_bucket():
    assert taxonomy.email_bucket("gmail.com") == "gmail"
    assert taxonomy.email_bucket("hotmail.co.uk") == "microsoft"
    assert taxonomy.email_bucket("comcast.net") == "other_free"
    assert taxonomy.email_bucket("vesta.com") == "other"
    assert taxonomy.email_bucket(float("nan")) == "unknown"


def test_encode_series_with_mapping():
    s = pd.Series(["grocery_pos", "shopping_net", "nope", None])
    codes = taxonomy.encode_series("merchant_category", s, taxonomy.SPARKOV_CATEGORY)
    assert codes.tolist() == [schema.cat_code("merchant_category", "grocery"),
                              schema.cat_code("merchant_category", "online_retail"), 0, 0]
