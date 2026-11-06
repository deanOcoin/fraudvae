"""v7: the fast handbook-shaped simulator writes the handbook's format, builds as fdh_sim, and is fast"""
import time

import numpy as np
import pandas as pd

from fraudvae.data.simulate import COLUMNS, simulate


def test_simulate_writes_the_handbook_format_that_fdh_sim_builds(tmp_path):
    from fraudvae.data.build import build_source
    from fraudvae.paths import ensure_dirs

    ensure_dirs()
    out = tmp_path / "fdh_sim" / "sim"
    info = simulate(20_000, out, days=60, seed=7, log=lambda m: None)
    files = sorted(out.glob("*.pkl"))
    assert len(files) == 60 and (out / "simulate.json").exists()
    df = pd.concat([pd.read_pickle(f) for f in files], ignore_index=True)
    assert list(df.columns) == COLUMNS and info["rows"] == len(df) and 15_000 < len(df) < 25_000
    assert df["TX_TIME_SECONDS"].is_monotonic_increasing and df["TX_DATETIME"].is_monotonic_increasing
    assert (df["TRANSACTION_ID"].to_numpy() == np.arange(len(df))).all() and df["TX_TIME_DAYS"].max() == 59
    assert (df["TX_AMOUNT"] > 0).all() and df["CUSTOMER_ID"].nunique() > 100 and df["TERMINAL_ID"].between(0, info["terminals"] - 1).all()
    rate = df["TX_FRAUD"].mean()
    assert 0.005 <= rate <= 0.03, rate
    assert {1, 2, 3} <= set(df["TX_FRAUD_SCENARIO"].unique()) and ((df["TX_FRAUD_SCENARIO"] > 0) == (df["TX_FRAUD"] == 1)).all()
    assert (df.loc[df["TX_FRAUD_SCENARIO"] == 1, "TX_AMOUNT"] > 220).all()
    # a customer pays at few terminals: the three preferred ones take most of the rows
    top = df.groupby("CUSTOMER_ID")["TERMINAL_ID"].agg(lambda s: s.value_counts().iloc[:3].sum() / len(s))
    assert top.median() > 0.6
    # the round trip through the adapter, as the corpus build does it
    stats, _ = build_source("fdh_sim", tmp_path)
    assert stats["rows"] == len(df) and stats["fraud"] == int(df["TX_FRAUD"].sum())


def test_simulate_does_ten_million_rows_a_minute(tmp_path):
    t0 = time.time()
    info = simulate(1_000_000, tmp_path / "fast", days=30, seed=1, log=lambda m: None)
    el = time.time() - t0
    assert 900_000 < info["rows"] < 1_100_000
    assert info["rows"] / el * 60 > 10_000_000, f"{info['rows'] / el * 60:,.0f} rows/min"
