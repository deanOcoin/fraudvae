"""the amlworld adapter on a tiny file in the kaggle layout (two columns called Account, patterns file)"""
import numpy as np
import pandas as pd

from fraudvae.data.adapters import amlworld


def _write(raw_dir):
    d = raw_dir / "amlworld"
    d.mkdir(parents=True)
    rows = [
        ("2022/09/01 00:20", "010", "8000EBD30", "010", "8000EBD30", "3697.34", "US Dollar", "3697.34", "US Dollar", "Reinvestment", 0),
        ("2022/09/01 00:21", "03208", "8000F4580", "001", "8000F5340", "0.01", "US Dollar", "0.01", "US Dollar", "Cheque", 0),
        ("2022/09/01 01:05", "03208", "8000F4580", "001", "8000F5340", "250.00", "Euro", "250.00", "Euro", "Credit Card", 0),
        ("2022/09/01 02:30", "021174", "800737690", "012", "80011F990", "2848.96", "Euro", "2848.96", "Euro", "ACH", 1),
        ("2022/09/01 04:33", "021174", "800737690", "020", "80020C5B0", "8630.40", "Euro", "8630.40", "Euro", "ACH", 1),
        ("2022/09/02 09:14", "021174", "800737690", "020", "80006A5E0", "35642.49", "Yuan", "4990.00", "US Dollar", "Wire", 1),
        ("2022/09/03 10:00", "010", "8000EBD30", "011", "8000ABCDE", "1.5", "Bitcoin", "1.5", "Bitcoin", "Bitcoin", 0),
    ]
    header = "Timestamp,From Bank,Account,To Bank,Account,Amount Received,Receiving Currency,Amount Paid,Payment Currency,Payment Format,Is Laundering\n"
    with open(d / "HI-Small_Trans.csv", "w") as f:
        f.write(header)
        for r in rows:
            f.write(",".join(map(str, r)) + "\n")
    with open(d / "HI-Small_Patterns.txt", "w") as f:
        f.write("BEGIN LAUNDERING ATTEMPT - FAN-OUT:  Max 16-degree Fan-Out\n")
        f.write(",".join(map(str, rows[3])) + "\n")
        f.write(",".join(map(str, rows[4])) + "\n")
        f.write("END LAUNDERING ATTEMPT - FAN-OUT\n\n")
        f.write("BEGIN LAUNDERING ATTEMPT - CYCLE:  Max 6 hops\n")
        f.write(",".join(map(str, rows[5])) + "\n")
        f.write("END LAUNDERING ATTEMPT - CYCLE\n")
    return raw_dir


def test_amlworld_adapter(tmp_path):
    raw = _write(tmp_path)
    df = pd.concat(list(amlworld.load(raw)), ignore_index=True)
    assert len(df) == 7 and df["is_fraud"].sum() == 3
    assert (df["source"] == "amlworld").all()
    # usd conversion: yuan paid in usd -> 4990, bitcoin 1.5 -> 30000
    assert np.isclose(df.loc[5, "amount"], 4990.0) and np.isclose(df.loc[6, "amount"], 30000.0)
    # cross-currency row is foreign (2), same-currency rows are not (1)
    assert df.loc[5, "is_foreign"] == 2 and df.loc[0, "is_foreign"] == 1
    assert df.loc[2, "payment_format"] == 5 and df.loc[6, "payment_format"] == 7   # credit_card, crypto
    assert df.loc[2, "hour"] == 1 and df.loc[5, "dow"] == 4                        # 2022-09-02 is a friday
    # patterns: fan-out = 1 on the two ACH legs, cycle = 5 on the wire
    assert df.loc[[3, 4], "fraud_scenario"].tolist() == [1, 1] and df.loc[5, "fraud_scenario"] == 5
    assert df.loc[0, "fraud_scenario"] == 0
    # velocity: second row of the 03208 payer sees one prior txn, 44 min earlier
    assert df.loc[2, "cust_txn_count_prior"] == 1 and np.isclose(df.loc[2, "secs_since_prev_txn"], 44 * 60)
    assert df.loc[2, "cust_new_merchant"] == 1 and df.loc[1, "cust_new_merchant"] == 2
