"""the server's memory of scored transactions: ordering, the per-customer cap, the profile
round-trip, and that two runs whose names sanitise alike do not share a table"""
import numpy as np

from fraudvae.server import entity_store
from fraudvae.server.entity_store import EntityStore, _table_name


def _row(i):
    return np.full(3, float(i), np.float32), np.ones(3, np.float32), np.array([i, i + 1], np.int64)


def test_last_is_ordered_and_strictly_before(tmp_path):
    s = EntityStore(tmp_path / "e.sqlite", "run-a")
    for i, ts in enumerate([30.0, 10.0, 20.0, 40.0]):
        s.append("fdh", "c1", ts, *_row(i))
    got = s.last("fdh", "c1", before_ts=40.0, n=2)
    assert [g[0] for g in got] == [20.0, 30.0]            # the two newest before 40, oldest first
    assert got[-1][1][0] == 0.0 and got[-1][3].tolist() == [0, 1]   # the row written at ts 30 comes back intact
    assert s.last("fdh", "c1", before_ts=10.0, n=5) == []
    assert s.last("fdh", "other", before_ts=100.0, n=5) == [] and s.count("fdh", "c1") == 4
    assert s.forget("fdh", "c1") == 4 and s.count("fdh", "c1") == 0
    s.close()


def test_rows_per_customer_are_capped(tmp_path, monkeypatch):
    monkeypatch.setattr(entity_store, "MAX_ROWS_PER_CUSTOMER", 5)
    s = EntityStore(tmp_path / "e.sqlite", "run-a")
    for i in range(12):
        s.append("fdh", "c1", float(i), *_row(i))
    s.append("fdh", "c2", 1.0, *_row(99))
    assert s.count("fdh", "c1") == 5 and s.count("fdh", "c2") == 1
    assert [g[0] for g in s.last("fdh", "c1", before_ts=1e9, n=10)] == [7.0, 8.0, 9.0, 10.0, 11.0]   # the newest survive
    s.close()


def test_profile_round_trip_and_separate_runs(tmp_path):
    a = EntityStore(tmp_path / "e.sqlite", "v5-h200-e")
    b = EntityStore(tmp_path / "e.sqlite", "v5_h200_e")
    assert a.table != b.table and _table_name("v5-h200-e") != _table_name("v5_h200_e")
    assert a.profile("fdh", "c1") == (None, 0)
    a.set_profile("fdh", "c1", {"mouse": [0.1, 0.2]}, 3)
    assert a.profile("fdh", "c1") == ({"mouse": [0.1, 0.2]}, 3)
    assert b.profile("fdh", "c1") == (None, 0)
    a.append("fdh", "c1", 5.0, *_row(1))
    assert b.count("fdh", "c1") == 0
    a.close(); b.close()


def test_unhashed_table_is_migrated(tmp_path):
    import sqlite3

    p = tmp_path / "e.sqlite"
    conn = sqlite3.connect(p)
    conn.execute("CREATE TABLE h_old_run (source TEXT, customer TEXT, ts REAL, num BLOB, mask BLOB, cat BLOB, added REAL)")
    r = _row(4)
    conn.execute("INSERT INTO h_old_run VALUES (?, ?, ?, ?, ?, ?, ?)", ("fdh", "c1", 3.0, r[0].tobytes(), r[1].tobytes(), r[2].tobytes(), 0.0))
    conn.commit(); conn.close()
    s = EntityStore(p, "old-run")
    assert s.count("fdh", "c1") == 1 and s.last("fdh", "c1", 10.0, 5)[0][0] == 3.0
    s.close()
