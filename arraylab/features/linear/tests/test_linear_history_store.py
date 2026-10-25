"""Every successful result has its own atomic persistent archive entry."""
import sqlite3
import subprocess
import sys
from contextlib import contextmanager

import pytest

import arraylab.application.records as history_io
from arraylab.application.records import load_history, save_history
from arraylab.features.linear.history_store import HistoryStoreError, LinearHistoryStore


def record(label="#1 stored"):
    return {"label": label, "angles": [-90., 0., 90.], "db": [-20., 0., -20.],
            "phase": None, "result": None}


def test_every_result_persists_beyond_five_and_restores_selected_without_calculation(tmp_path, monkeypatch):
    path = tmp_path / "中文 history.sqlite3"
    store = LinearHistoryStore(path)
    ids = [store.append(record(f"#{i+1} calculation"), i+1, generated=True)[0] for i in range(18)]
    store.select(ids[2])
    monkeypatch.setattr("arraylab.core.calculate", lambda *a, **k: pytest.fail("Must not recalculate stored results"))
    rows, sequence, selected = LinearHistoryStore(path).load()
    assert len(rows) == 18 and sequence == 18 and selected == ids[2]
    assert [row[1]["label"] for row in rows] == [f"#{i+1} calculation" for i in range(18)]


def test_concurrent_windows_append_and_delete_only_their_explicit_record(tmp_path):
    path = tmp_path / "shared.sqlite3"
    left, right = LinearHistoryStore(path), LinearHistoryStore(path)
    a, seq_a, name_a = left.append(record(), 1, generated=True)
    b, seq_b, name_b = right.append(record(), 1, generated=True)
    assert (seq_a, seq_b, name_a, name_b) == (1, 2, "#1 stored", "#2 stored")
    left.remove(a)
    assert [row[0] for row in right.load()[0]] == [b]
    right.remove(b)
    assert right.load() == ([], 2, -1)
    c, seq_c, name_c = LinearHistoryStore(path).append(record(), 1, generated=True)
    assert c > b and seq_c == 3 and name_c == "#3 stored"


def test_corrupt_or_foreign_database_is_never_overwritten(tmp_path):
    path = tmp_path / "existing.sqlite3"
    content = b"not a database; preserve this original"
    path.write_bytes(content)
    with pytest.raises((HistoryStoreError, ValueError)):
        LinearHistoryStore(path)
    assert path.read_bytes() == content


def test_malformed_record_rejects_complete_restore_without_changing_file(tmp_path):
    path = tmp_path / "invalid-row.sqlite3"
    store = LinearHistoryStore(path)
    store.append(record(), 1)
    with sqlite3.connect(path) as connection:
        connection.execute("UPDATE linear_results SET payload=?", (b'{"label":"incomplete"}',))
    original = path.read_bytes()
    with pytest.raises(ValueError):
        LinearHistoryStore(path).load()
    assert path.read_bytes() == original


def test_crashed_writer_hot_journal_recovers_committed_results(tmp_path):
    path = tmp_path / "crash-recovery.sqlite3"
    store = LinearHistoryStore(path)
    store.append(record("#1 committed"), 1, generated=True)
    before = store.load()
    process = subprocess.run([sys.executable, "-c", """
import os,sqlite3,sys
connection=sqlite3.connect(sys.argv[1])
connection.execute('PRAGMA cache_size=8')
connection.execute('BEGIN IMMEDIATE')
connection.execute('INSERT INTO linear_results(label,payload) VALUES (?,?)',('#2 uncommitted',b'x'*2000000))
connection.execute("UPDATE history_meta SET value='2' WHERE key='sequence'")
os._exit(77)
""", str(path)], timeout=20)
    assert process.returncode == 77
    assert path.with_name(path.name + "-journal").exists()
    # This is a real interrupted transaction, not a mocked read exception.
    assert LinearHistoryStore(path).load() == before
    assert not path.with_name(path.name + "-journal").exists()


def test_failed_write_preserves_all_committed_entries_and_sequence(tmp_path, monkeypatch):
    store = LinearHistoryStore(tmp_path / "failure.sqlite3")
    store.append(record(), 1, generated=True)
    before = store.load()

    @contextmanager
    def no_write(*args, **kwargs):
        raise sqlite3.OperationalError("simulated disk full")
        yield

    with monkeypatch.context() as patch:
        patch.setattr(store, "_connect", no_write)
        with pytest.raises(HistoryStoreError, match="自动保存失败"):
            store.append(record("#2 new"), 2, generated=True)
    assert store.load() == before


def test_individual_size_budget_does_not_evict_older_results(tmp_path, monkeypatch):
    path = tmp_path / "budget.sqlite3"
    store = LinearHistoryStore(path)
    store.append(record(), 1)
    before = path.read_bytes()
    import arraylab.features.linear.records as linear_records
    monkeypatch.setattr(linear_records, "MAX_BYTES", 10)
    with pytest.raises(ValueError, match="单条快照"):
        store.append(record("new"), 2)
    assert path.read_bytes() == before


def test_manual_linear_history_accepts_more_than_five_and_keeps_selection(tmp_path):
    path = tmp_path / "linear.arraylab.json"
    records = [record(f"#{i+1}") for i in range(12)]
    save_history(path, "linear", records, 8)
    restored = load_history(path)
    assert restored["records"] == records and restored["selected"] == 8


def test_manual_feed_history_limit_stays_five(tmp_path):
    # Quantity is checked before report structure, so no solver is needed here.
    with pytest.raises(ValueError, match="长度1–5"):
        save_history(tmp_path / "feed.arraylab.json", "feed", [{"label": "x", "report": {}}]*6, 0)
