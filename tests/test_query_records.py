"""Filtered record queries: schema compatibility, ordering and stable pagination."""
from datetime import datetime, timedelta, timezone
import logging
import sqlite3

import pandas as pd
import pytest
from sqlalchemy import create_engine

from signaldeck_sdk import Field
from signaldeck_plugin_sqlite.persistence.sqlite_store import SqliteStore, metadata


@pytest.fixture
def store(tmp_path):
    # Avoid spinning up the production asynchronous writer in a query test.
    db = SqliteStore.__new__(SqliteStore)
    db.url = f"sqlite:///{tmp_path / 'events.sqlite'}"
    db.engine = create_engine(db.url, future=True)
    metadata.create_all(db.engine)
    db._proc_cache = {}
    db._field_cache = {}
    db.fields = []
    db.logger = logging.getLogger(__name__)
    db._queue = None
    db._writer_task = None
    for name, dtype in (("room", "str"), ("type", "str"), ("id", "str"), ("payload", "str"), ("revision", "int")):
        db.register_field(Field(processor_name="socketio", name=name, dtype=dtype))
    yield db
    db.engine.dispose()


def add(store, at, name, kind, room="family"):
    store.insert_measurement_sync("socketio", {
        "date": at,
        "room": room,
        "type": kind,
        "id": name,
        "payload": f"payload-{name}",
        "revision": 1,
    }, {})


def test_filters_act_in_sql_before_pagination(store):
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    add(store, base, "a", "chat")
    add(store, base + timedelta(milliseconds=1), "b", "appointment")
    add(store, base + timedelta(milliseconds=2), "c", "chat")
    add(store, base + timedelta(milliseconds=3), "d", "appointment", "other")
    add(store, base + timedelta(milliseconds=4), "e", "appointment")
    page = store.query_records("socketio", ["id", "type", "room"], {}, filters={"room": "family", "type": "appointment"}, limit=1)
    assert page.records["id"].tolist() == ["b"]
    assert page.next_cursor is not None
    second = store.query_records("socketio", ["id", "type", "room"], {}, filters={"room": "family", "type": "appointment"}, limit=1, cursor=page.next_cursor)
    assert second.records["id"].tolist() == ["e"]
    assert second.next_cursor is None


def test_cursor_rejects_changed_filters_and_excludes_new_records(store):
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    for index in range(3):
        add(store, base + timedelta(milliseconds=index), str(index), "chat")
    first = store.query_records("socketio", ["id"], {}, filters={"type": "chat"}, limit=1)
    add(store, base + timedelta(milliseconds=3), "late", "chat")
    second = store.query_records("socketio", ["id"], {}, filters={"type": "chat"}, limit=2, cursor=first.next_cursor)
    assert second.records["id"].tolist() == ["1", "2"]
    assert second.next_cursor is None
    with pytest.raises(ValueError, match="cursor"):
        store.query_records("socketio", ["id"], {}, filters={"type": "appointment"}, limit=1, cursor=first.next_cursor)


def test_date_bounds_and_descending_order(store):
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    for index in range(4):
        add(store, base + timedelta(days=index), str(index), "chat")
    page = store.query_records("socketio", ["id"], {}, date_from=base + timedelta(days=1), date_to=base + timedelta(days=3), order="desc", limit=1)
    assert page.records["id"].tolist() == ["2"]
    page2 = store.query_records("socketio", ["id"], {}, date_from=base + timedelta(days=1), date_to=base + timedelta(days=3), order="desc", limit=1, cursor=page.next_cursor)
    assert page2.records["id"].tolist() == ["1"]
    assert page2.next_cursor is None


def test_unknown_field_fails_explicitly(store):
    with pytest.raises(ValueError, match="Unknown persist fields"):
        store.query_records("socketio", ["id"], {}, filters={"missing": "x"})


@pytest.mark.parametrize("order", ["asc", "desc"])
def test_cursor_preserves_submillisecond_timestamps(store, order):
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    # SQLite can store fractional milliseconds even in the declared BigInteger column.
    add(store, base + timedelta(microseconds=500), "first", "appointment")
    add(store, base + timedelta(microseconds=600), "second", "appointment")
    add(store, base + timedelta(microseconds=700), "third", "appointment")
    add(store, base + timedelta(milliseconds=1, microseconds=250), "fourth", "appointment")

    expected = ["first", "second", "third", "fourth"]
    if order == "desc":
        expected.reverse()

    seen = []
    cursor = None
    for _ in range(len(expected) + 1):
        page = store.query_records(
            "socketio", ["id"], {}, filters={"type": "appointment"},
            limit=1, cursor=cursor, order=order,
        )
        seen.extend(page.records["id"].tolist())
        if page.next_cursor is None:
            break
        assert page.next_cursor != cursor, "Cursor must advance between pages"
        cursor = page.next_cursor
    assert seen == expected
    assert page.next_cursor is None


def test_submillisecond_date_bounds_are_not_truncated(store):
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    add(store, base + timedelta(microseconds=500), "before", "appointment")
    add(store, base + timedelta(microseconds=600), "inside", "appointment")
    add(store, base + timedelta(microseconds=700), "after", "appointment")
    result = store.query_records(
        "socketio", ["id"], {},
        date_from=base + timedelta(microseconds=600),
        date_to=base + timedelta(microseconds=700),
    )
    assert result.records["id"].tolist() == ["inside"]


def test_unbounded_query_batches_payload_reads_under_sqlite_variable_limit(store):
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    for index in range(150):
        add(store, base + timedelta(milliseconds=index), f"event-{index}", "chat")

    # Simulate an SQLite build with a restrictive parameter count. The old
    # query bound every record ID at once and would fail with OperationalError.
    with store.engine.connect() as conn:
        raw = conn.connection.driver_connection
        previous_limit = raw.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, 128)
        try:
            result = store.query_records(
                "socketio", ["id", "payload"], {}, filters={"type": "chat"}
            )
            assert len(result.records) == 150
            assert result.records["id"].tolist() == [
                f"event-{index}" for index in range(150)
            ]
            assert result.records["payload"].iloc[-1] == "payload-event-149"
            assert result.next_cursor is None
        finally:
            raw.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, previous_limit)
