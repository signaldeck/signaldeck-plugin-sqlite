"""Filtered record queries: schema compatibility, ordering and stable pagination."""
from datetime import datetime, timedelta, timezone
import logging

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
