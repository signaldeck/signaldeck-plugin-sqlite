"""
SQLAlchemy-basierter Store (DB-agnostisch: SQLite oder Postgres)
----------------------------------------------------------------

- Normalisiertes Schema (Weg A): processors, fields, records, values_num
- Funktioniert mit SQLite (sqlite:///data.db) und Postgres
  (postgresql+psycopg://user:pass@host:5432/dbname)
- ID-Caches (im RAM) + get_or_create* mit Race-sicherem Fallback
- Sync-API + optionaler Async-Writer (asyncio.Queue + Batching)
- Tagesabfragen (Field-only und Wide/Pivot per Pandas)

Hinweis:
- Für SQLite werden sinnvolle PRAGMAs beim Connect gesetzt (WAL, synchronous=NORMAL)
- Zeitstempel werden als INTEGER Epoch (ms oder s) gespeichert (konfigurierbar)

Beispiel:
---------
store = SAStore("sqlite:///data.db")
store.start_writer()  # optional
store.enqueue_measurement("A", {"date": datetime.now(), "a": 1.2, "b": 3.4})
...
await store.shutdown_writer()
store.dispose()
"""
from __future__ import annotations

from dataclasses import dataclass
import asyncio, logging
from contextlib import nullcontext
import pandas as pd
from datetime import date, datetime, timedelta, timezone, time
from typing import Any, Dict, Iterable, List, Optional, Tuple
from signaldeck_sdk import Field
from signaldeck_sdk import DataStore
import numpy as np
from pathlib import Path
from zoneinfo import ZoneInfo
from contextlib import closing
import sqlite3

from sqlalchemy import (
    BigInteger,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    MetaData,
    String,
    Table,
    UniqueConstraint,
    Index,
    create_engine,
    text,
    select,
    insert,
    update
)
from sqlalchemy.engine import Engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session


# ---------------- Schema-Definition ----------------
metadata = MetaData()

processors = Table(
    "processors",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("name", String, nullable=False, unique=True),
)

fields = Table(
    "fields",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("processor_id", Integer, ForeignKey("processors.id", ondelete="CASCADE"), nullable=False),
    Column("name", String, nullable=False),
    Column("unit", String),
    Column("dtype", String, nullable=False),
    Column("display_name", String),
    UniqueConstraint("processor_id", "name", name="uq_fields_proc_name"),
)

records = Table(
    "records",
    metadata,
    Column("id", Integer, primary_key=True), 
    # Wir speichern Epoch als INTEGER (ms oder s, siehe Store-Optionen)
    Column("ts", BigInteger, nullable=False),
    Column("processor_id", Integer, ForeignKey("processors.id", ondelete="CASCADE"), nullable=False),
    UniqueConstraint("processor_id", "ts", name="uq_records_proc_ts"),
    Index("ix_records_proc_ts", "processor_id", "ts"),
)

values_num = Table(
    "values_num",
    metadata,
    Column("record_id", Integer, ForeignKey("records.id", ondelete="CASCADE"), nullable=False),
    Column("field_id", Integer, ForeignKey("fields.id", ondelete="CASCADE"), nullable=False),
    Column("value", Float),
    UniqueConstraint("record_id", "field_id", name="pk_values_num_record_field"),
    Index("ix_values_field", "field_id"),
)

values_str = Table(
    "values_str",
    metadata,
    Column("record_id", Integer, ForeignKey("records.id", ondelete="CASCADE"), nullable=False),
    Column("field_id", Integer, ForeignKey("fields.id", ondelete="CASCADE"), nullable=False),
    Column("value", String),
    UniqueConstraint("record_id", "field_id", name="pk_values_str_record_field"),
    Index("ix_values_str_field", "field_id"),
)


# ---------------- Hilfs-Datentyp ----------------
@dataclass
class _BatchItem:
    processor_name: str
    data: Dict[str, Any]
    config: Dict[str, Any]



# ---------------- Store ----------------
class   SqliteStore(DataStore):
    def __init__(
        self,
        loop,
        url: str = "sqlite:///data.db"
    ) -> None:
        """SQLAlchemy-Store.
        :param url: SQLAlchemy-URL (SQLite oder Postgres)
        :param use_uuid: pro Record eine UUID mitschreiben
        :param echo: SQLAlchemy echo
        :param create: Schema bei Bedarf erzeugen
        """
        super().__init__(loop,config={"url":url})
        self.url = url
        self.logger = logging.getLogger(__name__)
        self.engine: Engine = create_engine(self.url, echo=False, future=True)
        self._apply_sqlite_pragmas_if_needed()
        metadata.create_all(self.engine)

        # RAM-Caches
        self._proc_cache: Dict[str, int] = {}
        self._field_cache: Dict[Tuple[int, str], int] = {}
        self._warm_caches()

        # async Writer
        self._queue: Optional[asyncio.Queue[_BatchItem]] = None
        self._writer_task: Optional[asyncio.Task] = None
        self._writer_batch_max = 4
        self._writer_batch_interval = 0.2
        self.isProcessing=False

        self.start_writer(self._loop)

    # ---------- Engine/PRAGMA ----------
    def _apply_sqlite_pragmas_if_needed(self) -> None:
        # Nur für SQLite sinnvolle PRAGMAs setzen
        if self.url.startswith("sqlite:"):
            with self.engine.connect() as conn:
                conn.exec_driver_sql("PRAGMA foreign_keys=ON;")
                conn.exec_driver_sql("PRAGMA journal_mode=WAL;")
                conn.exec_driver_sql("PRAGMA synchronous=NORMAL;")
                conn.commit()

    def dispose(self) -> None:
        self.engine.dispose()

    # ---------- Caches ----------
    def _warm_caches(self) -> None:
        with self.engine.connect() as conn:
            for name, pid in conn.execute(select(processors.c.name, processors.c.id)):
                self._proc_cache[name] = int(pid)
            for pid, name, fid, t, dn, un in conn.execute(select(fields.c.processor_id, fields.c.name, fields.c.id, fields.c.dtype, fields.c.display_name, fields.c.unit)):
                self._field_cache[(int(pid), str(name))] = Field(fid,name,pid,un,t,dn)



    # get_or_create mit Race-sicherem Fallback (SELECT, INSERT, SELECT)
    def get_or_create_processor_id(self, name: str) -> int:
        pid = self._proc_cache.get(name)
        if pid is not None:
            return pid
        with self.engine.begin() as conn:
            row = conn.execute(select(processors.c.id).where(processors.c.name == name)).fetchone()
            if row:
                pid = int(row[0])
                self._proc_cache[name] = pid
                return pid
            try:
                conn.execute(insert(processors).values(name=name))
            except IntegrityError:
                pass
            pid = int(conn.execute(select(processors.c.id).where(processors.c.name == name)).scalar_one())
            self._proc_cache[name] = pid
            return pid

    def register_field(
            self, field: Field
        ) -> Field:
            super().register_field(field)
            processor_id = self.get_or_create_processor_id(field.processor_name)
            field.processor_id = processor_id
            f: Field = self.get_field(processor_id, field.name)
            if f is not None:
                field.id = f.id
                field.processor_id = f.processor_id
                if field != f:
                    #IDs und processor_name sind identisch. 
                    #Änderung in display_name, dtype, unit
                    updatedDType = f.dtype
                    if field.has_compatible_dtype(f):
                        updatedDType = field.dtype
                    else:
                        self.logger.warning(f'Changed dtype for "{field.name}": {f.dtype} -> {field.dtype}. Change is not compatible with existing data and cannot be done automatically')
                    with self.engine.begin() as conn:
                        try:
                            conn.execute(
                            update(fields)
                            .where(fields.c.id == f.id)
                            .values(dtype= updatedDType, display_name= field.display_name, unit= field.unit))
                        except:
                            self.logger.warning(f'Unable to update field {field.name}',stack_info=True)
                
                return field
            with self.engine.begin() as conn:
                try:
                    conn.execute(
                        insert(fields).values(
                        processor_id=processor_id,
                        name=field.name,
                        unit=field.unit,
                        dtype=field.dtype,
                        display_name=field.display_name,
                    )
                )
                except IntegrityError:
                    pass
            createdField = self.get_field(processor_id, field.name)
            field.id = createdField.id
            return field
    

    def get_field(
        self,
        processor_id: int,
        field_name: str
    ) -> Field:
        key = (processor_id, field_name)
        field = self._field_cache.get(key)
        
        if field is not None:
            return field
        res=None
        with self.engine.begin() as conn:
            row = conn.execute(
                select(fields.c.processor_id, fields.c.name, fields.c.id, fields.c.dtype, fields.c.display_name, fields.c.unit).where(
                    (fields.c.processor_id == processor_id) & (fields.c.name == field_name)
                )
            ).mappings().first()
            if row:
                res = Field(**row)
                self._field_cache[key] = res
        return res
            

    # ---------- Inserts ----------
    def insert_measurement_sync(
        self,
        processor_name: str,
        data: Dict[str, Any],
        config
    ) -> int:
        if "date" not in data:
            self.logger.error("Data must contain 'date'")
            raise ValueError("data must contain 'date'")
        ts_int=-1
        
        pid = self.get_or_create_processor_id(processor_name)

        # Felder vorbereiten (IDs auflösen)
        rows_num: List[Tuple[int, float]] = []
        rows_str: List[Tuple[int, float]] = []
        for k, v in data.items():
            if k == "date":
                ts_int = self.getTsFromDate(v)
                continue
            field: Field = self.get_field(pid, k)
            if field is None:
                raise ValueError(f"Field '{k}' not registered for processor '{processor_name}'")
            fid = field.id
            numeric, val = field.handleValueToDB(v,data,config,self.logger)
            if val is None:
                continue
            if numeric:
                rows_num.append((fid,val))
            else:
                rows_str.append((fid,val))
        if ts_int < 0:
            self.logger.warning(f'Timestamp <0 for record from processor "{processor_name}": {data}. Skip record')
            return None
            

        if rows_num or rows_str:
            with self.engine.begin() as conn:
                try:
                    rec_id = int(
                        conn.execute(
                            insert(records).values(ts=ts_int, processor_id=pid).returning(records.c.id)
                        ).scalar_one()
                    )
                except IntegrityError:
                    self.logger.warning(f'Duplicated data for processor "{processor_name}" (id={pid}) and timestamp: {data["date"].isoformat()} (={ts_int})')
                    return None
                if rows_num:
                    conn.execute(
                        insert(values_num),
                        [
                            {"record_id": rec_id, "field_id": fid, "value": v}
                            for (fid, v) in rows_num
                        ],
                    )
                if rows_str:
                    conn.execute(
                        insert(values_str),
                        [
                            {"record_id": rec_id, "field_id": fid, "value": v}
                            for (fid, v) in rows_str
                        ]
                    )
        return rec_id

    # ---------- Async Writer ----------
    def start_writer(
        self,
        loop: Optional[asyncio.AbstractEventLoop] = None,
        *,
        batch_max: int = 500,
        batch_interval: float = 0.2,
    ) -> None:
        if self._writer_task is not None:
            return
        if not loop or not loop.is_running():
            raise RuntimeError("start_writer: event loop must be running")

        self._writer_batch_max = batch_max
        self._writer_batch_interval = batch_interval

        async def _init():
            self._queue = asyncio.Queue()
            self._writer_task = asyncio.create_task(self._writer_loop(), name="sqlite-writer")

        fut = asyncio.run_coroutine_threadsafe(_init(), loop)
        fut.result()  # wirft, falls _init() scheitert

    async def _writer_loop(self) -> None:
        assert self._queue is not None
        q = self._queue
        pending: List[_BatchItem] = []
        try:
            counter = 0
            while True:
                try:
                    if counter > 0 and q.empty():
                        self.logger.info(f'Processed {counter} records. Queue is empty!')
                        counter=0
                        self.isProcessing=False
                    item = await asyncio.wait_for(q.get(), timeout=self._writer_batch_interval)
                    pending.append(item)
                    if not q.empty():
                        if counter == 0:
                            self.logger.info(f'Write queue has objects.. Start processing')
                            self.isProcessing=True
                    while not q.empty() and len(pending) < self._writer_batch_max:
                        pending.append(q.get_nowait())
                    counter += len(pending)
                    if counter % 100000 == 0:
                        self.logger.info(f'Processed {counter} records..')
                except asyncio.TimeoutError:
                    pass
                if not pending:
                    continue
                batch = pending
                pending = []
                await asyncio.to_thread(self._insert_batch_sync, batch)
        except asyncio.CancelledError:
            if pending:
                await asyncio.to_thread(self._insert_batch_sync, pending)
            raise
        except Exception as e:
            self.logger.error(e)

    def _insert_batch_sync(self, batch: Iterable[_BatchItem]) -> None:
        for item in batch:
            try:
                self.insert_measurement_sync(item.processor_name, item.data, item.config)
            except Exception:
                self.logger.warning("unable to save",exc_info=True,stack_info=True)


    def enqueue_measurement(self, processor_name: str, data: Dict[str, Any], config) -> None:
        if self._queue is None:
            print("No async queue, inserting sync")
            self.insert_measurement_sync(processor_name, data,config)
            return
        self._queue.put_nowait(_BatchItem(processor_name, data, config))

    async def shutdown_writer(self) -> None:
        if self._writer_task is not None:
            self._writer_task.cancel()
            try:
                await self._writer_task
            except asyncio.CancelledError:
                pass
            self._writer_task = None
            self._queue = None

    def getDateFromTs(self,ts,config):
        return datetime.fromtimestamp(ts / 1000, tz= ZoneInfo(config.get("timezone","Europe/Berlin")))

    def getTsFromDate(self,date: date):
        return date.timestamp() * 1000

    # ---------- Queries ----------

    def _last_record(self,processor_name,config):
        pid = self.get_or_create_processor_id(processor_name)

        latest_id = (
            select(records.c.id)
            .where(records.c.processor_id == pid)
            .order_by(records.c.ts.desc())
            .limit(1)
            .scalar_subquery()
        )

        stmt = (
            select(records.c.ts,fields.c.name, values_num.c.value)
                .select_from(
                    values_num
                    .join(records, records.c.id == values_num.c.record_id)
                    .join(fields,  fields.c.id == values_num.c.field_id)
                )
                .where(records.c.id == latest_id)
                .order_by(fields.c.name)
        )

        with self.engine.connect() as conn:
            try:
                rows = conn.execute(stmt).all()
                res ={}
                date_val=None
                for ts, field_name, val in rows:
                    if not date_val:
                        date_val = self.getDateFromTs(ts,config)
                    field = self.get_field(pid,field_name)
                    if field:
                        res[field.name]=field.handleValueFromDB(val,config)
                res[config.get("date_field","date")]=date_val
                return res
            except:
                self.logger.warning(f'Unable to get latest records for processor "{processor_name}"')
                return None

    def _day_range_epoch(self, day: date, config) -> Tuple[int, int]:
        start_dt = datetime(day.year, day.month, day.day, 0, 0, 0, tzinfo=day.tzinfo)
        end_dt = start_dt + timedelta(days=1)
        return int(self.getTsFromDate(start_dt)), int(self.getTsFromDate(end_dt))
    
    def _time_range_around(self,date: date, mins = 2)-> Tuple[int, int]:
        return int(self.getTsFromDate(date + timedelta(minutes=-mins))), int(self.getTsFromDate(date + timedelta(minutes=+mins)))

    def _get_values_for_field(self,processor_name: str, field_name: str, start: int, end:int, config: dict):
        pid = self.get_or_create_processor_id(processor_name)
        field = self.get_field(pid, field_name)
        val_table = values_num
        if field.is_str():
            val_table = values_str
        stmt = (
            select(records.c.ts, val_table.c.value)
            .select_from(val_table.join(records, records.c.id == values_num.c.record_id))
            .where(
                (val_table.c.field_id == field.id)
                & (records.c.processor_id == pid)
                & (records.c.ts.between(start, end))
            )
            .order_by(records.c.ts)
        )
        with self.engine.connect() as conn:
            rows = conn.execute(stmt).all()
        data= [(self.getDateFromTs(int(ts),config), field.handleValueFromDB(val,config)) for (ts, val) in rows]
        return pd.DataFrame(data,columns=["date",field_name]).dropna()



    def select_day_wide(self, processor_name: str, day: date):
        pid = self.get_or_create_processor_id(processor_name)
        start, end = self._day_range_epoch(day,{})
        stmt = text(
            """
            SELECT r.ts, f.name AS field, v.value
            FROM values_num v
            JOIN records r ON r.id = v.record_id
            JOIN fields  f ON f.id = v.field_id
            WHERE r.processor_id = :pid AND r.ts BETWEEN :t0 AND :t1
            ORDER BY r.ts
            """
        )
        with self.engine.connect() as conn:
            rows = conn.execute(stmt, dict(pid=pid, t0=start, t1=end)).all()
        try:
            df = pd.DataFrame(rows, columns=["ts", "field", "value"])  # type: ignore
            if df.empty:
                return df
            wide = df.pivot_table(index="ts", columns="field", values="value", aggfunc="last")
            wide.sort_index(inplace=True)
            wide.reset_index(inplace=True)
            return wide
        except Exception:
            pass
        return rows

    #-----DataStore methods
    def __get_best_value_for_date(self,processor_name,fieldName,targetDate,config):
        res=[]
        minWindow = config.get("continous_data",{}).get("window_minute", 60 * 12)
        while len(res)==0 and minWindow < 60 * 17:
            start, end = self._time_range_around(targetDate,mins= minWindow)
            res = self._get_values_for_field(processor_name,fieldName,start,end,config)
            minWindow += 240
        if res is None or len(res)==0:
            return None
        res["diff"]=np.abs(res["date"]-targetDate)
        res = res[["date",fieldName,"diff"]].dropna()
        res= res.loc[res["diff"].idxmin()]
        return res[fieldName]


    def save(self,processor_name,data,persist_config):
        dataToStore = data
        if "date" not in data:
            if "date_field" not in persist_config:
                self.logger.error(f"Unable to save data from '{processor_name}' without 'date'-Field. Either provide data with the field or configure a custom date-field by config: 'date_field'")
                return 
            dateField = persist_config.get("date_field","")
            dateVal = data[dateField]
            dataToStore = { k:data[k] for k in data.keys() if k != dateField }
            dataToStore["date"]= dateVal
        self.logger.info(f"Request saving data for processor '{processor_name}': {dataToStore}")
        self.enqueue_measurement(processor_name, dataToStore,persist_config)


    def get_first_from_day(self,processor_name,fieldName,askedDate,config):
        self.logger.info("get_first_from_day(%s, %s, %s)", processor_name, fieldName, askedDate)
        targetDate=self.assure_tz_aware(datetime.combine(askedDate, time.min),config)
        return self.__get_best_value_for_date(processor_name,fieldName,targetDate,config)
    
    def get_last_from_day(self,processor_name,fieldName,askedDate,config):
        self.logger.info("get_last_from_day(%s, %s, %s)", processor_name, fieldName, askedDate)
        targetDate = self.assure_tz_aware(datetime.combine(askedDate, time.max),config)
        return self.__get_best_value_for_date(processor_name,fieldName,targetDate,config)


    def get_full_day(self,processor_name,fieldName,askedDate,config):
        askedDate = self.assure_tz_aware(askedDate,config)
        self.logger.info("get_full_day(%s, %s, %s)", processor_name, fieldName, askedDate)
        start, end = self._day_range_epoch(askedDate,config)
        res = self._get_values_for_field(processor_name,fieldName,start,end,config)
        if res is None or len(res)==0:
            return None
        res=res.set_index("date")[fieldName]
        return res
    
    def get_best_value(self,processor_name,fieldName,askedDate,config):
        askedDate = self.assure_tz_aware(askedDate,config)
        self.logger.info("get_best_value(%s, %s, %s)", processor_name, fieldName, askedDate)
        return self.__get_best_value_for_date(processor_name,fieldName,askedDate,config)


    def get_current_vals(self,processor_name,config):
        return self._last_record(processor_name,config)
    
    def backup(self):
        dest = Path("backup") / "backup.sqlite"
        dest.parent.mkdir(parents=True, exist_ok=True)

        # (optional) laufende WAL aufräumen
        with self.engine.connect() as c:
            c.exec_driver_sql("PRAGMA wal_checkpoint(TRUNCATE);")

        # DBAPI-Connection greifen und Backup fahren
        with closing(self.engine.raw_connection()) as raw:
        # DB-API connection extrahieren (SA 1.4 vs 2.0 kompatibel)
            dbapi_conn = getattr(raw, "connection", None) or getattr(raw, "driver_connection", None)
            if dbapi_conn is None:
                raise RuntimeError("Could not access DB-API connection from raw_connection()")

            with sqlite3.connect(str(dest)) as dst:
                dbapi_conn.backup(dst, pages=0)  # konsistente Kopie, auch im WAL-Modus

        return str(dest)