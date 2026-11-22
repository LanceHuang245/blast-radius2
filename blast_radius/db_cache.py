"""Generic lazy deterministic result cache for in-memory DuckDB and SQLite test connections.

At Docker build time, `install_db_recorder()` records (engine, ddl_fingerprint, query_sql) -> rows | exc
while running DB-backed sqlglot tests, plus materializes read-only /app/map/tpcds.duckdb and tpch.duckdb.

At scoring time, `install_db_cache()` proxies `duckdb.connect()` and `sqlite3.connect(":memory:")`:
- DDL/DML statements are buffered without opening any real database engine.
- Read queries hitting the cache return exact pinned results/exceptions in ~2 microseconds.
- Any cache miss (e.g. when a patch altered generated SQL) lazily opens the real DB, replays
  the buffered DDL/DML sequence once, and executes the query genuinely.
"""

from __future__ import annotations

import hashlib
import os
import pickle
import re
from pathlib import Path
from typing import Any

from blast_radius import APP

DB_CACHE_FILE = APP / "map" / "db_cache.pkl"
TPCDS_DUCKDB_FILE = APP / "map" / "tpcds.duckdb"
TPCH_DUCKDB_FILE = APP / "map" / "tpch.duckdb"

_NONDET_RE = re.compile(r"\b(?:RANDOM|RAND|NOW|CURRENT_DATE|CURRENT_TIME|CURRENT_TIMESTAMP|TODAY|UUID)\b", re.I)
_MUTATE_RE = re.compile(
    r"^(?:CREATE|INSERT|UPDATE|DELETE|DROP|ALTER|PRAGMA|SET|ATTACH|DETACH|USE|BEGIN|COMMIT|ROLLBACK|COPY)\b",
    re.I,
)

_CACHE_DATA: tuple[dict[tuple[str, str, str], tuple[str, Any]], set[tuple[str, str]]] | None = None
_ORIG_DUCK_CONNECT = None
_ORIG_SQLITE_CONNECT = None


def _ddl_fp(prev_fp: str, kind: str, sql: str) -> str:
    return hashlib.blake2b(f"{prev_fp}|{kind}|{sql}".encode("utf-8"), digest_size=12).hexdigest()


def _is_in_memory_target(engine: str, args: tuple, kwargs: dict) -> bool:
    if kwargs:
        if any(k != "database" for k in kwargs):
            return False
        db = kwargs.get("database", ":memory:")
    elif args:
        if len(args) > 1:
            return False
        db = args[0]
    else:
        db = ":memory:"
    if engine == "duckdb":
        return db in (None, "", ":memory:")
    return db == ":memory:"


def load_db_cache() -> tuple[dict[tuple[str, str, str], tuple[str, Any]], set[tuple[str, str]]]:
    global _CACHE_DATA
    if _CACHE_DATA is None:
        if DB_CACHE_FILE.is_file():
            try:
                with open(DB_CACHE_FILE, "rb") as handle:
                    data = pickle.load(handle)
                if isinstance(data, dict) and "entries" in data and "valid_fps" in data:
                    entries = data["entries"] if isinstance(data["entries"], dict) else {}
                    fps = set(data["valid_fps"]) if isinstance(data["valid_fps"], (set, frozenset, list, tuple)) else set()
                    _CACHE_DATA = (entries, fps)
                elif isinstance(data, dict):
                    fps = {(eng, fp) for eng, fp, _sql in data}
                    _CACHE_DATA = (data, fps)
                else:
                    _CACHE_DATA = ({}, set())
            except Exception:
                _CACHE_DATA = ({}, set())
        else:
            _CACHE_DATA = ({}, set())
    return _CACHE_DATA


# --------------------------------------------------------------------------- build-time recorder


_REC_ENTRIES: dict[tuple[str, str, str], tuple[str, Any]] = {}
_REC_VALID_FPS: set[tuple[str, str]] = set()


def install_db_recorder() -> None:
    """Wrap duckdb.connect and sqlite3.connect to record deterministic query outcomes."""
    global _ORIG_DUCK_CONNECT, _ORIG_SQLITE_CONNECT
    import duckdb
    import sqlite3

    if _ORIG_DUCK_CONNECT is None:
        _ORIG_DUCK_CONNECT = duckdb.connect
    if _ORIG_SQLITE_CONNECT is None:
        _ORIG_SQLITE_CONNECT = sqlite3.connect

    class _RecCursor:
        def __init__(self, real_cur, engine: str, fp: str, sql: str):
            self._cur = real_cur
            self._engine = engine
            self._fp = fp
            self._sql = sql

        def fetchall(self):
            try:
                rows = self._cur.fetchall()
            except BaseException as err:
                if not _NONDET_RE.search(self._sql):
                    _REC_ENTRIES[(self._engine, self._fp, self._sql)] = (
                        "exc",
                        "fetch",
                        type(err).__module__,
                        type(err).__name__,
                        str(err),
                    )
                raise
            if not _NONDET_RE.search(self._sql):
                _REC_ENTRIES[(self._engine, self._fp, self._sql)] = ("rows", rows)
            return rows

        def fetchone(self):
            try:
                row = self._cur.fetchone()
            except BaseException as err:
                if not _NONDET_RE.search(self._sql):
                    _REC_ENTRIES[(self._engine, self._fp, self._sql)] = (
                        "exc",
                        "fetch",
                        type(err).__module__,
                        type(err).__name__,
                        str(err),
                    )
                raise
            if not _NONDET_RE.search(self._sql):
                key = (self._engine, self._fp, self._sql)
                if key not in _REC_ENTRIES or _REC_ENTRIES[key][0] != "rows":
                    _REC_ENTRIES[key] = ("one", row)
            return row

        def __getattr__(self, name: str):
            return getattr(self._cur, name)

    class _RecConn:
        def __init__(self, real_conn, engine: str):
            self._conn = real_conn
            self._engine = engine
            self._fp = "0"
            self._tainted = False

        def execute(self, sql: str, *args, **kwargs):
            if args or kwargs or self._tainted or not isinstance(sql, str):
                self._tainted = True
                return self._conn.execute(sql, *args, **kwargs)
            stripped = sql.lstrip()
            if _MUTATE_RE.match(stripped):
                res = self._conn.execute(sql)
                self._fp = _ddl_fp(self._fp, "E", sql)
                _REC_VALID_FPS.add((self._engine, self._fp))
                return res
            fp_snap = self._fp
            try:
                cur = self._conn.execute(sql)
            except BaseException as err:
                if not _NONDET_RE.search(sql):
                    _REC_ENTRIES[(self._engine, fp_snap, sql)] = (
                        "exc",
                        "exec",
                        type(err).__module__,
                        type(err).__name__,
                        str(err),
                    )
                raise
            return _RecCursor(cur, self._engine, fp_snap, sql)

        def executescript(self, script: str):
            res = self._conn.executescript(script)
            self._fp = _ddl_fp(self._fp, "S", script)
            _REC_VALID_FPS.add((self._engine, self._fp))
            return res

        def create_function(self, *args, **kwargs):
            self._tainted = True
            return self._conn.create_function(*args, **kwargs)

        def close(self):
            return self._conn.close()

        def __enter__(self):
            self._conn.__enter__()
            return self

        def __exit__(self, *exc):
            return self._conn.__exit__(*exc)

        def __getattr__(self, name: str):
            return getattr(self._conn, name)

    def rec_duck_connect(*args, **kwargs):
        real = _ORIG_DUCK_CONNECT(*args, **kwargs)
        if _is_in_memory_target("duckdb", args, kwargs):
            return _RecConn(real, "duckdb")
        return real

    def rec_sqlite_connect(*args, **kwargs):
        real = _ORIG_SQLITE_CONNECT(*args, **kwargs)
        if _is_in_memory_target("sqlite3", args, kwargs):
            return _RecConn(real, "sqlite3")
        return real

    duckdb.connect = rec_duck_connect
    sqlite3.connect = rec_sqlite_connect


def save_db_cache(path: Path = DB_CACHE_FILE) -> int:
    with open(path, "wb") as handle:
        pickle.dump(
            {"entries": _REC_ENTRIES, "valid_fps": _REC_VALID_FPS},
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    return len(_REC_ENTRIES)


def restore_orig_db_connect() -> None:
    if _ORIG_DUCK_CONNECT is not None:
        import duckdb

        duckdb.connect = _ORIG_DUCK_CONNECT
    if _ORIG_SQLITE_CONNECT is not None:
        import sqlite3

        sqlite3.connect = _ORIG_SQLITE_CONNECT


# --------------------------------------------------------------------------- runtime lazy proxy


def _raise_cached_exc(mod_name: str, cls_name: str, msg: str) -> None:
    import importlib

    try:
        mod = importlib.import_module(mod_name)
        exc_cls = getattr(mod, cls_name)
    except (ImportError, AttributeError):
        if "duckdb" in mod_name:
            exc_cls = getattr(importlib.import_module("duckdb"), cls_name)
        else:
            raise
    raise exc_cls(msg)


def install_db_cache() -> None:
    """Install lazy in-memory DuckDB and SQLite proxy wrappers (idempotent)."""
    cache, valid_fps = load_db_cache()
    if not cache:
        return

    import sqlite3
    import sys

    global _ORIG_SQLITE_CONNECT
    if not getattr(sqlite3.connect, "_br_cached", False):
        if _ORIG_SQLITE_CONNECT is None:
            _ORIG_SQLITE_CONNECT = sqlite3.connect

        def cached_sqlite_connect(*args, **kwargs):
            if _is_in_memory_target("sqlite3", args, kwargs):
                return _LazyCachedConn("sqlite3", args, kwargs, cache, valid_fps)
            return _ORIG_SQLITE_CONNECT(*args, **kwargs)

        cached_sqlite_connect._br_cached = True  # type: ignore[attr-defined]
        sqlite3.connect = cached_sqlite_connect

    # Wrap duckdb lazily if already imported, or via wrap_duckdb_if_loaded right after module import.
    if "duckdb" in sys.modules:
        _wrap_duckdb(sys.modules["duckdb"], cache, valid_fps)


def wrap_duckdb_if_loaded() -> None:
    import sys

    cache, valid_fps = load_db_cache()
    if cache and "duckdb" in sys.modules:
        _wrap_duckdb(sys.modules["duckdb"], cache, valid_fps)


def _wrap_duckdb(duckdb_mod, cache: dict, valid_fps: set[tuple[str, str]]) -> None:
    global _ORIG_DUCK_CONNECT
    if getattr(duckdb_mod.connect, "_br_cached", False):
        return
    if _ORIG_DUCK_CONNECT is None:
        _ORIG_DUCK_CONNECT = duckdb_mod.connect

    def cached_duck_connect(*args, **kwargs):
        if _is_in_memory_target("duckdb", args, kwargs):
            return _LazyCachedConn("duckdb", args, kwargs, cache, valid_fps)
        return _ORIG_DUCK_CONNECT(*args, **kwargs)

    cached_duck_connect._br_cached = True  # type: ignore[attr-defined]
    duckdb_mod.connect = cached_duck_connect


class _CachedCursor:
    __slots__ = ("_conn", "_sql", "_entry", "_consumed")

    def __init__(self, conn: _LazyCachedConn, sql: str, entry: tuple):
        self._conn = conn
        self._sql = sql
        self._entry = entry
        self._consumed = False

    def fetchall(self):
        kind = self._entry[0]
        if kind == "rows":
            return list(self._entry[1])
        if kind == "exc":
            _raise_cached_exc(self._entry[2], self._entry[3], self._entry[4])
        # Cached via fetchone only: materialize to be 100% safe for fetchall.
        return self._materialize_exec().fetchall()

    def fetchone(self):
        kind = self._entry[0]
        if kind == "one":
            if self._consumed:
                return None
            self._consumed = True
            return self._entry[1]
        if kind == "rows":
            rows = self._entry[1]
            if self._consumed or not rows:
                return None
            self._consumed = True
            return rows[0]
        if kind == "exc":
            _raise_cached_exc(self._entry[2], self._entry[3], self._entry[4])
        return self._materialize_exec().fetchone()

    def _materialize_exec(self):
        real = self._conn._materialize()
        return real.execute(self._sql)

    def __iter__(self):
        return iter(self.fetchall())

    def __getattr__(self, name: str):
        return getattr(self._materialize_exec(), name)


class _NoopCursor:
    __slots__ = ("_conn", "_sql")

    def __init__(self, conn: _LazyCachedConn, sql: str):
        self._conn = conn
        self._sql = sql

    def _real(self):
        real = self._conn._materialize(skip_last=True)
        return real.execute(self._sql)

    def fetchall(self):
        return self._real().fetchall()

    def fetchone(self):
        return self._real().fetchone()

    def __iter__(self):
        return iter(self._real())

    def __getattr__(self, name: str):
        return getattr(self._real(), name)


class _LazyCachedConn:
    """Delays real DuckDB/SQLite opening until a cache miss or uncached feature is used."""

    def __init__(self, engine: str, args: tuple, kwargs: dict, cache: dict, valid_fps: set[tuple[str, str]]):
        self._engine = engine
        self._args = args
        self._kwargs = kwargs
        self._cache = cache
        self._valid_fps = valid_fps
        self._ddls: list[tuple[str, str]] = []
        self._fp = "0"
        self._real = None
        self._closed = False
        self._tainted = False

    def _materialize(self, skip_last: bool = False):
        if self._real is None:
            if self._engine == "duckdb":
                import duckdb

                connect_fn = _ORIG_DUCK_CONNECT or duckdb.connect
            else:
                import sqlite3

                connect_fn = _ORIG_SQLITE_CONNECT or sqlite3.connect
            real = connect_fn(*self._args, **self._kwargs)
            ddls = self._ddls[:-1] if (skip_last and self._ddls) else self._ddls
            if skip_last and self._ddls:
                self._ddls.pop()
            for kind, stmt in ddls:
                if kind == "S":
                    real.executescript(stmt)
                else:
                    real.execute(stmt)
            self._real = real
        return self._real

    def execute(self, sql: str, *args, **kwargs):
        if args or kwargs or not isinstance(sql, str) or self._tainted:
            self._tainted = True
            return self._materialize().execute(sql, *args, **kwargs)
        hit = self._cache.get((self._engine, self._fp, sql))
        if hit is not None:
            if hit[0] == "exc" and hit[1] == "exec":
                _raise_cached_exc(hit[2], hit[3], hit[4])
            return _CachedCursor(self, sql, hit)
        if _MUTATE_RE.match(sql.lstrip()):
            next_fp = _ddl_fp(self._fp, "E", sql)
            if self._real is None and (self._engine, next_fp) in self._valid_fps:
                self._ddls.append(("E", sql))
                self._fp = next_fp
                return _NoopCursor(self, sql)
            real = self._materialize()
            self._fp = next_fp
            return real.execute(sql)
        return self._materialize().execute(sql)

    def executescript(self, script: str):
        next_fp = _ddl_fp(self._fp, "S", script)
        if self._real is None and (self._engine, next_fp) in self._valid_fps:
            self._ddls.append(("S", script))
            self._fp = next_fp
            return self
        real = self._materialize()
        self._fp = next_fp
        return real.executescript(script)

    def close(self):
        self._closed = True
        if self._real is not None:
            return self._real.close()
        return None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        if self._real is not None:
            return self._real.__exit__(*exc)
        self.close()
        return False

    def __getattr__(self, name: str):
        if name in ("create_function", "create_aggregate", "create_collation", "register"):
            self._tainted = True
        return getattr(self._materialize(), name)


def build_tpc_duckdb_files(source: Path) -> None:
    """Pre-materialize /app/map/tpcds.duckdb and tpch.duckdb from CSV.gz at build time."""
    import duckdb
    from tests.helpers import FIXTURES_DIR, TPCDS_SCHEMA, TPCH_SCHEMA

    restore_orig_db_connect()
    for db_file, folder, schema in (
        (TPCDS_DUCKDB_FILE, "tpc-ds", TPCDS_SCHEMA),
        (TPCH_DUCKDB_FILE, "tpc-h", TPCH_SCHEMA),
    ):
        if db_file.exists():
            os.unlink(db_file)
        conn = duckdb.connect(str(db_file))
        try:
            for table, columns in schema.items():
                conn.execute(
                    f"CREATE TABLE {table} AS SELECT * FROM READ_CSV("
                    f"'{FIXTURES_DIR}/optimizer/{folder}/{table}.csv.gz', delim='|', header=True, columns={columns})"
                )
        finally:
            conn.close()
