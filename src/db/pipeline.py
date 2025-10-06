"""
Database pipeline: schema creation, CSV ingestion, and arbitrary SQL execution.
Uses Polars for CSV reading and processing, SQLAlchemy for connection pooling,
and psycopg2 for efficient COPY operations.
"""

from __future__ import annotations

import io
import os
from typing import Optional, Dict, Tuple, Set, List

import polars as pl
import psycopg2
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from dotenv import load_dotenv, dotenv_values
from pathlib import Path
from src.utils.routing import env_config


# ---------- CONNECTIONS ----------
def _engine():
    """
    Create a SQLAlchemy engine with connection pooling.
    Uses env_config() to get connection parameters.
    """
    config = env_config()
    PG_HOST = config.get("POSTGRES_HOST", "localhost")
    PG_PORT = config.get("POSTGRES_PORT", 5432)
    PG_USER = config.get("POSTGRES_USER", "local")
    PG_PASSWORD = config.get("POSTGRES_PASSWORD", "root")
    PG_DB = config.get("POSTGRES_DB", "local_postgres_db")
    PG_SCHEMA = config.get("POSTGRES_SCHEMA", "public")
    url = f"postgresql+psycopg2://{PG_USER}:{PG_PASSWORD}@{PG_HOST}:{PG_PORT}/{PG_DB}"
    return create_engine(
        url,
        pool_pre_ping=True,
        future=True,
        connect_args={"options": f"-c search_path={PG_SCHEMA}"}  # PG_SCHEMA='public'
    )

def _conn():
    """ Create a psycopg2 connection. """
    config = env_config()
    PG_HOST = config.get("POSTGRES_HOST", "localhost")
    PG_PORT = config.get("POSTGRES_PORT", 5432)
    PG_USER = config.get("POSTGRES_USER", "local")
    PG_PASSWORD = config.get("POSTGRES_PASSWORD", "root")
    PG_DB = config.get("POSTGRES_DB", "local_postgres_db")
    return psycopg2.connect(
        host=PG_HOST, port=PG_PORT, user=PG_USER, password=PG_PASSWORD, dbname=PG_DB
    )

# ---------- UTILS ----------
def _qi(ident: str) -> str:
    """
    Quote identifier for Postgres.
    Simple quoting; does not handle schema-qualified names.
    
    Parameters:
    - `ident`: identifier to quote.

    Returns: quoted identifier.
    """
    return '"' + ident.replace('"', '""') + '"'

def _get_table_columns_meta(table_name: str, schema: Optional[str] = None):
    """
    Return (ordered_cols, cols_set, defaults_map) where defaults_map[col] is the column_default text (or None).

    Parameters:
    - `table_name`: target DB table name (case-insensitive).
    - `schema`: optional DB schema (defaults to env_config() PG_SCHEMA or 'public

    Returns:
        - `ordered_cols`: list of column names in table order.
        - `cols_set`: set of column names.
        - `defaults_map`: dict mapping column name to its default value text (or None).
    """
    config = env_config()
    PG_SCHEMA = config.get("POSTGRES_SCHEMA", "public")
    schema = schema or PG_SCHEMA
    q = """
    SELECT column_name, column_default
    FROM information_schema.columns
    WHERE table_schema = :s AND table_name = :t
    ORDER BY ordinal_position
    """
    df = run_sql(q, {"s": schema, "t": table_name.lower()})
    ordered = []
    defaults = {}
    if df is not None and df.height > 0:
        ordered = df["column_name"].to_list()
        # Build defaults map (can be None)
        defaults = {df["column_name"][i]: df["column_default"][i] for i in range(df.height)}
    return ordered, set(ordered), defaults

# ---------- PUBLIC API ----------
def create_schema_from_file(path: str | None = None) -> None:
    """
    Create schema by executing the SQL in `path`.
    If `path` is None, resolve SCHEMA_FILE from env_config().

    Parameters:
    - `path`: path to the SQL schema file (optional).

    Raises:
        FileNotFoundError if the schema file does not exist.
    """
    config = env_config()
    schema_file = Path(path) if path is not None else Path(str(config.get("SCHEMA_FILE", "src/db/sql/schema.sql")))
    schema_file = schema_file if schema_file.is_absolute() else (Path.cwd() / schema_file)

    if not schema_file.exists():
        raise FileNotFoundError(f"Schema file not found: {schema_file}")

    with open(schema_file, "r", encoding="utf-8") as f:
        ddl = f.read()
    with _conn() as conn:
        with conn.cursor() as cur:
            cur.execute(ddl)
        conn.commit()


# def run_sql(sql: str, params: Optional[dict] = None) -> Optional[pl.DataFrame]:
#     """
#     Execute arbitrary SQL; returns a Polars DataFrame for SELECT-like statements,
#     otherwise returns None.
#     """
#     eng = _engine()
#     with eng.connect() as con:
#         res = con.execute(text(sql), params or {})
#         try:
#             rows = res.fetchall()
#             cols = list(res.keys())
#             return pl.DataFrame(rows, schema=cols)
#         except Exception:
#             return None
        
def run_sql(sql: str, params: Optional[dict] = None) -> Optional[pl.DataFrame]:
    """
    Execute arbitrary SQL; returns a Polars DataFrame for SELECT-like statements,
    otherwise returns None. Uses a transactional block that auto-commits.

    Parameters:
    - `sql`: SQL statement to execute.
    - `params`: optional dict of parameters for parameterized queries.

    Returns:
        Polars DataFrame for queries that return rows (e.g., SELECT),
        otherwise None for statements like INSERT/UPDATE/DELETE.
    """
    eng = _engine()
    # .begin() starts a transaction and COMMITs on successful exit
    with eng.begin() as con:
        res = con.execute(text(sql), params or {})
        if res.returns_rows:  # True for SELECT/CTE that return rows
            rows = res.fetchall()
            cols = list(res.keys())
            return pl.DataFrame(rows, schema=cols)
        return None


def _copy_polars_df(df: pl.DataFrame, table_name: str, schema: Optional[str] = None) -> int:
    """
    COPY a Polars DataFrame into Postgres quickly.

    Parameters:
    - `df`: Polars DataFrame to copy.
    - `table_name`: target DB table name (case-insensitive).
    - `schema`: optional DB schema (defaults to env_config() PG_SCHEMA or 'public

    Returns:
        Returns the total row count in the table after ingestion.
    """
    config = env_config()
    PG_SCHEMA = config.get("POSTGRES_SCHEMA", "public")
    schema = schema or PG_SCHEMA

    # Ensure CSV uses '.' for decimals; Polars uses '.' internally, so this is fine.
    buf = io.StringIO()
    df.write_csv(buf)  # header included by default
    buf.seek(0)

    col_list = ", ".join(_qi(c) for c in df.columns)
    copy_sql = f"COPY {_qi(schema)}.{_qi(table_name)} ({col_list}) FROM STDIN WITH (FORMAT CSV, HEADER TRUE)"

    with _conn() as conn:
        with conn.cursor() as cur:
            cur.copy_expert(copy_sql, buf)
        conn.commit()

    cnt = run_sql(f"SELECT COUNT(*) AS n FROM {_qi(schema)}.{_qi(table_name)}")
    return int(cnt["n"][0]) if cnt is not None else -1

def ingest_csv_polars(
    table_name: str,
    csv_path: str,
    schema: Optional[str] = None,
    encoding: str = "utf-8",
    rename_map: Optional[Dict[str, str]] = None,
    fill_defaults: Optional[Dict[str, object]] = None,
    separator: str = "|",
    try_parse_dates: bool = True,
    decimal_comma: bool = True,
    null_values: Optional[list[str]] = None,
) -> int:
    """
    Ingest a CSV file into the specified table using Polars for reading and processing.
    Parameters:
    - `table_name`: target DB table name (case-insensitive).
    - `csv_path`: path to the CSV file.
    - `schema`: optional DB schema (defaults to env_config() PG_SCHEMA or 'public').
    - `encoding`: file encoding (default 'utf-8').
    - `rename_map`: optional mapping of CSV column names to DB column names (case-insensitive).
    - `fill_defaults`: optional mapping of column names to default values to use when missing
      (only applied if the column has no DB default).
    - `separator`: CSV delimiter (default '|').
    - `try_parse_dates`: whether to attempt date parsing (default True).
    - `decimal_comma`: whether to treat ',' as decimal point (default True).
    - `null_values`: list of strings to treat as NULL (default None).

    Returns:
        Returns the total row count in the table after ingestion.
    """
    df = pl.read_csv(
        csv_path,
        separator=separator,
        try_parse_dates=try_parse_dates,
        decimal_comma=decimal_comma,
        null_values=null_values,
        encoding=encoding,
    )

    # normalize headers to lowercase to match DB
    df = df.rename({c: c.lower() for c in df.columns})

    if rename_map:
        rename_map = {k.lower(): v.lower() for k, v in rename_map.items()}
        df = df.rename(rename_map)

    table_cols_ordered, table_cols_set, defaults_map = _get_table_columns_meta(table_name, schema=schema)

    # 1) Drop any DF columns that are backed by a DB-generated default (e.g., serial/identity)
    #    so the DB default will be used instead of passing explicit values/NULLs.
    cols_with_nextval = {c for c, d in defaults_map.items() if isinstance(d, str) and "nextval(" in d}
    df = df.drop([c for c in df.columns if c in cols_with_nextval])

    # 2) Add missing columns ONLY if they have no DB default
    fill_defaults = { (k.lower() if isinstance(k, str) else k): v for k, v in (fill_defaults or {}).items() }
    for col in table_cols_set:
        if col not in df.columns:
            if defaults_map.get(col) is not None:
                # let the DB fill its default (e.g., BIGSERIAL)
                continue
            df = df.with_columns(pl.lit(fill_defaults.get(col, None)).alias(col))

    # 3) Align to table order (omitting columns we intentionally left out)
    df = df.select([c for c in table_cols_ordered if c in df.columns])

    return _copy_polars_df(df, table_name.lower(), schema=schema)

def ingest_from_paths_polars(
    products_files: list[str],
    transactions_files: list[str],
    schema: Optional[str] = None,
    encoding: str = "utf-8",
) -> dict:
    """
    Load Product first, then Transactions (one or many CSVs).
    Applies your parsing rules (| delimiter, decimal comma, null tokens).

    Parameters:
    - `products_files`: list of paths to Product CSV files.
    - `transactions_files`: list of paths to Transactions CSV files.
    - `schema`: optional DB schema (defaults to env_config() PG_SCHEMA or 'public
    - `encoding`: file encoding (default 'utf-8').

    Returns:
        A dict mapping file stem to total row count in the corresponding table after ingestion.
    """
    stats = {}
    for p in products_files:
        stats[Path(p["file_path"]).stem] = ingest_csv_polars(
            "Product",
            p["file_path"],
            schema=schema,
            encoding=encoding,
            separator=p.get("delimiter", "|"),
            try_parse_dates=p.get("try_parse_dates", True),
            decimal_comma=p.get("decimal_comma", True),
            null_values=p.get("null_value_list", ["#NO VALUE", ""]),
        )

    for p in transactions_files:
        stats[Path(p["file_path"]).stem] = ingest_csv_polars(
            "Transactions",
            p["file_path"],
            schema=schema,
            encoding=encoding,
            separator=p.get("delimiter", "|"),
            try_parse_dates=p.get("try_parse_dates", True),
            decimal_comma=p.get("decimal_comma", True),
            null_values=p.get("null_value_list", ["#NO VALUE", ""]),
            # Fill Numero_TPV when missing:
            fill_defaults={"Numero_TPV": None},
        )
    return stats
