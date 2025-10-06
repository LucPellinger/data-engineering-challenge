"""
Module to run SQL against the database from python script level.
"""

# src/etl/ingest/run_sql.py
import sys
from src.db.pipeline import run_sql

if __name__ == "__main__":

    sql_queries = {
        "query_1": "select count(*) from product;",
    }

    sql = " ".join(sys.argv[1:]) or sql_queries["query_1"]
    df = run_sql(sql)
    if df is not None:
        print(df)
    else:
        print("OK")