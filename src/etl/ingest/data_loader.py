"""
Data loading script to ingest product and transaction data into the database.
"""

# src/etl/ingest/data_loader.py
from pathlib import Path
from src.db.pipeline import ingest_from_paths_polars
from src.assets.metadata.transactions import transactions_files
from src.assets.metadata.products import products_files

if __name__ == "__main__":
    stats = ingest_from_paths_polars(
        products_files=products_files,
        transactions_files=transactions_files,
    )
    print("Load stats:", stats)