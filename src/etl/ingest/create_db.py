"""
Module to create the database schema from a SQL file.
"""

# src/etl/create_db.py
from ...db.pipeline import create_schema_from_file

if __name__ == "__main__":
    create_schema_from_file("src/db/sql/schema.sql")
    print("Schema created.")
