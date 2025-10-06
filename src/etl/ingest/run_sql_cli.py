"""
Command-line interface to run SQL files using src.db.pipeline.run_sql.

usage:
List available queries:
python -m src.etl.ingest.run_sql_cli --list


Run by name (filename without .sql):
python -m src.etl.ingest.run_sql_cli --name total_margin_signature_last_2_months

Run by explicit path:
python -m src.etl.ingest.run_sql_cli --path path/to/query.sql

Use a different queries directory:
python -m src.etl.ingest.run_sql_cli --dir src/etl/ingest/custom_sql --name some_query

"""



# src/etl/ingest/run_sql.py
import argparse
import sys
from pathlib import Path
from src.db.pipeline import run_sql

DEFAULT_QUERIES_DIR = Path(__file__).resolve().parent.parent.parent / "sql" / "queries"

def load_queries(queries_dir: Path):
    """Return a dict {query_name: file_path} for all .sql in the directory."""
    if not queries_dir.exists():
        return {}
    return {p.stem: p for p in queries_dir.glob("*.sql")}

def read_sql_from_file(path: Path) -> str:
    """Read and return the content of a SQL file."""
    if not path.exists():
        raise FileNotFoundError(f"SQL file not found: {path}")
    return path.read_text(encoding="utf-8")

def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Run an SQL file using src.db.pipeline.run_sql"
    )
    parser.add_argument(
        "-n", "--name",
        help="Name of the query file (without .sql) from the queries directory",
    )
    parser.add_argument(
        "-p", "--path",
        help="Path to a specific .sql file (overrides --name)"
    )
    parser.add_argument(
        "-d", "--dir",
        default=str(DEFAULT_QUERIES_DIR),
        help=f"Directory containing .sql queries (default: {DEFAULT_QUERIES_DIR})"
    )
    parser.add_argument(
        "-l", "--list",
        action="store_true",
        help="List available queries and exit"
    )

    args = parser.parse_args(argv)
    queries_dir = Path(args.dir)
    registry = load_queries(queries_dir)

    if args.list:
        if not registry:
            print(f"No .sql files found in: {queries_dir}")
        else:
            print(f"Queries in {queries_dir}:")
            for name in sorted(registry):
                print(f"  - {name}")
        return

    sql_text = None

    if args.path:
        sql_text = read_sql_from_file(Path(args.path))
    elif args.name:
        if args.name not in registry:
            available = ", ".join(sorted(registry.keys())) or "(none)"
            raise SystemExit(
                f"Unknown query '{args.name}'. Available: {available}"
            )
        sql_text = read_sql_from_file(registry[args.name])
    else:
        # Fallback: if there is a query called 'query_1.sql', use it; else first file
        if "query_1" in registry:
            sql_text = read_sql_from_file(registry["query_1"])
        elif registry:
            first = next(iter(registry.values()))
            print(f"No name/path provided; running first query found: {first.name}")
            sql_text = read_sql_from_file(first)
        else:
            # Final fallback to inline default (matches your original behavior)
            sql_text = "select count(*) from product;"

    df = run_sql(sql_text)
    if df is not None:
        print(df)
    else:
        print("OK")

if __name__ == "__main__":
    main()
