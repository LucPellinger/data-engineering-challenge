"""
Dataset Profiler
Fast, memory-safe dataset profiling (CSV, Parquet, Excel) with Polars.
Handles large files via lazy scans, robustly detects delimiters, infers schemas,
and computes quick stats.
Saves JSON/CSV artifacts and small samples for easy reporting.

Usage:
  python -m src.utils.dataset_profiler_v2 --data-dir ./data --files data1.csv data2.parquet data3.xlsx --out-dir ./_profile_out
"""

from __future__ import annotations
from pathlib import Path
import argparse
import json
import os
import subprocess
from typing import Iterable, List, Dict, Tuple, Optional, Mapping
import polars as pl
from .logger import ModuleLogger
import logging

import re


# Initialize logger
logger = ModuleLogger.get(
    __name__, 
    log_dir="logs", 
    filename="dataset_profiler.log", 
    level=logging.INFO, 
    overwrite=True
).logger

# --------------------------- Errors ------------------------------------------

class DatasetProfilerError(Exception):
    """Base class for all dataset-profiler errors."""

class UnsupportedFormatError(DatasetProfilerError):
    def __init__(self, path: Path, suffix: str):
        super().__init__(f"Unsupported file format for {path} (suffix={suffix!r})")
        self.path, self.suffix = path, suffix

class DelimiterDetectionError(DatasetProfilerError):
    def __init__(self, path: Path, original: Exception):
        super().__init__(f"Failed to detect delimiter for {path}: {original}")
        self.path, self.original = path, original

class ScanBuildError(DatasetProfilerError):
    def __init__(self, path: Path, fmt: str, original: Exception):
        super().__init__(f"Failed to build scan for {path} (format={fmt}): {original}")
        self.path, self.fmt, self.original = path, fmt, original

class RowCountError(DatasetProfilerError):
    def __init__(self, path: Path, fmt: str, original: Exception):
        super().__init__(f"Failed to count rows for {path} (format={fmt}): {original}")
        self.path, self.fmt, self.original = path, fmt, original

class SchemaSampleError(DatasetProfilerError):
    def __init__(self, path: Path, original: Exception):
        super().__init__(f"Failed to get schema/sample for {path}: {original}")
        self.path, self.original = path, original

class ArtifactWriteError(DatasetProfilerError):
    def __init__(self, out_dir: Path, original: Exception):
        super().__init__(f"Failed to write artifacts to {out_dir}: {original}")
        self.out_dir, self.original = out_dir, original

class ColumnStatsError(DatasetProfilerError):
    def __init__(self, path: Path, columns, original: Exception):
        super().__init__(f"Failed to compute column stats for {path} (cols={columns}): {original}")
        self.path, self.columns, self.original = path, columns, original


# --------------------------- Utils -------------------------------------------

def infer_format(path: Path) -> str:
    s = path.suffix.lower()
    if s in {".csv", ".tsv", ".txt"}: return "csv"
    if s in {".parquet"}: return "parquet"
    if s in {".xlsx", ".xls"}: return "excel"
    raise UnsupportedFormatError(path, s)

def detect_delimiter(path: Path, candidates: Iterable[str] = (",", "\t", ";", "|")) -> str:
    logger.info(f"Detecting delimiter for: {path}")
    try:
        with open(path, "rb") as f:
            chunk = f.read(64 * 1024)
        text = chunk.decode("utf-8", errors="ignore")
        counts = {d: text.count(d) for d in candidates}
        ordered = sorted(counts.items(), key=lambda kv: (-kv[1], list(candidates).index(kv[0])))
        delim = ordered[0][0] if ordered and ordered[0][1] > 0 else ","
        logger.success(f"Detected delimiter: {delim!r} (counts={counts})")
        return delim
    except Exception as e:
        logger.exception(f"Delimiter detection failed for {path}: {e}")
        # Raise a typed error (caller can decide whether to continue)
        raise DelimiterDetectionError(path, e) from e


def fast_line_count(path: Path, has_header: bool = True) -> int:
    """
    Use `wc -l` if available, fallback to buffered Python read.
    Returns data rows (excludes header by default).
    """
    logger.info(f"Counting lines for: {path}")
    try:
        out = subprocess.check_output(["wc", "-l", str(path)], text=True).strip()
        total = int(out.split()[0])
        logger.success(f"Line count (data rows): {total}")
    except Exception as e:
        logger.warning(f"Line count failed for {path}: {e}")
        total = 0
        with open(path, "rb", buffering=1024 * 1024) as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                total += chunk.count(b"\n")
    if has_header and total > 0:
        total -= 1
    return total


def make_scan(
    path: Path,
    infer_rows: int = 5000,
    null_values: Iterable[str] = ("", "NA", "NaN"),
    delimiter: str = ",",
    schema_overrides: Optional[Dict[str, pl.datatypes.DataType]] = None,
    sheet_name: Optional[str] = None,
    strict: bool = False,  # NEW
) -> pl.LazyFrame:
    """
    Create a Polars lazy scan for the given file path.

    Parameters:
     - `path`: Path to the input file.
     - `infer_rows`: Number of rows to use for schema inference.
     - `null_values`: List of strings to treat as null values.
     - `delimiter`: Column delimiter for CSV files.
     - `schema_overrides`: Optional schema overrides for specific columns.
     - `sheet_name`: Optional sheet name for Excel files.
     - `strict`: Whether to enforce strict format checking.

    Returns:
        - A Polars LazyFrame representing the dataset.

    Raises:
        - `UnsupportedFormatError`: If the file format is unsupported.
        - `DelimiterDetectionError`: If delimiter detection fails (when `delimiter='auto'`).
        - `ScanBuildError`: If building the scan fails for any reason.
    """

    fmt = None
    try:
        fmt = infer_format(path)
    except UnsupportedFormatError as e:
        logger.exception(str(e))
        if strict:
            raise
        
        fmt = "csv"

    try:
        if fmt == "csv":
            if delimiter == "auto":
                try:
                    delimiter = detect_delimiter(path)
                except DelimiterDetectionError as e:
                    if strict:
                        raise
                    logger.warning(f"{e} — defaulting to ','")
                    delimiter = ","
            logger.info(f"Building CSV lazy scan for: {path} (delimiter={delimiter!r})")
            return pl.scan_csv(
                path,
                has_header=True,
                separator=delimiter,
                infer_schema_length=infer_rows,
                null_values=list(null_values),
                ignore_errors=True,
                low_memory=True,
                try_parse_dates=True,
                schema_overrides=schema_overrides or {},
            )

        if fmt == "parquet":
            logger.info(f"Building Parquet lazy scan for: {path}")
            return pl.scan_parquet(path, low_memory=True)

        # Excel (no scan_* API)
        logger.info(f"Reading Excel eagerly for: {path} (sheet={sheet_name})")
        try:
            # calamine supports top-level infer_schema_length
            df = pl.read_excel(
                path,
                sheet_name=sheet_name,
                has_header=True,
                schema_overrides=schema_overrides,
                infer_schema_length=infer_rows,   # <-- move here
                drop_empty_rows=True,
                drop_empty_cols=True,
                engine="calamine",
            )
        except Exception as calamine_err:
            if strict:
                raise
            logger.warning(
                f"Calamine failed for {path}: {calamine_err} — falling back to xlsx2csv engine"
            )
            # xlsx2csv also supports top-level infer_schema_length
            df = pl.read_excel(
                path,
                sheet_name=sheet_name,
                has_header=True,
                schema_overrides=schema_overrides,
                infer_schema_length=infer_rows,   # <-- still top-level
                drop_empty_rows=True,
                drop_empty_cols=True,
                engine="xlsx2csv",
                # read_options={}  # (optional) pass read_csv()-style options if needed
            )

        return df.lazy()

    except Exception as e:
        logger.exception(f"Scan build failed for {path}: {e}")
        raise ScanBuildError(path, fmt or "unknown", e) from e


# --------------------------- Core ops ----------------------------------------

def peek_schema_and_sample(
    path: Path,
    infer_rows: int = 5000,
    sample_rows: int = 10,
    null_values: Iterable[str] = ("", "NA", "NaN"),
    delimiter: str = "auto",
    schema_overrides: Optional[Dict[str, pl.datatypes.DataType]] = None,
    sheet_name: Optional[str] = None,
    strict: bool = False,
) -> Tuple[Dict[str, pl.DataType], pl.DataFrame, Optional[str]]:
    """
    Infer schema and fetch a small sample from the dataset.
    
    Parameters:
    - `path`: Path to the input file.
    - `infer_rows`: Number of rows to use for schema inference.
    - `sample_rows`: Number of rows to sample from the dataset.
    - `null_values`: List of strings to treat as null values.
    - `delimiter`: Column delimiter for CSV files.
    - `schema_overrides`: Optional schema overrides for specific columns.
    - `sheet_name`: Optional sheet name for Excel files.
    - `strict`: Whether to enforce strict format checking.

    Returns:
        - A tuple containing:
            - A dictionary mapping column names to their inferred data types.
            - A Polars DataFrame containing a small sample of the dataset.
            - The resolved delimiter used for CSV files (if applicable).
    """
    try:
        fmt = infer_format(path)
    except UnsupportedFormatError as e:
        logger.exception(str(e))
        if strict:
            raise
        fmt = "csv"  # best-effort

    resolved_delim = None
    if fmt == "csv":
        if delimiter == "auto":
            try:
                resolved_delim = detect_delimiter(path)
            except DelimiterDetectionError as e:
                if strict:
                    raise
                logger.warning(f"{e} — defaulting to ','")
                resolved_delim = ","
        else:
            resolved_delim = delimiter

    try:
        scan = make_scan(
            path=path,
            infer_rows=infer_rows,
            null_values=null_values,
            delimiter=resolved_delim or ",",
            schema_overrides=schema_overrides,
            sheet_name=sheet_name,
            strict=strict,
        )
        schema = scan.collect_schema()
        sample = scan.fetch(sample_rows)
        return schema, sample, resolved_delim
    except Exception as e:
        logger.exception(f"peek_schema_and_sample failed for {path}: {e}")
        raise SchemaSampleError(path, e) from e

def count_rows(path: Path, has_header: bool = True, strict: bool = False) -> int:
    """
    Count the number of data rows in the dataset.   
    
    Parameters:
    - `path`: Path to the input file.
    - `has_header`: Whether the file has a header row (only relevant for CSV).
    - `strict`: Whether to enforce strict format checking.

    Returns:
        - The number of data rows in the dataset.
    
    Raises:
        - `UnsupportedFormatError`: If the file format is unsupported.
        - `RowCountError`: If counting rows fails for any reason.
    """
    try:
        fmt = infer_format(path)
    except UnsupportedFormatError as e:
        logger.exception(str(e))
        if strict:
            raise
        fmt = "csv"

    try:
        if fmt == "csv":
            return fast_line_count(path, has_header=has_header)
        if fmt == "parquet":
            return int(pl.scan_parquet(path).select(pl.len()).collect().item())
        # Excel
        return int(pl.read_excel(path).height)

    except Exception as e:
        logger.exception(f"Row count failed for {path}: {e}")
        if strict:
            raise RowCountError(path, fmt, e) from e
        logger.warning(f"Continuing with rows=0 for {path} due to error.")
        return 0


def profile_files(
    files: List[Path],
    infer_rows: int = 5000,
    sample_rows: int = 10,
    null_values: Iterable[str] = ("", "NA", "NaN"),
    delimiter: str = "auto",
    schema_overrides: Optional[Dict[str, pl.datatypes.DataType]] = None,
    sheet_name: Optional[str] = None,
    fail_fast: bool = False,   # NEW
    strict: bool = False,      # NEW
) -> List[Dict]:
    """
    Profile a list of dataset files, returning a list of profile dictionaries.
    Each profile dictionary contains:
        - `file`: File name.
        - `path`: Full file path.
        - `rows`: Number of data rows.
        - `n_cols`: Number of columns.
        - `columns`: List of column names.
        - `dtypes`: Dictionary of column data types.
        - `delimiter`: Delimiter used in the file (None for non-CSV).
        - `format`: File format (e.g., CSV, Parquet).
        - `sample`: Sample data (if applicable).
        - `error`: Error message (if profiling failed).

    Parameters:
    - `files`: List of file paths to profile.
    - `infer_rows`: Number of rows to use for schema inference.
    - `sample_rows`: Number of rows to sample from each dataset.
    - `null_values`: List of strings to treat as null values.
    - `delimiter`: Column delimiter for CSV files.
    - `schema_overrides`: Optional schema overrides for specific columns.
    - `sheet_name`: Optional sheet name for Excel files.
    - `fail_fast`: Whether to abort immediately on first file error.
    - `strict`: Whether to enforce strict format checking.
    
    Returns:
        - A list of profile dictionaries, one per file.

    Raises:
        - `DatasetProfilerError`: If profiling fails and `fail_fast` is True.
        - `RowCountError`: If row counting fails and `fail_fast` is True.
        - `SchemaSampleError`: If schema/sample fetching fails and `fail_fast` is True.

    """

    logger.info(f"Profiling {len(files)} file(s)")
    profiles = []
    for f in files:
        try:
            logger.info(f"Profiling file: {f}")
            schema, sample, delim = peek_schema_and_sample(
                f,
                infer_rows=infer_rows,
                sample_rows=sample_rows,
                null_values=null_values,
                delimiter=delimiter,
                schema_overrides=schema_overrides,
                sheet_name=sheet_name,
                strict=strict,
            )
            n_rows = count_rows(f, has_header=True, strict=strict)
            prof = {
                "file": f.name,
                "path": str(f),
                "rows": n_rows,
                "n_cols": len(schema),
                "columns": list(schema.keys()),
                "dtypes": {k: str(v) for k, v in schema.items()},
                "delimiter": delim,             # None for non-CSV
                "format": infer_format(f),      # may raise, but we reached here
                #"sample": sample,
                "error": None,                  # NEW
            }
            profiles.append(prof)
            logger.success(f"Profiled {f.name}: rows={n_rows:,}, cols={len(schema)}")
            logger.info(f"...3")
            logger.info(f"...2")
            logger.info(f"...1")
        except DatasetProfilerError as e:
            logger.exception(f"Profiling failed for {f}: {e}")
            if fail_fast:
                raise
            profiles.append({
                "file": f.name,
                "path": str(f),
                "rows": 0,
                "n_cols": 0,
                "columns": [],
                "dtypes": {},
                "delimiter": None,
                "format": None,
                #"sample": pl.DataFrame(),  # empty placeholder
                "error": type(e).__name__ + ": " + str(e),
            })
    return profiles



def compare_schemas(profiles: List[Dict]) -> Tuple[List[str], List[str], pl.DataFrame]:
    """
    Return (union_cols, inter_cols, presence_df[bools])

    Parameters:
    - `profiles`: List of profile dictionaries (as returned by `profile_files`).

    Returns:
        - A tuple containing:
            - `union_cols`: List of columns present in any file.
            - `inter_cols`: List of columns present in all files.
            - `presence_df`: DataFrame indicating column presence across files.

    Raises:
        - None (assumes valid input profiles).

    """
    logger.info("Comparing schemas across files")
    col_sets = [set(p["columns"]) for p in profiles]
    union_cols = sorted(set().union(*col_sets)) if col_sets else []
    inter_cols = sorted(set.intersection(*col_sets)) if col_sets else []

    # Presence matrix
    presence_rows = []
    for col in union_cols:
        row = {"column": col}
        for p in profiles:
            row[p["file"]] = col in set(p["columns"])
        presence_rows.append(row)
    presence_df = pl.DataFrame(presence_rows) if presence_rows else pl.DataFrame({"column": []})
    logger.success(f"Union cols={len(union_cols)}, Intersection cols={len(inter_cols)}")
    return union_cols, inter_cols, presence_df


def quick_column_stats(
    files: List[Path],
    columns: List[str],
    infer_rows: int = 5000,
    null_values: Iterable[str] = ("", "NA", "NaN"),
    delimiter: str = "auto",
    approx_unique: bool = True,
    strict: bool = False,
) -> Dict[str, pl.DataFrame]:
    """
    Compute quick stats (null counts, distinct counts) for specified columns across multiple files.
    Parameters:
    - `files`: List of file paths to analyze.
    - `columns`: List of columns to compute stats for.
    - `infer_rows`: Number of rows to use for schema inference.
    - `null_values`: List of strings to treat as null values.
    - `delimiter`: Column delimiter for CSV files.
    - `approx_unique`: Whether to use approximate distinct counts for speed.
    - `strict`: Whether to enforce strict format checking.
    Returns:
        - A dictionary mapping file names to DataFrames containing the stats.

    Raises:
        - `DatasetProfilerError`: If stats computation fails and `strict` is True.
    """
    results: Dict[str, pl.DataFrame] = {}
    for f in files:
        try:
            fmt = infer_format(f)
        except UnsupportedFormatError as e:
            logger.exception(str(e))
            if strict:
                raise
            fmt = "csv"

        try:
            resolved_delim = None
            if fmt == "csv":
                if delimiter == "auto":
                    try:
                        resolved_delim = detect_delimiter(f)
                    except DelimiterDetectionError as e:
                        if strict:
                            raise
                        logger.warning(f"{e} — defaulting to ','")
                        resolved_delim = ","
                else:
                    resolved_delim = delimiter

            scan = make_scan(
                f,
                infer_rows=infer_rows,
                null_values=null_values,
                delimiter=resolved_delim or ",",
                strict=strict,
            )

            exprs = []
            for c in columns:
                if c in scan.columns:
                    exprs.append(pl.col(c).null_count().alias(f"{c}__nulls"))
                    exprs.append(
                        (pl.col(c).approx_n_unique() if approx_unique else pl.col(c).n_unique())
                        .alias(f"{c}__{'n_approx_unique' if approx_unique else 'n_unique'}")
                    )

            if exprs:
                results[f.name] = scan.select(exprs).collect(streaming=True)
        except Exception as e:
            logger.exception(f"Column stats failed for {f}: {e}")
            if strict:
                raise ColumnStatsError(f, columns, e) from e
            # else skip this file
    return results



# --------------------------- I/O helpers -------------------------------------

def save_profiles_artifacts(
    profiles: List[Dict],
    union_cols: List[str],
    inter_cols: List[str],
    presence_df: pl.DataFrame,
    out_dir: Path,
    save_samples: bool = True,
) -> None:
    """
    Save profiling artifacts to the specified output directory:
    - profiles.json: List of profile dictionaries (excluding samples).
    - union_columns.json: List of columns in the union schema.
    - intersection_columns.json: List of columns in the intersection schema.
    - presence_matrix.csv: CSV file indicating column presence across files.
    - {file}_sample.csv: Sample data for each file (if `save_samples` is True).

    Parameters:
    - `profiles`: List of profile dictionaries (as returned by `profile_files`).
    - `union_cols`: List of columns in the union schema.
    - `inter_cols`: List of columns in the intersection schema.
    - `presence_df`: DataFrame indicating column presence across files.
    - `out_dir`: Directory to save the artifacts.
    - `save_samples`: Whether to save sample data for each file.

    Raises:
        - `ArtifactWriteError`: If writing artifacts fails for any reason.
    """
    try:
        logger.info(f"Writing artifacts to: {out_dir}")
        out_dir.mkdir(parents=True, exist_ok=True)

        json_profiles = []
        for p in profiles:
            p_copy = {k: v for k, v in p.items() if k != "sample"}
            json_profiles.append(p_copy)
        (out_dir / "profiles.json").write_text(json.dumps(json_profiles, indent=2))
        (out_dir / "union_columns.json").write_text(json.dumps(union_cols, indent=2))
        (out_dir / "intersection_columns.json").write_text(json.dumps(inter_cols, indent=2))
        if presence_df.height > 0:
            presence_df.write_csv(out_dir / "presence_matrix.csv")

        if save_samples:
            for p in profiles:
                try:
                    sample = p.get("sample")
                    if isinstance(sample, pl.DataFrame) and sample.height > 0:
                        (out_dir / f"{Path(p['file']).stem}_sample.csv").write_text(sample.write_csv())
                except Exception as e:
                    logger.warning(f"Failed to write sample for {p.get('file')}: {e}")

        logger.success("Artifacts written (JSON, CSV, samples)")
    except Exception as e:
        logger.exception(f"Artifacts write failed: {e}")
        raise ArtifactWriteError(out_dir, e) from e



# --------------------------- CLI ---------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Fast, memory-safe dataset profiling (CSV, Parquet, Excel) with Polars."
    )
    ap.add_argument(
        "--data-dir",
        type=Path,
        default=Path("."),
        help="Directory containing the data files (CSV, Parquet, or Excel)."
    )
    ap.add_argument(
        "--files",
        type=str,
        nargs="+",
        default=["data1.csv", "data2.parquet", "data3.xlsx"],
        help="File names (relative to --data-dir) or absolute paths. Supports CSV, Parquet, Excel."
    )
    ap.add_argument(
        "--infer-rows",
        type=int,
        default=5000,
        help="Rows to scan for schema inference."
    )
    ap.add_argument(
        "--sample-rows",
        type=int,
        default=10,
        help="Rows to fetch for small preview samples."
    )
    ap.add_argument(
        "--null-values",
        type=str,
        nargs="*",
        default=["", "NA", "NaN"],
        help="Values to treat as nulls."
    )
    ap.add_argument(
        "--delimiter",
        type=str,
        default="auto",
        help="CSV delimiter (e.g. ',', '\\t', ';', '|') or 'auto' to detect. Ignored for Parquet/Excel."
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("./_profile_out"),
        help="Where to save artifacts."
    )
    ap.add_argument(
        "--approx-unique",
        action="store_true",
        help="Use approx_n_unique() for faster distinct counts in column stats."
    )
    ap.add_argument(
        "--important-cols",
        type=str,
        nargs="*",
        default=None,
        help="Subset of columns for quick stats (nulls + distinct). Defaults to first 10 of union."
    )
    ap.add_argument(
        "--sheet-name",
        type=str,
        default=None,
        help="Excel sheet to read (if multiple)."
    )
    ap.add_argument(
        "--fail-fast",
        action="store_true",
        help="Abort immediately on first file error."
    )
    ap.add_argument(
        "--strict",
        action="store_true",
        help="Raise on delimiter detection or unsupported/malformed formats instead of falling back."
    )
    return ap.parse_args()



def main():
    try:
        logger.info("Starting dataset profiling pipeline")
        args = parse_args()
        logger.success("CLI arguments parsed")

        files = [Path(f) if Path(f).is_absolute() else args.data_dir / f for f in args.files]
        files = [f for f in files if f.exists()]
        if not files:
            raise SystemExit("No input files found. Check --data-dir/--files.")

        profiles = profile_files(
            files=files,
            infer_rows=args.infer_rows,
            sample_rows=args.sample_rows,
            null_values=args.null_values,
            delimiter=args.delimiter,
            sheet_name=args.sheet_name,
            fail_fast=args.fail_fast,
            strict=args.strict,
        )

        # Print overview (include error column if present)
        for p in profiles:
            fmt = p.get("format") or "unknown"
            delim_display = p["delimiter"] if p.get("delimiter") is not None else "N/A"
            print(f"\n=== {p['file']} ===")
            print(f"Format: {fmt} | Rows: {p['rows']:,} | Columns: {p['n_cols']} | Delimiter: {delim_display!r}")
            if p.get("error"):
                print(f"ERROR: {p['error']}")
                continue
            head_cols = p["columns"][:10]
            print("First columns:", head_cols, "..." if p["n_cols"] > 10 else "")
            try:
                from IPython.display import display  # type: ignore
                display(p["sample"])
            except Exception:
                print(p["sample"])

        union_cols, inter_cols, presence_df = compare_schemas([p for p in profiles if not p.get("error")])
        print(f"\n# of columns (union): {len(union_cols)}")
        print(f"# of columns (intersection): {len(inter_cols)}")

        important_cols = args.important_cols or union_cols[:10]
        if important_cols:
            print("\nImportant columns:", important_cols)
            stats = quick_column_stats(
                files=[Path(p["path"]) for p in profiles if not p.get("error")],
                columns=important_cols,
                infer_rows=args.infer_rows,
                null_values=args.null_values,
                delimiter=args.delimiter,
                approx_unique=args.approx_unique,
                strict=args.strict,
            )
            for fname, df in stats.items():
                print(f"\n[Stats] {fname}")
                try:
                    from IPython.display import display  # type: ignore
                    display(df)
                except Exception:
                    print(df)

        save_profiles_artifacts(
            profiles=profiles,
            union_cols=union_cols,
            inter_cols=inter_cols,
            presence_df=presence_df,
            out_dir=args.out_dir,
            save_samples=True,
        )
        logger.info(f"Artifacts written to: {args.out_dir.resolve()}")
        logger.success("Profiling complete")

    except SystemExit as e:
        # argparse and our explicit exits come here
        logger.error(str(e))
        raise
    except Exception as e:
        # last-resort safety net with traceback
        logger.exception(f"Fatal error: {e}")
        raise

def entry_error_proportions(
    df: pl.DataFrame,
    columns: Optional[List[str]] = None,
    null_tokens: Optional[Set[str]] = None,
    strict: bool = False,
    regex_cfg: Optional[Mapping[str, str]] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Compute entry error proportions per column, efficiently:
      - Uses regex patterns (US/EU numeric, int, float-dot, bool, ISO date, time).
      - Processes only unique values (value_counts), weighted by frequency.
      - Skips columns not present unless strict=True.

    Parameters:
    - df : pl.DataFrame
    - columns : list[str] | None
        Columns to check; if None, analyze all columns.
    - null_tokens : set[str] | None
        Placeholders treated as "no value" (e.g. {"#NO VALUE", "na", "null"}).
        NOTE: Empty string "" is handled separately as `empty_error_*`.
    - strict : bool
        If True, raise on missing columns; otherwise they are skipped with a warning.
    - regex_cfg : Mapping[str, str] | None
        Optional mapping providing regex patterns. Expected keys:
        "RE_INT", "RE_US_NUMERIC", "RE_EU_NUMERIC", "RE_FLOAT_DOT",
        "RE_BOOL", "RE_ISO_DATE", "RE_ISO_DATETIME".
        Falls back to sane defaults if not supplied.

    Returns:
    - Dict[str, Dict[str, float]]
    """
    logger.info("Computing entry error proportions (unique-weighted)")

    # --- choose columns & handle missing
    if columns is None:
        target_cols = df.columns
        missing = []
    else:
        missing = [c for c in columns if c not in df.columns]
        target_cols = [c for c in columns if c in df.columns]
        if missing:
            msg = f"Skipping {len(missing)} missing columns: {missing}"
            if strict:
                raise KeyError(msg)
            else:
                logger.warning(msg)

    # --- null tokens (treat empty string separately)
    tokens = set(null_tokens or {"#NO VALUE", "na", "n/a", "none", "null", "nan", "missing"})
    tokens_no_empty = {t for t in tokens if t != ""}

    # --- compile regexes once
    defaults = {
        "RE_INT": r"^[+-]?\d+$",
        "RE_US_NUMERIC": r"^[+-]?(\d{1,3}(,\d{3})*|\d+)(\.\d+)?$",
        "RE_EU_NUMERIC": r"^[+-]?(\d{1,3}(\.\d{3})*|\d+)(,\d+)?$",
        "RE_FLOAT_DOT": r"^[+-]?(\d{1,3}(,\d{3})*|\d+)?(\.\d+)$|^[+-]?\.\d+$|^[+-]?\d+\.$",
        "RE_BOOL": r"^(true|false|t|f|yes|no|y|n|0|1)$",
        "RE_ISO_DATE": r"^\d{4}-\d{2}-\d{2}$",
        "RE_ISO_DATETIME": r"^\d{4}-\d{2}-\d{2}[ tT]\d{2}:\d{2}(:\d{2})?(\.\d+)?(Z|[+-]\d{2}:\d{2})?$",
    }
    rcfg = {**defaults, **(regex_cfg or {})}

    re_int          = re.compile(rcfg["RE_INT"])
    re_us_numeric   = re.compile(rcfg["RE_US_NUMERIC"])
    re_eu_numeric   = re.compile(rcfg["RE_EU_NUMERIC"])
    re_float_dot    = re.compile(rcfg["RE_FLOAT_DOT"])
    re_bool         = re.compile(rcfg["RE_BOOL"], re.IGNORECASE)
    re_iso_date     = re.compile(rcfg["RE_ISO_DATE"])
    re_iso_datetime = re.compile(rcfg["RE_ISO_DATETIME"])
    # time-only (HH:MM[:SS][.ms]) – derived locally (not in .env)
    re_time_only    = re.compile(r"^\d{2}:\d{2}(:\d{2})?(\.\d+)?$")

    results: Dict[str, Dict[str, float]] = {}

    for col in target_cols:
        # string view
        s = df[col].cast(pl.Utf8, strict=False)

        # process uniques with weights
        vc = s.value_counts(sort=False)  # DataFrame: [col, count]
        if vc.height == 0:
            # empty column
            results[col] = {
                "total": 0,
                "numeric_valid_count": 0,
                "comma_error_count": 0,
                "no_value_error_count": 0,
                "empty_error_count": 0,
                "other_error_count": 0,
                "boolean_valid_count": 0,
                "date_valid_count": 0,
                "hour_valid_count": 0,
                "numeric_valid_proportion": 0.0,
                "comma_error_proportion": 0.0,
                "no_value_error_proportion": 0.0,
                "empty_error_proportion": 0.0,
                "other_error_proportion": 0.0,
                "boolean_valid_proportion": 0.0,
                "date_valid_proportion": 0.0,
                "hour_valid_proportion": 0.0,
            }
            continue

        val_col = col
        count_col = "count" if "count" in vc.columns else "counts"

        values = vc.get_column(val_col).to_list()
        counts = vc.get_column(count_col).to_list()
        total = int(sum(counts))

        # accumulators
        comma_errors = 0
        no_value_errors = 0
        empty_errors = 0
        numeric_valid = 0
        boolean_valid = 0
        date_valid = 0
        hour_valid = 0

        for v, c in zip(values, counts):
            # v may be None if the column has nulls
            if v is None:
                # keep None in "other" bucket (not counted explicitly)
                continue

            # make sure it's a string
            v_str: str = v

            # null/empty
            if v_str == "":
                empty_errors += c
                continue

            if v_str in tokens_no_empty:
                no_value_errors += c
                continue

            # numeric validity checks
            has_comma = "," in v_str
            eu_ok = re_eu_numeric.fullmatch(v_str) is not None
            us_ok = re_us_numeric.fullmatch(v_str) is not None
            int_ok = re_int.fullmatch(v_str) is not None
            float_dot_ok = re_float_dot.fullmatch(v_str) is not None

            # if comma present but not valid EU or US numeric → comma error
            if has_comma and not (eu_ok or us_ok):
                comma_errors += c

            if int_ok or us_ok or eu_ok or float_dot_ok:
                numeric_valid += c

            # booleans/dates/times (informational counts)
            if re_bool.fullmatch(v_str):
                boolean_valid += c
            if re_iso_date.fullmatch(v_str):
                date_valid += c
            if re_time_only.fullmatch(v_str):
                hour_valid += c
            # if you also want to acknowledge ISO datetimes (date+time), you could
            # increment date_valid/hour_valid or track a separate counter:
            # if re_iso_datetime.fullmatch(v_str): ...

        other_errors = max(0, total - (comma_errors + no_value_errors + empty_errors + numeric_valid))

        results[col] = {
            "total": total,
            "numeric_valid_count": numeric_valid,
            "comma_error_count": comma_errors,
            "no_value_error_count": no_value_errors,
            "empty_error_count": empty_errors,
            "other_error_count": other_errors,
            "boolean_valid_count": boolean_valid,
            "date_valid_count": date_valid,
            "hour_valid_count": hour_valid,

            "numeric_valid_proportion": (numeric_valid / total) if total else 0.0,
            "comma_error_proportion": (comma_errors / total) if total else 0.0,
            "no_value_error_proportion": (no_value_errors / total) if total else 0.0,
            "empty_error_proportion": (empty_errors / total) if total else 0.0,
            "other_error_proportion": (other_errors / total) if total else 0.0,
            "boolean_valid_proportion": (boolean_valid / total) if total else 0.0,
            "date_valid_proportion": (date_valid / total) if total else 0.0,
            "hour_valid_proportion": (hour_valid / total) if total else 0.0,
        }

    logger.success("Finished computing error proportions")
    return results


if __name__ == "__main__":
    main()
