"""Utility functions for reporting and profiling datasets."""

import json
from pathlib import Path
import polars as pl
from typing import List, Dict, Any
from src.utils.dataset_profiling import profile_files, compare_schemas




def dump_json_string(json_str: str, file_path: str | Path, indent: int = 2) -> None:
    """
    Take a JSON string and dump it into a JSON file.

    Parameters:
        json_str: A valid JSON string.
        file_path: Path to the output JSON file.
        indent: Indentation level for pretty-printing (default=2).

    Raises:
        ValueError: If the input string is not valid JSON.
    """
    try:
        data = json.loads(json_str)   # parse to Python object
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON string: {e}") from e

    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)  # create dirs if needed

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)


def create_profile_report(files: list[Path], output_path: Path, json_name: str = "profile_report.json") -> dict:
    """Create and store a JSON profile report."""
    try:
        profiles = profile_files(files)

    except Exception as e:
        print(f"Error profiling files: {e}")
        return {"error": str(e)}

    try:
        union_cols, inter_cols, presence_df = compare_schemas(profiles)
    except Exception as e:
        print(f"Error comparing schemas: {e}")
        return {"error": str(e)}

    try:
        analysis_transactions = {
        "05_profiles": profiles,
        "04_union_columns": union_cols,
        "03_intersection_columns": inter_cols,
        "02_missing_cols_per_file": str(presence_df.filter(pl.any_horizontal(pl.col(pl.Boolean) == False))),
        "01_number_samples": sum([p['rows'] for p in profiles])
        }
    except Exception as e:
        print(f"Error creating analysis dictionary: {e}")
        return {"error": str(e)}

    # dump to JSON file
    try:
        json_str = json.dumps(analysis_transactions, indent=2)
        dump_json_string(json_str, output_path / json_name)
    except Exception as e:
        print(f"Error dumping JSON string: {e}")
        return {"error": str(e)}

    return analysis_transactions