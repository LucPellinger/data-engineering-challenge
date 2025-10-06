"""Utility functions for routing and file path management."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping, Dict, Any
from dotenv import load_dotenv, dotenv_values, find_dotenv


def env_config() -> Mapping[str, str]:
    """ 
    Load environment variables from a .env file and return as a dictionary. 
    Adds default regex patterns and paths to the config dictionary.

    Returns:
        A dictionary with environment variable names as keys and their values as values.
    Raises:
        FileNotFoundError: If the .env file is not found.
        ValueError: If there is an issue loading the .env file.
    """
    
    # Relative Paths (assumed from project root)
    PRODUCT_DIR  = "src/assets/dataset/raw/products"
    DATA_DIR     = "src/assets/dataset/raw/transactions"
    OUTPUT_DIR   = "src/assets/analysis"
    PROCESSED_DIR= "src/assets/dataset/processed"
    PROCESSED_TRANSACTIONS_DIR= "src/assets/dataset/processed/transactions"
    PROCESSED_PRODUCTS_DIR= "src/assets/dataset/processed/products"
    METADATA_DIR = "src/assets/metadata"
    SCHEMA_FILE  = "src/db/sql/schema.sql"  # <-- new default

    # Data Cleaning Parameters (unchanged)
    RE_INT          = r"^[+-]?\d+$"
    RE_FLOAT_DOT    = r"^[+-]?(\d{1,3}(,\d{3})*|\d+)?(\.\d+)$|^[+-]?\.\d+$|^[+-]?\d+\.$"
    RE_FLOAT_COMMA  = r"^[+-]?(\d{1,3}(\.\d{3})*|\d+)?(,\d+)$|^[+-]?,\d+$|^[+-]?\d+,$"
    RE_EU_NUMERIC   = r"^[+-]?(\d{1,3}(\.\d{3})*|\d+)(,\d+)?$"
    RE_US_NUMERIC   = r"^[+-]?(\d{1,3}(,\d{3})*|\d+)(\.\d+)?$"
    RE_BOOL         = r"^(true|false|t|f|yes|no|y|n|0|1)$"
    RE_ISO_DATE     = r"^\d{4}-\d{2}-\d{2}$"
    RE_ISO_DATETIME = r"^\d{4}-\d{2}-\d{2}[ tT]\d{2}:\d{2}(:\d{2})?(\.\d+)?(Z|[+-]\d{2}:\d{2})?$"

    DEFAULT_NULL_TOKENS = {
        "#NO VALUE", "#NO_VALUE", "#no_value", "na", "n/a", "none", "null", "nan", "missing", ""
    }

    config: dict[str, object] = {
        "PRODUCT_DIR": PRODUCT_DIR,
        "DATA_DIR": DATA_DIR,
        "OUTPUT_DIR": OUTPUT_DIR,
        "PROCESSED_DIR": PROCESSED_DIR,
        "PROCESSED_DIR_TRANSACTIONS": PROCESSED_TRANSACTIONS_DIR,
        "PROCESSED_DIR_PRODUCTS": PROCESSED_PRODUCTS_DIR,
        "METADATA_DIR": METADATA_DIR,
        "SCHEMA_FILE": SCHEMA_FILE,
        "RE_INT": RE_INT,
        "RE_FLOAT_DOT": RE_FLOAT_DOT,
        "RE_FLOAT_COMMA": RE_FLOAT_COMMA,
        "RE_EU_NUMERIC": RE_EU_NUMERIC,
        "RE_US_NUMERIC": RE_US_NUMERIC,
        "RE_BOOL": RE_BOOL,
        "RE_ISO_DATE": RE_ISO_DATE,
        "RE_ISO_DATETIME": RE_ISO_DATETIME,
        "DEFAULT_NULL_TOKENS": DEFAULT_NULL_TOKENS,
    }

    # Load .env from the **project root**
    env_path = find_dotenv()   # automatically searches upward
    if not env_path:
        raise FileNotFoundError(".env file not found")
    env_path = env_path# Path("..") / ".env"
    dotenv_available = load_dotenv(dotenv_path=env_path)
    if dotenv_available:
        env_values = dotenv_values(dotenv_path=env_path)  # dict[str, str]
        config.update(env_values)
    else:
        raise FileNotFoundError(f".env file not found at {env_path}")

    # Optional: quick sanity prints (can be removed)
    print(f"Environment variables loaded from {env_path}: {dotenv_available}")
    print(f"Check if Dataset Directory is correct: {config.get('DATA_DIR')}")

    return config

def get_project_path(field_key: str, config: Mapping[str, str]) -> Path:
    """
    Construct project paths relative to the project root.

    Args:
        field_key: One of 'DATA_DIR', 'OUTPUT_DIR', 'PROCESSED_DIR', 'PRODUCT_DIR'.
        config: dict-like object with keys 'DATA_DIR', 'OUTPUT_DIR',
                'PROCESSED_DIR', and 'PRODUCT_DIR'.

    Returns:
        A dictionary with paths resolved against the project root.
    """
    try:
        root_dir = Path(os.getcwd()).parent

        if field_key == "ROOT":
            return root_dir
        elif field_key not in config:
            raise ValueError(f"Invalid field_key: {field_key}. Must be one of {list(config.keys())}.")

        path = root_dir / Path(config.get(field_key))
    except Exception as e:
        raise ValueError(f"Error constructing path for {field_key}: {e}")
    
    return path

def create_file_path_list(dir_path: Path, extensions: list[str]) -> list[Path]:
    """Create a list of file paths in a directory with specified extensions."""
    try:
        if not dir_path.exists() or not dir_path.is_dir():
            raise ValueError(f"Directory {dir_path} does not exist or is not a directory.")
    except Exception as e:
        raise ValueError(f"Error accessing directory {dir_path}: {e}")
    
    return [f for f in dir_path.iterdir() if f.is_file() and f.suffix in extensions]
