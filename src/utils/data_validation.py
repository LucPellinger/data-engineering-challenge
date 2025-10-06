from __future__ import annotations

import polars as pl
from typing import Dict, Iterable, List, Optional, Tuple, Union, Set, Literal
from pprint import pprint
import json
from pathlib import Path

import re
from collections import defaultdict

## ---------- Column exploration ----------
#def column_description(df: pl.DataFrame, column: str) -> dict:
#    """
#    Build a small data dictionary for a given DataFrame column.
#    """
#    if column not in df.columns:
#        raise KeyError(f"Column '{column}' not found in DataFrame")
#
#    s = df[column]
#    col_dict = {
#        "column_name": column,
#        "column_dtype": str(s.dtype),
#        "number_uniques": s.n_unique(),
#        "number_missing_values": s.null_count(),
#        "values_list": s.unique().to_list(),
#    }
#    pprint(col_dict, sort_dicts=False)
#    return col_dict
#
#
#def build_pl_series(input_list: list, col_name: str = "series") -> pl.Series:
#    """Helper to build a Polars Series from a Python list."""
#    return pl.Series(col_name, input_list)
#

# ---------- Core validation helpers ----------
def _col_invalid_expr(
    col: str,
    pat: str,
    *,
    accept_null: bool = False,
    treat_empty_as_invalid: bool = True,
) -> pl.Expr:
    """
    Expression that evaluates to True when the value in `col` is INVALID
    against regex `pat`, with configurable treatment of NULL/empty strings.
    """
    # Cast to Utf8 so str functions are available regardless of original dtype.
    x = pl.col(col).cast(pl.Utf8)

    # Match check (full-string because patterns are anchored in the schema).
    matches = x.str.contains(pat).fill_null(accept_null)

    # Empty-string handling
    if treat_empty_as_invalid:
        is_empty = (x == "").fill_null(False)
        # invalid if empty OR not matches
        return is_empty | ~matches
    else:
        return ~matches


def invalid_row_mask(
    df: pl.DataFrame,
    patterns: Dict[str, str],
    *,
    accept_nulls: Iterable[str] = (),
    treat_empty_as_invalid: bool = True,
) -> pl.Expr:
    """
    Build a single boolean expression for the dataframe that is True when
    any of the pattern-checked columns is invalid.
    """
    accept_nulls = set(accept_nulls)
    exprs = [
        _col_invalid_expr(
            col,
            pat,
            accept_null=(col in accept_nulls),
            treat_empty_as_invalid=treat_empty_as_invalid,
        )
        for col, pat in patterns.items()
        if col in df.columns  # ignore patterns for columns not present
    ]
    if not exprs:
        # If no matching columns, nothing is invalid.
        return pl.lit(False)
    return pl.any_horizontal(exprs)


def per_column_invalid_counts(
    df: pl.DataFrame,
    patterns: Dict[str, str],
    *,
    accept_nulls: Iterable[str] = (),
    treat_empty_as_invalid: bool = True,
) -> pl.DataFrame:
    """
    Return a 1-row DataFrame of invalid counts per column in `patterns`.
    Missing columns are skipped.
    """
    accept_nulls = set(accept_nulls)
    exprs = []
    for col, pat in patterns.items():
        if col not in df.columns:
            continue
        exprs.append(
            _col_invalid_expr(
                col,
                pat,
                accept_null=(col in accept_nulls),
                treat_empty_as_invalid=treat_empty_as_invalid,
            )
            .sum()
            .alias(f"{col}__invalid_count")
        )
    if not exprs:
        return pl.DataFrame()
    return df.select(exprs)


# def invalid_rows_with_bad_columns(
#     df: pl.DataFrame,
#     patterns: Dict[str, str],
#     *,
#     accept_nulls: Iterable[str] = (),
#     treat_empty_as_invalid: bool = True,
#     bad_cols_field: str = "bad_columns",
#     drop_temp_cols: bool = True,
# ) -> pl.DataFrame:
#     """
#     Return only invalid rows plus a `bad_columns` list that shows which
#     columns failed validation for each row.
#     """
#     accept_nulls = set(accept_nulls)
#     tmp_cols = []
# 
#     with_flags = df
#     for col, pat in patterns.items():
#         if col not in df.columns:
#             continue
#         tmp = f"__bad__{col}"
#         tmp_cols.append(tmp)
#         with_flags = with_flags.with_columns(
#             pl.when(
#                 _col_invalid_expr(
#                     col,
#                     pat,
#                     accept_null=(col in accept_nulls),
#                     treat_empty_as_invalid=treat_empty_as_invalid,
#                 )
#             )
#             .then(pl.lit(col))
#             .otherwise(None)
#             .alias(tmp)
#         )
# 
#     if not tmp_cols:
#         # Nothing to validate; return empty result with the same schema + field
#         return df.head(0).with_columns(pl.lit(pl.Series([], dtype=pl.List(pl.Utf8))).alias(bad_cols_field))
# 
#     result = (
#         with_flags
#         .with_columns(pl.concat_list(tmp_cols).list.drop_nulls().alias(bad_cols_field))
#         .filter(pl.col(bad_cols_field).list.len() > 0)
#     )
#     if drop_temp_cols:
#         result = result.drop(tmp_cols)
#     return result


# def validate_and_summarize(
#     df: pl.DataFrame,
#     patterns: Dict[str, str],
#     *,
#     accept_nulls: Iterable[str] = (),
#     treat_empty_as_invalid: bool = True,
# ) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, dict]:
#     """
#     Convenience wrapper:
#       - returns (valid_df, invalid_df, per_col_counts_df, summary_dict)
#     """
#     mask = invalid_row_mask(
#         df,
#         patterns,
#         accept_nulls=accept_nulls,
#         treat_empty_as_invalid=treat_empty_as_invalid,
#     )
#     invalid_df = df.filter(mask)
#     valid_df = df.filter(~mask)
# 
#     per_col = per_column_invalid_counts(
#         df,
#         patterns,
#         accept_nulls=accept_nulls,
#         treat_empty_as_invalid=treat_empty_as_invalid,
#     )
# 
#     total_rows = df.height
#     invalid_rows = invalid_df.height
#     summary = {
#         "total_rows": total_rows,
#         "invalid_rows": invalid_rows,
#         "valid_rows": total_rows - invalid_rows,
#         "invalid_pct": (invalid_rows / total_rows * 100.0) if total_rows else 0.0,
#     }
#     return valid_df, invalid_df, per_col, summary



def invalid_values_by_column(
    df: pl.DataFrame,
    patterns: Dict[str, str],
    *,
    accept_nulls: Iterable[str] = (),
    treat_empty_as_invalid: bool = True,
    include_empty: bool = False,
    limit_per_column: Optional[int] = None,
    drop_placeholders: Optional[Iterable[object]] = None,
    as_json: bool = False,
) -> Union[dict, str]:
    """
    Build a mapping of column -> unique invalid values (based on regex patterns).

    Returns a Python dict by default, or a JSON string if `as_json=True`.

    Args:
        df: Polars DataFrame.
        patterns: dict of {column: anchored regex}.
        accept_nulls: columns where NULL should be considered valid.
        treat_empty_as_invalid: if True, "" counts as invalid.
        include_empty: if False, columns with no invalids are omitted.
        limit_per_column: cap the number of values per column (useful for huge sets).
        drop_placeholders: optional iterable of values to remove from results
                           (e.g., ['#NO VALUE']) — comparison is done on raw values.
        as_json: return a JSON string instead of a dict.

    Notes:
        - Values are returned in their original dtype (e.g., None for NULL, date/time objects).
        - When `as_json=True`, non-JSON-serializable values (e.g., date/time) are stringified.
    """
    accept_nulls = set(accept_nulls)
    drop_set = set(drop_placeholders) if drop_placeholders is not None else None

    out: Dict[str, List[object]] = {}

    for col, pat in patterns.items():
        if col not in df.columns:
            continue

        invalid_expr = _col_invalid_expr(
            col,
            pat,
            accept_null=(col in accept_nulls),
            treat_empty_as_invalid=treat_empty_as_invalid,
        )

        # Collect unique invalid values from the ORIGINAL column (preserve dtype)
        if df.height == 0:
            uniques: List[object] = []
        else:
            uniques = (
                df.filter(invalid_expr)
                  .get_column(col)
                  .unique()
                  .to_list()
            )

        # Optionally drop placeholders (e.g., "#NO VALUE")
        if drop_set is not None and uniques:
            uniques = [v for v in uniques if v not in drop_set]

        # Optionally limit the number of returned values
        if limit_per_column is not None and len(uniques) > limit_per_column:
            uniques = uniques[:limit_per_column]

        if include_empty or uniques:
            out[col] = uniques

    if as_json:
        # Stringify non-serializable values like date/time
        return json.dumps(out, ensure_ascii=False, default=str)
    return out


def discover_placeholder_values(
    bad_maps: Iterable[Union[str, Dict[str, List[object]]]],
    *,
    only_strings: bool = True,
    min_files: int = 1,
    min_columns: int = 1,
    seeds: Optional[Iterable[str]] = None,
    extra_regexes: Optional[Iterable[str]] = None,
    normalize: Literal["none", "lower_trim"] = "lower_trim",
    return_variants: bool = True,
) -> Tuple[Set[str], Dict[str, dict]]:
    """
    Aggregate invalid-value JSONs/dicts from multiple files and return a set of
    placeholder-like tokens.

    Args:
        ...
        normalize: 
            - "none": keep exact tokens (case/spacing preserved)
            - "lower_trim": lowercase + trim + collapse spaces for grouping
        return_variants:
            If True, the stats dict includes 'samples' of original forms
            so you can recover exact tokens even when normalizing.

    Returns:
        (placeholders, stats)
          - placeholders: set of tokens; exact if normalize="none",
                          normalized if normalize="lower_trim".
          - stats: token -> {files, columns, count, samples}
    """
    import json, re
    from collections import defaultdict

    def _normalize_token(s: str) -> str:
        if normalize == "none":
            return s
        # lower_trim
        s = s.strip()
        s = re.sub(r"\s+", " ", s)
        return s.lower()

    rx_list = [re.compile(r) for r in (extra_regexes or [])]
    seed_set = set(_normalize_token(s) for s in (seeds or []))

    stats: Dict[str, dict] = defaultdict(lambda: {
        "files": set(),
        "columns": set(),
        "count": 0,
        "samples": set(),  # original forms
    })

    for file_idx, m in enumerate(bad_maps):
        if isinstance(m, str):
            m = json.loads(m)
        if not isinstance(m, dict):
            continue

        for col, values in m.items():
            if not isinstance(values, list):
                continue
            for v in values:
                if only_strings and not isinstance(v, str):
                    continue
                s = v if isinstance(v, str) else str(v)
                key = _normalize_token(s)

                stats[key]["files"].add(file_idx)
                stats[key]["columns"].add(col)
                stats[key]["count"] += 1
                if return_variants and len(stats[key]["samples"]) < 10:
                    stats[key]["samples"].add(s)

    def looks_like_placeholder(token: str) -> bool:
        if token in seed_set:
            return True
        if any(rx.match(token) for rx in rx_list):
            return True
        # simple heuristics
        if token in {"", "na", "n/a", "null", "none", "nan", "inf", "infinity",
                     "unk", "unknown", "not available", "not applicable"}:
            return True
        if re.fullmatch(r"^[#\.\-_/\\]+$", token):
            return True
        if re.fullmatch(r"^0{2,}$", token):
            return True
        return False

    placeholders: Set[str] = set()
    for tok, st in stats.items():
        if len(st["files"]) >= min_files and len(st["columns"]) >= min_columns:
            if looks_like_placeholder(tok):
                placeholders.add(tok)

    return placeholders, stats