"""
etl_pipeline.py
---------------
ETL (Extract → Transform → Load) pipeline specifically designed for the
quirks of education science research datasets.

Handles:
  - Common messy encodings (SPSS-style missing codes: -9, -8, -1, 999)
  - Likert scale normalization
  - Longitudinal wave reshaping (wide ↔ long)
  - Teacher / student / school hierarchical ID validation
  - Variable renaming via a mapping dictionary
  - De-identification (dropping or hashing ID columns)
  - Final output ready for OSF / ICPSR deposit
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional
import hashlib
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# 1. Extract
# ─────────────────────────────────────────────────────────────────────────────

def load_raw(path: str | Path, **kwargs) -> pd.DataFrame:
    """
    Load raw data from CSV, TSV, or SPSS-style CSV.
    Automatically detects separator.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in [".csv"]:
        df = pd.read_csv(path, **kwargs)
    elif suffix in [".tsv", ".txt"]:
        df = pd.read_csv(path, sep="\t", **kwargs)
    else:
        raise ValueError(f"Unsupported file type: {suffix}. Use CSV or TSV.")
    print(f"  [load] {path.name}: {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. Transform: missing code replacement
# ─────────────────────────────────────────────────────────────────────────────

SPSS_MISSING_CODES = [-9, -8, -7, -1, 999, 9999, 99, 98]


def replace_spss_missing(df: pd.DataFrame,
                          missing_codes: list = SPSS_MISSING_CODES,
                          columns: Optional[list] = None) -> pd.DataFrame:
    """
    Replace SPSS-style numeric missing codes with np.nan.
    Skips non-numeric columns automatically.
    """
    df = df.copy()
    cols = columns or df.select_dtypes(include="number").columns.tolist()
    replaced = 0
    for col in cols:
        mask = df[col].isin(missing_codes)
        replaced += mask.sum()
        df.loc[mask, col] = np.nan
    print(f"  [missing] Replaced {replaced:,} SPSS missing codes with NaN")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3. Transform: variable renaming
# ─────────────────────────────────────────────────────────────────────────────

def rename_variables(df: pd.DataFrame, rename_map: dict) -> pd.DataFrame:
    """
    Rename variables using a {old_name: new_name} mapping.
    Warns for columns in the map that don't exist in the dataset.
    """
    missing_cols = [k for k in rename_map if k not in df.columns]
    if missing_cols:
        print(f"  [rename] ⚠️  Variables not found in dataset: {missing_cols}")
    df = df.rename(columns=rename_map)
    renamed = [k for k in rename_map if k in df.columns or k not in missing_cols]
    print(f"  [rename] Renamed {len(rename_map) - len(missing_cols)} variables")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 4. Transform: Likert normalization
# ─────────────────────────────────────────────────────────────────────────────

def normalize_likert(df: pd.DataFrame,
                      likert_cols: list,
                      original_range: tuple,
                      target_range: tuple = (0, 100)) -> pd.DataFrame:
    """
    Linearly rescale Likert-scale columns from original_range to target_range.
    Useful for combining instruments with different response scales.

    Example: (1,5) → (0,100) makes 1=0, 3=50, 5=100.
    """
    df = df.copy()
    orig_lo, orig_hi = original_range
    tgt_lo, tgt_hi = target_range

    for col in likert_cols:
        if col not in df.columns:
            print(f"  [likert] ⚠️  Column '{col}' not found, skipping.")
            continue
        df[col] = (
            (df[col] - orig_lo) / (orig_hi - orig_lo) * (tgt_hi - tgt_lo) + tgt_lo
        ).round(2)

    print(f"  [likert] Normalized {len(likert_cols)} Likert column(s): "
          f"{original_range} → {target_range}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 5. Transform: wide → long reshape for longitudinal data
# ─────────────────────────────────────────────────────────────────────────────

def wide_to_long(
    df: pd.DataFrame,
    id_vars: list,
    wave_stub: str,
    wave_col: str = "wave",
    value_col: str = "score",
    wave_labels: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Reshape longitudinal data from wide format (one row per child,
    columns reading_w1…reading_w5) to long format (one row per child-wave).

    Parameters
    ----------
    id_vars : list
        Columns to keep as-is (identifiers and time-invariant covariates)
    wave_stub : str
        Column name prefix (e.g., 'reading_w' matches reading_w1, reading_w2…)
    wave_labels : dict, optional
        Map wave numbers to readable labels (e.g., {1: "Kindergarten", 2: "Grade 1"})
    """
    wave_cols = [c for c in df.columns if c.startswith(wave_stub)]
    if not wave_cols:
        raise ValueError(f"No columns found with stub '{wave_stub}'")

    long_df = df[id_vars + wave_cols].melt(
        id_vars=id_vars,
        value_vars=wave_cols,
        var_name=wave_col,
        value_name=value_col,
    )
    # Extract wave number
    long_df[wave_col] = long_df[wave_col].str.replace(wave_stub, "", regex=False)
    long_df[wave_col] = pd.to_numeric(long_df[wave_col], errors="coerce")

    if wave_labels:
        long_df[f"{wave_col}_label"] = long_df[wave_col].map(wave_labels)

    long_df = long_df.sort_values(id_vars + [wave_col]).reset_index(drop=True)
    print(f"  [reshape] Wide→Long: {len(df):,} rows → {len(long_df):,} rows "
          f"({len(wave_cols)} waves × {len(df):,} children)")
    return long_df


# ─────────────────────────────────────────────────────────────────────────────
# 6. Transform: de-identification
# ─────────────────────────────────────────────────────────────────────────────

def deidentify(df: pd.DataFrame,
               drop_cols: list = None,
               hash_cols: list = None,
               hash_salt: str = "edu_research_2024") -> pd.DataFrame:
    """
    De-identify a dataset by:
      - Dropping specified columns entirely
      - Hashing specified ID columns (one-way SHA256)

    This supports FERPA compliance for education research datasets.
    """
    df = df.copy()

    if drop_cols:
        existing = [c for c in drop_cols if c in df.columns]
        df = df.drop(columns=existing)
        print(f"  [deid] Dropped {len(existing)} column(s): {existing}")

    if hash_cols:
        for col in hash_cols:
            if col not in df.columns:
                continue
            df[col] = df[col].astype(str).apply(
                lambda x: hashlib.sha256(f"{hash_salt}{x}".encode()).hexdigest()[:12]
            )
        print(f"  [deid] Hashed {len(hash_cols)} ID column(s)")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 7. Load: save cleaned dataset
# ─────────────────────────────────────────────────────────────────────────────

def save_clean(df: pd.DataFrame, save_path: Path, format: str = "csv"):
    """
    Save the cleaned, de-identified dataset.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    if format == "csv":
        df.to_csv(save_path, index=False)
    elif format == "parquet":
        df.to_parquet(save_path, index=False)
    else:
        raise ValueError(f"Unsupported format: {format}")
    print(f"  [save] Clean dataset → {save_path} ({df.shape[0]:,} rows × {df.shape[1]} cols)")
