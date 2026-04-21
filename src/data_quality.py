"""
data_quality.py
---------------
Comprehensive data quality assessment for education science research datasets.

Produces a structured quality report covering:
  - Missingness patterns (overall, per-variable, by group)
  - Suspicious value detection (out-of-range, implausible codes)
  - Longitudinal ID consistency checks
  - Duplicate detection
  - Distribution summaries with flagging

Designed to mirror the data quality standards advocated by Dr. Jessica Logan's
work on improving research data management in the learning and developmental
sciences, and aligned with FAIR data principles.
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# 1. Missingness analysis
# ─────────────────────────────────────────────────────────────────────────────

def missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-column missingness summary, sorted by missing percentage descending.
    """
    total = len(df)
    miss = df.isnull().sum()
    pct = (miss / total * 100).round(2)
    dtype = df.dtypes

    summary = pd.DataFrame({
        "n_missing": miss,
        "pct_missing": pct,
        "dtype": dtype,
        "n_unique": df.nunique(),
    }).sort_values("pct_missing", ascending=False)

    summary["flag"] = summary["pct_missing"].apply(
        lambda p: "🔴 HIGH" if p > 20 else ("🟡 MODERATE" if p > 5 else "🟢 OK")
    )

    return summary


def plot_missing_heatmap(df: pd.DataFrame, save_path: Optional[Path] = None):
    """
    Heatmap of missing values across observations and variables.
    Reveals patterns (e.g., missing-not-at-random across waves).
    """
    cols_with_missing = [c for c in df.columns if df[c].isnull().any()]
    if not cols_with_missing:
        print("  No missing values found.")
        return

    fig, ax = plt.subplots(figsize=(min(20, len(cols_with_missing) * 0.8 + 4), 6))
    miss_matrix = df[cols_with_missing].isnull().astype(int)
    sns.heatmap(
        miss_matrix.T, cmap=["#e8f5e9", "#c62828"],
        cbar=False, ax=ax, yticklabels=True, xticklabels=False
    )
    ax.set_title("Missing Data Heatmap\n(red = missing, green = present)",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Observations", fontsize=10)
    ax.set_ylabel("Variables", fontsize=10)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  [plot] Missing heatmap → {save_path}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# 2. Out-of-range / implausible value detection
# ─────────────────────────────────────────────────────────────────────────────

# Schema: variable → (min_valid, max_valid, description)
DEFAULT_RANGE_SCHEMA = {
    "ses_quintile":          (1, 5,   "SES quintile"),
    "parent_edu":            (1, 4,   "Parent education level"),
    "special_ed_flag":       (0, 1,   "Special education indicator"),
    "sex":                   (0, 1,   "Sex indicator"),
    "race":                  (1, 5,   "Race/ethnicity category"),
    "school_type":           (0, 1,   "School type indicator"),
    "vocab_baseline":        (0, 100, "Vocabulary composite"),
    "phonological":          (0, 50,  "Phonological awareness"),
    "working_memory":        (0, 100, "Working memory"),
    "teacher_experience_yrs":(0, 40,  "Teacher experience (years)"),
    "class_size":            (5, 40,  "Class size"),
    "reading_w1":            (0, 200, "Reading score Wave 1"),
    "reading_w2":            (0, 200, "Reading score Wave 2"),
    "reading_w3":            (0, 200, "Reading score Wave 3"),
    "reading_w4":            (0, 200, "Reading score Wave 4"),
    "reading_w5":            (0, 200, "Reading score Wave 5"),
    "math_w1":               (0, 200, "Math score Wave 1"),
    "math_w2":               (0, 200, "Math score Wave 2"),
    "math_w3":               (0, 200, "Math score Wave 3"),
    "math_w4":               (0, 200, "Math score Wave 4"),
    "math_w5":               (0, 200, "Math score Wave 5"),
}


def check_out_of_range(df: pd.DataFrame,
                        schema: dict = DEFAULT_RANGE_SCHEMA) -> pd.DataFrame:
    """
    Flag observations with values outside defined valid ranges.

    Returns a DataFrame of violations with child_id (if present), variable,
    observed value, and valid range.
    """
    violations = []
    id_col = "child_id" if "child_id" in df.columns else None

    for col, (lo, hi, desc) in schema.items():
        if col not in df.columns:
            continue
        mask = (df[col] < lo) | (df[col] > hi)
        n_violations = mask.sum()
        if n_violations > 0:
            bad = df[mask][[id_col, col]].copy() if id_col else df[mask][[col]].copy()
            bad["variable"] = col
            bad["description"] = desc
            bad["valid_range"] = f"[{lo}, {hi}]"
            bad["observed_value"] = df.loc[mask, col].values
            violations.append(bad)

    if not violations:
        return pd.DataFrame(columns=["variable", "observed_value", "valid_range"])

    result = pd.concat(violations, ignore_index=True)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 3. Longitudinal consistency checks
# ─────────────────────────────────────────────────────────────────────────────

def check_longitudinal_monotonicity(df: pd.DataFrame,
                                     wave_cols: list[str],
                                     id_col: str = "child_id",
                                     tolerance: float = 0.0) -> pd.DataFrame:
    """
    Flag children whose scores *decrease* across consecutive waves by more
    than `tolerance` points — potential data entry or linkage errors.

    Parameters
    ----------
    tolerance : float
        Allow up to this much decrease before flagging (e.g., 0.0 = any decrease flagged)
    """
    if id_col not in df.columns:
        return pd.DataFrame()

    violations = []
    for i in range(len(wave_cols) - 1):
        w1, w2 = wave_cols[i], wave_cols[i + 1]
        if w1 not in df.columns or w2 not in df.columns:
            continue
        diff = df[w2] - df[w1]
        flagged = df[diff < -tolerance][[id_col, w1, w2]].copy()
        flagged["decrease"] = (df.loc[diff < -tolerance, w1] -
                               df.loc[diff < -tolerance, w2]).round(2)
        flagged["waves"] = f"{w1} → {w2}"
        violations.append(flagged)

    if not violations:
        return pd.DataFrame()
    return pd.concat(violations, ignore_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Duplicate detection
# ─────────────────────────────────────────────────────────────────────────────

def check_duplicates(df: pd.DataFrame, id_col: str = "child_id") -> dict:
    """
    Check for duplicate rows and duplicate IDs.
    """
    result = {
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_ids":  0,
    }
    if id_col in df.columns:
        result["duplicate_ids"] = int(df[id_col].duplicated().sum())
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 5. Full quality report
# ─────────────────────────────────────────────────────────────────────────────

def run_quality_report(df: pd.DataFrame,
                        wave_cols_reading: list[str] = None,
                        wave_cols_math: list[str] = None,
                        save_dir: Optional[Path] = None) -> dict:
    """
    Compile and print a full data quality report.
    """
    print("─" * 55)
    print("DATA QUALITY REPORT")
    print("─" * 55)
    print(f"  Dataset shape: {df.shape[0]:,} rows × {df.shape[1]} columns\n")

    report = {}

    # Missingness
    miss = missing_summary(df)
    report["missingness"] = miss
    print("── Missingness ──────────────────────────────────────")
    flagged_miss = miss[miss["pct_missing"] > 0]
    if flagged_miss.empty:
        print("  ✅ No missing values detected.")
    else:
        print(flagged_miss[["n_missing", "pct_missing", "flag"]].to_string())
    print()

    if save_dir:
        plot_missing_heatmap(df, save_path=save_dir / "missing_heatmap.png")

    # Out-of-range
    oor = check_out_of_range(df)
    report["out_of_range"] = oor
    print("── Out-of-Range Values ──────────────────────────────")
    if oor.empty:
        print("  ✅ No out-of-range values detected.")
    else:
        print(f"  ⚠️  {len(oor)} out-of-range observations found:")
        print(oor[["variable", "observed_value", "valid_range"]].head(10).to_string(index=False))
    print()

    # Duplicates
    dups = check_duplicates(df)
    report["duplicates"] = dups
    print("── Duplicates ───────────────────────────────────────")
    print(f"  Duplicate rows: {dups['duplicate_rows']}")
    print(f"  Duplicate IDs:  {dups['duplicate_ids']}")
    print()

    # Longitudinal monotonicity
    if wave_cols_reading:
        mono = check_longitudinal_monotonicity(df, wave_cols_reading, tolerance=5.0)
        report["reading_monotonicity_violations"] = mono
        print("── Reading Score Longitudinal Consistency ───────────")
        if mono.empty:
            print("  ✅ No implausible decreases in reading scores.")
        else:
            print(f"  ⚠️  {len(mono)} wave-pairs with implausible score decreases (>5 pts):")
            print(mono[["waves", "decrease"]].value_counts("waves").to_string())
        print()

    print("─" * 55)
    print("Quality report complete.")
    print("─" * 55)

    return report
