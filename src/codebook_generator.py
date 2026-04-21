"""
codebook_generator.py
---------------------
Automatically generates a structured, APA-style data codebook from a messy
CSV research dataset — ready for deposit on OSF, ICPSR, or institutional
repositories.

Output formats:
  - CSV codebook (machine-readable)
  - Markdown codebook (human-readable, renders on GitHub/OSF)

Aligned with FAIR data principles (Findable, Accessible, Interoperable,
Reusable) and the data sharing standards advocated by Dr. Jessica Logan's
research on improving data management practices in the learning sciences.

Each codebook entry includes:
  - Variable name, label, type
  - Value range / unique values
  - Missing count and percentage
  - Descriptive statistics (for numeric)
  - Value labels (for categorical / Likert)
"""

from __future__ import annotations
from pathlib import Path
from datetime import date
from typing import Optional
import json

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Variable metadata registry
# Supply this to annotate your variables properly
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_VARIABLE_METADATA = {
    "child_id": {
        "label": "Unique child identifier",
        "type": "ID",
        "notes": "De-identified. Do not use as analysis variable.",
    },
    "sex": {
        "label": "Child sex",
        "type": "categorical",
        "value_labels": {0: "Male", 1: "Female"},
    },
    "race": {
        "label": "Child race/ethnicity",
        "type": "categorical",
        "value_labels": {
            1: "White, non-Hispanic",
            2: "Black, non-Hispanic",
            3: "Hispanic",
            4: "Asian",
            5: "Other",
        },
    },
    "ses_quintile": {
        "label": "Socioeconomic status composite quintile",
        "type": "ordinal",
        "value_labels": {1: "Lowest SES", 2: "Low SES", 3: "Middle SES",
                         4: "High SES", 5: "Highest SES"},
        "notes": "Derived from income, parent education, and occupational prestige.",
    },
    "parent_edu": {
        "label": "Highest parental education level",
        "type": "ordinal",
        "value_labels": {
            1: "Less than high school",
            2: "High school diploma or GED",
            3: "Some college or associate degree",
            4: "Bachelor's degree or higher",
        },
    },
    "special_ed_flag": {
        "label": "Special education services indicator",
        "type": "binary",
        "value_labels": {0: "No IEP / no special ed services",
                         1: "Has IEP or receives special ed services"},
    },
    "vocab_baseline": {
        "label": "Vocabulary composite score at kindergarten entry",
        "type": "continuous",
        "scale": "0–100",
        "notes": "Higher = stronger vocabulary. Adapted from PPVT-style assessment.",
    },
    "phonological": {
        "label": "Phonological awareness score at kindergarten entry",
        "type": "continuous",
        "scale": "0–50",
        "notes": "Composite of rhyme, alliteration, and phoneme segmentation tasks.",
    },
    "working_memory": {
        "label": "Working memory composite at kindergarten entry",
        "type": "continuous",
        "scale": "0–100",
        "notes": "Verbal and visual-spatial working memory tasks.",
    },
    "teacher_experience_yrs": {
        "label": "Classroom teacher years of experience",
        "type": "continuous",
        "scale": "0–40 years",
    },
    "class_size": {
        "label": "Kindergarten classroom size",
        "type": "continuous",
        "scale": "Number of students",
    },
    "school_type": {
        "label": "School type",
        "type": "categorical",
        "value_labels": {0: "Public school", 1: "Private school"},
    },
    "reading_w1": {"label": "Reading IRT score — Kindergarten (Wave 1)",    "type": "continuous", "scale": "0–200"},
    "reading_w2": {"label": "Reading IRT score — Grade 1 (Wave 2)",         "type": "continuous", "scale": "0–200"},
    "reading_w3": {"label": "Reading IRT score — Grade 2 (Wave 3)",         "type": "continuous", "scale": "0–200"},
    "reading_w4": {"label": "Reading IRT score — Grade 3 (Wave 4)",         "type": "continuous", "scale": "0–200"},
    "reading_w5": {"label": "Reading IRT score — Grade 4 (Wave 5)",         "type": "continuous", "scale": "0–200"},
    "math_w1":    {"label": "Math IRT score — Kindergarten (Wave 1)",       "type": "continuous", "scale": "0–200"},
    "math_w2":    {"label": "Math IRT score — Grade 1 (Wave 2)",            "type": "continuous", "scale": "0–200"},
    "math_w3":    {"label": "Math IRT score — Grade 2 (Wave 3)",            "type": "continuous", "scale": "0–200"},
    "math_w4":    {"label": "Math IRT score — Grade 3 (Wave 4)",            "type": "continuous", "scale": "0–200"},
    "math_w5":    {"label": "Math IRT score — Grade 4 (Wave 5)",            "type": "continuous", "scale": "0–200"},
}


# ─────────────────────────────────────────────────────────────────────────────
# Core codebook builder
# ─────────────────────────────────────────────────────────────────────────────

def build_codebook(
    df: pd.DataFrame,
    metadata: dict = DEFAULT_VARIABLE_METADATA,
    dataset_name: str = "Education Research Dataset",
) -> pd.DataFrame:
    """
    Build a structured codebook DataFrame from a research dataset.

    Parameters
    ----------
    df : pd.DataFrame
        The research dataset.
    metadata : dict
        Variable-level metadata (labels, value labels, notes, etc.)
    dataset_name : str
        Name of the dataset for the codebook header.

    Returns
    -------
    pd.DataFrame with one row per variable.
    """
    rows = []
    n = len(df)

    for i, col in enumerate(df.columns, start=1):
        meta = metadata.get(col, {})
        series = df[col]

        n_missing = int(series.isnull().sum())
        pct_missing = round(n_missing / n * 100, 2)
        n_valid = n - n_missing

        row = {
            "variable_number": i,
            "variable_name": col,
            "label": meta.get("label", ""),
            "type": meta.get("type", _infer_type(series)),
            "scale": meta.get("scale", ""),
            "n_valid": n_valid,
            "n_missing": n_missing,
            "pct_missing": pct_missing,
            "notes": meta.get("notes", ""),
        }

        # Descriptive stats for numeric
        if pd.api.types.is_numeric_dtype(series) and meta.get("type") not in ["categorical", "binary", "ID"]:
            row["min"] = round(series.min(), 3) if n_valid > 0 else ""
            row["max"] = round(series.max(), 3) if n_valid > 0 else ""
            row["mean"] = round(series.mean(), 3) if n_valid > 0 else ""
            row["sd"] = round(series.std(), 3) if n_valid > 0 else ""
            row["median"] = round(series.median(), 3) if n_valid > 0 else ""
        else:
            row["min"] = row["max"] = row["mean"] = row["sd"] = row["median"] = ""

        # Value labels / unique values
        val_labels = meta.get("value_labels", {})
        if val_labels:
            row["value_labels"] = "; ".join(
                [f"{k}={v}" for k, v in sorted(val_labels.items())]
            )
        elif series.nunique() <= 10:
            row["value_labels"] = str(sorted(series.dropna().unique().tolist()))
        else:
            row["value_labels"] = f"{series.nunique()} unique values"

        rows.append(row)

    return pd.DataFrame(rows)


def _infer_type(series: pd.Series) -> str:
    if pd.api.types.is_float_dtype(series):
        return "continuous"
    if pd.api.types.is_integer_dtype(series):
        n_unique = series.nunique()
        return "binary" if n_unique <= 2 else "categorical" if n_unique <= 10 else "continuous"
    return "string"


# ─────────────────────────────────────────────────────────────────────────────
# Export: CSV codebook
# ─────────────────────────────────────────────────────────────────────────────

def export_csv_codebook(codebook_df: pd.DataFrame, save_path: Path):
    codebook_df.to_csv(save_path, index=False)
    print(f"  [codebook] CSV saved → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Export: Markdown codebook (OSF/GitHub ready)
# ─────────────────────────────────────────────────────────────────────────────

def export_markdown_codebook(
    codebook_df: pd.DataFrame,
    dataset_name: str,
    dataset_description: str,
    n_observations: int,
    save_path: Path,
    author: str = "0xafraidoftime",
    repository_url: str = "https://github.com/0xafraidoftime/edu-research-data-pipeline",
):
    """
    Generate a publication-ready Markdown codebook following OSF/ICPSR conventions.
    """
    today = date.today().strftime("%B %d, %Y")

    lines = [
        f"# Data Codebook: {dataset_name}",
        "",
        f"**Generated:** {today}  ",
        f"**Author:** {author}  ",
        f"**Repository:** {repository_url}  ",
        f"**N Observations:** {n_observations:,}  ",
        f"**N Variables:** {len(codebook_df)}  ",
        "",
        "---",
        "",
        "## Dataset Description",
        "",
        dataset_description,
        "",
        "---",
        "",
        "## Variable Index",
        "",
    ]

    # Index table
    lines.append("| # | Variable | Label | Type |")
    lines.append("|---|---|---|---|")
    for _, row in codebook_df.iterrows():
        lines.append(
            f"| {row['variable_number']} | `{row['variable_name']}` | "
            f"{row['label']} | {row['type']} |"
        )
    lines += ["", "---", "", "## Variable Details", ""]

    # Detailed entries
    for _, row in codebook_df.iterrows():
        lines.append(f"### {row['variable_number']}. `{row['variable_name']}`")
        lines.append("")
        if row["label"]:
            lines.append(f"**Label:** {row['label']}  ")
        lines.append(f"**Type:** {row['type']}  ")
        if row["scale"]:
            lines.append(f"**Scale:** {row['scale']}  ")
        lines.append(f"**Valid N:** {row['n_valid']:,} | "
                     f"**Missing:** {row['n_missing']} ({row['pct_missing']}%)  ")
        if row["mean"] != "":
            lines.append(
                f"**Mean:** {row['mean']} | **SD:** {row['sd']} | "
                f"**Median:** {row['median']} | "
                f"**Range:** [{row['min']}, {row['max']}]  "
            )
        if row["value_labels"]:
            lines.append(f"**Values:** {row['value_labels']}  ")
        if row["notes"]:
            lines.append(f"**Notes:** {row['notes']}  ")
        lines.append("")

    lines += [
        "---",
        "",
        "## Licensing & Citation",
        "",
        "This dataset is synthetic and freely available for research and educational use.",
        "Please cite this repository if you use it in your work.",
        "",
        f"> {author} ({date.today().year}). *{dataset_name}*. {repository_url}",
        "",
        "---",
        "",
        "## FAIR Data Compliance",
        "",
        "| Principle | Status |",
        "|---|---|",
        "| **F**indable — persistent identifier | ✅ GitHub/OSF URL |",
        "| **A**ccessible — open access | ✅ MIT License |",
        "| **I**nteroperable — standard formats | ✅ CSV + Markdown |",
        "| **R**eusable — rich metadata | ✅ This codebook |",
    ]

    with open(save_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"  [codebook] Markdown saved → {save_path}")
