"""
test_pipeline.py
----------------
Unit tests for the education research data pipeline.
"""
import sys
from pathlib import Path
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from etl_pipeline import (
    replace_spss_missing, rename_variables, normalize_likert,
    wide_to_long, deidentify, SPSS_MISSING_CODES
)
from data_quality import (
    missing_summary, check_out_of_range, check_duplicates,
    check_longitudinal_monotonicity
)
from codebook_generator import build_codebook


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "child_id": [1, 2, 3, 4, 5],
        "sex": [0, 1, 0, 1, 0],
        "ses_quintile": [1, 3, 5, -9, 2],      # -9 is SPSS missing
        "vocab_baseline": [45.0, 60.0, 999, 55.0, 70.0],  # 999 is SPSS missing
        "reading_w1": [80, 90, 85, 78, 92],
        "reading_w2": [88, 95, 91, 82, 98],
        "reading_w3": [60, 100, 95, 50, 102],   # 60 and 50 are regressions
    })


def test_replace_spss_missing(sample_df):
    df = replace_spss_missing(sample_df)
    assert df["ses_quintile"].isna().sum() == 1
    assert df["vocab_baseline"].isna().sum() == 1


def test_rename_variables(sample_df):
    df = rename_variables(sample_df, {"child_id": "id", "sex": "gender"})
    assert "id" in df.columns
    assert "gender" in df.columns
    assert "child_id" not in df.columns


def test_normalize_likert():
    df = pd.DataFrame({"q1": [1, 2, 3, 4, 5], "q2": [1, 3, 5, 2, 4]})
    result = normalize_likert(df, ["q1", "q2"], (1, 5), (0, 100))
    assert result["q1"].min() == pytest.approx(0.0)
    assert result["q1"].max() == pytest.approx(100.0)


def test_wide_to_long(sample_df):
    long_df = wide_to_long(
        sample_df, id_vars=["child_id", "sex"],
        wave_stub="reading_w", wave_col="wave", value_col="score"
    )
    assert len(long_df) == 5 * 3  # 5 children × 3 waves
    assert "score" in long_df.columns


def test_deidentify_hash(sample_df):
    df = deidentify(sample_df, hash_cols=["child_id"])
    assert df["child_id"].dtype == object
    assert len(df["child_id"].iloc[0]) == 12


def test_missing_summary(sample_df):
    df = replace_spss_missing(sample_df)
    summary = missing_summary(df)
    assert "pct_missing" in summary.columns
    assert summary.loc["ses_quintile", "pct_missing"] == pytest.approx(20.0)


def test_check_duplicates(sample_df):
    result = check_duplicates(sample_df)
    assert result["duplicate_rows"] == 0
    assert result["duplicate_ids"] == 0


def test_codebook_columns(sample_df):
    cb = build_codebook(sample_df)
    assert "variable_name" in cb.columns
    assert "pct_missing" in cb.columns
    assert len(cb) == len(sample_df.columns)
