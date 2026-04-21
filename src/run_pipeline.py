"""
run_pipeline.py
---------------
End-to-end runner for the education research data pipeline.

Demonstrates a full real-world workflow:
  1. Generate a "messy" synthetic ECLS-K style dataset (with SPSS missing codes,
     inconsistent naming, a few out-of-range values injected)
  2. Run ETL: replace missing codes, rename, normalize Likert scales, de-identify
  3. Run data quality checks and generate quality report
  4. Reshape to long format for longitudinal analysis
  5. Auto-generate CSV + Markdown codebooks
  6. Save all outputs (clean data, codebooks, QC report)

Usage:
    cd src
    python run_pipeline.py
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from etl_pipeline import (
    load_raw, replace_spss_missing, rename_variables,
    normalize_likert, wide_to_long, deidentify, save_clean,
)
from data_quality import run_quality_report
from codebook_generator import (
    build_codebook, export_csv_codebook, export_markdown_codebook,
    DEFAULT_VARIABLE_METADATA,
)

OUT   = Path(__file__).parent.parent / "outputs"
DATA  = Path(__file__).parent.parent / "data"
BOOKS = Path(__file__).parent.parent / "data" / "codebooks"

for p in [OUT, DATA, BOOKS]:
    p.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic "messy" data generator
# ─────────────────────────────────────────────────────────────────────────────

def generate_messy_dataset(n: int = 500) -> pd.DataFrame:
    """
    Generate a realistic messy education dataset with:
      - SPSS missing codes (-9, 999)
      - Inconsistent column naming (e.g., READ_W1 instead of reading_w1)
      - A handful of out-of-range injected values
      - Likert-scale teacher survey on 1-5 scale
    """
    rng = np.random.default_rng(42)

    ses  = rng.choice([1,2,3,4,5], n)
    pedu = rng.choice([1,2,3,4], n, p=[0.15,0.30,0.35,0.20])
    sped = rng.choice([0,1], n, p=[0.87,0.13])
    sex  = rng.choice([0,1], n)
    race = rng.choice([1,2,3,4,5], n, p=[0.52,0.15,0.23,0.04,0.06])

    vocab = np.clip(rng.normal(50,15,n) - (3-ses)*2.5, 5, 100).round(2)
    phono = np.clip(rng.normal(25,8,n) + (ses-3)*1.5, 0, 50).round(2)
    wmem  = np.clip(rng.normal(50,12,n), 10, 100).round(2)

    def reading(boost):
        base = 40+boost+0.25*vocab+0.40*phono+0.10*wmem+(ses-3)*3.5+(pedu-2)*2-sped*8+sex*2
        return np.clip(rng.normal(base, 8), 0, 200).round(1)

    df = pd.DataFrame({
        "CHILD_ID":     np.arange(1, n+1),
        "SEX":          sex,
        "RACE":         race,
        "SES_QUINTILE": ses,
        "PARENT_EDU":   pedu,
        "SPED_FLAG":    sped,
        "VOCAB":        vocab,
        "PHONO":        phono,
        "WMEM":         wmem,
        "TCH_EXP":      np.clip(rng.poisson(8, n), 0, 35),
        "CLASS_SIZE":   rng.integers(12, 30, n),
        "SCHOOL_TYPE":  rng.choice([0,1], n, p=[0.75,0.25]),
        # Likert teacher survey (1-5 scale; will be normalized)
        "TCH_SUPPORT_1": rng.choice([1,2,3,4,5], n),
        "TCH_SUPPORT_2": rng.choice([1,2,3,4,5], n),
        "TCH_SUPPORT_3": rng.choice([1,2,3,4,5], n),
        # Reading scores — using SPSS-style naming
        "READ_W1": reading(0),
        "READ_W2": reading(15),
        "READ_W3": reading(28),
        "READ_W4": reading(39),
        "READ_W5": reading(49),
    })

    # Inject SPSS missing codes
    for col in ["VOCAB", "PHONO", "WMEM", "READ_W1", "READ_W3"]:
        idx = rng.choice(n, size=int(n * 0.05), replace=False)
        df.loc[idx, col] = -9

    # Inject a few out-of-range values
    df.loc[rng.choice(n, 3, replace=False), "SES_QUINTILE"] = 99
    df.loc[rng.choice(n, 2, replace=False), "VOCAB"] = 150  # impossible on 0-100 scale

    print(f"  [generate] Messy dataset: {df.shape}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

RENAME_MAP = {
    "CHILD_ID":     "child_id",
    "SEX":          "sex",
    "RACE":         "race",
    "SES_QUINTILE": "ses_quintile",
    "PARENT_EDU":   "parent_edu",
    "SPED_FLAG":    "special_ed_flag",
    "VOCAB":        "vocab_baseline",
    "PHONO":        "phonological",
    "WMEM":         "working_memory",
    "TCH_EXP":      "teacher_experience_yrs",
    "CLASS_SIZE":   "class_size",
    "SCHOOL_TYPE":  "school_type",
    "READ_W1":      "reading_w1",
    "READ_W2":      "reading_w2",
    "READ_W3":      "reading_w3",
    "READ_W4":      "reading_w4",
    "READ_W5":      "reading_w5",
}

WAVE_LABELS = {
    1: "Kindergarten",
    2: "Grade 1",
    3: "Grade 2",
    4: "Grade 3",
    5: "Grade 4",
}

DATASET_DESCRIPTION = """
Synthetic dataset modeled on the Early Childhood Longitudinal Study —
Kindergarten Cohort (ECLS-K). Contains demographic, SES, cognitive, and
classroom variables for 500 children tracked from Kindergarten through
Grade 4. All data is entirely synthetic; no real child records are included.

Designed to demonstrate a full ETL + data quality + codebook pipeline
for education science research, aligned with FAIR data principles and
the data management practices advocated by researchers in the learning
and developmental sciences.
"""


def main():
    print("=" * 60)
    print("Education Research Data Pipeline")
    print("Extract → Transform → Quality Check → Codebook → Load")
    print("=" * 60)

    # 1. Generate messy raw data
    print("\n[1/6] Generating synthetic messy dataset...")
    raw_df = generate_messy_dataset(n=500)
    raw_df.to_csv(DATA / "raw" / "ecls_raw.csv", index=False)

    # 2. ETL
    print("\n[2/6] Running ETL pipeline...")

    # Replace SPSS missing codes
    df = replace_spss_missing(raw_df)

    # Rename to clean snake_case
    df = rename_variables(df, RENAME_MAP)

    # Normalize teacher support Likert (1-5 → 0-100)
    df = normalize_likert(
        df,
        likert_cols=["TCH_SUPPORT_1", "TCH_SUPPORT_2", "TCH_SUPPORT_3"],
        original_range=(1, 5),
        target_range=(0, 100),
    )

    # Compute teacher support composite
    tch_cols = ["TCH_SUPPORT_1", "TCH_SUPPORT_2", "TCH_SUPPORT_3"]
    df["tch_support_composite"] = df[tch_cols].mean(axis=1).round(2)
    df = df.drop(columns=tch_cols)

    # De-identify: hash child_id
    df = deidentify(df, hash_cols=["child_id"])

    # Save clean wide dataset
    save_clean(df, DATA / "processed" / "ecls_clean_wide.csv")

    # 3. Data quality check
    print("\n[3/6] Running data quality checks...")
    qc_report = run_quality_report(
        df,
        wave_cols_reading=["reading_w1","reading_w2","reading_w3","reading_w4","reading_w5"],
        save_dir=OUT,
    )

    # Save QC summary
    qc_report["missingness"].to_csv(OUT / "qc_missingness.csv")
    if not qc_report.get("out_of_range", pd.DataFrame()).empty:
        qc_report["out_of_range"].to_csv(OUT / "qc_out_of_range.csv", index=False)

    # 4. Reshape to long format
    print("\n[4/6] Reshaping to longitudinal long format...")
    id_vars = ["child_id", "sex", "race", "ses_quintile", "parent_edu",
               "special_ed_flag", "vocab_baseline", "phonological",
               "working_memory", "teacher_experience_yrs", "class_size",
               "school_type", "tch_support_composite"]
    id_vars_present = [c for c in id_vars if c in df.columns]

    long_df = wide_to_long(
        df,
        id_vars=id_vars_present,
        wave_stub="reading_w",
        wave_col="wave",
        value_col="reading_score",
        wave_labels=WAVE_LABELS,
    )
    save_clean(long_df, DATA / "processed" / "ecls_clean_long.csv")

    # 5. Generate codebooks
    print("\n[5/6] Generating codebooks...")
    codebook_df = build_codebook(
        df,
        metadata=DEFAULT_VARIABLE_METADATA,
        dataset_name="Synthetic ECLS-K Education Dataset",
    )

    export_csv_codebook(codebook_df, BOOKS / "codebook.csv")
    export_markdown_codebook(
        codebook_df,
        dataset_name="Synthetic ECLS-K Education Dataset",
        dataset_description=DATASET_DESCRIPTION.strip(),
        n_observations=len(df),
        save_path=BOOKS / "CODEBOOK.md",
    )

    # 6. Summary
    print(f"\n{'='*60}")
    print("✓ Pipeline complete.")
    print(f"\nOutputs:")
    print(f"  data/raw/ecls_raw.csv                  ← original messy data")
    print(f"  data/processed/ecls_clean_wide.csv      ← cleaned, wide format")
    print(f"  data/processed/ecls_clean_long.csv      ← cleaned, long format")
    print(f"  data/codebooks/codebook.csv             ← machine-readable codebook")
    print(f"  data/codebooks/CODEBOOK.md              ← OSF/GitHub-ready codebook")
    print(f"  outputs/qc_missingness.csv              ← missingness report")
    print(f"  outputs/missing_heatmap.png             ← missing data heatmap")


if __name__ == "__main__":
    main()
