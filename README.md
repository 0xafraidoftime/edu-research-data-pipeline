# edu-research-data-pipeline

> A complete **ETL + data quality + auto-codebook pipeline** purpose-built for education science research datasets — handling SPSS missing codes, Likert normalization, longitudinal reshaping, de-identification, FAIR-compliant codebook generation, and full data quality reporting.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![pandas](https://img.shields.io/badge/pandas-2.0%2B-green)](https://pandas.pydata.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![FAIR Data](https://img.shields.io/badge/FAIR-Data%20Principles-blue)](https://www.go-fair.org/fair-principles/)

---

## Motivation

Education science researchers routinely work with datasets that are:
- Full of **SPSS-style missing codes** (-9, -8, 999) that look like valid values
- Named with **inconsistent conventions** across data collection waves
- **Wide-format longitudinal** data that needs reshaping for analysis
- Deposited on OSF or ICPSR **without proper codebooks** — harming reproducibility

This pipeline directly addresses the data management challenges that Dr. Jessica Logan (Vanderbilt University / Peabody College) has identified as central to improving **research transparency and statistical conclusion validity** in the learning and developmental sciences.

---

## What This Pipeline Does

```
Messy Raw CSV (SPSS codes, bad names, out-of-range values)
        │
        ▼
 EXTRACT ──────────────────────────────────────────────────
  └── Load CSV/TSV with auto-detection
        │
        ▼
 TRANSFORM ────────────────────────────────────────────────
  ├── Replace SPSS missing codes (-9, 999, etc.) with NaN
  ├── Rename variables to clean snake_case
  ├── Normalize Likert scales (e.g., 1-5 → 0-100)
  ├── Compute composite scores
  ├── De-identify (drop or hash ID columns; FERPA-aligned)
  └── Reshape wide → long (longitudinal data)
        │
        ▼
 QUALITY CHECK ────────────────────────────────────────────
  ├── Missingness report (per-variable + heatmap)
  ├── Out-of-range value detection (schema-validated)
  ├── Longitudinal monotonicity check
  └── Duplicate row / duplicate ID detection
        │
        ▼
 CODEBOOK GENERATION ──────────────────────────────────────
  ├── CSV codebook (machine-readable, for analysis)
  └── Markdown codebook (OSF/ICPSR/GitHub-ready)
        │
        ▼
 LOAD ─────────────────────────────────────────────────────
  ├── ecls_clean_wide.csv  (analysis-ready)
  ├── ecls_clean_long.csv  (longitudinal format)
  └── CODEBOOK.md          (deposit-ready documentation)
```

---

## Project Structure

```
edu-research-data-pipeline/
│
├── data/
│   ├── raw/
│   │   └── ecls_raw.csv              # Original messy input
│   ├── processed/
│   │   ├── ecls_clean_wide.csv       # Cleaned, wide format
│   │   └── ecls_clean_long.csv       # Cleaned, long (longitudinal) format
│   └── codebooks/
│       ├── codebook.csv              # Machine-readable codebook
│       └── CODEBOOK.md               # OSF/GitHub-ready Markdown codebook
│
├── src/
│   ├── etl_pipeline.py               # ETL functions
│   ├── data_quality.py               # Quality checks + visualisation
│   ├── codebook_generator.py         # Auto-codebook generation
│   └── run_pipeline.py               # End-to-end runner
│
├── outputs/
│   ├── missing_heatmap.png           # Missing data visualisation
│   ├── qc_missingness.csv            # Per-variable missingness report
│   └── qc_out_of_range.csv           # Out-of-range violations
│
├── tests/
│   └── test_pipeline.py
│
├── requirements.txt
└── README.md
```

---

## Setup & Usage

```bash
git clone https://github.com/0xafraidoftime/edu-research-data-pipeline.git
cd edu-research-data-pipeline
pip install -r requirements.txt

cd src
python run_pipeline.py
```

---

## Key Modules

### `etl_pipeline.py`

| Function | What it does |
|---|---|
| `load_raw()` | Load CSV/TSV with auto-detection |
| `replace_spss_missing()` | Replace -9, -8, 999, etc. with NaN |
| `rename_variables()` | Rename via a mapping dictionary |
| `normalize_likert()` | Linearly rescale Likert columns |
| `wide_to_long()` | Reshape longitudinal data |
| `deidentify()` | Drop or SHA256-hash ID columns |
| `save_clean()` | Save to CSV or Parquet |

### `data_quality.py`

| Function | What it checks |
|---|---|
| `missing_summary()` | Per-column missingness with severity flags |
| `plot_missing_heatmap()` | Visual missingness pattern |
| `check_out_of_range()` | Schema-validated range checking |
| `check_longitudinal_monotonicity()` | Implausible score decreases across waves |
| `check_duplicates()` | Duplicate rows and IDs |
| `run_quality_report()` | Full report, all checks combined |

### `codebook_generator.py`

| Function | Output |
|---|---|
| `build_codebook()` | Structured DataFrame with per-variable stats |
| `export_csv_codebook()` | Machine-readable CSV codebook |
| `export_markdown_codebook()` | OSF/ICPSR/GitHub-ready Markdown |

---

## Codebook Format (auto-generated)

Each variable entry includes:

- Variable name, label, type (continuous / ordinal / categorical / binary / ID)
- Valid N and missing count/percentage
- For numeric: mean, SD, median, min, max
- Value labels (e.g., 1=Male, 2=Female)
- Notes on derivation or instrument
- FAIR compliance checklist

---

## De-identification (FERPA Alignment)

The pipeline supports two de-identification strategies:
- **Drop**: Remove columns entirely (e.g., names, raw school IDs)
- **Hash**: Replace IDs with 12-character SHA256 hashes — preserving linkage capability within a study while preventing re-identification

---

## References

- Logan, J. A. R. (research on data management practices in learning sciences, Vanderbilt)
- Wilkinson, M. D., et al. (2016). The FAIR Guiding Principles for scientific data management and stewardship. *Scientific Data, 3*, 160018.
- ICPSR Data Management & Curation resources: https://www.icpsr.umich.edu/
- OSF (Open Science Framework): https://osf.io/

---

## Author

**0xafraidoftime** — [GitHub](https://github.com/0xafraidoftime)

Built in alignment with Dr. Jessica Logan's work on improving research data management and sharing practices in the educational and developmental sciences at Vanderbilt University's Peabody College.

---

## License

MIT License — see [LICENSE](LICENSE) for details.
