# Staged classification of adolescent suicidal behavior

This repository contains analysis scripts and metadata tables for a cross-sectional staged classification-model development and external-validation study of adolescent suicidal behavior.

The development analysis uses KYRBS 2020-2024. External validation uses the 2023 national YRBSS. Raw survey files, derived individual-level analytic datasets, trained model objects, and manuscript documents are not redistributed in this public repository.

## Repository contents

- `analysis/`: model development, external validation, confidence intervals, and sensitivity-analysis scripts.
- `scripts/`: figure-generation script.
- `metadata/`: predictor lists, variable-structure summaries, and YRBSS harmonization mapping tables.
- `data/`: local-only data location; files in this directory are ignored by git except for `data/README.md`.
- `results/`: local-only analysis outputs; files in this directory are ignored by git except for `.gitkeep`.
- `figures/`: local-only rendered figures; files in this directory are ignored by git except for `.gitkeep`.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Expected local inputs

Place the KYRBS model-ready input at:

```text
data/kyrbs2020_2024_primary_grade_comparable_grade3_6_40predictors_input.csv
```

Place the YRBSS raw files at:

```text
data/yrbss_raw/2023XXH-SPSS.sps
data/yrbss_raw/XXH2023_YRBS_Data.dat
```

## Main execution order

Build the YRBSS harmonized model input:

```bash
python analysis/build_yrbss2023_selected_40predictor_reference.py
```

Run staged KYRBS model development:

```bash
python analysis/ml_stagewise_kyrbs.py \
  --input data/kyrbs2020_2024_primary_grade_comparable_grade3_6_40predictors_input.csv \
  --predictor-list metadata/kyrbs2020_2024_40predictor_list.csv \
  --out-dir results/ml_results_kyrbs2020_2024_primary_grade_comparable_grade3_6_40predictors_6models_80trials \
  --trials 80
```

Run frozen-threshold external validation:

```bash
python analysis/run_yrbss_external_validation_primary_grade_comparable40.py
```

Generate cluster-bootstrap confidence intervals:

```bash
python analysis/bootstrap_ci_primary_grade_comparable_internal_external.py
```

Build sensitivity-analysis tables:

```bash
python analysis/build_harmonization_sensitivity_comparison.py
python analysis/build_construct_contamination_sensitivity_comparison.py
```

Generate figures from stored outputs:

```bash
python scripts/create_ajp_style_figures.py
```

## Data-use note

KYRBS and YRBSS data are publicly accessible through their respective providers, but redistribution of raw or derived individual-level analytic files is not handled by this repository. Users should obtain the source files directly and comply with the relevant data-use conditions.
