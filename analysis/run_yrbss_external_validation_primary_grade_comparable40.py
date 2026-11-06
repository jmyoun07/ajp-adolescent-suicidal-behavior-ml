#!/usr/bin/env python3
"""External validation for the KYRBS grade 3-6 grade-comparable 40-predictor models."""

from pathlib import Path

import run_yrbss_external_validation_final40 as ev


REPO_ROOT = Path(__file__).resolve().parents[1]

ev.RESULTS_DIR = (
    REPO_ROOT
    / "results"
    / "ml_results_kyrbs2020_2024_primary_grade_comparable_grade3_6_40predictors_6models_80trials"
)
ev.MODEL_DIR = ev.RESULTS_DIR / "models"
ev.KYRBS_INPUT = REPO_ROOT / "data" / "kyrbs2020_2024_primary_grade_comparable_grade3_6_40predictors_input.csv"
ev.OUT_DIR = REPO_ROOT / "results" / "yrbss2023_external_validation_primary_grade_comparable40"
ev.FIG_DIR = ev.OUT_DIR / "figures"


if __name__ == "__main__":
    ev.main()
