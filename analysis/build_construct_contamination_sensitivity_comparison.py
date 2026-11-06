#!/usr/bin/env python3
"""Compare primary GRADE 3-6 40-predictor model vs M_SAD/M_STR-excluded sensitivity."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = REPO_ROOT / "results"
OUT_DIR = RESULTS_ROOT / "primary_grade_comparable_construct_contamination_sensitivity_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_SETS = [
    {
        "model_set": "Primary 40 predictors",
        "analysis_role": "Primary protocol model",
        "predictor_count": 40,
        "internal_dir": RESULTS_ROOT
        / "ml_results_kyrbs2020_2024_primary_grade_comparable_grade3_6_40predictors_6models_80trials",
        "external_dir": RESULTS_ROOT / "yrbss2023_external_validation_primary_grade_comparable40",
    },
    {
        "model_set": "Excluding M_SAD and M_STR",
        "analysis_role": "Construct-contamination sensitivity",
        "predictor_count": 38,
        "internal_dir": RESULTS_ROOT
        / "ml_results_kyrbs2020_2024_primary_grade_comparable_38predictors_excluding_M_SAD_M_STR_6models_80trials",
        "external_dir": RESULTS_ROOT
        / "yrbss2023_external_validation_primary_grade_comparable_38predictors_excluding_M_SAD_M_STR",
    },
]

TRANSITION_LABELS = {
    "T1_ideation_vs_none": "T1: ideation vs none",
    "T2_plan_given_ideation": "T2: plan among ideators",
    "T3_attempt_given_plan": "T3: attempt among planners",
}
TRANSITION_ORDER = list(TRANSITION_LABELS)
METRICS = ["auroc", "auprc", "sensitivity", "specificity", "ppv", "npv", "f1", "brier"]


def weighted_metrics(y: np.ndarray, proba: np.ndarray, weight: np.ndarray, threshold: float) -> dict[str, float]:
    y = y.astype(int)
    pred = (proba >= threshold).astype(int)
    tp = float(np.sum(weight[(y == 1) & (pred == 1)]))
    fp = float(np.sum(weight[(y == 0) & (pred == 1)]))
    tn = float(np.sum(weight[(y == 0) & (pred == 0)]))
    fn = float(np.sum(weight[(y == 1) & (pred == 0)]))
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else math.nan
    specificity = tn / (tn + fp) if (tn + fp) > 0 else math.nan
    ppv = tp / (tp + fp) if (tp + fp) > 0 else math.nan
    npv = tn / (tn + fn) if (tn + fn) > 0 else math.nan
    f1 = 2 * ppv * sensitivity / (ppv + sensitivity) if (ppv + sensitivity) > 0 else math.nan
    return {
        "n": int(len(y)),
        "events": int(np.sum(y == 1)),
        "weighted_prevalence": float(np.average(y, weights=weight)),
        "auroc": float(roc_auc_score(y, proba, sample_weight=weight)),
        "auprc": float(average_precision_score(y, proba, sample_weight=weight)),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "ppv": float(ppv),
        "npv": float(npv),
        "f1": float(f1),
        "brier": float(brier_score_loss(y, proba, sample_weight=weight)),
        "threshold": float(threshold),
    }


def internal_rows(meta: dict[str, Any]) -> list[dict[str, Any]]:
    best = pd.read_csv(meta["internal_dir"] / "best_model_by_transition.csv")
    pred = pd.read_csv(meta["internal_dir"] / "test_predictions.csv")
    rows = []
    for _, best_row in best.iterrows():
        transition = best_row["transition"]
        model = best_row["model"]
        g = pred[pred["transition"].eq(transition) & pred["model"].eq(model)]
        metrics = weighted_metrics(
            g["y_true"].to_numpy(int),
            g["predicted_probability"].to_numpy(float),
            g["survey_weight"].to_numpy(float),
            float(g["threshold"].iloc[0]),
        )
        rows.append(
            {
                "dataset": "KYRBS internal test",
                "transition": transition,
                "transition_label": TRANSITION_LABELS[transition],
                "model_set": meta["model_set"],
                "analysis_role": meta["analysis_role"],
                "predictor_count": meta["predictor_count"],
                "best_model": model,
                "best_validation_auprc": float(best_row["best_validation_auprc"]),
                **metrics,
            }
        )
    return rows


def external_rows(meta: dict[str, Any]) -> list[dict[str, Any]]:
    pred = pd.read_csv(meta["external_dir"] / "external_validation_predictions.csv")
    rows = []
    for transition, g in pred.groupby("transition", sort=False):
        model = g["model"].iloc[0]
        metrics = weighted_metrics(
            g["y_true"].to_numpy(int),
            g["proba_original"].to_numpy(float),
            g["weight"].to_numpy(float),
            float(g["threshold"].iloc[0]),
        )
        rows.append(
            {
                "dataset": "YRBSS external validation",
                "transition": transition,
                "transition_label": TRANSITION_LABELS[transition],
                "model_set": meta["model_set"],
                "analysis_role": meta["analysis_role"],
                "predictor_count": meta["predictor_count"],
                "best_model": model,
                "best_validation_auprc": math.nan,
                **metrics,
            }
        )
    return rows


def markdown_table(df: pd.DataFrame) -> str:
    display = df.copy()
    for col in display.columns:
        if pd.api.types.is_float_dtype(display[col]):
            display[col] = display[col].map(lambda x: "" if pd.isna(x) else f"{x:.3f}")
        else:
            display[col] = display[col].map(lambda x: "" if pd.isna(x) else str(x))
    widths = [
        max([len(str(c))] + [len(str(v)) for v in display[c].tolist()])
        for c in display.columns
    ]

    def row(values: list[Any]) -> str:
        return "| " + " | ".join(str(v).ljust(widths[i]) for i, v in enumerate(values)) + " |"

    lines = [row(list(display.columns)), "| " + " | ".join("-" * w for w in widths) + " |"]
    lines.extend(row(list(r)) for r in display.to_numpy().tolist())
    return "\n".join(lines)


def main() -> None:
    rows = []
    for meta in MODEL_SETS:
        rows.extend(internal_rows(meta))
        rows.extend(external_rows(meta))
    comparison = pd.DataFrame(rows)
    comparison["transition"] = pd.Categorical(comparison["transition"], TRANSITION_ORDER, ordered=True)
    comparison["dataset"] = pd.Categorical(
        comparison["dataset"], ["KYRBS internal test", "YRBSS external validation"], ordered=True
    )
    comparison["model_set"] = pd.Categorical(
        comparison["model_set"], ["Primary 40 predictors", "Excluding M_SAD and M_STR"], ordered=True
    )
    comparison = comparison.sort_values(["transition", "dataset", "model_set"]).reset_index(drop=True)
    comparison.to_csv(OUT_DIR / "construct_contamination_internal_external_comparison.csv", index=False, encoding="utf-8-sig")

    display_cols = [
        "transition_label",
        "dataset",
        "model_set",
        "best_model",
        "auroc",
        "auprc",
        "sensitivity",
        "specificity",
        "ppv",
        "npv",
        "f1",
        "brier",
        "threshold",
    ]
    manuscript = comparison[display_cols].copy()
    manuscript.to_csv(OUT_DIR / "construct_contamination_manuscript_table.csv", index=False, encoding="utf-8-sig")

    primary = comparison[comparison["model_set"].eq("Primary 40 predictors")]
    sensitivity = comparison[comparison["model_set"].eq("Excluding M_SAD and M_STR")]
    merge_cols = ["dataset", "transition", "transition_label"]
    deltas = primary.merge(sensitivity, on=merge_cols, suffixes=("_primary40", "_excluding_M_SAD_M_STR"))
    for metric in METRICS:
        deltas[f"delta_{metric}_excluding_minus_primary"] = (
            deltas[f"{metric}_excluding_M_SAD_M_STR"] - deltas[f"{metric}_primary40"]
        )
        deltas[f"relative_delta_{metric}_percent"] = (
            100 * deltas[f"delta_{metric}_excluding_minus_primary"] / deltas[f"{metric}_primary40"]
        )
    deltas.to_csv(OUT_DIR / "construct_contamination_performance_deltas.csv", index=False, encoding="utf-8-sig")

    delta_display_cols = [
        "transition_label",
        "dataset",
        "best_model_primary40",
        "best_model_excluding_M_SAD_M_STR",
        "auroc_primary40",
        "auroc_excluding_M_SAD_M_STR",
        "delta_auroc_excluding_minus_primary",
        "auprc_primary40",
        "auprc_excluding_M_SAD_M_STR",
        "delta_auprc_excluding_minus_primary",
        "sensitivity_primary40",
        "sensitivity_excluding_M_SAD_M_STR",
        "delta_sensitivity_excluding_minus_primary",
        "specificity_primary40",
        "specificity_excluding_M_SAD_M_STR",
        "delta_specificity_excluding_minus_primary",
        "f1_primary40",
        "f1_excluding_M_SAD_M_STR",
        "delta_f1_excluding_minus_primary",
        "brier_primary40",
        "brier_excluding_M_SAD_M_STR",
        "delta_brier_excluding_minus_primary",
    ]
    delta_display = deltas[delta_display_cols].copy()
    delta_display.to_csv(OUT_DIR / "construct_contamination_delta_summary.csv", index=False, encoding="utf-8-sig")

    t1_deltas = delta_display[delta_display["transition_label"].eq("T1: ideation vs none")]
    detailed_summary = f"""# Construct-Contamination Sensitivity Analysis

## Design

- Cohort: KYRBS 2020-2024 GRADE 3-6, protocol-primary grade-comparable cohort.
- Outcome: strict continuation-ratio T1/T2/T3.
- Sensitivity predictor set: primary 40 predictors excluding only `M_SAD` and `M_STR`.
- Rationale: `M_SAD` and `M_STR` are conceptually proximal mental-health predictors and may inflate suicide-risk prediction, especially T1 ideation.
- Pipeline: same six models, Optuna 80 trials, preprocessing, missingness handling, split design, and threshold optimization as the primary model.
- External validation: YRBSS 2023 frozen model application; no YRBSS refit.

## Main Comparison

{markdown_table(manuscript)}

## Performance Deltas

Delta is calculated as `excluding M_SAD/M_STR minus primary 40 predictors`.

{markdown_table(delta_display)}

## T1-Specific Interpretation

{markdown_table(t1_deltas)}

## Interpretation

Excluding `M_SAD` and `M_STR` produces the expected major performance drop for T1 ideation. Internally, T1 AUPRC falls from the primary 40-predictor model to the M_SAD/M_STR-excluded model, and YRBSS T1 AUROC/AUPRC also drops materially. This supports the interpretation that these proximal mental-health predictors carry substantial signal for suicidal ideation prediction.

This does not invalidate the primary model. In a clinical or public-health screening model, sadness/hopelessness and stress are legitimate upstream predictors if measured before or alongside screening. However, because they are conceptually close to suicidal ideation, the manuscript should explicitly report the M_SAD/M_STR-excluded model as a robustness analysis addressing construct proximity or potential criterion contamination.

For T2 and T3, the performance loss is much smaller and less consistent than T1. This pattern is conceptually coherent: sadness/stress are most proximal to ideation onset, while progression from ideation to plan/attempt is influenced by broader behavioral, violence, sleep, substance-use, and contextual predictors.

External validation caveat: excluding M_SAD/M_STR does not solve YRBSS threshold transportability. T2 specificity collapses to zero under the frozen KYRBS threshold in the sensitivity model, and T3 specificity remains low. Therefore, the sensitivity analysis mainly addresses construct contamination, not calibration or threshold portability.

Recommended manuscript positioning:

1. Keep the 40-predictor GRADE 3-6 model as the primary protocol model.
2. Present the M_SAD/M_STR-excluded model as a construct-contamination sensitivity analysis.
3. Emphasize the T1 AUPRC/AUROC decline as evidence that proximal mental-health predictors contribute materially to ideation prediction.
4. Avoid replacing the primary model with the sensitivity model, because the sensitivity model intentionally removes clinically meaningful predictors and shows worse T1 transportability.
"""
    (OUT_DIR / "construct_contamination_detailed_summary.md").write_text(detailed_summary, encoding="utf-8")
    (OUT_DIR / "construct_contamination_summary.md").write_text(
        f"""# Construct-Contamination Sensitivity Summary

## Main Comparison

{markdown_table(manuscript)}

## Delta Summary

{markdown_table(delta_display)}
""",
        encoding="utf-8",
    )

    predictor_list = BASE / "kyrbs2020_2024_primary_grade_comparable_38predictor_list_excluding_M_SAD_M_STR.csv"
    if predictor_list.exists():
        (OUT_DIR / predictor_list.name).write_bytes(predictor_list.read_bytes())

    print(f"Saved outputs to {OUT_DIR}")
    print(manuscript.to_string(index=False))
    print()
    print(delta_display.to_string(index=False))


if __name__ == "__main__":
    main()
