#!/usr/bin/env python3
"""Compare primary GRADE 3-6 40/28/21 predictor harmonization sensitivity models."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = REPO_ROOT / "results"
OUT_DIR = RESULTS_ROOT / "primary_grade_comparable_harmonization_sensitivity_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_SETS = [
    {
        "model_set": "40 predictors",
        "predictor_count": 40,
        "harmonization_scope": "Full primary model: all selected KYRBS predictors, including YRBSS unavailable variables",
        "mapping_categories": "Direct + Derived + Proxy + Not available",
        "internal_dir": RESULTS_ROOT
        / "ml_results_kyrbs2020_2024_primary_grade_comparable_grade3_6_40predictors_6models_80trials",
        "external_dir": RESULTS_ROOT / "yrbss2023_external_validation_primary_grade_comparable40",
    },
    {
        "model_set": "28 predictors",
        "predictor_count": 28,
        "harmonization_scope": "YRBSS-harmonizable variables including proxy mappings",
        "mapping_categories": "Direct + Derived + Proxy",
        "internal_dir": RESULTS_ROOT
        / "ml_results_kyrbs2020_2024_primary_grade_comparable_28predictors_direct_derived_proxy_6models_80trials",
        "external_dir": RESULTS_ROOT
        / "yrbss2023_external_validation_primary_grade_comparable_28predictors_direct_derived_proxy",
    },
    {
        "model_set": "21 predictors",
        "predictor_count": 21,
        "harmonization_scope": "Higher-fidelity YRBSS-harmonizable variables only",
        "mapping_categories": "Direct + Derived",
        "internal_dir": RESULTS_ROOT
        / "ml_results_kyrbs2020_2024_primary_grade_comparable_21predictors_direct_derived_6models_80trials",
        "external_dir": RESULTS_ROOT
        / "yrbss2023_external_validation_primary_grade_comparable_21predictors_direct_derived",
    },
]

TRANSITION_LABELS = {
    "T1_ideation_vs_none": "T1: ideation vs none",
    "T2_plan_given_ideation": "T2: plan among ideators",
    "T3_attempt_given_plan": "T3: attempt among planners",
}
TRANSITION_ORDER = list(TRANSITION_LABELS.keys())


def weighted_metrics(
    y: np.ndarray,
    proba: np.ndarray,
    weight: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    y = y.astype(int)
    proba = proba.astype(float)
    weight = weight.astype(float)
    pred = (proba >= threshold).astype(int)
    tp = float(np.sum(weight[(y == 1) & (pred == 1)]))
    fp = float(np.sum(weight[(y == 0) & (pred == 1)]))
    tn = float(np.sum(weight[(y == 0) & (pred == 0)]))
    fn = float(np.sum(weight[(y == 1) & (pred == 0)]))
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else math.nan
    specificity = tn / (tn + fp) if (tn + fp) > 0 else math.nan
    ppv = tp / (tp + fp) if (tp + fp) > 0 else math.nan
    npv = tn / (tn + fn) if (tn + fn) > 0 else math.nan
    f1 = (
        2 * ppv * sensitivity / (ppv + sensitivity)
        if (ppv + sensitivity) > 0
        else math.nan
    )
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
                **{k: meta[k] for k in ["model_set", "predictor_count", "harmonization_scope", "mapping_categories"]},
                "dataset": "KYRBS internal test",
                "transition": transition,
                "transition_label": TRANSITION_LABELS[transition],
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
                **{k: meta[k] for k in ["model_set", "predictor_count", "harmonization_scope", "mapping_categories"]},
                "dataset": "YRBSS external validation",
                "transition": transition,
                "transition_label": TRANSITION_LABELS[transition],
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


def copy_predictor_lists() -> None:
    files = [
        BASE / "kyrbs2020_2024_allgrades_40predictor_list_from_structure_who_adolescent_bmi_smartphone_vtrt.csv",
        BASE / "kyrbs2020_2024_primary_grade_comparable_direct_derived_proxy_28_predictor_list.csv",
        BASE / "kyrbs2020_2024_primary_grade_comparable_direct_derived_21_predictor_list.csv",
        BASE / "primary_grade_comparable_harmonization_sensitivity_mapping_counts.csv",
    ]
    for path in files:
        if path.exists():
            out = OUT_DIR / path.name
            out.write_bytes(path.read_bytes())


def main() -> None:
    copy_predictor_lists()
    rows = []
    for meta in MODEL_SETS:
        rows.extend(internal_rows(meta))
        rows.extend(external_rows(meta))
    comparison = pd.DataFrame(rows)
    comparison["transition"] = pd.Categorical(comparison["transition"], TRANSITION_ORDER, ordered=True)
    comparison["model_set"] = pd.Categorical(
        comparison["model_set"], ["40 predictors", "28 predictors", "21 predictors"], ordered=True
    )
    comparison["dataset"] = pd.Categorical(
        comparison["dataset"], ["KYRBS internal test", "YRBSS external validation"], ordered=True
    )
    comparison = comparison.sort_values(["transition", "dataset", "model_set"]).reset_index(drop=True)
    comparison.to_csv(OUT_DIR / "harmonization_sensitivity_internal_external_comparison.csv", index=False, encoding="utf-8-sig")

    internal = comparison[comparison["dataset"].eq("KYRBS internal test")].copy()
    external = comparison[comparison["dataset"].eq("YRBSS external validation")].copy()
    internal.to_csv(OUT_DIR / "harmonization_sensitivity_internal_only.csv", index=False, encoding="utf-8-sig")
    external.to_csv(OUT_DIR / "harmonization_sensitivity_external_only.csv", index=False, encoding="utf-8-sig")

    merge_cols = ["transition", "transition_label", "model_set", "predictor_count", "mapping_categories"]
    paired = internal.merge(
        external,
        on=merge_cols,
        suffixes=("_internal", "_external"),
    )
    for metric in ["auroc", "auprc", "sensitivity", "specificity", "ppv", "npv", "f1", "brier"]:
        paired[f"delta_{metric}_external_minus_internal"] = (
            paired[f"{metric}_external"] - paired[f"{metric}_internal"]
        )
    paired.to_csv(OUT_DIR / "harmonization_sensitivity_paired_internal_external_deltas.csv", index=False, encoding="utf-8-sig")

    display_cols = [
        "transition_label",
        "dataset",
        "model_set",
        "predictor_count",
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
    display = comparison[display_cols].copy()
    display.to_csv(OUT_DIR / "harmonization_sensitivity_manuscript_table.csv", index=False, encoding="utf-8-sig")

    external_specificity = external[
        ["transition_label", "model_set", "sensitivity", "specificity", "ppv", "npv", "f1", "brier"]
    ].copy()
    detailed_summary = f"""# YRBSS Harmonization Sensitivity Analysis

## Analysis Definition

- Primary development cohort: KYRBS 2020-2024 GRADE 3-6, hierarchy-inconsistent suicide pattern excluded.
- Outcomes: strict continuation-ratio T1, T2, T3.
- ML pipeline: same six-model Optuna 80-trial pipeline, same split logic, same threshold optimization strategy.
- External validation: YRBSS 2023 high-school data, frozen KYRBS models and frozen KYRBS thresholds. No YRBSS refit for the validation estimates in the main comparison.

## Predictor Sets

- 40 predictors: original primary GRADE 3-6 model, including variables unavailable in YRBSS and represented as all-missing during external validation.
- 28 predictors: Direct + Derived + Proxy harmonizable predictors.
- 21 predictors: Direct + Derived predictors only; proxy mappings removed.

## Main Performance Table

{markdown_table(display)}

## External Threshold Behavior

{markdown_table(external_specificity)}

## Interpretation

The 40-predictor model remains the strongest primary KYRBS development model conceptually because it follows the protocol-derived Korean predictor set and avoids tailoring the development model to YRBSS availability. However, its external validation partially tests missing-column transportability rather than a fully harmonized feature space.

The 28-predictor Direct+Derived+Proxy model is the most defensible harmonization sensitivity model for the manuscript. It removes YRBSS-unavailable predictors while preserving clinically and behaviorally important proxy constructs such as smartphone use, loneliness/mental-health proxy, sleep sufficiency proxy, violence proxy, residence proxy, and vigorous activity proxy. It retains relatively similar T1/T2 external discrimination to the 40-predictor model, although T3 threshold transportability is poor.

The 21-predictor Direct+Derived-only model is the strictest harmonization analysis and avoids low-fidelity proxy variables, but it loses substantial internal discrimination for T1 and T2. It is useful as a conservative lower-bound sensitivity analysis, not as the preferred model for the main manuscript.

Important caveat: none of the harmonized models solves the YRBSS T2/T3 specificity problem under KYRBS-derived frozen thresholds. For T2, external specificity remains low across all reduced predictor sets. For T3, the 28-predictor model has near-zero specificity; the 21-predictor model improves specificity relative to 28 predictors but remains low. This supports reporting threshold transportability as a key limitation rather than claiming recalibration-free clinical classification portability.

Recommended manuscript positioning:

1. Primary model: 40 predictors, because it is protocol-driven and maximizes KYRBS construct validity.
2. Main harmonization sensitivity: 28 predictors, because it best balances YRBSS applicability and construct coverage.
3. Conservative harmonization sensitivity: 21 predictors, because it excludes proxy mappings but sacrifices performance.
"""
    (OUT_DIR / "harmonization_sensitivity_detailed_summary.md").write_text(detailed_summary, encoding="utf-8")
    (OUT_DIR / "harmonization_sensitivity_summary.md").write_text(
        f"""# Harmonization Sensitivity Summary

{markdown_table(display)}
""",
        encoding="utf-8",
    )
    print(f"Saved outputs to {OUT_DIR}")
    print(display.to_string(index=False))


if __name__ == "__main__":
    main()
