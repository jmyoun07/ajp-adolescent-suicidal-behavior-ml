#!/usr/bin/env python3
"""Bootstrap CIs for primary KYRBS GRADE 3-6 internal and YRBSS external validation."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


REPO_ROOT = Path(__file__).resolve().parents[1]
INTERNAL_DIR = (
    REPO_ROOT
    / "results"
    / "ml_results_kyrbs2020_2024_primary_grade_comparable_grade3_6_40predictors_6models_80trials"
)
EXTERNAL_DIR = REPO_ROOT / "results" / "yrbss2023_external_validation_primary_grade_comparable40"
KYRBS_INPUT = REPO_ROOT / "data" / "kyrbs2020_2024_primary_grade_comparable_grade3_6_40predictors_input.csv"

N_BOOT = 1000
RANDOM_SEED = 20260707

TRANSITION_INFO = {
    "T1_ideation_vs_none": {
        "subset_query": "sui_pattern_inconsistent == 0",
        "label": "T1: ideation vs none",
    },
    "T2_plan_given_ideation": {
        "subset_query": "(sui_pattern_inconsistent == 0) & (sui_ideation == 1)",
        "label": "T2: plan among ideators",
    },
    "T3_attempt_given_plan": {
        "subset_query": "(sui_pattern_inconsistent == 0) & (sui_plan == 1)",
        "label": "T3: attempt among planners",
    },
}

METRICS = [
    "auroc",
    "auprc",
    "sensitivity",
    "specificity",
    "ppv",
    "npv",
    "f1",
    "brier",
]


def weighted_metrics(y: np.ndarray, proba: np.ndarray, w: np.ndarray, threshold: float) -> dict[str, float]:
    y = y.astype(int)
    proba = proba.astype(float)
    w = w.astype(float)
    pred = (proba >= threshold).astype(int)

    tp = float(np.sum(w[(y == 1) & (pred == 1)]))
    fp = float(np.sum(w[(y == 0) & (pred == 1)]))
    tn = float(np.sum(w[(y == 0) & (pred == 0)]))
    fn = float(np.sum(w[(y == 1) & (pred == 0)]))

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else math.nan
    specificity = tn / (tn + fp) if (tn + fp) > 0 else math.nan
    ppv = tp / (tp + fp) if (tp + fp) > 0 else math.nan
    npv = tn / (tn + fn) if (tn + fn) > 0 else math.nan
    f1 = 2 * ppv * sensitivity / (ppv + sensitivity) if (ppv + sensitivity) > 0 else math.nan

    out = {
        "n": int(len(y)),
        "events": int(np.sum(y == 1)),
        "weighted_prevalence": float(np.average(y, weights=w)),
        "auroc": math.nan,
        "auprc": math.nan,
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "ppv": float(ppv),
        "npv": float(npv),
        "f1": float(f1),
        "brier": float(brier_score_loss(y, proba, sample_weight=w)),
    }
    if np.unique(y).size == 2:
        out["auroc"] = float(roc_auc_score(y, proba, sample_weight=w))
        out["auprc"] = float(average_precision_score(y, proba, sample_weight=w))
    return out


def cluster_bootstrap(pred: pd.DataFrame, seed_offset: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(RANDOM_SEED + seed_offset)
    groups = [
        g.index.to_numpy(dtype=int)
        for _, g in pred.groupby("cluster", sort=False, observed=True)
    ]
    point = weighted_metrics(
        pred["y_true"].to_numpy(int),
        pred["proba"].to_numpy(float),
        pred["weight"].to_numpy(float),
        float(pred["threshold"].iloc[0]),
    )
    point["bootstrap_success_n"] = 0

    boot_rows = []
    for b in range(N_BOOT):
        sampled_groups = rng.integers(0, len(groups), size=len(groups))
        idx = np.concatenate([groups[i] for i in sampled_groups])
        boot = pred.loc[idx]
        if boot["y_true"].nunique() < 2:
            continue
        metrics = weighted_metrics(
            boot["y_true"].to_numpy(int),
            boot["proba"].to_numpy(float),
            boot["weight"].to_numpy(float),
            float(pred["threshold"].iloc[0]),
        )
        metrics["bootstrap_id"] = b
        boot_rows.append(metrics)

    boot_df = pd.DataFrame(boot_rows)
    ci_rows = []
    for metric in METRICS:
        vals = boot_df[metric].replace([np.inf, -np.inf], np.nan).dropna()
        ci_rows.append(
            {
                "metric": metric,
                "estimate": float(point[metric]),
                "ci_lower": float(vals.quantile(0.025)) if len(vals) else math.nan,
                "ci_upper": float(vals.quantile(0.975)) if len(vals) else math.nan,
                "bootstrap_success_n": int(len(vals)),
            }
        )
    ci_df = pd.DataFrame(ci_rows)
    return ci_df, boot_df


def load_internal_predictions() -> pd.DataFrame:
    best = pd.read_csv(INTERNAL_DIR / "best_model_by_transition.csv")
    predictions = pd.read_csv(INTERNAL_DIR / "test_predictions.csv")
    kyrbs = pd.read_csv(KYRBS_INPUT)
    kyrbs = kyrbs[kyrbs["sui_stage_highest"].notna()].copy()
    kyrbs["sui_stage_highest"] = kyrbs["sui_stage_highest"].astype(int)

    pieces = []
    for _, row in best.iterrows():
        transition = row["transition"]
        model = row["model"]
        pred = predictions[
            predictions["transition"].eq(transition) & predictions["model"].eq(model)
        ].copy()
        subset = kyrbs.loc[kyrbs.eval(TRANSITION_INFO[transition]["subset_query"])].reset_index(drop=True)
        subset_rows = subset.iloc[pred["row_id"].astype(int).to_numpy()]
        pred["cluster"] = (
            subset_rows["YEAR"].astype(str).str.replace(r"\.0$", "", regex=True).to_numpy()
            + "_"
            + subset_rows["CLUSTER"].astype(str).str.replace(r"\.0$", "", regex=True).to_numpy()
        )
        pred = pred.rename(
            columns={
                "survey_weight": "weight",
                "predicted_probability": "proba",
            }
        )
        pred["transition_label"] = TRANSITION_INFO[transition]["label"]
        pieces.append(
            pred[
                [
                    "transition",
                    "transition_label",
                    "model",
                    "cluster",
                    "weight",
                    "y_true",
                    "proba",
                    "threshold",
                ]
            ]
        )
    return pd.concat(pieces, ignore_index=True)


def load_external_predictions() -> pd.DataFrame:
    pred = pd.read_csv(EXTERNAL_DIR / "external_validation_predictions.csv")
    pred = pred.rename(columns={"proba_original": "proba"})
    pred["transition_label"] = pred["transition"].map(
        {k: v["label"] for k, v in TRANSITION_INFO.items()}
    )
    return pred[
        [
            "transition",
            "transition_label",
            "model",
            "cluster",
            "weight",
            "y_true",
            "proba",
            "threshold",
        ]
    ].copy()


def summarize_dataset(predictions: pd.DataFrame, dataset: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    ci_parts = []
    boot_parts = []
    for seed_offset, (transition, pred) in enumerate(predictions.groupby("transition", sort=False)):
        ci, boot = cluster_bootstrap(pred.reset_index(drop=True), seed_offset)
        model = pred["model"].iloc[0]
        label = pred["transition_label"].iloc[0]
        for df in (ci, boot):
            df.insert(0, "dataset", dataset)
            df.insert(1, "transition", transition)
            df.insert(2, "transition_label", label)
            df.insert(3, "model", model)
        ci.insert(4, "n", int(len(pred)))
        ci.insert(5, "events", int(pred["y_true"].sum()))
        ci.insert(6, "threshold", float(pred["threshold"].iloc[0]))
        ci_parts.append(ci)
        boot_parts.append(boot)
    return pd.concat(ci_parts, ignore_index=True), pd.concat(boot_parts, ignore_index=True)


def pivot_for_comparison(ci: pd.DataFrame) -> pd.DataFrame:
    internal = ci[ci["dataset"].eq("KYRBS internal test")].copy()
    external = ci[ci["dataset"].eq("YRBSS external validation")].copy()
    merged = internal.merge(
        external,
        on=["transition", "transition_label", "metric"],
        how="outer",
        suffixes=("_internal", "_external"),
    )
    merged["delta_external_minus_internal"] = (
        merged["estimate_external"] - merged["estimate_internal"]
    )
    cols = [
        "transition",
        "transition_label",
        "metric",
        "model_internal",
        "model_external",
        "n_internal",
        "events_internal",
        "n_external",
        "events_external",
        "estimate_internal",
        "ci_lower_internal",
        "ci_upper_internal",
        "estimate_external",
        "ci_lower_external",
        "ci_upper_external",
        "delta_external_minus_internal",
        "bootstrap_success_n_internal",
        "bootstrap_success_n_external",
    ]
    return merged[cols]


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    display = df.copy()
    for col in display.columns:
        if pd.api.types.is_float_dtype(display[col]):
            display[col] = display[col].map(lambda x: "" if pd.isna(x) else f"{x:.3f}")
        else:
            display[col] = display[col].map(lambda x: "" if pd.isna(x) else str(x))
    widths = [
        max([len(str(col))] + [len(str(v)) for v in display[col].tolist()])
        for col in display.columns
    ]

    def row(values: list[str]) -> str:
        return "| " + " | ".join(str(v).ljust(widths[i]) for i, v in enumerate(values)) + " |"

    lines = [row(list(display.columns)), "| " + " | ".join("-" * w for w in widths) + " |"]
    lines.extend(row(list(r)) for r in display.to_numpy().tolist())
    return "\n".join(lines)


def main() -> None:
    internal_pred = load_internal_predictions()
    external_pred = load_external_predictions()
    internal_ci, internal_boot = summarize_dataset(internal_pred, "KYRBS internal test")
    external_ci, external_boot = summarize_dataset(external_pred, "YRBSS external validation")
    all_ci = pd.concat([internal_ci, external_ci], ignore_index=True)
    all_boot = pd.concat([internal_boot, external_boot], ignore_index=True)
    comparison = pivot_for_comparison(all_ci)

    internal_ci.to_csv(
        INTERNAL_DIR / "primary_grade_comparable_internal_test_bootstrap_ci.csv",
        index=False,
        encoding="utf-8-sig",
    )
    external_ci.to_csv(
        EXTERNAL_DIR / "primary_grade_comparable_external_validation_bootstrap_ci.csv",
        index=False,
        encoding="utf-8-sig",
    )
    all_ci.to_csv(
        EXTERNAL_DIR / "primary_grade_comparable_internal_external_bootstrap_ci_long.csv",
        index=False,
        encoding="utf-8-sig",
    )
    all_boot.to_csv(
        EXTERNAL_DIR / "primary_grade_comparable_internal_external_bootstrap_estimates_long.csv",
        index=False,
        encoding="utf-8-sig",
    )
    comparison.to_csv(
        EXTERNAL_DIR / "primary_grade_comparable_internal_vs_external_bootstrap_ci_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )

    compact = comparison[
        [
            "transition_label",
            "metric",
            "estimate_internal",
            "ci_lower_internal",
            "ci_upper_internal",
            "estimate_external",
            "ci_lower_external",
            "ci_upper_external",
            "delta_external_minus_internal",
        ]
    ].copy()
    text = f"""# Primary GRADE 3-6 Bootstrap Confidence Intervals

- Bootstrap method: cluster bootstrap with 1000 resamples.
- Internal KYRBS test clustering: `YEAR + CLUSTER`.
- External YRBSS clustering: `CLUSTER`.
- Estimates are survey-weighted and use the frozen validation/calibration thresholds from the primary GRADE 3-6 models.
- CI method: percentile 2.5th to 97.5th percentiles.

{dataframe_to_markdown(compact)}
"""
    (EXTERNAL_DIR / "primary_grade_comparable_bootstrap_ci_summary.md").write_text(
        text, encoding="utf-8"
    )

    print("Saved bootstrap CI outputs")
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
