#!/usr/bin/env python3
"""External validation of frozen KYRBS 2020-2024 40-predictor models on YRBSS 2023."""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
BASE = REPO_ROOT
LOCAL_PKGS = REPO_ROOT / "ml_pkgs"
if LOCAL_PKGS.exists():
    sys.path.insert(0, str(LOCAL_PKGS))

os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".matplotlib"))
os.environ.setdefault("MPLBACKEND", "Agg")

import joblib
import matplotlib.pyplot as plt
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

import ml_stagewise_kyrbs as kyrbs_ml


RESULTS_DIR = REPO_ROOT / "results" / "ml_results_kyrbs2020_2024_allgrades_40predictors_who_adolescent_bmi_smartphone_vtrt_6models_80trials"
MODEL_DIR = RESULTS_DIR / "models"
KYRBS_INPUT = REPO_ROOT / "data" / "kyrbs2020_2024_primary_grade_comparable_grade3_6_40predictors_input.csv"
PREDICTOR_LIST = REPO_ROOT / "metadata" / "kyrbs2020_2024_40predictor_list.csv"
YRBSS_INPUT = REPO_ROOT / "results" / "yrbss2023_selected_40predictor_reference" / "yrbss2023_selected_40predictor_model_input.csv"
YRBSS_MAPPING = REPO_ROOT / "metadata" / "yrbss2023_40predictor_mapping_table.csv"
OUT_DIR = REPO_ROOT / "results" / "yrbss2023_external_validation_final40"
FIG_DIR = OUT_DIR / "figures"

TRANSITION_INFO = {
    "T1_ideation_vs_none": {
        "subset_query": "sui_pattern_inconsistent == 0",
        "positive_query": "sui_ideation == 1",
        "label": "T1: ideation vs none",
    },
    "T2_plan_given_ideation": {
        "subset_query": "(sui_pattern_inconsistent == 0) & (sui_ideation == 1)",
        "positive_query": "sui_plan == 1",
        "label": "T2: plan given ideation",
    },
    "T3_attempt_given_plan": {
        "subset_query": "(sui_pattern_inconsistent == 0) & (sui_plan == 1)",
        "positive_query": "sui_attempt == 1",
        "label": "T3: attempt given plan",
    },
}

METRICS_FOR_CI = [
    "auroc",
    "auprc",
    "sensitivity",
    "specificity",
    "precision",
    "f1",
    "balanced_accuracy",
    "brier",
    "calibration_intercept",
    "calibration_slope",
]


def normalized_weight(w: pd.Series | np.ndarray) -> np.ndarray:
    arr = np.asarray(w, dtype=float)
    arr = np.where(np.isfinite(arr), arr, np.nan)
    if np.isnan(arr).all():
        return np.ones_like(arr, dtype=float)
    arr = np.where(np.isnan(arr), np.nanmedian(arr), arr)
    mean = np.mean(arr)
    if not np.isfinite(mean) or mean <= 0:
        return np.ones_like(arr, dtype=float)
    return arr / mean


def calibration_slope_intercept(y_true: np.ndarray, proba: np.ndarray, sample_weight: np.ndarray) -> tuple[float, float]:
    eps = 1e-6
    p = np.clip(proba, eps, 1 - eps)
    x = logit(p).reshape(-1, 1)
    try:
        lr = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
        lr.fit(x, y_true.astype(int), sample_weight=sample_weight)
        return float(lr.intercept_[0]), float(lr.coef_[0][0])
    except Exception:
        return math.nan, math.nan


def evaluate(y: np.ndarray, proba: np.ndarray, sample_weight: np.ndarray, threshold: float) -> dict[str, float]:
    pred = (proba >= threshold).astype(int)
    cm = confusion_matrix(y, pred, sample_weight=sample_weight, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else math.nan
    intercept, slope = calibration_slope_intercept(y, proba, sample_weight)
    return {
        "n": int(len(y)),
        "events": int(np.sum(y == 1)),
        "weighted_prevalence": float(np.average(y, weights=sample_weight)),
        "auroc": float(roc_auc_score(y, proba, sample_weight=sample_weight)),
        "auprc": float(average_precision_score(y, proba, sample_weight=sample_weight)),
        "sensitivity": float(recall_score(y, pred, sample_weight=sample_weight, zero_division=0)),
        "specificity": float(specificity),
        "precision": float(precision_score(y, pred, sample_weight=sample_weight, zero_division=0)),
        "f1": float(f1_score(y, pred, sample_weight=sample_weight, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred, sample_weight=sample_weight)),
        "brier": float(brier_score_loss(y, proba, sample_weight=sample_weight)),
        "calibration_intercept": float(intercept),
        "calibration_slope": float(slope),
        "threshold": float(threshold),
    }


def fit_platt_on_kyrbs_calibration(transition_name: str, model_bundle: dict, predictors: list[str]) -> LogisticRegression:
    df = pd.read_csv(KYRBS_INPUT)
    df = df[df["sui_stage_highest"].notna()].copy()
    df["sui_stage_highest"] = df["sui_stage_highest"].astype(int)
    X_all = kyrbs_ml.clean_x_for_model(df[predictors])
    survey_w_all = normalized_weight(df["W"])
    primary_stage_mask = df.eval("sui_pattern_inconsistent == 0").to_numpy(dtype=bool)
    master_splits = kyrbs_ml.make_master_splits(df, primary_stage_mask)

    info = TRANSITION_INFO[transition_name]
    subset_mask = df.eval(info["subset_query"]).to_numpy(dtype=bool)
    y = df.loc[subset_mask].eval(info["positive_query"]).to_numpy(dtype=int)
    subset_global_idx = np.flatnonzero(subset_mask)
    calibration_idx = kyrbs_ml.local_indices_from_master(subset_global_idx, master_splits["calibration"])
    X = X_all.loc[subset_mask].reset_index(drop=True)
    w = survey_w_all[subset_mask]
    selected = model_bundle["selected_features"]
    pipe = model_bundle["pipeline"]
    proba = pipe.predict_proba(X.iloc[calibration_idx][selected])[:, 1]
    p = np.clip(proba, 1e-6, 1 - 1e-6)
    lr = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
    lr.fit(logit(p).reshape(-1, 1), y[calibration_idx], sample_weight=w[calibration_idx])
    return lr


def apply_platt(platt: LogisticRegression, proba: np.ndarray) -> np.ndarray:
    p = np.clip(proba, 1e-6, 1 - 1e-6)
    return platt.predict_proba(logit(p).reshape(-1, 1))[:, 1]


def cluster_bootstrap(pred_df: pd.DataFrame, proba_col: str, threshold: float, n_boot: int = 1000, seed: int = 20260706) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    clusters = pred_df["cluster"].dropna().unique()
    rows = []
    for b in range(n_boot):
        sampled = rng.choice(clusters, size=len(clusters), replace=True)
        pieces = []
        for draw_id, cluster in enumerate(sampled):
            part = pred_df[pred_df["cluster"].eq(cluster)].copy()
            part["_draw_id"] = draw_id
            pieces.append(part)
        boot = pd.concat(pieces, ignore_index=True)
        if boot["y_true"].nunique() < 2:
            continue
        try:
            metrics = evaluate(
                boot["y_true"].to_numpy(int),
                boot[proba_col].to_numpy(float),
                boot["weight"].to_numpy(float),
                threshold,
            )
            metrics["bootstrap_id"] = b
            rows.append(metrics)
        except Exception:
            continue
    boot_df = pd.DataFrame(rows)
    ci_rows = []
    for metric in METRICS_FOR_CI:
        vals = boot_df[metric].replace([np.inf, -np.inf], np.nan).dropna()
        ci_rows.append(
            {
                "metric": metric,
                "bootstrap_success_n": int(len(vals)),
                "ci_lower": float(vals.quantile(0.025)) if len(vals) else math.nan,
                "ci_upper": float(vals.quantile(0.975)) if len(vals) else math.nan,
            }
        )
    return pd.DataFrame(ci_rows)


def calibration_bins(pred_df: pd.DataFrame, proba_col: str, n_bins: int = 10) -> pd.DataFrame:
    df = pred_df[["transition", "y_true", "weight", proba_col]].copy()
    df["bin"] = pd.qcut(df[proba_col], q=min(n_bins, df[proba_col].nunique()), duplicates="drop")
    rows = []
    for (transition, bin_label), g in df.groupby(["transition", "bin"], observed=True):
        w = g["weight"].to_numpy(float)
        rows.append(
            {
                "transition": transition,
                "probability": proba_col,
                "bin": str(bin_label),
                "n": int(len(g)),
                "mean_predicted": float(np.average(g[proba_col], weights=w)),
                "observed": float(np.average(g["y_true"], weights=w)),
            }
        )
    return pd.DataFrame(rows)


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    display = df.copy()
    for col in display.columns:
        if pd.api.types.is_float_dtype(display[col]):
            display[col] = display[col].map(lambda x: "" if pd.isna(x) else f"{x:.6f}")
        else:
            display[col] = display[col].map(lambda x: "" if pd.isna(x) else str(x))
    headers = list(display.columns)
    rows = display.values.tolist()
    widths = []
    for i, header in enumerate(headers):
        widths.append(max(len(str(header)), *(len(str(row[i])) for row in rows)) if rows else len(str(header)))

    def fmt_row(values: list[str]) -> str:
        return "| " + " | ".join(str(v).ljust(widths[i]) for i, v in enumerate(values)) + " |"

    lines = [fmt_row(headers), "| " + " | ".join("-" * w for w in widths) + " |"]
    lines.extend(fmt_row(row) for row in rows)
    return "\n".join(lines)


def plot_curves(all_predictions: pd.DataFrame) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for transition, g in all_predictions.groupby("transition"):
        label = TRANSITION_INFO[transition]["label"]
        y = g["y_true"].to_numpy(int)
        w = g["weight"].to_numpy(float)
        p = g["proba_original"].to_numpy(float)

        fpr, tpr, _ = roc_curve(y, p, sample_weight=w)
        roc_auc = roc_auc_score(y, p, sample_weight=w)
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, label=f"AUROC={roc_auc:.3f}")
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
        plt.xlabel("1 - Specificity")
        plt.ylabel("Sensitivity")
        plt.title(f"{label} ROC curve")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(FIG_DIR / f"{transition}_roc_curve.png", dpi=300)
        plt.close()

        precision, recall, _ = precision_recall_curve(y, p, sample_weight=w)
        auprc = average_precision_score(y, p, sample_weight=w)
        plt.figure(figsize=(6, 5))
        plt.plot(recall, precision, label=f"AUPRC={auprc:.3f}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"{label} precision-recall curve")
        plt.legend(loc="upper right")
        plt.tight_layout()
        plt.savefig(FIG_DIR / f"{transition}_pr_curve.png", dpi=300)
        plt.close()

        bins = calibration_bins(g, "proba_original")
        plt.figure(figsize=(6, 5))
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
        plt.plot(bins["mean_predicted"], bins["observed"], marker="o")
        plt.xlabel("Mean predicted probability")
        plt.ylabel("Observed event rate")
        plt.title(f"{label} calibration plot")
        plt.tight_layout()
        plt.savefig(FIG_DIR / f"{transition}_calibration_plot.png", dpi=300)
        plt.close()


def attach_ci(summary: pd.DataFrame, bootstrap: pd.DataFrame) -> pd.DataFrame:
    summary = summary.copy()
    for metric in METRICS_FOR_CI:
        summary[f"{metric}_ci"] = ""
    for i, row in summary.iterrows():
        ci_rows = bootstrap[
            bootstrap["transition"].eq(row["transition"])
            & bootstrap["probability_scale"].eq(row["probability_scale"])
        ]
        for metric in METRICS_FOR_CI:
            ci = ci_rows[ci_rows["metric"].eq(metric)]
            if not ci.empty:
                lo = ci.iloc[0]["ci_lower"]
                hi = ci.iloc[0]["ci_upper"]
                summary.loc[i, f"{metric}_ci"] = f"{lo:.6f}-{hi:.6f}"
    return summary


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    predictors = pd.read_csv(PREDICTOR_LIST)["predictor"].astype(str).tolist()
    best = pd.read_csv(RESULTS_DIR / "best_model_by_transition.csv")
    yrbss = pd.read_csv(YRBSS_INPUT)
    mapping = pd.read_csv(YRBSS_MAPPING)

    missing_cols = [c for c in predictors if c not in yrbss.columns]
    if missing_cols:
        raise ValueError("YRBSS input is missing predictor columns: " + ", ".join(missing_cols))

    X_all = kyrbs_ml.clean_x_for_model(yrbss[predictors])
    weights = normalized_weight(yrbss["W"])

    summary_rows = []
    prediction_rows = []
    bootstrap_rows = []
    calibration_bin_rows = []

    for _, best_row in best.iterrows():
        transition = best_row["transition"]
        model_name = best_row["model"]
        bundle = joblib.load(MODEL_DIR / f"{transition}__{model_name}.joblib")
        threshold = float(bundle["threshold"])
        info = TRANSITION_INFO[transition]

        subset_mask = yrbss.eval(info["subset_query"]).to_numpy(dtype=bool)
        y = yrbss.loc[subset_mask].eval(info["positive_query"]).to_numpy(dtype=int)
        X = X_all.loc[subset_mask].reset_index(drop=True)
        w = weights[subset_mask]
        clusters = yrbss.loc[subset_mask, "CLUSTER"].to_numpy()

        selected = list(bundle["selected_features"])
        pipe = bundle["pipeline"]
        proba = pipe.predict_proba(X[selected])[:, 1]
        platt = fit_platt_on_kyrbs_calibration(transition, bundle, predictors)
        proba_platt = apply_platt(platt, proba)

        pred_df = pd.DataFrame(
            {
                "transition": transition,
                "model": model_name,
                "row_id": yrbss.index[subset_mask].to_numpy(),
                "cluster": clusters,
                "weight": w,
                "y_true": y,
                "proba_original": proba,
                "proba_platt": proba_platt,
                "threshold": threshold,
                "predicted_label": (proba >= threshold).astype(int),
            }
        )
        prediction_rows.append(pred_df)

        for scale, col in [("original", "proba_original"), ("platt_kyrbs_calibration", "proba_platt")]:
            metrics = evaluate(y, pred_df[col].to_numpy(float), w, threshold)
            metrics.update(
                {
                    "transition": transition,
                    "transition_label": info["label"],
                    "model": model_name,
                    "probability_scale": scale,
                    "selected_feature_count": len(selected),
                    "threshold_source": "frozen KYRBS calibration split threshold",
                    "external_dataset": "YRBSS 2023 national high school",
                }
            )
            summary_rows.append(metrics)
            ci = cluster_bootstrap(pred_df, col, threshold, n_boot=1000)
            ci.insert(0, "probability_scale", scale)
            ci.insert(0, "transition", transition)
            bootstrap_rows.append(ci)
            calibration_bin_rows.append(calibration_bins(pred_df, col))

    predictions = pd.concat(prediction_rows, ignore_index=True)
    bootstrap = pd.concat(bootstrap_rows, ignore_index=True)
    calibration_bins_df = pd.concat(calibration_bin_rows, ignore_index=True)
    summary = attach_ci(pd.DataFrame(summary_rows), bootstrap)

    original_summary = summary[summary["probability_scale"].eq("original")].copy()
    perf_cols = [
        "transition",
        "model",
        "n",
        "events",
        "weighted_prevalence",
        "auroc",
        "auroc_ci",
        "auprc",
        "auprc_ci",
        "sensitivity",
        "sensitivity_ci",
        "specificity",
        "specificity_ci",
        "precision",
        "precision_ci",
        "f1",
        "f1_ci",
        "balanced_accuracy",
        "balanced_accuracy_ci",
        "brier",
        "brier_ci",
        "threshold",
    ]
    performance_summary = original_summary[perf_cols]

    calibration_summary = summary[
        [
            "transition",
            "model",
            "probability_scale",
            "n",
            "events",
            "weighted_prevalence",
            "brier",
            "brier_ci",
            "calibration_intercept",
            "calibration_intercept_ci",
            "calibration_slope",
            "calibration_slope_ci",
            "auroc",
            "auprc",
        ]
    ]

    internal = pd.read_csv(RESULTS_DIR / "best_model_by_transition.csv")
    comparison = internal[
        ["transition", "model", "test_auroc", "test_auprc", "test_sensitivity", "test_specificity", "test_f1", "test_brier"]
    ].merge(
        performance_summary[["transition", "model", "auroc", "auprc", "sensitivity", "specificity", "f1", "brier"]],
        on=["transition", "model"],
        how="left",
        suffixes=("_kyrbs_internal", "_yrbss_external"),
    )
    comparison["delta_auroc_external_minus_internal"] = comparison["auroc"] - comparison["test_auroc"]
    comparison["delta_auprc_external_minus_internal"] = comparison["auprc"] - comparison["test_auprc"]
    comparison["delta_f1_external_minus_internal"] = comparison["f1"] - comparison["test_f1"]
    comparison["delta_brier_external_minus_internal"] = comparison["brier"] - comparison["test_brier"]

    unavailable = mapping[mapping["mapping_level"].eq("Unavailable")]["kyrbs_variable"].tolist()
    low_proxy = mapping[mapping["mapping_level"].isin(["Low proxy", "Low", "Partial"])][
        ["kyrbs_variable", "yrbss_source", "mapping_level", "harmonization_rule"]
    ]

    predictions.to_csv(OUT_DIR / "external_validation_predictions.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT_DIR / "external_validation_metrics_all_probability_scales.csv", index=False, encoding="utf-8-sig")
    performance_summary.to_csv(OUT_DIR / "external_validation_performance_summary.csv", index=False, encoding="utf-8-sig")
    calibration_summary.to_csv(OUT_DIR / "external_validation_calibration_summary.csv", index=False, encoding="utf-8-sig")
    bootstrap.to_csv(OUT_DIR / "external_validation_cluster_bootstrap_ci.csv", index=False, encoding="utf-8-sig")
    calibration_bins_df.to_csv(OUT_DIR / "external_validation_calibration_bins.csv", index=False, encoding="utf-8-sig")
    comparison.to_csv(OUT_DIR / "kyrbs_internal_vs_yrbss_external_comparison.csv", index=False, encoding="utf-8-sig")
    low_proxy.to_csv(OUT_DIR / "harmonization_proxy_variables_for_review.csv", index=False, encoding="utf-8-sig")
    (OUT_DIR / "manuscript_ready_external_validation_performance_table.md").write_text(
        dataframe_to_markdown(performance_summary), encoding="utf-8"
    )
    (OUT_DIR / "manuscript_ready_external_validation_calibration_table.md").write_text(
        dataframe_to_markdown(calibration_summary), encoding="utf-8"
    )
    (OUT_DIR / "manuscript_ready_internal_external_comparison_table.md").write_text(
        dataframe_to_markdown(comparison), encoding="utf-8"
    )
    plot_curves(predictions)

    text = f"""# YRBSS 2023 External Validation of Final KYRBS 40-Predictor Models

## Validation Mode

- Development model: final KYRBS 2020-2024 all-grade 40-predictor models.
- External dataset: YRBSS 2023 national high school public-use data.
- Model training on YRBSS: none.
- Hyperparameter changes: none.
- Threshold changes: none. Frozen KYRBS calibration thresholds were applied.
- Primary metrics use original frozen probabilities. Platt-scaled probabilities are reported as recalibration diagnostics only; Platt scaling was fit on the KYRBS calibration split, not on YRBSS.

## Harmonization Notes

- YRBSS is grades 9-12 only, while the development model here is the final KYRBS all-grade model. This should be reported as a transportability/external-validation analysis with age/grade-scope caveats.
- Unavailable KYRBS predictors retained as all-missing YRBSS columns: {", ".join(unavailable)}.
- Low-fidelity proxy variables are listed in `harmonization_proxy_variables_for_review.csv`.

## Primary External Validation Performance

{dataframe_to_markdown(performance_summary)}

## Internal vs External Comparison

{dataframe_to_markdown(comparison)}
"""
    (OUT_DIR / "external_validation_summary.md").write_text(text, encoding="utf-8")

    print(f"Output directory: {OUT_DIR}")
    print(performance_summary.to_string(index=False))
    print()
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
