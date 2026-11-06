#!/usr/bin/env python3
"""Protocol-driven staged machine-learning analysis for KYRBS 2020-2024.

This script implements:
- staged suicidal behavior modeling
- stage-wise binary transitions
- Optuna tuning with the same number of trials for every model
- missingness threshold and class balancing included in tuning
- validation-set threshold optimization for the rare positive class
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
LOCAL_PKGS = REPO_ROOT / "ml_pkgs"
if LOCAL_PKGS.exists():
    sys.path.insert(0, str(LOCAL_PKGS))

MPLCONFIGDIR = REPO_ROOT / ".matplotlib"
MPLCONFIGDIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

import numpy as np
import optuna
import pandas as pd
from scipy.special import logit
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)


RANDOM_SEED = 20250701
MISSING_THRESHOLDS = [0.05, 0.10, 0.20, 0.40]
CLASS_BALANCE_MODES = ["none", "sqrt_balanced", "balanced"]

OUTCOME_VARS = {
    "M_SUI_CON",
    "M_SUI_PLN",
    "M_SUI_ATT",
    "sui_ideation",
    "sui_plan",
    "sui_attempt",
    "sui_any",
    "sui_stage_highest",
    "sui_pattern_inconsistent",
    "t1_ideation_or_higher_vs_none",
    "t2_plan_or_higher_given_ideation",
    "t3_attempt_given_plan",
}
DESIGN_VARS = {"STRATA", "CLUSTER", "W", "FPC"}
ID_VARS = {"OBS", "YEAR"}

CONTINUOUS_VARS = {
    "AGE",
    "AGE_M",
    "HT",
    "WT",
    "bmi",
    "sleep_weekday_min",
    "sleep_weekend_min",
    "sleep_daily_min",
    "smartphone_avg_day_min",
    "smartphone_daily_min",
    "sedentary_daily_min",
    "INT_SPWD_TM",
    "INT_SPWK_TM",
    "PA_SWD_S",
    "PA_SWD_N",
    "PA_SWK_S",
    "PA_SWK_N",
}


@dataclass(frozen=True)
class Transition:
    name: str
    subset_query: str
    positive_query: str
    description: str


TRANSITIONS = [
    Transition(
        name="T1_ideation_vs_none",
        subset_query="sui_pattern_inconsistent == 0",
        positive_query="sui_ideation == 1",
        description="none -> ideation among hierarchy-consistent suicide-response patterns",
    ),
    Transition(
        name="T2_plan_given_ideation",
        subset_query="(sui_pattern_inconsistent == 0) & (sui_ideation == 1)",
        positive_query="sui_plan == 1",
        description="ideation -> plan among adolescents with ideation and hierarchy-consistent responses",
    ),
    Transition(
        name="T3_attempt_given_plan",
        subset_query="(sui_pattern_inconsistent == 0) & (sui_plan == 1)",
        positive_query="sui_attempt == 1",
        description="plan -> attempt among adolescents with a suicide plan and hierarchy-consistent responses",
    ),
]


def weighted_class_factor(y: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return np.ones_like(y, dtype=float)
    counts = np.bincount(y.astype(int), minlength=2).astype(float)
    counts[counts == 0] = 1.0
    factors = len(y) / (2.0 * counts)
    out = factors[y.astype(int)]
    if mode == "sqrt_balanced":
        out = np.sqrt(out)
    return out / np.mean(out)


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


def stratified_group_holdout(
    idx: np.ndarray,
    stratify_values: np.ndarray,
    group_values: np.ndarray,
    n_splits: int,
    seed_offset: int,
) -> tuple[np.ndarray, np.ndarray]:
    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=RANDOM_SEED + seed_offset,
    )
    placeholder_x = np.zeros(len(idx))
    train_rel, holdout_rel = next(
        splitter.split(placeholder_x, stratify_values, group_values)
    )
    return idx[train_rel], idx[holdout_rel]


def make_master_splits(
    df: pd.DataFrame,
    eligible_mask: np.ndarray,
) -> dict[str, np.ndarray]:
    """Create one CLUSTER-disjoint split shared by all stage transitions.

    Proportions are approximately 60% train, 10% tuning validation,
    10% threshold calibration, and 20% untouched test.
    """
    eligible_idx = np.flatnonzero(eligible_mask)

    def y_groups(rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        y = df.iloc[rows]["sui_stage_highest"].astype(int).to_numpy()
        if "YEAR" in df.columns:
            groups = (
                df.iloc[rows]["YEAR"].astype(str).str.replace(r"\.0$", "", regex=True)
                + "_"
                + df.iloc[rows]["CLUSTER"].astype(str)
            ).to_numpy()
        else:
            groups = df.iloc[rows]["CLUSTER"].astype(str).to_numpy()
        return y, groups

    y_all, groups_all = y_groups(eligible_idx)
    train_val_cal_idx, test_idx = stratified_group_holdout(
        eligible_idx, y_all, groups_all, n_splits=5, seed_offset=0
    )

    y_train_val_cal, groups_train_val_cal = y_groups(train_val_cal_idx)
    train_idx, val_cal_idx = stratified_group_holdout(
        train_val_cal_idx,
        y_train_val_cal,
        groups_train_val_cal,
        n_splits=4,
        seed_offset=101,
    )

    y_val_cal, groups_val_cal = y_groups(val_cal_idx)
    tune_validation_idx, calibration_idx = stratified_group_holdout(
        val_cal_idx,
        y_val_cal,
        groups_val_cal,
        n_splits=2,
        seed_offset=202,
    )

    return {
        "train": np.asarray(train_idx, dtype=int),
        "tuning_validation": np.asarray(tune_validation_idx, dtype=int),
        "calibration": np.asarray(calibration_idx, dtype=int),
        "test": np.asarray(test_idx, dtype=int),
    }


def local_indices_from_master(
    subset_global_idx: np.ndarray,
    master_split_idx: np.ndarray,
) -> np.ndarray:
    split_set = set(np.asarray(master_split_idx, dtype=int).tolist())
    return np.asarray(
        [local_i for local_i, global_i in enumerate(subset_global_idx) if global_i in split_set],
        dtype=int,
    )


def stratified_limit_indices(
    idx: np.ndarray,
    y: np.ndarray,
    max_n: int,
    seed_offset: int = 0,
) -> np.ndarray:
    if max_n <= 0 or len(idx) <= max_n:
        return idx
    sampled_idx, _ = train_test_split(
        idx,
        train_size=max_n,
        random_state=RANDOM_SEED + seed_offset,
        stratify=y[idx],
    )
    return np.asarray(sampled_idx)


def select_features_by_missingness(
    X_train: pd.DataFrame,
    all_feature_cols: list[str],
    missing_threshold: float,
) -> list[str]:
    miss = X_train[all_feature_cols].isna().mean()
    selected = [c for c in all_feature_cols if miss[c] <= missing_threshold]
    if not selected:
        raise ValueError("No features remained after missingness filtering.")
    return selected


def make_preprocessor(
    feature_cols: list[str],
    min_frequency: int | None,
) -> ColumnTransformer:
    numeric_cols = [c for c in feature_cols if c in CONTINUOUS_VARS]
    categorical_cols = [c for c in feature_cols if c not in CONTINUOUS_VARS]

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler(with_mean=False)),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="__MISSING__")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    min_frequency=min_frequency,
                    sparse_output=True,
                ),
            ),
        ]
    )
    transformers = []
    if numeric_cols:
        transformers.append(("num", numeric_pipe, numeric_cols))
    if categorical_cols:
        transformers.append(("cat", categorical_pipe, categorical_cols))

    return ColumnTransformer(transformers=transformers, remainder="drop")


def suggest_common_params(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "missing_threshold": trial.suggest_categorical(
            "missing_threshold", MISSING_THRESHOLDS
        ),
        "min_frequency": trial.suggest_categorical(
            "min_frequency", [None, 5, 10, 20]
        ),
        "class_balance": trial.suggest_categorical(
            "class_balance", CLASS_BALANCE_MODES
        ),
    }


def make_model(model_name: str, trial: optuna.Trial) -> Any:
    if model_name == "logistic_regression":
        penalty = trial.suggest_categorical("penalty", ["l1", "l2"])
        return LogisticRegression(
            C=trial.suggest_float("C", 1e-3, 50.0, log=True),
            penalty=penalty,
            solver="liblinear",
            max_iter=500,
            tol=1e-4,
            random_state=RANDOM_SEED,
        )

    if model_name == "random_forest":
        max_depth_choice = trial.suggest_categorical(
            "max_depth_choice", ["none", "4", "6", "8", "12", "16"]
        )
        max_depth = None if max_depth_choice == "none" else int(max_depth_choice)
        return RandomForestClassifier(
            n_estimators=trial.suggest_int("n_estimators", 40, 140, step=10),
            max_depth=max_depth,
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 80, log=True),
            min_samples_split=trial.suggest_int("min_samples_split", 2, 80, log=True),
            max_features=trial.suggest_categorical(
                "max_features", ["sqrt", "log2", 0.30, 0.50, 0.80]
            ),
            bootstrap=True,
            max_samples=trial.suggest_float("max_samples", 0.45, 0.90),
            n_jobs=-1,
            random_state=RANDOM_SEED,
        )

    if model_name == "extra_trees":
        max_depth_choice = trial.suggest_categorical(
            "max_depth_choice", ["none", "4", "6", "8", "12", "16"]
        )
        max_depth = None if max_depth_choice == "none" else int(max_depth_choice)
        return ExtraTreesClassifier(
            n_estimators=trial.suggest_int("n_estimators", 40, 160, step=10),
            max_depth=max_depth,
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 80, log=True),
            min_samples_split=trial.suggest_int("min_samples_split", 2, 80, log=True),
            max_features=trial.suggest_categorical(
                "max_features", ["sqrt", "log2", 0.30, 0.50, 0.80]
            ),
            bootstrap=True,
            max_samples=trial.suggest_float("max_samples", 0.45, 0.90),
            n_jobs=-1,
            random_state=RANDOM_SEED,
        )

    if model_name == "xgboost":
        return XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            n_estimators=trial.suggest_int("n_estimators", 80, 420, step=20),
            max_depth=trial.suggest_int("max_depth", 2, 7),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.25, log=True),
            subsample=trial.suggest_float("subsample", 0.55, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.55, 1.0),
            min_child_weight=trial.suggest_float("min_child_weight", 1.0, 30.0, log=True),
            gamma=trial.suggest_float("gamma", 0.0, 6.0),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 50.0, log=True),
            n_jobs=-1,
            random_state=RANDOM_SEED,
        )

    if model_name == "lightgbm":
        max_depth_choice = trial.suggest_categorical(
            "max_depth_choice", ["-1", "3", "5", "7", "9", "12"]
        )
        max_depth = int(max_depth_choice)
        return LGBMClassifier(
            objective="binary",
            n_estimators=trial.suggest_int("n_estimators", 80, 500, step=20),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.25, log=True),
            num_leaves=trial.suggest_int("num_leaves", 8, 96, log=True),
            max_depth=max_depth,
            min_child_samples=trial.suggest_int("min_child_samples", 5, 120, log=True),
            subsample=trial.suggest_float("subsample", 0.55, 1.0),
            subsample_freq=1,
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.55, 1.0),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 50.0, log=True),
            random_state=RANDOM_SEED,
            n_jobs=-1,
            verbose=-1,
        )

    if model_name == "catboost":
        return CatBoostClassifier(
            loss_function="Logloss",
            eval_metric="Logloss",
            iterations=trial.suggest_int("iterations", 80, 500, step=20),
            depth=trial.suggest_int("depth", 3, 8),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.25, log=True),
            l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1.0, 30.0, log=True),
            random_strength=trial.suggest_float("random_strength", 1e-3, 10.0, log=True),
            bagging_temperature=trial.suggest_float("bagging_temperature", 0.0, 5.0),
            rsm=trial.suggest_float("rsm", 0.55, 1.0),
            random_seed=RANDOM_SEED,
            thread_count=-1,
            verbose=False,
            allow_writing_files=False,
        )

    raise ValueError(f"Unknown model: {model_name}")


def build_pipeline(
    model_name: str,
    params: dict[str, Any],
    feature_cols: list[str],
) -> Pipeline:
    pre = make_preprocessor(feature_cols, params["min_frequency"])

    class FixedTrial:
        def __init__(self, p: dict[str, Any]):
            self.params = p

        def suggest_categorical(self, name: str, choices: list[Any]) -> Any:
            return self.params[name]

        def suggest_float(
            self,
            name: str,
            low: float,
            high: float,
            log: bool = False,
        ) -> float:
            return self.params[name]

        def suggest_int(
            self,
            name: str,
            low: int,
            high: int,
            step: int = 1,
            log: bool = False,
        ) -> int:
            return int(self.params[name])

    model = make_model(model_name, FixedTrial(params))
    return Pipeline(steps=[("preprocess", pre), ("model", model)])


def get_positive_proba(pipe: Pipeline, X: pd.DataFrame) -> np.ndarray:
    if hasattr(pipe.named_steps["model"], "predict_proba"):
        return pipe.predict_proba(X)[:, 1]
    scores = pipe.decision_function(X)
    return 1.0 / (1.0 + np.exp(-scores))


def weighted_confusion(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sample_weight: np.ndarray,
) -> tuple[float, float, float, float]:
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    w = np.asarray(sample_weight, dtype=float)
    tp = np.sum(w[(y_true == 1) & (y_pred == 1)])
    fp = np.sum(w[(y_true == 0) & (y_pred == 1)])
    tn = np.sum(w[(y_true == 0) & (y_pred == 0)])
    fn = np.sum(w[(y_true == 1) & (y_pred == 0)])
    return tp, fp, tn, fn


def optimize_threshold_balanced_accuracy(
    y_true: np.ndarray,
    proba: np.ndarray,
    sample_weight: np.ndarray,
) -> tuple[float, dict[str, float]]:
    grid = np.unique(
        np.concatenate(
            [
                np.linspace(0.01, 0.99, 99),
                np.quantile(proba, np.linspace(0.01, 0.99, 99)),
            ]
        )
    )
    best = {"threshold": 0.5, "balanced_accuracy": -1.0, "f1": -1.0}
    for threshold in grid:
        pred = (proba >= threshold).astype(int)
        tp, fp, tn, fn = weighted_confusion(y_true, pred, sample_weight)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        f1 = (
            2.0 * precision * sensitivity / (precision + sensitivity)
            if (precision + sensitivity) > 0
            else 0.0
        )
        balanced_accuracy = (sensitivity + specificity) / 2.0
        if (balanced_accuracy > best["balanced_accuracy"]) or (
            np.isclose(balanced_accuracy, best["balanced_accuracy"]) and f1 > best["f1"]
        ):
            best = {
                "threshold": float(threshold),
                "f1": float(f1),
                "balanced_accuracy": float(balanced_accuracy),
                "precision": float(precision),
                "sensitivity": float(sensitivity),
                "specificity": float(specificity),
            }
    return best["threshold"], best


def calibration_slope_intercept(
    y_true: np.ndarray,
    proba: np.ndarray,
    sample_weight: np.ndarray,
) -> tuple[float, float]:
    eps = 1e-6
    p = np.clip(proba, eps, 1.0 - eps)
    x = logit(p).reshape(-1, 1)
    try:
        lr = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
        lr.fit(x, y_true, sample_weight=sample_weight)
        return float(lr.intercept_[0]), float(lr.coef_[0][0])
    except Exception:
        return math.nan, math.nan


def evaluate_predictions(
    y_true: np.ndarray,
    proba: np.ndarray,
    sample_weight: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    pred = (proba >= threshold).astype(int)
    tp, fp, tn, fn = weighted_confusion(y_true, pred, sample_weight)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else math.nan
    intercept, slope = calibration_slope_intercept(y_true, proba, sample_weight)
    prevalence = np.average(y_true, weights=sample_weight)
    return {
        "n": int(len(y_true)),
        "events": int(np.sum(y_true == 1)),
        "weighted_prevalence": float(prevalence),
        "auroc": float(roc_auc_score(y_true, proba, sample_weight=sample_weight)),
        "auprc": float(average_precision_score(y_true, proba, sample_weight=sample_weight)),
        "brier": float(brier_score_loss(y_true, proba, sample_weight=sample_weight)),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, pred, sample_weight=sample_weight)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, pred, sample_weight=sample_weight)
        ),
        "sensitivity": float(recall_score(y_true, pred, sample_weight=sample_weight)),
        "specificity": float(specificity),
        "precision": float(
            precision_score(y_true, pred, sample_weight=sample_weight, zero_division=0)
        ),
        "f1": float(f1_score(y_true, pred, sample_weight=sample_weight, zero_division=0)),
        "calibration_intercept": intercept,
        "calibration_slope": slope,
    }


def net_benefit(y: np.ndarray, proba: np.ndarray, threshold: float, w: np.ndarray) -> float:
    pred = proba >= threshold
    tp = np.sum(w[(y == 1) & pred])
    fp = np.sum(w[(y == 0) & pred])
    n = np.sum(w)
    return float(tp / n - fp / n * (threshold / (1.0 - threshold)))


def clean_x_for_model(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    for col in X.columns:
        if col not in CONTINUOUS_VARS:
            X[col] = X[col].where(X[col].notna(), "__MISSING__").astype(str)
    return X


def aggregate_feature_importance(
    pipe: Pipeline,
    importance: np.ndarray,
    selected_features: list[str],
    transition: str,
    model_name: str,
) -> pd.DataFrame:
    pre = pipe.named_steps["preprocess"]
    try:
        transformed_names = pre.get_feature_names_out()
    except Exception:
        transformed_names = np.array([f"feature_{i}" for i in range(len(importance))])

    selected_sorted = sorted(selected_features, key=len, reverse=True)

    def original_name(transformed: str) -> str:
        name = transformed.split("__", 1)[-1]
        if name.startswith("missingindicator_"):
            return name.replace("missingindicator_", "")
        for col in selected_sorted:
            if name == col or name.startswith(col + "_"):
                return col
        return name

    rows = []
    for tname, val in zip(transformed_names, importance):
        rows.append(
            {
                "transition": transition,
                "model": model_name,
                "original_variable": original_name(str(tname)),
                "transformed_feature": str(tname),
                "importance": float(abs(val)),
            }
        )
    df = pd.DataFrame(rows)
    return (
        df.groupby(["transition", "model", "original_variable"], as_index=False)[
            "importance"
        ]
        .sum()
        .sort_values(["transition", "model", "importance"], ascending=[True, True, False])
    )


def model_feature_importance(
    pipe: Pipeline,
    selected_features: list[str],
    transition: str,
    model_name: str,
) -> pd.DataFrame | None:
    model = pipe.named_steps["model"]
    if hasattr(model, "feature_importances_"):
        imp = np.asarray(model.feature_importances_, dtype=float)
        return aggregate_feature_importance(pipe, imp, selected_features, transition, model_name)
    if hasattr(model, "coef_"):
        coef = np.asarray(model.coef_).ravel()
        return aggregate_feature_importance(pipe, coef, selected_features, transition, model_name)
    return None


def xgboost_shap_importance(
    pipe: Pipeline,
    X: pd.DataFrame,
    selected_features: list[str],
    transition: str,
    max_rows: int = 2000,
) -> pd.DataFrame | None:
    try:
        import shap
    except Exception:
        return None
    try:
        sample = X.sample(n=min(max_rows, len(X)), random_state=RANDOM_SEED)
        Xt = pipe.named_steps["preprocess"].transform(sample)
        if hasattr(Xt, "toarray"):
            Xt_dense = Xt.toarray()
        else:
            Xt_dense = np.asarray(Xt)
        explainer = shap.TreeExplainer(pipe.named_steps["model"])
        values = explainer.shap_values(Xt_dense)
        if isinstance(values, list):
            values = values[-1]
        mean_abs = np.mean(np.abs(values), axis=0)
        out = aggregate_feature_importance(
            pipe, mean_abs, selected_features, transition, "xgboost_shap"
        )
        return out
    except Exception as exc:
        print(f"[WARN] SHAP failed for {transition}: {exc}", flush=True)
        return None


def run_one_model(
    model_name: str,
    transition: Transition,
    X: pd.DataFrame,
    X_missing_basis: pd.DataFrame,
    y: np.ndarray,
    survey_w: np.ndarray,
    all_feature_cols: list[str],
    train_idx: np.ndarray,
    tune_validation_idx: np.ndarray,
    calibration_idx: np.ndarray,
    test_idx: np.ndarray,
    n_trials: int,
    out_dir: Path,
    max_tune_train: int,
    max_tune_val: int,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None, pd.DataFrame]:
    X_train = X.iloc[train_idx]
    X_missing_train = X_missing_basis.iloc[train_idx]
    X_calibration = X.iloc[calibration_idx]
    X_test = X.iloc[test_idx]
    y_train = y[train_idx]
    y_calibration = y[calibration_idx]
    y_test = y[test_idx]
    w_train_survey = survey_w[train_idx]
    w_calibration_survey = survey_w[calibration_idx]
    w_test_survey = survey_w[test_idx]
    tune_train_idx = stratified_limit_indices(train_idx, y, max_tune_train, seed_offset=11)
    tune_val_idx = stratified_limit_indices(
        tune_validation_idx, y, max_tune_val, seed_offset=17
    )
    X_tune_train = X.iloc[tune_train_idx]
    X_tune_val = X.iloc[tune_val_idx]
    y_tune_train = y[tune_train_idx]
    y_tune_val = y[tune_val_idx]
    w_tune_train_survey = survey_w[tune_train_idx]
    w_tune_val_survey = survey_w[tune_val_idx]

    def objective(trial: optuna.Trial) -> float:
        params = suggest_common_params(trial)
        selected_features = select_features_by_missingness(
            X_missing_train, all_feature_cols, params["missing_threshold"]
        )
        model = make_model(model_name, trial)
        pipe = Pipeline(
            steps=[
                ("preprocess", make_preprocessor(selected_features, params["min_frequency"])),
                ("model", model),
            ]
        )
        class_factor = weighted_class_factor(y_tune_train, params["class_balance"])
        fit_weight = w_tune_train_survey * class_factor
        fit_weight = fit_weight / np.mean(fit_weight)
        try:
            pipe.fit(
                X_tune_train[selected_features],
                y_tune_train,
                model__sample_weight=fit_weight,
            )
            val_proba = get_positive_proba(pipe, X_tune_val[selected_features])
            auprc = average_precision_score(
                y_tune_val,
                val_proba,
                sample_weight=w_tune_val_survey,
            )
            if not np.isfinite(auprc):
                return 0.0
            return float(auprc)
        except Exception as exc:
            trial.set_user_attr("failed_reason", str(exc)[:500])
            return 0.0

    sampler = optuna.samplers.TPESampler(seed=RANDOM_SEED)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    started = time.time()

    def progress_callback(study_obj: optuna.Study, trial_obj: optuna.trial.FrozenTrial) -> None:
        done = trial_obj.number + 1
        if done == 1 or done % 10 == 0 or done == n_trials:
            best = study_obj.best_value if len(study_obj.trials) else float("nan")
            print(
                f"      trial {done:03d}/{n_trials} best validation AUPRC={best:.4f}",
                flush=True,
            )

    study.optimize(
        objective,
        n_trials=n_trials,
        n_jobs=1,
        show_progress_bar=False,
        callbacks=[progress_callback],
    )
    elapsed = time.time() - started

    trials_df = study.trials_dataframe(attrs=("number", "value", "params", "state"))
    trials_df.insert(0, "transition", transition.name)
    trials_df.insert(1, "model", model_name)
    trials_df["elapsed_seconds_for_study"] = elapsed

    best_params = dict(study.best_params)
    selected_features = select_features_by_missingness(
        X_missing_train, all_feature_cols, best_params["missing_threshold"]
    )

    # Final fit uses train + tuning-validation after hyperparameter decisions.
    # The calibration split is held out from fitting and used only for threshold choice.
    train_tune_idx = np.concatenate([train_idx, tune_validation_idx])
    X_train_tune = X.iloc[train_tune_idx]
    y_train_tune = y[train_tune_idx]
    w_train_tune_survey = survey_w[train_tune_idx]
    final_fit_weight = w_train_tune_survey * weighted_class_factor(
        y_train_tune, best_params["class_balance"]
    )
    final_fit_weight = final_fit_weight / np.mean(final_fit_weight)

    final_pipe = build_pipeline(model_name, best_params, selected_features)
    final_pipe.fit(
        X_train_tune[selected_features],
        y_train_tune,
        model__sample_weight=final_fit_weight,
    )

    calib_proba = get_positive_proba(final_pipe, X_calibration[selected_features])
    threshold, threshold_info = optimize_threshold_balanced_accuracy(
        y_calibration, calib_proba, w_calibration_survey
    )
    calibration_metrics = evaluate_predictions(
        y_calibration, calib_proba, w_calibration_survey, threshold
    )
    test_proba = get_positive_proba(final_pipe, X_test[selected_features])
    test_metrics = evaluate_predictions(y_test, test_proba, w_test_survey, threshold)

    dca_thresholds = [0.05, 0.10, 0.15, 0.20, 0.30]
    dca = {
        f"net_benefit_{thr:.2f}": net_benefit(y_test, test_proba, thr, w_test_survey)
        for thr in dca_thresholds
    }

    summary = {
        "transition": transition.name,
        "transition_description": transition.description,
        "model": model_name,
        "n_trials": n_trials,
        "tune_train_n": int(len(tune_train_idx)),
        "tune_validation_n": int(len(tune_val_idx)),
        "train_n": int(len(train_idx)),
        "calibration_n": int(len(calibration_idx)),
        "test_n": int(len(test_idx)),
        "best_validation_auprc": float(study.best_value),
        "best_tuning_validation_auprc": float(study.best_value),
        "selected_feature_count": len(selected_features),
        "dropped_feature_count": len(all_feature_cols) - len(selected_features),
        "elapsed_seconds": elapsed,
        "best_params": best_params,
        "threshold_optimization_metric": "calibration balanced accuracy / Youden index",
        "threshold_calibration_f1": threshold_info.get("f1", math.nan),
        "threshold_calibration_balanced_accuracy": threshold_info.get(
            "balanced_accuracy", math.nan
        ),
        **{f"calibration_{k}": v for k, v in calibration_metrics.items()},
        **{f"test_{k}": v for k, v in test_metrics.items()},
        **dca,
    }

    importance_df = model_feature_importance(
        final_pipe, selected_features, transition.name, model_name
    )
    shap_df = None
    if model_name == "xgboost":
        shap_df = xgboost_shap_importance(
            final_pipe, X_test[selected_features], selected_features, transition.name
        )

    pred_df = pd.DataFrame(
        {
            "transition": transition.name,
            "model": model_name,
            "row_id": test_idx,
            "y_true": y_test,
            "survey_weight": w_test_survey,
            "predicted_probability": test_proba,
            "threshold": threshold,
            "predicted_label": (test_proba >= threshold).astype(int),
        }
    )

    model_out = out_dir / "models"
    model_out.mkdir(parents=True, exist_ok=True)
    try:
        import joblib

        joblib.dump(
            {
                "pipeline": final_pipe,
                "selected_features": selected_features,
                "threshold": threshold,
                "transition": transition.name,
                "model": model_name,
                "best_params": best_params,
            },
            model_out / f"{transition.name}__{model_name}.joblib",
        )
    except Exception as exc:
        print(f"[WARN] Could not save model {transition.name}/{model_name}: {exc}")

    return summary, trials_df, importance_df, shap_df, pred_df


def write_protocol_summary(
    out_path: Path,
    input_csv: Path,
    predictor_list: Path | None,
    n_trials: int,
    model_names: list[str],
    all_feature_cols: list[str],
    max_tune_train: int,
    max_tune_val: int,
    grade_scope: str,
    transitions: list[Transition] | None = None,
) -> None:
    transitions = transitions or TRANSITIONS
    text = f"""# KYRBS 2025 Protocol-Based ML Analysis Decisions

Input data: `{input_csv}`
Predictor list: `{predictor_list}` if provided, otherwise all non-outcome/non-design variables in the input data.
Grade scope: `{grade_scope}`

Protocol interpretation:
- Outcome is modeled as staged suicidal behavior, not a single flat outcome.
- Requested continuation-ratio conditional stage classifications in this run:
{chr(10).join(f"  - {t.name}: {t.description}" for t in transitions)}
- Primary stage-wise analyses exclude inconsistent suicide-response patterns
  (plan/attempt reported without ideation) and should describe them separately
  or evaluate them in sensitivity analyses.
- Descriptive epidemiology should use the complex survey variables, but this ML script uses `W` as sample weight during model fitting/evaluation and excludes `STRATA`, `CLUSTER`, `FPC` from predictors.

Predictor handling:
- Excluded from predictors: IDs, survey design variables, raw suicide items, and derived suicide outcomes.
- Candidate predictor count before missingness thresholding: {len(all_feature_cols)}
- Categorical/ordinal survey codes are one-hot encoded.
- Continuous variables are median-imputed with missing indicators.
- Categorical missingness is retained as an explicit missing category.
- Feature missingness thresholding is computed on the raw predictor matrix before
  categorical missing values are converted to an explicit missing category.

Optimization:
- Models: {", ".join(model_names)}
- Trials per model per transition: {n_trials}
- Total studies: {len(model_names) * len(transitions)}
- Total trials: {len(model_names) * len(transitions) * n_trials}
- Tuning-validation objective: weighted AUPRC.
- Optuna tuning uses stratified tuning subsets for computational feasibility: up to {max_tune_train} training rows and {max_tune_val} tuning-validation rows per transition/model.
- Tuned preprocessing thresholds: feature missingness cutoff in {MISSING_THRESHOLDS}; one-hot rare-category min_frequency in [None, 5, 10, 20].
- Tuned imbalance handling: {CLASS_BALANCE_MODES}.
- Final models are fit on train + tuning-validation data.
- Final classification threshold: selected on a separate calibration split by weighted balanced accuracy / Youden index, balancing sensitivity and specificity. Decision-curve net benefit is exported separately for clinical screening threshold discussion.

Split:
- A single master split is shared by every transition: approximately 60% train, 10% tuning-validation, 10% threshold calibration, and 20% untouched test.
- The master split is stratified by suicidal-behavior stage and group-disjoint by year-specific `YEAR + CLUSTER` to reduce train/test contamination.
- Hyperparameter tuning uses train -> tuning-validation only.
- Threshold selection uses calibration only.
- Final test metrics are computed once after tuning.
"""
    out_path.write_text(text, encoding="utf-8")


def safe_artifact_name(*parts: str) -> str:
    return "__".join(parts).replace("/", "_").replace(" ", "_")


def write_incremental_outputs(
    out_dir: Path,
    summary: dict[str, Any],
    trials_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    imp_df: pd.DataFrame | None,
    shap_df: pd.DataFrame | None,
) -> None:
    """Persist each completed model immediately so long runs are recoverable."""
    name = safe_artifact_name(summary["transition"], summary["model"])
    incremental_dir = out_dir / "incremental"
    for child in ["summaries", "trials", "predictions", "importances", "shap"]:
        (incremental_dir / child).mkdir(parents=True, exist_ok=True)

    summary_payload = dict(summary)
    summary_payload["best_params"] = json.dumps(
        summary_payload["best_params"], ensure_ascii=False
    )
    pd.DataFrame([summary_payload]).to_csv(
        incremental_dir / "summaries" / f"{name}.csv", index=False
    )
    trials_df.to_csv(incremental_dir / "trials" / f"{name}.csv", index=False)
    pred_df.to_csv(incremental_dir / "predictions" / f"{name}.csv", index=False)
    if imp_df is not None:
        imp_df.to_csv(incremental_dir / "importances" / f"{name}.csv", index=False)
    if shap_df is not None:
        shap_df.to_csv(incremental_dir / "shap" / f"{name}.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="data/kyrbs2020_2024_primary_grade_comparable_grade3_6_40predictors_input.csv",
        help="KYRBS 2020-2024 model-ready analysis CSV.",
    )
    parser.add_argument(
        "--predictor-list",
        default=None,
        help="Optional CSV containing the exact predictor list. Uses a 'predictor' column if present, otherwise the first column.",
    )
    parser.add_argument(
        "--out-dir",
        default="results/ml_results_kyrbs2020_2024_primary_grade_comparable_grade3_6_40predictors_6models_80trials",
    )
    parser.add_argument("--trials", type=int, default=80)
    parser.add_argument("--max-tune-train", type=int, default=18000)
    parser.add_argument("--max-tune-val", type=int, default=6000)
    parser.add_argument(
        "--grade-scope",
        choices=["all", "high_school"],
        default="all",
        help="Use all grades or high-school grades only. high_school keeps GRADE 4-6.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=[
            "logistic_regression",
            "random_forest",
            "extra_trees",
            "xgboost",
            "lightgbm",
            "catboost",
        ],
    )
    parser.add_argument(
        "--transitions",
        nargs="+",
        choices=[t.name for t in TRANSITIONS],
        default=None,
        help="Optional subset of transition names to run. Defaults to all transitions.",
    )
    args = parser.parse_args()

    input_csv = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv)
    df = df[df["sui_stage_highest"].notna()].copy()
    df["sui_stage_highest"] = df["sui_stage_highest"].astype(int)
    if args.grade_scope == "high_school":
        df = df[pd.to_numeric(df["GRADE"], errors="coerce").between(4, 6)].copy()
        if df.empty:
            raise ValueError("No rows remain after high_school grade filtering.")

    excluded = OUTCOME_VARS | DESIGN_VARS | ID_VARS
    predictor_list_path = Path(args.predictor_list) if args.predictor_list else None
    if predictor_list_path is not None:
        predictor_df = pd.read_csv(predictor_list_path)
        predictor_col = "predictor" if "predictor" in predictor_df.columns else predictor_df.columns[0]
        requested_features = predictor_df[predictor_col].dropna().astype(str).tolist()
        missing_features = [c for c in requested_features if c not in df.columns]
        if missing_features:
            raise ValueError(
                "Predictor list contains variables not found in input data: "
                + ", ".join(missing_features)
            )
        all_feature_cols = [c for c in requested_features if c not in excluded]
    else:
        all_feature_cols = [c for c in df.columns if c not in excluded]
    X_missing_basis_all = df[all_feature_cols].copy()
    X_all = clean_x_for_model(df[all_feature_cols])
    survey_w_all = normalized_weight(df["W"])
    selected_transitions = [
        t for t in TRANSITIONS if args.transitions is None or t.name in args.transitions
    ]

    primary_stage_mask = df.eval("sui_pattern_inconsistent == 0").to_numpy(dtype=bool)
    master_splits = make_master_splits(df, primary_stage_mask)

    write_protocol_summary(
        out_dir / "protocol_ml_decisions.md",
        input_csv,
        predictor_list_path,
        args.trials,
        args.models,
        all_feature_cols,
        args.max_tune_train,
        args.max_tune_val,
        args.grade_scope,
        selected_transitions,
    )

    summaries = []
    trial_histories = []
    importances = []
    shap_importances = []
    predictions = []
    transition_rows = []

    print(
        f"Running protocol ML: {len(selected_transitions)} transitions x "
        f"{len(args.models)} models x {args.trials} trials",
        flush=True,
    )

    for transition in selected_transitions:
        subset_mask = df.eval(transition.subset_query).to_numpy(dtype=bool)
        y = df.loc[subset_mask].eval(transition.positive_query).to_numpy(dtype=int)
        subset_global_idx = np.flatnonzero(subset_mask)
        X = X_all.loc[subset_mask].reset_index(drop=True)
        X_missing_basis = X_missing_basis_all.loc[subset_mask].reset_index(drop=True)
        w = survey_w_all[subset_mask]

        train_idx = local_indices_from_master(subset_global_idx, master_splits["train"])
        tune_validation_idx = local_indices_from_master(
            subset_global_idx, master_splits["tuning_validation"]
        )
        calibration_idx = local_indices_from_master(
            subset_global_idx, master_splits["calibration"]
        )
        test_idx = local_indices_from_master(subset_global_idx, master_splits["test"])
        split_map = {
            "train": train_idx,
            "tuning_validation": tune_validation_idx,
            "calibration": calibration_idx,
            "test": test_idx,
        }
        for split_name, split_idx in split_map.items():
            if len(split_idx) == 0 or len(np.unique(y[split_idx])) < 2:
                raise ValueError(
                    f"{transition.name} has an unusable {split_name} split: "
                    f"n={len(split_idx)}, classes={np.unique(y[split_idx]).tolist()}"
                )
        transition_rows.append(
            {
                "transition": transition.name,
                "description": transition.description,
                "n_total": int(len(y)),
                "events_total": int(np.sum(y == 1)),
                "event_rate_unweighted": float(np.mean(y)),
                "n_train": int(len(train_idx)),
                "events_train": int(np.sum(y[train_idx] == 1)),
                "n_tuning_validation": int(len(tune_validation_idx)),
                "events_tuning_validation": int(np.sum(y[tune_validation_idx] == 1)),
                "n_calibration": int(len(calibration_idx)),
                "events_calibration": int(np.sum(y[calibration_idx] == 1)),
                "n_test": int(len(test_idx)),
                "events_test": int(np.sum(y[test_idx] == 1)),
            }
        )
        print(
            f"\n[{transition.name}] n={len(y):,}, events={np.sum(y == 1):,} "
            f"({np.mean(y):.3f})",
            flush=True,
        )

        for model_name in args.models:
            print(
                f"  - Optimizing {model_name} with {args.trials} trials...",
                flush=True,
            )
            summary, trials_df, imp_df, shap_df, pred_df = run_one_model(
                model_name=model_name,
                transition=transition,
                X=X,
                X_missing_basis=X_missing_basis,
                y=y,
                survey_w=w,
                all_feature_cols=all_feature_cols,
                train_idx=train_idx,
                tune_validation_idx=tune_validation_idx,
                calibration_idx=calibration_idx,
                test_idx=test_idx,
                n_trials=args.trials,
                out_dir=out_dir,
                max_tune_train=args.max_tune_train,
                max_tune_val=args.max_tune_val,
            )
            summaries.append(summary)
            trial_histories.append(trials_df)
            predictions.append(pred_df)
            if imp_df is not None:
                importances.append(imp_df)
            if shap_df is not None:
                shap_importances.append(shap_df)
            write_incremental_outputs(
                out_dir=out_dir,
                summary=summary,
                trials_df=trials_df,
                pred_df=pred_df,
                imp_df=imp_df,
                shap_df=shap_df,
            )
            print(
                "    test AUROC={:.3f}, AUPRC={:.3f}, sensitivity={:.3f}, "
                "specificity={:.3f}, threshold={:.3f}".format(
                    summary["test_auroc"],
                    summary["test_auprc"],
                    summary["test_sensitivity"],
                    summary["test_specificity"],
                    summary["test_threshold"],
                ),
                flush=True,
            )

    summary_df = pd.DataFrame(summaries)
    best_params = {
        f"{row['transition']}__{row['model']}": row["best_params"]
        for row in summaries
    }
    summary_export = summary_df.drop(columns=["best_params"])
    summary_export.to_csv(out_dir / "model_performance_summary.csv", index=False)
    pd.concat(trial_histories, ignore_index=True).to_csv(
        out_dir / "optuna_trial_history.csv", index=False
    )
    pd.DataFrame(transition_rows).to_csv(out_dir / "transition_splits.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_csv(
        out_dir / "test_predictions.csv", index=False
    )
    if importances:
        pd.concat(importances, ignore_index=True).to_csv(
            out_dir / "feature_importance_by_model.csv", index=False
        )
    if shap_importances:
        pd.concat(shap_importances, ignore_index=True).to_csv(
            out_dir / "xgboost_shap_importance.csv", index=False
        )
    with open(out_dir / "best_hyperparameters.json", "w", encoding="utf-8") as f:
        json.dump(best_params, f, ensure_ascii=False, indent=2)

    # Model selection must be completed before looking at test performance.
    # The held-out test set is kept for final reporting only.
    leaderboard = (
        summary_export.sort_values(
            ["transition", "best_validation_auprc"], ascending=[True, False]
        )
        .groupby("transition")
        .head(1)
    )
    leaderboard.to_csv(out_dir / "best_model_by_transition.csv", index=False)

    print("\nSaved outputs to", out_dir, flush=True)
    print(leaderboard[["transition", "model", "test_auroc", "test_auprc", "test_sensitivity", "test_specificity"]], flush=True)


if __name__ == "__main__":
    os.environ.setdefault("PYTHONHASHSEED", str(RANDOM_SEED))
    main()
