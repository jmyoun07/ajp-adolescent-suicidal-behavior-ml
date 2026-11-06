from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ML_PKGS = ROOT / "ml_pkgs"
if ML_PKGS.exists():
    sys.path.insert(0, str(ML_PKGS))

import joblib
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
import shap


OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)

INTERNAL_DIR = ROOT / "results" / "ml_results_kyrbs2020_2024_primary_grade_comparable_grade3_6_40predictors_6models_80trials"
EXTERNAL_DIR = ROOT / "results" / "yrbss2023_external_validation_primary_grade_comparable40"
CAL_DIR = ROOT / "results" / "primary_grade_comparable_calibration_analysis_40predictors_corrected"

INTERNAL_PRED = INTERNAL_DIR / "test_predictions.csv"
EXTERNAL_PRED = EXTERNAL_DIR / "external_validation_predictions.csv"
CAL_POINTS = CAL_DIR / "calibration_curve_points.csv"
KDATA = ROOT / "data" / "kyrbs2020_2024_primary_grade_comparable_grade3_6_40predictors_input.csv"

SELECTED = {
    "T1_ideation_vs_none": "lightgbm",
    "T2_plan_given_ideation": "random_forest",
    "T3_attempt_given_plan": "xgboost",
}

LABELS = {
    "T1_ideation_vs_none": "T1: Ideation vs none",
    "T2_plan_given_ideation": "T2: Plan among ideators",
    "T3_attempt_given_plan": "T3: Attempt among planners",
}

SHORT = {
    "T1_ideation_vs_none": "T1",
    "T2_plan_given_ideation": "T2",
    "T3_attempt_given_plan": "T3",
}

VAR_LABELS = {
    "AGE_M": "Age in months",
    "M_LON": "Loneliness",
    "M_STR": "Stress",
    "M_SAD": "Depressive sadness",
    "V_TRT": "Violence-related treatment",
    "PR_HT": "Perceived health",
    "M_SLP_EN": "Sleep satisfaction",
    "sleep_daily_min": "Sleep duration",
    "smartphone_daily_min": "Smartphone time",
    "CITY": "City",
    "PA_MSC": "Muscle exercise",
    "vigorous_activity_days": "Vigorous activity",
    "F_BR": "Breakfast",
    "PA_TOT": "Physical activity",
    "drug_use_ever": "Ever drug use",
    "current_tobacco_type": "Current tobacco type",
    "AC_AMNT": "Alcohol amount",
    "AC_DAYS": "Alcohol use days",
    "AC_DRUNK": "Binge drinking",
    "TC_AMNT": "Cigarette amount",
    "E_RES": "Residence type",
    "E_S_RCRD": "School record",
    "SEX": "Sex",
    "sedentary_daily_min": "Sedentary time",
}

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 11,
        "axes.titlesize": 11,
        "axes.labelsize": 11,
        "xtick.labelsize": 9.6,
        "ytick.labelsize": 9.6,
        "legend.fontsize": 8.7,
        "figure.dpi": 300,
        "savefig.dpi": 600,
        "axes.linewidth": 1.2,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

COL_INTERNAL = "#1f5aa6"
COL_EXTERNAL = "#c94f24"
COL_ACCENT = "#2f9e7e"
GRID = "#d9dde4"
SHAP_CMAP = shap.plots.colors.red_blue
MODEL_COLORS = {
    "logistic_regression": "#414141",
    "random_forest": "#2f9e7e",
    "extra_trees": "#81b29a",
    "xgboost": "#d95f02",
    "lightgbm": "#1f5aa6",
    "catboost": "#8b5fbf",
}
MODEL_LABELS = {
    "logistic_regression": "Logistic",
    "random_forest": "RF",
    "extra_trees": "Extra trees",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
    "catboost": "CatBoost",
}


def savefig(fig: mpl.figure.Figure, filename: str) -> None:
    path = OUT / filename
    fig.savefig(path, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)
    # Flatten to a true RGB (no alpha channel) PNG on a solid white background,
    # per Elsevier artwork guidance, while preserving the embedded DPI metadata.
    from PIL import Image

    with Image.open(path) as im:
        dpi = im.info.get("dpi")
        if im.mode in ("RGBA", "LA"):
            bg = Image.new("RGB", im.size, (255, 255, 255))
            bg.paste(im, mask=im.split()[-1])
            im = bg
        else:
            im = im.convert("RGB")
        save_kwargs = {"dpi": dpi} if dpi else {}
        im.save(path, **save_kwargs)
    print(path)


def add_panel_label(ax: plt.Axes, label: str, fontsize: float = 12.5) -> None:
    ax.text(
        -0.12,
        1.06,
        label,
        transform=ax.transAxes,
        fontsize=fontsize,
        fontweight="bold",
        va="top",
        ha="left",
    )


def draw_band(ax, x, y, w, h, label, facecolor, edgecolor) -> None:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.0,rounding_size=0.12",
        linewidth=1.3,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linestyle=(0, (4, 2)),
        zorder=1,
    )
    ax.add_patch(patch)
    ax.text(
        x + 0.14,
        y + h - 0.20,
        label,
        fontsize=9.6,
        fontweight="bold",
        color=edgecolor,
        ha="left",
        va="top",
        style="italic",
    )


def draw_module(ax, x, y, w, h, title, subtitle, facecolor, edgecolor, title_size=10.6, sub_size=8.7):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.045,rounding_size=0.09",
        linewidth=1.4,
        facecolor=facecolor,
        edgecolor=edgecolor,
        zorder=3,
    )
    ax.add_patch(patch)
    if subtitle:
        ax.text(x + w / 2, y + h * 0.62, title, ha="center", va="center", fontsize=title_size, fontweight="bold", zorder=4)
        ax.text(
            x + w / 2,
            y + h * 0.27,
            subtitle,
            ha="center",
            va="center",
            fontsize=sub_size,
            color="#30343b",
            linespacing=1.15,
            zorder=4,
        )
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", fontsize=title_size, fontweight="bold", zorder=4)
    return (x, y, w, h)


def arrow(ax, p0, p1, color="#5a616b", style="-|>", lw=1.15, connectionstyle="arc3,rad=0.0", scale=10, zorder=2):
    ax.add_patch(
        FancyArrowPatch(
            p0,
            p1,
            arrowstyle=style,
            mutation_scale=scale,
            linewidth=lw,
            color=color,
            connectionstyle=connectionstyle,
            zorder=zorder,
        )
    )


def figure1_workflow() -> None:
    fig, ax = plt.subplots(figsize=(13.0, 8.1))
    ax.axis("off")
    ax.set_xlim(0, 13.0)
    ax.set_ylim(0, 8.9)

    ARROW = "#5a616b"

    # --- Row 1: data sources + central staged-target panel -----------------
    kyrbs_box = draw_module(
        ax, 0.3, 7.55, 2.9, 1.0,
        "KYRBS 2020-2024", "Model development data",
        facecolor="#d9ecff", edgecolor="#1f5aa6", title_size=11.6, sub_size=9.2,
    )
    yrbss_box = draw_module(
        ax, 9.8, 7.55, 2.9, 1.0,
        "YRBSS 2023", "External validation data",
        facecolor="#ffe6cf", edgecolor="#c9691f", title_size=11.6, sub_size=9.2,
    )
    draw_band(ax, 3.5, 7.25, 6.0, 1.62, "STAGED PREDICTION TARGETS", "#eaf5e3", "#4f8a3f")
    stage_defs = [
        ("T1", "Ideation vs none"),
        ("T2", "Plan among ideators"),
        ("T3", "Attempt among planners"),
    ]
    stage_w = 1.78
    stage_gap = 0.14
    stage_x0 = 3.5 + (6.0 - (stage_w * 3 + stage_gap * 2)) / 2
    stage_boxes = []
    for i, (tag, text) in enumerate(stage_defs):
        sx = stage_x0 + i * (stage_w + stage_gap)
        b = draw_module(ax, sx, 7.55, stage_w, 0.92, tag, text, facecolor="#f7fbf4", edgecolor="#4f8a3f", title_size=11.5, sub_size=8.5)
        stage_boxes.append(b)
        if i < 2:
            arrow(ax, (sx + stage_w + 0.01, 7.55 + 0.46), (sx + stage_w + stage_gap - 0.01, 7.55 + 0.46), color="#4f8a3f", lw=1.0, scale=8)
    arrow(ax, (kyrbs_box[0] + kyrbs_box[2], 7.55 + 0.5), (3.5, 8.05), color=ARROW)
    arrow(ax, (3.5 + 6.0, 8.05), (yrbss_box[0], 7.55 + 0.5), color=ARROW)

    # --- Row 2: model development (left/center) + external validation (right, tall) ---
    dev_band = (0.3, 4.75, 9.4, 2.35)
    draw_band(ax, *dev_band, "MODEL DEVELOPMENT LAYER", "#f1ecfb", "#6a4fb0")
    ext_band = (9.8, 2.55, 2.9, 4.55)
    draw_band(ax, *ext_band, "EXTERNAL VALIDATION LAYER", "#ffe6cf", "#c9691f")

    dev_mods = [
        ("Preprocessing", "impute, encode,\nscale"),
        ("Train / tune /\ncalibrate / test split", "stratified,\nsurvey-weighted"),
        ("Candidate models", "logistic, RF, extra\ntrees, XGBoost,\nLightGBM, CatBoost"),
        ("Threshold\nselection", "validation-set\nAUPRC"),
    ]
    n = len(dev_mods)
    mod_w = 2.08
    mod_gap = (dev_band[2] - 0.5 - n * mod_w) / (n - 1)
    mod_x0 = dev_band[0] + 0.35
    mod_boxes = []
    for i, (title, sub) in enumerate(dev_mods):
        mx = mod_x0 + i * (mod_w + mod_gap)
        b = draw_module(ax, mx, 5.15, mod_w, 1.55, title, sub, facecolor="#ffffff", edgecolor="#6a4fb0", title_size=10.3, sub_size=8.3)
        mod_boxes.append(b)
        if i > 0:
            px, py, pw, ph = mod_boxes[i - 1]
            arrow(ax, (px + pw + 0.02, py + ph / 2), (mx - 0.02, py + ph / 2), color="#6a4fb0", lw=1.05, scale=9)

    # staged targets -> preprocessing (development)
    arrow(ax, (stage_x0 + (stage_w * 3 + stage_gap * 2) / 2, 7.55), (mod_boxes[0][0] + mod_boxes[0][2] / 2, 5.15 + 1.55), color=ARROW)

    ext_mods = [
        ("Frozen models", "no refitting"),
        ("Applied to YRBSS\n2023 test set", "held-out\ntransportability check"),
    ]
    ey0 = 5.85
    ext_boxes = []
    for i, (title, sub) in enumerate(ext_mods):
        b = draw_module(ax, ext_band[0] + 0.2, ey0 - i * 1.28, ext_band[2] - 0.4, 1.05, title, sub, facecolor="#fff7ee", edgecolor="#c9691f", title_size=10.0, sub_size=8.3)
        ext_boxes.append(b)
    arrow(ax, (ext_boxes[0][0] + ext_boxes[0][2] / 2, ext_boxes[0][1]), (ext_boxes[1][0] + ext_boxes[1][2] / 2, ext_boxes[1][1] + ext_boxes[1][3]), color="#c9691f", lw=1.05, scale=9)
    # threshold selection -> frozen models (dev -> external, "frozen" hand-off)
    arrow(
        ax,
        (mod_boxes[-1][0] + mod_boxes[-1][2], mod_boxes[-1][1] + mod_boxes[-1][3] * 0.62),
        (ext_boxes[0][0], ext_boxes[0][1] + ext_boxes[0][3] * 0.5),
        color=ARROW,
        connectionstyle="arc3,rad=-0.12",
    )
    # YRBSS data -> external validation layer
    arrow(ax, (yrbss_box[0] + yrbss_box[2] / 2, 7.55), (ext_boxes[0][0] + ext_boxes[0][2] * 0.8, ext_boxes[0][1] + ext_boxes[0][3]), color=ARROW, connectionstyle="arc3,rad=0.15")

    # --- Row 3: explainability layer (left, below model development) -------
    exp_band = (0.3, 2.55, 9.4, 1.85)
    draw_band(ax, *exp_band, "EXPLAINABILITY LAYER", "#dff2ee", "#1f8a72")
    exp_mods = [
        ("SHAP explanations", "TreeExplainer on\nselected pipeline"),
        ("Aggregation to original\npredictors", "encoded columns\nsummed per variable"),
    ]
    exp_w = 3.55
    exp_gap = 0.5
    exp_x0 = exp_band[0] + 0.9
    exp_boxes = []
    for i, (title, sub) in enumerate(exp_mods):
        ex = exp_x0 + i * (exp_w + exp_gap)
        b = draw_module(ax, ex, 2.85, exp_w, 1.15, title, sub, facecolor="#ffffff", edgecolor="#1f8a72", title_size=10.1, sub_size=8.3)
        exp_boxes.append(b)
        if i > 0:
            px, py, pw, ph = exp_boxes[i - 1]
            arrow(ax, (px + pw + 0.02, py + ph / 2), (ex - 0.02, py + ph / 2), color="#1f8a72", lw=1.05, scale=9)
    # candidate models / threshold selection -> SHAP explanations
    arrow(
        ax,
        (mod_boxes[2][0] + mod_boxes[2][2] / 2, mod_boxes[2][1]),
        (exp_boxes[0][0] + exp_boxes[0][2] * 0.3, exp_boxes[0][1] + exp_boxes[0][3]),
        color=ARROW,
        connectionstyle="arc3,rad=0.1",
    )

    # --- Row 4: outputs -----------------------------------------------------
    out_y, out_h = 0.35, 1.55
    outputs = [
        (0.3, 3.55, "Internal performance", "AUROC, AUPRC,\ncalibration on\nKYRBS test split", "#1f5aa6"),
        (4.15, 3.9, "Stage-specific\npredictors", "top ranked SHAP\npredictors per\ntransition", "#1f8a72"),
        (8.35, 4.35, "External\ntransportability", "AUROC, AUPRC on\nYRBSS 2023,\nfrozen models", "#c9691f"),
    ]
    out_boxes = []
    for x, w, title, sub, edge in outputs:
        b = draw_module(ax, x, out_y, w, out_h, title, sub, facecolor="#f8f9fb", edgecolor=edge, title_size=10.3, sub_size=8.4)
        out_boxes.append(b)

    # model development (test split) -> internal performance, routed via left margin (elbow, avoids all boxes)
    corridor_x = 0.82
    split_mid_x = mod_boxes[1][0] + mod_boxes[1][2] / 2
    stub_y = dev_band[1] - 0.10
    ax.plot([split_mid_x, split_mid_x], [mod_boxes[1][1], stub_y], color=ARROW, lw=1.15, zorder=2)
    ax.plot([split_mid_x, corridor_x], [stub_y, stub_y], color=ARROW, lw=1.15, zorder=2)
    ax.plot([corridor_x, corridor_x], [stub_y, out_y + out_h + 0.06], color=ARROW, lw=1.15, zorder=2)
    arrow(ax, (corridor_x, out_y + out_h + 0.06), (corridor_x, out_y + out_h), color=ARROW, lw=1.15, scale=9)

    # explainability -> stage-specific predictors
    arrow(
        ax,
        (exp_boxes[1][0] + exp_boxes[1][2] / 2, exp_boxes[1][1]),
        (out_boxes[1][0] + out_boxes[1][2] / 2, out_y + out_h),
        color=ARROW,
    )
    # external validation -> external transportability
    arrow(
        ax,
        (ext_boxes[1][0] + ext_boxes[1][2] / 2, ext_boxes[1][1]),
        (out_boxes[2][0] + out_boxes[2][2] / 2, out_y + out_h),
        color=ARROW,
    )

    ax.text(0.3, out_y + out_h + 0.14, "OUTPUTS", fontsize=9.6, fontweight="bold", color="#5a616b", ha="left", va="bottom", style="italic")
    fig.tight_layout(pad=0.3)
    savefig(fig, "Figure1_preprocessing_workflow.png")


def load_curve_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    internal = pd.read_csv(INTERNAL_PRED)
    external = pd.read_csv(EXTERNAL_PRED, encoding="utf-8-sig")
    return internal, external


def plot_roc_pr(kind: str) -> None:
    """Visual redesign only: same ROC/PR curve data (roc_curve/precision_recall_curve
    computed from the same predictions), same AUROC/AUPRC, same model set. Only line
    width, transparency, label sizes, legend, spines, and gridlines are restyled to
    match the finalized Fig. 4 style."""
    internal, external = load_curve_data()
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 3.15), sharex=False, sharey=False)
    for idx, (transition, model) in enumerate(SELECTED.items()):
        ax = axes[idx]
        dint_all = internal[internal["transition"] == transition].copy()
        dint = dint_all[dint_all["model"] == model].copy()
        dext = external[(external["transition"] == transition) & (external["model"] == model)].copy()
        if kind == "roc":
            for model_i in MODEL_LABELS:
                dm = dint_all[dint_all["model"] == model_i]
                if dm.empty:
                    continue
                fpr_m, tpr_m, _ = roc_curve(
                    dm["y_true"], dm["predicted_probability"], sample_weight=dm["survey_weight"]
                )
                auc_m = roc_auc_score(
                    dm["y_true"], dm["predicted_probability"], sample_weight=dm["survey_weight"]
                )
                is_selected = model_i == model
                ax.plot(
                    fpr_m,
                    tpr_m,
                    lw=1.0 if is_selected else 0.55,
                    alpha=0.9 if is_selected else 0.42,
                    color=MODEL_COLORS[model_i],
                    label=MODEL_LABELS[model_i],
                    zorder=3 if is_selected else 2,
                )
            fpr_e, tpr_e, _ = roc_curve(dext["y_true"], dext["proba_original"], sample_weight=dext["weight"])
            auc_e = roc_auc_score(dext["y_true"], dext["proba_original"], sample_weight=dext["weight"])
            ax.plot(
                fpr_e,
                tpr_e,
                lw=1.1,
                color=COL_EXTERNAL,
                ls=(0, (4, 2)),
                label="YRBSS selected",
                zorder=4,
            )
            ax.plot([0, 1], [0, 1], ls=(0, (1.5, 1.5)), lw=0.7, color="#a3a3a3", zorder=1)
            ax.set_xlabel("False positive rate", fontsize=8.5)
            if idx == 0:
                ax.set_ylabel("True positive rate", fontsize=8.5)
            ax.set_xlim(-0.02, 1.02)
            ax.set_ylim(-0.02, 1.02)
        else:
            for model_i in MODEL_LABELS:
                dm = dint_all[dint_all["model"] == model_i]
                if dm.empty:
                    continue
                prec_m, rec_m, _ = precision_recall_curve(
                    dm["y_true"], dm["predicted_probability"], sample_weight=dm["survey_weight"]
                )
                ap_m = average_precision_score(
                    dm["y_true"], dm["predicted_probability"], sample_weight=dm["survey_weight"]
                )
                is_selected = model_i == model
                ax.plot(
                    rec_m,
                    prec_m,
                    lw=1.0 if is_selected else 0.55,
                    alpha=0.9 if is_selected else 0.42,
                    color=MODEL_COLORS[model_i],
                    label=MODEL_LABELS[model_i],
                    zorder=3 if is_selected else 2,
                )
            prec_e, rec_e, _ = precision_recall_curve(dext["y_true"], dext["proba_original"], sample_weight=dext["weight"])
            ap_e = average_precision_score(dext["y_true"], dext["proba_original"], sample_weight=dext["weight"])
            ax.plot(
                rec_e,
                prec_e,
                lw=1.1,
                color=COL_EXTERNAL,
                ls=(0, (4, 2)),
                label="YRBSS selected",
                zorder=4,
            )
            ax.set_xlabel("Recall", fontsize=8.5)
            if idx == 0:
                ax.set_ylabel("Precision", fontsize=8.5)
            ax.set_xlim(-0.02, 1.02)
            ax.set_ylim(-0.02, 1.02)
        ax.set_title(SHORT[transition], pad=4, fontweight="bold", fontsize=8.5)
        add_panel_label(ax, chr(ord("A") + idx), fontsize=8.5)
        ax.tick_params(axis="both", labelsize=7.3, length=2.2, width=0.7)
        ax.grid(True, color="#f2f2f2", linewidth=0.45)
        for spine in ax.spines.values():
            spine.set_linewidth(0.7)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, -0.04),
        fontsize=7.6,
        handlelength=1.4,
        columnspacing=1.1,
    )
    fig.tight_layout(w_pad=1.2, pad=0.6, rect=[0, 0.13, 1, 1])
    stem = "Figure2_roc_curves" if kind == "roc" else "Figure3_precision_recall_curves"
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight", facecolor="white", edgecolor="none")
    savefig(fig, f"{stem}.png")


def figure4_calibration() -> None:
    """Visual redesign only: same calibration_curve_points.csv values, same bins,
    same transitions/datasets. Only line width, marker size, label sizes, legend,
    spines, and gridlines are restyled."""
    cal = pd.read_csv(CAL_POINTS, encoding="utf-8-sig")
    cal = cal[cal["method"] == "as_is_frozen"].copy()
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.85), sharex=False, sharey=False)
    for idx, transition in enumerate(SELECTED):
        ax = axes[idx]
        for dataset, color, marker in [
            ("KYRBS internal test", COL_INTERNAL, "o"),
            ("YRBSS external validation", COL_EXTERNAL, "s"),
        ]:
            d = cal[(cal["transition"] == transition) & (cal["dataset"] == dataset)].sort_values("mean_predicted")
            ax.plot(
                d["mean_predicted"],
                d["observed"],
                marker=marker,
                ms=3.2,
                mew=0,
                lw=1.2,
                color=color,
                label="KYRBS" if dataset.startswith("KYRBS") else "YRBSS",
                zorder=3,
            )
        ax.plot([0, 1], [0, 1], ls=(0, (1.5, 1.5)), lw=0.8, color="#a3a3a3", label="Ideal", zorder=1)
        ax.set_title(SHORT[transition], pad=4, fontweight="bold", fontsize=8.5)
        ax.set_xlabel("Mean predicted risk", fontsize=8.5)
        if idx == 0:
            ax.set_ylabel("Observed event rate", fontsize=8.5)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.tick_params(axis="both", labelsize=7.3, length=2.2, width=0.7)
        ax.grid(True, color="#f2f2f2", linewidth=0.45, zorder=0)
        for spine in ax.spines.values():
            spine.set_linewidth(0.7)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        add_panel_label(ax, chr(ord("A") + idx), fontsize=8.5)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center", ncol=3, frameon=False,
        bbox_to_anchor=(0.5, -0.06), fontsize=7.6, handlelength=1.4, columnspacing=1.3,
    )
    fig.tight_layout(w_pad=1.4, pad=0.5, rect=[0, 0.04, 1, 1])
    fig.savefig(OUT / "Figure4_calibration_curves.pdf", bbox_inches="tight", facecolor="white", edgecolor="none")
    savefig(fig, "Figure4_calibration_curves.png")


def original_variable(feature_name: str, selected: list[str]) -> str:
    name = feature_name.split("__", 1)[-1]
    if name.startswith("missingindicator_"):
        name = name.replace("missingindicator_", "", 1)
    for var in sorted(selected, key=len, reverse=True):
        if name == var or name.startswith(f"{var}_"):
            return var
    return name


def shap_values_positive(explainer: shap.TreeExplainer, x_matrix: np.ndarray) -> np.ndarray:
    values = explainer.shap_values(x_matrix, check_additivity=False)
    if isinstance(values, list):
        values = values[1] if len(values) > 1 else values[0]
    values = np.asarray(values)
    if values.ndim == 3:
        values = values[:, :, 1]
    return values


SHAP_SAMPLE_CAP = {
    "T1_ideation_vs_none": 5000,
    "T2_plan_given_ideation": 4082,
    "T3_attempt_given_plan": 1259,
}


def compute_shap_panels() -> dict[str, dict[str, object]]:
    rng = np.random.default_rng(20260709)
    data = pd.read_csv(KDATA)
    pred = pd.read_csv(INTERNAL_PRED)
    panels: dict[str, dict[str, object]] = {}
    for transition, model_name in SELECTED.items():
        bundle_path = INTERNAL_DIR / "models" / f"{transition}__{model_name}.joblib"
        bundle = joblib.load(bundle_path)
        pipe = bundle["pipeline"]
        selected = bundle["selected_features"]
        rows = pred[(pred["transition"] == transition) & (pred["model"] == model_name)]["row_id"].astype(int).to_numpy()
        cap = SHAP_SAMPLE_CAP.get(transition, 3000)
        if len(rows) > cap:
            rows = rng.choice(rows, size=cap, replace=False)
        x_raw = data.iloc[rows][selected].copy()
        pre = pipe.named_steps["preprocess"]
        model = pipe.named_steps["model"]
        x_trans = pre.transform(x_raw)
        if hasattr(x_trans, "toarray"):
            x_trans = x_trans.toarray()
        feature_names = list(pre.get_feature_names_out())
        explainer = shap.TreeExplainer(model)
        values = shap_values_positive(explainer, x_trans)

        groups: dict[str, list[int]] = {}
        for j, fname in enumerate(feature_names):
            groups.setdefault(original_variable(fname, selected), []).append(j)
        agg = {}
        for var, cols in groups.items():
            agg[var] = values[:, cols].sum(axis=1)
        agg_df = pd.DataFrame(agg)
        panels[transition] = {
            "model": model_name,
            "x_raw": x_raw,
            "agg": agg_df,
            "importance": agg_df.abs().mean(),
        }
    return panels


def _feature_color_values(raw_col: pd.Series) -> np.ndarray:
    """Min-max normalize (not rank) so binary/skewed predictors map to the true
    Low/High extremes of the colormap rather than compressing toward the middle."""
    coded = pd.to_numeric(raw_col, errors="coerce")
    if coded.notna().sum() == 0:
        coded = raw_col.astype("category").cat.codes.replace(-1, np.nan)
    coded = coded.astype(float)
    vmin, vmax = coded.min(), coded.max()
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        return np.full(len(coded), 0.5)
    scaled = (coded - vmin) / (vmax - vmin)
    return np.clip(scaled.fillna(0.5).to_numpy(), 0, 1)


def _row_jitter(values: np.ndarray, rng: np.random.Generator, bins: int = 150, row_frac: float = 0.38) -> np.ndarray:
    """Density-normalized jitter shared identically by every row in every panel: the
    row's own densest bin always fills to row_frac, so visual "fullness" is comparable
    across T1/T2/T3 regardless of how discrete or continuous the SHAP values happen to
    be for a given model. Placement within a bin is random (not a sorted ladder), which
    avoids the diagonal stripe/hatching artifact a deterministic ladder produces on
    long-tailed rows. Only the y-offset (display only) is computed here; x stays exactly
    the plotted SHAP value."""
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n == 0:
        return np.array([])
    vmin, vmax = np.nanmin(values), np.nanmax(values)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        return rng.normal(0, 0.02, size=n)
    edges = np.linspace(vmin, vmax, bins + 1)
    bin_ids = np.clip(np.digitize(values, edges) - 1, 0, bins - 1)
    counts = np.bincount(bin_ids, minlength=bins)
    max_count = max(int(counts.max()), 1)
    offsets = np.zeros(n)
    for b in range(bins):
        idx = np.where(bin_ids == b)[0]
        k = len(idx)
        if k == 0:
            continue
        spread = row_frac * math.sqrt(k / max_count)
        offsets[idx] = rng.uniform(-spread, spread, size=k)
    return offsets


def _draw_shap_panel(fig, ax, cax, transition: str, panel: dict) -> None:
    model_name = panel["model"]
    x_raw = panel["x_raw"]
    agg_df = panel["agg"]
    top_vars = panel["importance"].sort_values(ascending=False).head(10).index.tolist()
    plot_vars = top_vars[::-1]  # most important ends up drawn at the top row

    rng = np.random.default_rng(20260709)
    all_vals = agg_df[top_vars].to_numpy()
    data_min = float(np.nanmin(all_vals))
    data_max = float(np.nanmax(all_vals))
    margin = 0.06 * max(data_max - data_min, 1e-9)

    for yi, var in enumerate(plot_vars):
        vals = agg_df[var].to_numpy()
        yy = yi + _row_jitter(vals, rng)
        colors = _feature_color_values(x_raw[var])
        ax.scatter(
            vals, yy,
            c=colors, cmap=SHAP_CMAP, vmin=0, vmax=1,
            s=0.6, alpha=0.75, linewidths=0, zorder=3,
        )

    ax.axvline(0, color="#8a8a8a", lw=0.8, zorder=1)
    ax.set_yticks(range(len(plot_vars)))
    ax.set_yticklabels([VAR_LABELS.get(v, v) for v in plot_vars])
    ax.set_ylim(-0.7, len(plot_vars) - 0.3)
    ax.set_xlim(data_min - margin, data_max + margin)
    ax.set_facecolor("white")
    _panel_model_display = {"lightgbm": "LightGBM", "xgboost": "XGBoost", "random_forest": "random forest"}
    display_name = _panel_model_display.get(model_name, model_name.replace("_", " "))
    ax.set_title(f"{SHORT[transition]} ({display_name})", pad=7, fontweight="bold", fontsize=11.3)
    ax.set_xlabel("SHAP value", fontsize=10.3)
    ax.tick_params(axis="y", labelsize=8.8, length=0)
    ax.tick_params(axis="x", labelsize=8.8)
    ax.grid(True, axis="y", color="#eeeeee", linewidth=0.5, zorder=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_linewidth(0.8)

    cax.set_facecolor("white")
    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    cb = mpl.colorbar.ColorbarBase(cax, cmap=SHAP_CMAP, norm=norm)
    cb.set_ticks([0, 1])
    cb.set_ticklabels(["Low", "High"])
    cb.ax.tick_params(labelsize=8.2, length=2, pad=2)
    cb.outline.set_linewidth(0.4)
    cb.set_label("Feature value", fontsize=8.4, labelpad=6)


def figure5_shap() -> None:
    """Visual redesign only: same compute_shap_panels() output, same independent
    per-transition top-10 selection, same real SHAP values as the prior release.
    Vertically stacked layout, one panel per row (T1 top, T2 middle, T3 bottom),
    each with its own slim colorbar to the right."""
    panels = compute_shap_panels()

    fig = plt.figure(figsize=(9.6, 12.0))
    fig.patch.set_facecolor("white")
    fig.patch.set_edgecolor("none")
    gs = fig.add_gridspec(
        3, 2,
        width_ratios=[1, 0.032],
        height_ratios=[1, 1, 1],
        wspace=0.05,
        hspace=0.5,
        left=0.20, right=0.90, top=0.965, bottom=0.058,
    )

    positions = {
        "T1_ideation_vs_none": 0,
        "T2_plan_given_ideation": 1,
        "T3_attempt_given_plan": 2,
    }
    for transition, panel in panels.items():
        row = positions[transition]
        ax = fig.add_subplot(gs[row, 0])
        cax = fig.add_subplot(gs[row, 1])
        _draw_shap_panel(fig, ax, cax, transition, panel)

    fig.savefig(OUT / "Figure5_shap_beeswarm.pdf", bbox_inches="tight", facecolor="white", edgecolor="none")
    savefig(fig, "Figure5_shap_beeswarm.png")


def main() -> None:
    figure1_workflow()
    plot_roc_pr("roc")
    plot_roc_pr("pr")
    figure4_calibration()
    figure5_shap()


if __name__ == "__main__":
    main()
