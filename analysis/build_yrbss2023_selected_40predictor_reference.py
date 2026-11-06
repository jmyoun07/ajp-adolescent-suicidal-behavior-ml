#!/usr/bin/env python3
"""Build a YRBSS 2023 selected dataset guided by the final KYRBS 40 predictors."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
YRBSS_DIR = REPO_ROOT / "data" / "yrbss_raw"
SPS_PATH = YRBSS_DIR / "2023XXH-SPSS.sps"
DAT_PATH = YRBSS_DIR / "XXH2023_YRBS_Data.dat"
KYRBS_STRUCTURE_PATH = REPO_ROOT / "metadata" / "kyrbs2020_2024_40predictor_structure.csv"
OUT_DIR = REPO_ROOT / "results" / "yrbss2023_selected_40predictor_reference"

DESIGN_COLS = ["STRATA", "CLUSTER", "W", "FPC"]
OUTCOME_SOURCE_COLS = ["M_SUI_CON", "M_SUI_PLN", "M_SUI_ATT"]
DERIVED_OUTCOME_COLS = [
    "sui_ideation",
    "sui_plan",
    "sui_attempt",
    "sui_any",
    "sui_stage_highest",
    "sui_pattern_inconsistent",
]
PROXY_COLS = ["social_media_frequency_proxy", "violence_exposure_proxy"]


def parse_sps(sps_path: Path) -> tuple[list[tuple[str, int, int, bool]], dict[str, str]]:
    text = sps_path.read_text(encoding="latin-1")
    data_block = text.split("DATA LIST", 1)[1].split("EXECUTE.", 1)[0]
    specs = []
    for name, start, end, alpha in re.findall(r"(\w+)\s+(\d+)-(\d+)(?:\s+\((A)\))?", data_block):
        specs.append((name, int(start), int(end), bool(alpha)))
    label_block = text.split("VARIABLE LABELS", 1)[1].split("VALUE LABELS", 1)[0]
    labels = dict(re.findall(r'(\w+)\s+"([^"]*)"', label_block))
    return specs, labels


def read_yrbss_fixed_width(specs: list[tuple[str, int, int, bool]]) -> pd.DataFrame:
    colspecs = [(start - 1, end) for _, start, end, _ in specs]
    names = [name for name, _, _, _ in specs]
    df = pd.read_fwf(DAT_PATH, colspecs=colspecs, names=names, dtype=str, keep_default_na=False)
    alpha_cols = {name for name, _, _, alpha in specs if alpha}
    alpha_cols.update({"q6orig", "q7orig", "site"})
    for col in df.columns:
        stripped = df[col].astype(str).str.strip()
        if col in alpha_cols:
            df[col] = stripped.replace("", pd.NA)
        else:
            df[col] = pd.to_numeric(stripped.replace("", np.nan), errors="coerce")
    return df


def yes_no_yrbss(series: pd.Series) -> pd.Series:
    return pd.Series(np.select([series.eq(1), series.eq(2)], [1, 0], default=np.nan), index=series.index)


def yes_no_to_kyrbs_1no_2yes(series: pd.Series) -> pd.Series:
    return pd.Series(np.select([series.eq(1), series.eq(2)], [2, 1], default=np.nan), index=series.index)


def attempt_to_kyrbs(series: pd.Series) -> pd.Series:
    # KYRBS-like suicide attempt source: 1=no, 2=yes.
    return pd.Series(np.select([series.eq(1), series.between(2, 5)], [1, 2], default=np.nan), index=series.index)


def current_any_from_days(series: pd.Series) -> pd.Series:
    return pd.Series(np.select([series.eq(1), series.between(2, 7)], [0, 1], default=np.nan), index=series.index)


def current_tobacco_type(df: pd.DataFrame) -> pd.Series:
    cig = current_any_from_days(df["Q33"])
    evp = current_any_from_days(df["Q36"])
    out = pd.Series(np.nan, index=df.index)
    valid = cig.notna() & evp.notna()
    out.loc[valid & cig.eq(0) & evp.eq(0)] = 0
    out.loc[valid & cig.eq(1) & evp.eq(0)] = 1
    out.loc[valid & cig.eq(0) & evp.eq(1)] = 2
    out.loc[valid & cig.eq(1) & evp.eq(1)] = 3
    return out


def alcohol_days_kyrbs_skip(df: pd.DataFrame) -> pd.Series:
    out = pd.Series(np.nan, index=df.index)
    out.loc[df["Q41"].eq(1)] = 9999
    out.loc[df["Q42"].between(1, 7)] = df.loc[df["Q42"].between(1, 7), "Q42"]
    return out


def alcohol_amount_kyrbs_skip(df: pd.DataFrame) -> pd.Series:
    out = pd.Series(np.nan, index=df.index)
    no_current = df["Q41"].eq(1) | df["Q42"].eq(1) | df["Q44"].eq(1)
    out.loc[no_current] = 9999
    out.loc[df["Q44"].between(2, 5)] = df.loc[df["Q44"].between(2, 5), "Q44"] - 1
    out.loc[df["Q44"].between(6, 8)] = 5
    return out


def alcohol_drunk_kyrbs_skip(df: pd.DataFrame) -> pd.Series:
    out = pd.Series(np.nan, index=df.index)
    no_current = df["Q41"].eq(1) | df["Q42"].eq(1)
    out.loc[no_current] = 9999
    out.loc[df["Q43"].eq(1)] = 1
    out.loc[df["Q43"].eq(2)] = 2
    out.loc[df["Q43"].eq(3)] = 3
    out.loc[df["Q43"].between(4, 7)] = 4
    return out


def cigarette_amount_kyrbs_skip(df: pd.DataFrame) -> pd.Series:
    out = pd.Series(np.nan, index=df.index)
    no_current = df["Q31"].eq(2) | df["Q33"].eq(1) | df["Q34"].eq(1)
    out.loc[no_current] = 9999
    out.loc[df["Q34"].between(2, 7)] = df.loc[df["Q34"].between(2, 7), "Q34"] - 1
    return out


def any_drug_use_ever(df: pd.DataFrame) -> pd.Series:
    # YRBSS drug items use 1=0 times and larger values=ever used.
    drug_cols = ["Q49", "Q50", "Q51", "Q52", "Q53", "Q54", "Q55", "Q92", "Q93"]
    ever_matrix = pd.DataFrame(index=df.index)
    for col in drug_cols:
        ever_matrix[col] = np.select([df[col].eq(1), df[col].gt(1)], [0, 1], default=np.nan)
    out = ever_matrix.max(axis=1, skipna=True)
    out[ever_matrix.isna().all(axis=1)] = np.nan
    # KYRBS drug_use_ever is 1=no, 2=yes in current project convention.
    return pd.Series(np.select([out.eq(0), out.eq(1)], [1, 2], default=np.nan), index=df.index)


def weekly_frequency_to_kyrbs(series: pd.Series) -> pd.Series:
    return series.where(series.between(1, 7), np.nan)


def social_media_to_daily_minutes_proxy(series: pd.Series) -> pd.Series:
    # Not equivalent to smartphone minutes. Midpoints are retained only as a documented proxy.
    midpoint = {
        1: 0.0,
        2: 1.0 / 30.0 * 5,
        3: 1.0 / 7.0 * 5,
        4: 3.0 / 7.0 * 5,
        5: 5.0,
        6: 30.0,
        7: 60.0,
        8: 120.0,
    }
    return series.map(midpoint)


def build_selected(df: pd.DataFrame, predictors: list[str]) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)

    out["STRATA"] = df["stratum"]
    out["CLUSTER"] = df["psu"]
    out["W"] = df["weight"]
    out["FPC"] = np.nan

    out["M_SUI_CON"] = yes_no_to_kyrbs_1no_2yes(df["Q27"])
    out["M_SUI_PLN"] = yes_no_to_kyrbs_1no_2yes(df["Q28"])
    out["M_SUI_ATT"] = attempt_to_kyrbs(df["Q29"])

    out["sui_ideation"] = yes_no_yrbss(df["Q27"])
    out["sui_plan"] = yes_no_yrbss(df["Q28"])
    out["sui_attempt"] = pd.Series(np.select([df["Q29"].eq(1), df["Q29"].between(2, 5)], [0, 1], default=np.nan), index=df.index)
    out["sui_any"] = out[["sui_ideation", "sui_plan", "sui_attempt"]].max(axis=1, skipna=True)
    out.loc[out[["sui_ideation", "sui_plan", "sui_attempt"]].isna().all(axis=1), "sui_any"] = np.nan
    out["sui_stage_highest"] = np.nan
    out.loc[out["sui_ideation"].eq(0), "sui_stage_highest"] = 0
    out.loc[out["sui_ideation"].eq(1), "sui_stage_highest"] = 1
    out.loc[out["sui_plan"].eq(1), "sui_stage_highest"] = 2
    out.loc[out["sui_attempt"].eq(1), "sui_stage_highest"] = 3
    out["sui_pattern_inconsistent"] = (
        (out["sui_plan"].eq(1) & out["sui_ideation"].eq(0))
        | (out["sui_attempt"].eq(1) & out["sui_plan"].eq(0))
    ).astype(float)
    out.loc[out[["sui_ideation", "sui_plan", "sui_attempt"]].isna().any(axis=1), "sui_pattern_inconsistent"] = np.nan

    for predictor in predictors:
        out[predictor] = np.nan

    out["SEX"] = pd.Series(np.select([df["Q2"].eq(2), df["Q2"].eq(1)], [1, 2], default=np.nan), index=df.index)
    out["AGE_M"] = df["Q1"].map({1: 144, 2: 156, 3: 168, 4: 180, 5: 192, 6: 204, 7: 216})
    out["GRADE"] = df["Q3"].map({1: 3, 2: 4, 3: 5, 4: 6})
    out["SCHOOL"] = pd.Series(pd.NA, index=df.index, dtype="object")
    out.loc[df["Q3"].isin([1, 2, 3, 4]), "SCHOOL"] = "일반계고"
    out["E_S_RCRD"] = df["Q87"].where(df["Q87"].between(1, 5), np.nan)
    out["E_RES"] = df["Q86"].map({1: 1, 2: 3, 3: 5, 4: 5, 5: 5, 6: 5, 7: 5})
    out["PR_BI"] = df["Q66"].where(df["Q66"].between(1, 5), np.nan)
    out["adolescent_bmi_category"] = pd.cut(
        df["bmipct"],
        bins=[-np.inf, 5, 85, 95, np.inf],
        labels=[1, 2, 3, 4],
    ).astype(float)
    out["WC_MN"] = df["Q67"].where(df["Q67"].between(1, 4), np.nan)
    # No direct hospital-treatment item exists. Q16 is retained as a low-fidelity violence exposure proxy.
    out["V_TRT"] = df["Q16"].where(df["Q16"].between(1, 8), np.nan)
    out["M_SAD"] = yes_no_to_kyrbs_1no_2yes(df["Q26"])
    out["M_LON"] = df["Q84"].where(df["Q84"].between(1, 5), np.nan)
    out["M_SLP_EN"] = df["Q85"].map({1: 5, 2: 4, 3: 3, 4: 2, 5: 1, 6: 1, 7: 1})
    out["sleep_daily_min"] = df["Q85"].map({1: 240, 2: 300, 3: 360, 4: 420, 5: 480, 6: 540, 7: 600})
    out["smartphone_daily_min"] = social_media_to_daily_minutes_proxy(df["Q80"])
    out["AC_DAYS"] = alcohol_days_kyrbs_skip(df)
    out["AC_AMNT"] = alcohol_amount_kyrbs_skip(df)
    out["AC_DRUNK"] = alcohol_drunk_kyrbs_skip(df)
    out["current_tobacco_type"] = current_tobacco_type(df)
    out["TC_AMNT"] = cigarette_amount_kyrbs_skip(df)
    out["drug_use_ever"] = any_drug_use_ever(df)
    out["PA_TOT"] = df["Q76"].where(df["Q76"].between(1, 8), np.nan)
    out["vigorous_activity_days"] = df["Q76"].where(df["Q76"].between(1, 8), np.nan)
    out["PA_MSC"] = df["Q97"].where(df["Q97"].between(1, 8), np.nan)
    out["F_BR"] = df["Q75"].where(df["Q75"].between(1, 8), np.nan)
    out["F_FRUIT"] = weekly_frequency_to_kyrbs(df["Q69"])
    out["sweet_drink_freq"] = weekly_frequency_to_kyrbs(df["Q74"])
    out["F_WAT"] = df["Q96"].where(df["Q96"].between(1, 7), np.nan)

    out["social_media_frequency_proxy"] = df["Q80"].where(df["Q80"].between(1, 8), np.nan)
    out["violence_exposure_proxy"] = df["Q16"].where(df["Q16"].between(1, 8), np.nan)

    return out[DESIGN_COLS + OUTCOME_SOURCE_COLS + DERIVED_OUTCOME_COLS + predictors + PROXY_COLS]


def pct_missing(series: pd.Series) -> float:
    return float(series.isna().mean() * 100)


def value_summary(series: pd.Series) -> str:
    nonmissing = series.dropna()
    if nonmissing.empty:
        return ""
    if pd.api.types.is_numeric_dtype(nonmissing) and nonmissing.nunique() > 12:
        return f"range {nonmissing.min():.4g}-{nonmissing.max():.4g}; median {nonmissing.median():.4g}; mean {nonmissing.mean():.4g}"
    counts = nonmissing.value_counts(dropna=True).sort_index()
    return "; ".join(f"{idx}={cnt:,}" for idx, cnt in counts.items())


def mapping_rows(harmonized: pd.DataFrame, predictors: list[str]) -> pd.DataFrame:
    direct = {
        "SEX": ("Q2", "High", "YRBSS sex recoded to KYRBS convention: 1=male, 2=female."),
        "AGE_M": ("Q1", "Moderate", "Age category converted to approximate month age: 12-18 years as 144-216 months."),
        "GRADE": ("Q3", "Moderate", "YRBSS grades 9/10/11/12 mapped to KYRBS grade-comparable 3/4/5/6."),
        "SCHOOL": ("Q3", "Partial", "YRBSS high-school sample coded as 일반계고 where grade is 9-12."),
        "E_S_RCRD": ("Q87", "Moderate", "School grades A-F mapped to 1-5; none/not sure set missing."),
        "E_RES": ("Q86", "Low", "Housing status mapped roughly to KYRBS residence categories."),
        "PR_BI": ("Q66", "High", "Perceived weight mapped directly to 1-5."),
        "adolescent_bmi_category": ("bmipct", "Moderate", "CDC BMI percentile category: <5th, 5-<85th, 85-<95th, >=95th."),
        "WC_MN": ("Q67", "Moderate", "Weight intention retained as 1-4."),
        "V_TRT": ("Q16", "Low proxy", "YRBSS lacks violence-related hospital treatment; physical fight frequency used only as proxy."),
        "M_SAD": ("Q26", "High", "Sad/hopeless binary recoded to KYRBS 1=no, 2=yes."),
        "M_LON": ("Q84", "Low proxy", "YRBSS current mental health frequency used as closest mental-health frequency proxy, not loneliness."),
        "M_SLP_EN": ("Q85", "Low proxy", "Sleep hours reverse-mapped to sleep sufficiency proxy."),
        "sleep_daily_min": ("Q85", "Moderate", "School-night sleep categories converted to minutes."),
        "smartphone_daily_min": ("Q80", "Low proxy", "YRBSS social media frequency converted to approximate minutes; not equivalent to smartphone use minutes."),
        "AC_DAYS": ("Q41; Q42", "High", "Past-30-day alcohol use days retained as ordinal 1-7; never-drinkers recoded to KYRBS skip code 9999."),
        "AC_AMNT": ("Q41; Q42; Q44", "Moderate", "No current drinking recoded to 9999; YRBSS drink amount categories collapsed to KYRBS-compatible 1-5."),
        "AC_DRUNK": ("Q41; Q42; Q43", "Moderate", "No current drinking recoded to 9999; binge-drinking days collapsed to KYRBS-compatible 1-4."),
        "current_tobacco_type": ("Q33; Q36", "Moderate", "Derived from current cigarette and electronic vapor product use."),
        "TC_AMNT": ("Q31; Q33; Q34", "Moderate", "Non-current cigarette smoking recoded to 9999; daily cigarette amount collapsed to KYRBS-compatible 1-6."),
        "drug_use_ever": ("Q49-Q55; Q92-Q93", "Moderate", "Any selected lifetime/current illicit or nonmedical drug use recoded to 1=no, 2=yes."),
        "PA_TOT": ("Q76", "High", "60+ minutes physical activity days retained as ordinal 1-8."),
        "vigorous_activity_days": ("Q76", "Low proxy", "No vigorous-specific item; total physical activity days used as proxy."),
        "PA_MSC": ("Q97", "High", "Muscle strengthening days retained as ordinal 1-8."),
        "F_BR": ("Q75", "High", "Breakfast days retained as ordinal 1-8."),
        "F_FRUIT": ("Q69", "Moderate", "Fruit frequency retained as ordinal 1-7."),
        "sweet_drink_freq": ("Q74", "Moderate", "Soda/pop frequency used as narrowed sweet-drink analogue."),
        "F_WAT": ("Q96", "Moderate", "Plain water frequency retained as ordinal 1-7."),
    }
    rows = []
    for predictor in predictors:
        source, level, rule = direct.get(
            predictor,
            ("", "Unavailable", "No sufficiently direct 2023 national YRBSS counterpart; retained as missing for schema compatibility."),
        )
        rows.append(
            {
                "kyrbs_variable": predictor,
                "yrbss_source": source,
                "mapping_level": level,
                "harmonization_rule": rule,
                "missing_rate_percent": round(pct_missing(harmonized[predictor]), 3),
                "missing_count": int(harmonized[predictor].isna().sum()),
                "value_summary": value_summary(harmonized[predictor]),
            }
        )
    return pd.DataFrame(rows)


def stage_counts(harmonized: pd.DataFrame) -> pd.DataFrame:
    rows = []
    specs = [
        ("T1_ideation_vs_none", harmonized["sui_pattern_inconsistent"].eq(0), "sui_ideation"),
        ("T2_plan_given_ideation", harmonized["sui_pattern_inconsistent"].eq(0) & harmonized["sui_ideation"].eq(1), "sui_plan"),
        ("T3_attempt_given_plan", harmonized["sui_pattern_inconsistent"].eq(0) & harmonized["sui_plan"].eq(1), "sui_attempt"),
    ]
    for transition, mask, outcome in specs:
        subset = harmonized.loc[mask]
        rows.append(
            {
                "transition": transition,
                "analysis_n": int(mask.sum()),
                "event_n": int(subset[outcome].eq(1).sum()),
                "unweighted_prevalence": float(subset[outcome].mean()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    specs, labels = parse_sps(SPS_PATH)
    raw = read_yrbss_fixed_width(specs)
    kyrbs_structure = pd.read_csv(KYRBS_STRUCTURE_PATH)
    predictors = kyrbs_structure["variable"].astype(str).tolist()

    selected = build_selected(raw, predictors)
    mapping = mapping_rows(selected, predictors)
    stages = stage_counts(selected)

    structure = pd.DataFrame(
        {
            "variable": selected.columns,
            "role": [
                "survey_design" if c in DESIGN_COLS else
                "outcome_source" if c in OUTCOME_SOURCE_COLS else
                "derived_outcome" if c in DERIVED_OUTCOME_COLS else
                "proxy_auxiliary" if c in PROXY_COLS else
                "selected_predictor"
                for c in selected.columns
            ],
            "yrbss_label": [labels.get(c, "") for c in selected.columns],
            "missing_rate_percent": [round(pct_missing(selected[c]), 3) for c in selected.columns],
            "missing_count": [int(selected[c].isna().sum()) for c in selected.columns],
            "value_summary": [value_summary(selected[c]) for c in selected.columns],
        }
    )

    selected.to_csv(OUT_DIR / "yrbss2023_selected_40predictor_reference_dataset.csv", index=False, encoding="utf-8-sig")
    selected[DESIGN_COLS + OUTCOME_SOURCE_COLS + DERIVED_OUTCOME_COLS + predictors].to_csv(
        OUT_DIR / "yrbss2023_selected_40predictor_model_input.csv", index=False, encoding="utf-8-sig"
    )
    selected[predictors].to_csv(OUT_DIR / "yrbss2023_selected_40predictors_only.csv", index=False, encoding="utf-8-sig")
    mapping.to_csv(OUT_DIR / "yrbss2023_selected_40predictor_mapping_table.csv", index=False, encoding="utf-8-sig")
    structure.to_csv(OUT_DIR / "yrbss2023_selected_40predictor_structure_table.csv", index=False, encoding="utf-8-sig")
    stages.to_csv(OUT_DIR / "yrbss2023_selected_40predictor_stage_counts.csv", index=False, encoding="utf-8-sig")

    report = [
        "# YRBSS 2023 selected dataset guided by KYRBS final 40 predictors",
        "",
        f"- Raw syntax: `{SPS_PATH}`",
        f"- Raw fixed-width data: `{DAT_PATH}`",
        f"- Reference structure: `{KYRBS_STRUCTURE_PATH}`",
        f"- Rows: {len(selected):,}",
        f"- Selected KYRBS-reference predictors: {len(predictors):,}",
        f"- Total model-input columns: {len(DESIGN_COLS + OUTCOME_SOURCE_COLS + DERIVED_OUTCOME_COLS + predictors):,}",
        "",
        "## Mapping level counts",
        mapping["mapping_level"].value_counts().to_string(),
        "",
        "## Stage counts",
        stages.to_string(index=False),
        "",
        "## Important limitations",
        "- Several KYRBS predictors do not have direct 2023 national YRBSS counterparts and are retained as missing columns for schema compatibility.",
        "- `smartphone_daily_min` is represented only by a documented low-fidelity proxy from YRBSS social media frequency (`Q80`).",
        "- `V_TRT` is represented only by a low-fidelity proxy from physical fight frequency (`Q16`) because YRBSS lacks violence-related hospital-treatment frequency.",
        "- The output is a variable-selection/harmonization dataset, not a claim that every retained variable is perfectly equivalent across surveys.",
    ]
    (OUT_DIR / "yrbss2023_selected_40predictor_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    print("Selected dataset shape:", selected.shape)
    print("Model input shape:", (len(selected), len(DESIGN_COLS + OUTCOME_SOURCE_COLS + DERIVED_OUTCOME_COLS + predictors)))
    print("Predictor-only shape:", selected[predictors].shape)
    print("Output directory:", OUT_DIR)
    print(mapping["mapping_level"].value_counts().to_string())
    print(stages.to_string(index=False))


if __name__ == "__main__":
    main()
