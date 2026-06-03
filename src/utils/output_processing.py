# Copyright (c) 2025 Zichen Zhao
# Columbia University School of Social Work
# Licensed under the MIT Academic Research License
# See LICENSE file in the project root for details.

from __future__ import annotations
import os
import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from src.commonconst import *

def _sanitize_filename(name: str) -> str:
    name = str(name).strip().lower()
    name = re.sub(r"[^\w\s-]", "", name)
    name = re.sub(r"[\s/]+", "_", name)
    return name

def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def _cleanup_plots_directory():
    """Keep Plots/ figure-only by removing stale CSVs and deprecated pass/fail figures."""
    _ensure_dir(PLOTS_DIR)
    for filename in os.listdir(PLOTS_DIR):
        lower_name = filename.lower()
        if lower_name.endswith(".csv") or lower_name == "not_hate_passfail.png":
            try:
                os.remove(os.path.join(PLOTS_DIR, filename))
            except OSError:
                pass

def _remove_overall_average_row(df: pd.DataFrame) -> pd.DataFrame:
    if "Chatbot" not in df.columns:
        return df.copy()
    cleaned_df = df.copy()
    cleaned_df["Chatbot"] = cleaned_df["Chatbot"].astype(str).str.strip()
    return cleaned_df[cleaned_df["Chatbot"].str.lower() != OVERALL_AVERAGE_LABEL.lower()].copy()

def _coerce_metric_column(plot_df: pd.DataFrame, metric: str) -> pd.DataFrame:
    cleaned_df = plot_df.copy()
    cleaned_df[metric] = pd.to_numeric(cleaned_df[metric], errors="coerce")
    cleaned_df = cleaned_df.dropna(subset=[metric])
    return cleaned_df

def append_overall_average_row(df: pd.DataFrame, label: str = OVERALL_AVERAGE_LABEL) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    summary_df = df.copy()
    numeric_cols = summary_df.select_dtypes(include="number").columns.tolist()

    if not numeric_cols:
        return summary_df

    overall_row = {}
    for col in summary_df.columns:
        if col == "Chatbot":
            overall_row[col] = label
        elif col in numeric_cols:
            overall_row[col] = round(float(summary_df[col].mean()), 4)
        else:
            overall_row[col] = ""

    return pd.concat([summary_df, pd.DataFrame([overall_row])], ignore_index=True)


def plot_metric_bar(df: pd.DataFrame, metric: str, output_dir: str):
    if metric not in df.columns:
        print(f"[WARN] Metric '{metric}' not found in dataframe.")
        return

    _ensure_dir(output_dir)

    plot_df = _remove_overall_average_row(df)

    if "Chatbot" not in plot_df.columns:
        print("[WARN] 'Chatbot' column not found in dataframe.")
        return

    plot_df = plot_df[["Chatbot", metric]].copy()
    plot_df = _coerce_metric_column(plot_df, metric)

    if plot_df.empty:
        print(f"[WARN] No non-null numeric values found for '{metric}'.")
        return

    reference_metric_map = {
        "Negative Sentiment Probability": "Reference Negative Sentiment Probability",
        "Flesch Reading Ease": "Reference Flesch Reading Ease",
    }
    reference_value = None
    reference_col = reference_metric_map.get(metric)
    if reference_col and reference_col in df.columns:
        reference_values = pd.to_numeric(df[reference_col], errors="coerce").dropna()
        if not reference_values.empty:
            reference_value = float(reference_values.iloc[0])

    plt.figure(figsize=PLOT_FIGSIZE)
    plt.bar(plot_df["Chatbot"], plot_df[metric])
    if reference_value is not None:
        plt.axhline(
            y=reference_value,
            linestyle="--",
            linewidth=1.8,
            label=f"Human reference = {reference_value:.4f}",
        )
        plt.legend()
    plt.xticks(rotation=ROTATION, ha="right")
    plt.ylabel(metric)
    plt.title(metric)
    plt.tight_layout()

    output_path = os.path.join(output_dir, f"{_sanitize_filename(metric)}.png")
    plt.savefig(output_path, dpi=DPI)
    plt.close()


def plot_not_hate_metric(not_hate_df: pd.DataFrame):
    _ensure_dir(PLOTS_DIR)
    clean_df = _remove_overall_average_row(not_hate_df)

    required_cols = ["Chatbot", "Non-Hateful Language Probability"]
    missing_cols = [col for col in required_cols if col not in clean_df.columns]
    if missing_cols:
        print(f"[WARN] Missing Non-Hateful Language columns: {missing_cols}")
        return

    plot_df = clean_df[["Chatbot", "Non-Hateful Language Probability"]].copy()
    plot_df["Non-Hateful Language Probability"] = pd.to_numeric(
        plot_df["Non-Hateful Language Probability"], errors="coerce"
    )
    plot_df = plot_df.dropna(subset=["Non-Hateful Language Probability"])

    reference_value = None
    if "Reference Non-Hateful Language Probability" in not_hate_df.columns:
        reference_values = pd.to_numeric(
            not_hate_df["Reference Non-Hateful Language Probability"], errors="coerce"
        ).dropna()
        if not reference_values.empty:
            reference_value = float(reference_values.iloc[0])

    if not plot_df.empty:
        plt.figure(figsize=PLOT_FIGSIZE)
        plt.bar(plot_df["Chatbot"], plot_df["Non-Hateful Language Probability"])
        if reference_value is not None:
            plt.axhline(
                y=reference_value,
                linestyle="--",
                linewidth=1.8,
                label=f"Human reference = {reference_value:.4f}",
            )
            plt.legend()
        plt.xticks(rotation=ROTATION, ha="right")
        plt.ylabel("Non-Hateful Language Probability")
        plt.title("Non-Hateful Language Probability")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, "non_hateful_language_probability.png"), dpi=DPI)
        plt.close()
    else:
        print("[WARN] Non-Hateful Language Probability plot skipped because the dataframe is empty.")


def plot_urgency_dimension(urgency_df: pd.DataFrame):
    _ensure_dir(PLOTS_DIR)
    clean_df = _remove_overall_average_row(urgency_df)

    required_cols = ["Chatbot", "Crisis-Response Reference Similarity"]
    missing_cols = [col for col in required_cols if col not in clean_df.columns]
    if missing_cols:
        print(f"[WARN] Missing urgency columns: {missing_cols}")
        return

    plot_df = clean_df[["Chatbot", "Crisis-Response Reference Similarity"]].copy()
    plot_df["Crisis-Response Reference Similarity"] = pd.to_numeric(
        plot_df["Crisis-Response Reference Similarity"], errors="coerce"
    )
    plot_df = plot_df.dropna(subset=["Crisis-Response Reference Similarity"])

    if plot_df.empty:
        print("[WARN] Urgency plot skipped because the dataframe is empty.")
        return

    plt.figure(figsize=PLOT_FIGSIZE)
    plt.bar(plot_df["Chatbot"], plot_df["Crisis-Response Reference Similarity"])
    plt.xticks(rotation=ROTATION, ha="right")
    plt.ylabel("Crisis-Response Reference Similarity")
    plt.title("Crisis-Response Reference Similarity")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "crisis_response_reference_similarity.png"), dpi=DPI)
    plt.close()


def plot_risk_factor_dimension(risk_factor_df: pd.DataFrame):
    _ensure_dir(PLOTS_DIR)
    clean_df = _remove_overall_average_row(risk_factor_df)

    required_cols = ["Chatbot", "Risk-Assessment Reference Similarity"]
    missing_cols = [col for col in required_cols if col not in clean_df.columns]
    if missing_cols:
        print(f"[WARN] Missing risk-assessment columns: {missing_cols}")
        return

    plot_df = clean_df[["Chatbot", "Risk-Assessment Reference Similarity"]].copy()
    plot_df["Risk-Assessment Reference Similarity"] = pd.to_numeric(
        plot_df["Risk-Assessment Reference Similarity"], errors="coerce"
    )
    plot_df = plot_df.dropna(subset=["Risk-Assessment Reference Similarity"])

    if plot_df.empty:
        print("[WARN] Risk-factor plot skipped because the dataframe is empty.")
        return

    plt.figure(figsize=PLOT_FIGSIZE)
    plt.bar(plot_df["Chatbot"], plot_df["Risk-Assessment Reference Similarity"])
    plt.xticks(rotation=ROTATION, ha="right")
    plt.ylabel("Risk-Assessment Reference Similarity")
    plt.title("Risk-Assessment Reference Similarity")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "risk_assessment_reference_similarity.png"), dpi=DPI)
    plt.close()


# backward-compatible wrappers
def plot_identity_dimension(identity_df: pd.DataFrame):
    plot_urgency_dimension(identity_df)


def plot_safety_dimension(safety_df: pd.DataFrame):
    plot_risk_factor_dimension(safety_df)

def build_overall_summary_table(
    evaluation_df: pd.DataFrame,
    not_hate_df: pd.DataFrame | None = None,
    urgency_df: pd.DataFrame | None = None,
    risk_factor_df: pd.DataFrame | None = None,
    identity_df: pd.DataFrame | None = None,
    safety_df: pd.DataFrame | None = None,
    include_overall_average: bool = False,
) -> pd.DataFrame:
    summary_df = _remove_overall_average_row(evaluation_df).copy()
    if "Response" in summary_df.columns:
        summary_df = summary_df.drop(columns=["Response"])
    # Accept both the new split dataframes and older argument names.
    if urgency_df is None and identity_df is not None:
        urgency_df = identity_df
    if risk_factor_df is None and safety_df is not None:
        risk_factor_df = safety_df

    for component_df in [not_hate_df, urgency_df, risk_factor_df]:
        if component_df is not None:
            component_clean = _remove_overall_average_row(component_df).copy()
            summary_df = summary_df.merge(component_clean, on="Chatbot", how="left")
    existing_cols = [col for col in OVERALL_SUMMARY_COLUMNS if col in summary_df.columns]
    remaining_cols = [col for col in summary_df.columns if col not in existing_cols]
    summary_df = summary_df[existing_cols + remaining_cols]
    if include_overall_average:
        summary_df = append_overall_average_row(
            summary_df,
            label=OVERALL_AVERAGE_LABEL,
        )
    return summary_df

def save_overall_summary_table(summary_df: pd.DataFrame, output_path: str):
    summary_df.to_csv(output_path, index=False)




def process_all_outputs(
    evaluation_df: pd.DataFrame,
    integrated_responses: pd.DataFrame | None = None,
    not_hate_df: pd.DataFrame | None = None,
    urgency_df: pd.DataFrame | None = None,
    risk_factor_df: pd.DataFrame | None = None,
    identity_df: pd.DataFrame | None = None,
    safety_df: pd.DataFrame | None = None,
):
    _cleanup_plots_directory()
    for metric in VISUALIZATION_METRICS:
        plot_metric_bar(evaluation_df, metric, PLOTS_DIR)

    # Accept both new split arguments and old positional identity/safety calls.
    if urgency_df is None and identity_df is not None:
        urgency_df = identity_df
    if risk_factor_df is None and safety_df is not None:
        risk_factor_df = safety_df

    if not_hate_df is not None:
        plot_not_hate_metric(not_hate_df)
    if urgency_df is not None:
        plot_urgency_dimension(urgency_df)
    if risk_factor_df is not None:
        plot_risk_factor_dimension(risk_factor_df)
