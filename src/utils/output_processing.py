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
from scipy import stats
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


# =================================
# ROBUSTNESS / INFERENTIAL ANALYSIS: RANDOMIZED-BLOCK ANOVA
# =================================

def _get_available_robustness_metrics(df: pd.DataFrame) -> list[str]:
    """Return the 7 benchmark metrics that are present and numerically usable."""
    available = []
    for metric in ROBUSTNESS_METRICS:
        if metric in df.columns:
            values = pd.to_numeric(df[metric], errors="coerce")
            if values.notna().sum() >= 2:
                available.append(metric)
    return available


def _topic_sort_key_for_robustness(topic: str):
    if topic in ROBUSTNESS_TOPIC_ORDER:
        return (ROBUSTNESS_TOPIC_ORDER.index(topic), topic)
    if topic in CANONICAL_TOPIC_ORDER:
        return (len(ROBUSTNESS_TOPIC_ORDER) + CANONICAL_TOPIC_ORDER.index(topic), topic)
    return (len(ROBUSTNESS_TOPIC_ORDER) + len(CANONICAL_TOPIC_ORDER) + 1, topic)


def _run_randomized_block_anova(metric_df: pd.DataFrame, metric: str) -> dict:
    """
    Run randomized-block ANOVA without replication.

    Rows are chatbot systems, columns are response domains. The chatbot-system effect
    is the factor of interest; response domain is the blocking factor. This keeps the
    existing metric calculations unchanged while improving the inferential model so
    domain-level differences are accounted for rather than treated as independent noise.
    """
    clean_df = metric_df[["Chatbot", "Topic", metric]].copy()
    clean_df[metric] = pd.to_numeric(clean_df[metric], errors="coerce")
    clean_df = clean_df.dropna(subset=[metric])

    matrix_df = clean_df.pivot_table(
        index="Chatbot",
        columns="Topic",
        values=metric,
        aggfunc="mean",
    )

    # Classical randomized-block ANOVA requires a complete chatbot x domain matrix.
    # This drops only incomplete rows/columns for the specific metric being tested.
    matrix_df = matrix_df.dropna(axis=0, how="any").dropna(axis=1, how="any")

    k_chatbots = int(matrix_df.shape[0])
    b_domains = int(matrix_df.shape[1])
    total_observations = int(k_chatbots * b_domains)

    if k_chatbots < 2 or b_domains < 2:
        return {
            "Number of Chatbot Groups": k_chatbots,
            "Number of Response Domains": b_domains,
            "Total Domain-Level Observations": total_observations,
            "df_chatbot": np.nan,
            "df_domain": np.nan,
            "df_residual": np.nan,
            "F Statistic": np.nan,
            "p-value": np.nan,
            "Eta Squared": np.nan,
            "Partial Eta Squared": np.nan,
            "Included Chatbots": list(matrix_df.index),
            "Included Response Domains": list(matrix_df.columns),
            "Interpretation": "Insufficient complete chatbot-by-domain observations for randomized-block ANOVA",
        }

    values = matrix_df.to_numpy(dtype=float)
    grand_mean = float(values.mean())
    chatbot_means = values.mean(axis=1)
    domain_means = values.mean(axis=0)

    ss_total = float(((values - grand_mean) ** 2).sum())
    ss_chatbot = float(b_domains * ((chatbot_means - grand_mean) ** 2).sum())
    ss_domain = float(k_chatbots * ((domain_means - grand_mean) ** 2).sum())
    ss_residual = float(ss_total - ss_chatbot - ss_domain)
    # Guard against tiny negative residuals from floating point precision.
    if ss_residual < 0 and abs(ss_residual) < 1e-12:
        ss_residual = 0.0

    df_chatbot = k_chatbots - 1
    df_domain = b_domains - 1
    df_residual = df_chatbot * df_domain

    ms_chatbot = ss_chatbot / df_chatbot if df_chatbot > 0 else np.nan
    ms_residual = ss_residual / df_residual if df_residual > 0 else np.nan

    if pd.isna(ms_residual) or ms_residual <= 0:
        f_statistic = np.nan
        p_value = np.nan
    else:
        f_statistic = ms_chatbot / ms_residual
        p_value = stats.f.sf(f_statistic, df_chatbot, df_residual)

    eta_squared = ss_chatbot / ss_total if ss_total > 0 else np.nan
    partial_eta_squared = (
        ss_chatbot / (ss_chatbot + ss_residual)
        if (ss_chatbot + ss_residual) > 0
        else np.nan
    )

    return {
        "Number of Chatbot Groups": k_chatbots,
        "Number of Response Domains": b_domains,
        "Total Domain-Level Observations": total_observations,
        "df_chatbot": df_chatbot,
        "df_domain": df_domain,
        "df_residual": df_residual,
        "F Statistic": round(float(f_statistic), 4) if pd.notna(f_statistic) else np.nan,
        "p-value": round(float(p_value), 6) if pd.notna(p_value) else np.nan,
        "Eta Squared": round(float(eta_squared), 4) if pd.notna(eta_squared) else np.nan,
        "Partial Eta Squared": round(float(partial_eta_squared), 4) if pd.notna(partial_eta_squared) else np.nan,
        "Included Chatbots": list(matrix_df.index),
        "Included Response Domains": list(matrix_df.columns),
        "Interpretation": (
            "Statistically significant chatbot-system effect (p < .05)"
            if pd.notna(p_value) and float(p_value) < 0.05
            else "No statistically significant chatbot-system effect at p < .05"
            if pd.notna(p_value)
            else "Insufficient residual variation for randomized-block ANOVA"
        ),
    }


def generate_topic_level_metric_scores_for_anova(integrated_responses: pd.DataFrame) -> pd.DataFrame:
    """
    Build the domain-level score table used by randomized-block ANOVA.

    This table keeps the existing metric calculations unchanged, but stores one row per
    chatbot x response-domain pair. The ANOVA then compares chatbot systems while
    accounting for the same set of response domains being scored across systems.
    """
    if integrated_responses is None or integrated_responses.empty:
        print("[WARN] ANOVA skipped: integrated responses not provided.")
        return pd.DataFrame()

    from src.utils.evaluation_algo import (
        calculate_average_rouge,
        calculate_meteor,
        evaluate_negative_tone_probability,
        evaluate_readability_score,
        get_not_hate_probability,
        get_reference_alignment_score,
        prepare_aggregated_views,
    )

    views = prepare_aggregated_views(integrated_responses)
    reference_topic_map = views["reference_topic_map"]
    chatbot_topic_df = views["chatbot_topic_df"]

    target_topics = [
        topic for topic in ROBUSTNESS_TOPIC_ORDER
        if topic in reference_topic_map
    ]
    target_topics = sorted(target_topics, key=_topic_sort_key_for_robustness)

    if not target_topics:
        print("[WARN] ANOVA skipped: none of the formal benchmark response domains were found.")
        return pd.DataFrame()

    rows = []
    for _, row in chatbot_topic_df.iterrows():
        chatbot = str(row["Chatbot"]).strip()
        topic = str(row[TOPIC_COL]).strip()
        response_text = str(row["TopicResponse"]).strip()

        if topic not in target_topics:
            continue

        reference_text = str(reference_topic_map.get(topic, "")).strip()
        if not reference_text or not response_text:
            continue

        base_row = {
            "Chatbot": chatbot,
            "Topic": topic,
            "ROUGE Lexical Overlap": calculate_average_rouge(reference_text, response_text),
            "METEOR Lexical-Semantic Alignment": calculate_meteor(reference_text, response_text),
            "Negative Sentiment Probability": evaluate_negative_tone_probability(response_text),
            "Flesch Reading Ease": evaluate_readability_score(response_text),
            "Non-Hateful Language Probability": round(float(get_not_hate_probability(response_text)), 4),
            "Crisis-Response Reference Similarity": np.nan,
            "Risk-Assessment Reference Similarity": np.nan,
        }

        if topic in URGENCY_REFERENCE_TOPICS:
            base_row["Crisis-Response Reference Similarity"] = round(
                float(get_reference_alignment_score(response_text, reference_text)), 4
            )

        if topic in RISK_FACTOR_REFERENCE_TOPICS:
            base_row["Risk-Assessment Reference Similarity"] = round(
                float(get_reference_alignment_score(response_text, reference_text)), 4
            )

        rows.append(base_row)

    topic_level_df = pd.DataFrame(rows)
    if topic_level_df.empty:
        return topic_level_df

    topic_level_df["Topic"] = pd.Categorical(
        topic_level_df["Topic"],
        categories=target_topics,
        ordered=True,
    )
    topic_level_df = topic_level_df.sort_values(["Chatbot", "Topic"]).reset_index(drop=True)
    return topic_level_df


def generate_randomized_block_anova_by_metric(topic_level_df: pd.DataFrame) -> pd.DataFrame:
    """
    Run randomized-block ANOVA for each benchmark metric.

    For each metric, the null hypothesis is that the mean domain-level score is equal
    across chatbot systems after accounting for response-domain differences.
    """
    if topic_level_df is None or topic_level_df.empty:
        print("[WARN] ANOVA skipped: domain-level metric table is empty.")
        return pd.DataFrame()

    metric_cols = _get_available_robustness_metrics(topic_level_df)
    if not metric_cols:
        print("[WARN] ANOVA skipped: no numeric robustness metrics available.")
        return pd.DataFrame()

    rows = []
    for metric in metric_cols:
        result = _run_randomized_block_anova(topic_level_df, metric)
        rows.append({"Metric": metric, **result})

    return pd.DataFrame(rows)


# Backward-compatible wrapper for older scripts that still call the previous name.
def generate_oneway_anova_by_metric(topic_level_df: pd.DataFrame) -> pd.DataFrame:
    return generate_randomized_block_anova_by_metric(topic_level_df)


def save_randomized_block_anova_outputs(
    anova_df: pd.DataFrame,
    topic_level_df: pd.DataFrame,
):
    _ensure_dir(ROBUSTNESS_DIR)
    if topic_level_df is not None and not topic_level_df.empty:
        topic_level_df.to_csv(TOPIC_LEVEL_METRIC_SCORES_CSV_PATH, index=False)
    if anova_df is not None and not anova_df.empty:
        anova_df.to_csv(RANDOMIZED_BLOCK_ANOVA_CSV_PATH, index=False)


# Backward-compatible wrapper for older scripts that still call the previous name.
def save_oneway_anova_outputs(
    anova_df: pd.DataFrame,
    topic_level_df: pd.DataFrame,
):
    save_randomized_block_anova_outputs(anova_df=anova_df, topic_level_df=topic_level_df)


def plot_randomized_block_anova_p_values(anova_df: pd.DataFrame):
    if anova_df is None or anova_df.empty or "p-value" not in anova_df.columns:
        return

    plot_df = anova_df.copy()
    plot_df["p-value"] = pd.to_numeric(plot_df["p-value"], errors="coerce")
    plot_df = plot_df.dropna(subset=["p-value"])
    if plot_df.empty:
        return

    # Use -log10(p) so smaller p-values are easier to read visually.
    plot_df["-log10(p-value)"] = -np.log10(plot_df["p-value"].clip(lower=1e-300))

    _ensure_dir(PLOTS_DIR)
    plt.figure(figsize=(max(9, len(plot_df) * 1.2), 5.8))
    plt.bar(plot_df["Metric"], plot_df["-log10(p-value)"])
    plt.axhline(y=-np.log10(0.05), linestyle="--", linewidth=1.8, label="p = .05")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("-log10(p-value)")
    plt.title("Randomized-Block ANOVA by Benchmark Metric")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RANDOMIZED_BLOCK_ANOVA_PLOT_PATH, dpi=DPI)
    plt.close()


# Backward-compatible wrapper for older scripts that still call the previous name.
def plot_oneway_anova_p_values(anova_df: pd.DataFrame):
    plot_randomized_block_anova_p_values(anova_df)


def run_robustness_outputs(
    evaluation_df: pd.DataFrame,
    integrated_responses: pd.DataFrame | None = None,
):
    """
    Inferential output is intentionally limited to randomized-block ANOVA.

    Removed from the active pipeline:
    - Spearman metric-correlation matrix
    - leave-one-topic-out sensitivity checks
    - normalized sensitivity summaries
    """
    if integrated_responses is None:
        print("[WARN] ANOVA skipped: integrated_responses is required for domain-level ANOVA.")
        return

    topic_level_df = generate_topic_level_metric_scores_for_anova(integrated_responses)
    anova_df = generate_randomized_block_anova_by_metric(topic_level_df)
    save_randomized_block_anova_outputs(anova_df=anova_df, topic_level_df=topic_level_df)
    plot_randomized_block_anova_p_values(anova_df)


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

    run_robustness_outputs(
        evaluation_df=evaluation_df,
        integrated_responses=integrated_responses,
    )
