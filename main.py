# Copyright (c) 2025 Zichen Zhao
# Columbia University School of Social Work
# Licensed under the MIT Academic Research License
# See LICENSE file in the project root for details.

"""
Main execution script for the benchmark pipeline.
"""
from __future__ import annotations
import pandas as pd
from src.commonconst import *
from src.data.data_processing import (
    extract_text_from_docx,
    save_processed_files,
)
from src.utils.evaluation_algo import (
    ensure_output_dirs,
    generate_evaluation_scores,
    generate_not_hate_metric_scores,
    generate_urgency_dimension_scores,
    generate_risk_factor_dimension_scores,
    save_evaluation_to_csv,
)
from src.utils.output_processing import process_all_outputs

def append_component_scores_to_evaluation(
    evaluation_df: pd.DataFrame,
    not_hate_df: pd.DataFrame,
    urgency_df: pd.DataFrame,
    risk_factor_df: pd.DataFrame,
) -> pd.DataFrame:
    merged_df = evaluation_df.copy()
    component_dfs = [not_hate_df, urgency_df, risk_factor_df]
    for component_df in component_dfs:
        clean_component_df = component_df.copy()
        merge_cols = [col for col in clean_component_df.columns if col != "Response"]
        clean_component_df = clean_component_df[merge_cols]
        duplicate_cols = [
            col for col in clean_component_df.columns
            if col != "Chatbot" and col in merged_df.columns
        ]
        if duplicate_cols:
            merged_df = merged_df.drop(columns=duplicate_cols)
        merged_df = merged_df.merge(clean_component_df, on="Chatbot", how="left")
    return merged_df

def main():
    ensure_output_dirs()
    # Step 1: load raw docx text
    reference_text = extract_text_from_docx(REFERENCE_DOCX_PATH)
    chatbot_text = extract_text_from_docx(CHATBOT_DOCX_PATH)
    # Step 2: process and save all intermediate files
    save_processed_files(
        chatbot_text=chatbot_text,
        reference_text=reference_text,
        chatbot_output_path=CHATBOT_PROCESSED_CSV_PATH,
        reference_output_path=REFERENCE_PROCESSED_CSV_PATH,
        integrated_output_path=INTEGRATED_OUTPUT_CSV_PATH,
    )
    # Step 3: load integrated responses
    integrated_responses = pd.read_csv(INTEGRATED_OUTPUT_CSV_PATH)
    # Step 4: primary continuous metrics
    evaluation_df = generate_evaluation_scores(
        integrated_responses,
        include_overall_average=True,
    )
    # Step 5: split benchmark components
    not_hate_df = generate_not_hate_metric_scores(
        integrated_responses,
        include_overall_average=True,
    )
    urgency_df = generate_urgency_dimension_scores(
        integrated_responses,
        include_overall_average=True,
    )
    risk_factor_df = generate_risk_factor_dimension_scores(
        integrated_responses,
        include_overall_average=True,
    )
    # Step 6: append split component scores back into the main evaluation CSV
    evaluation_df = append_component_scores_to_evaluation(
        evaluation_df=evaluation_df,
        not_hate_df=not_hate_df,
        urgency_df=urgency_df,
        risk_factor_df=risk_factor_df,
    )
    save_evaluation_to_csv(OUTPUT_CSV_PATH, evaluation_df)
    # Step 7: benchmark plots only.
    process_all_outputs(
        evaluation_df=evaluation_df,
        integrated_responses=integrated_responses,
        not_hate_df=not_hate_df,
        urgency_df=urgency_df,
        risk_factor_df=risk_factor_df,
    )
    print("Benchmark evaluation complete.")
    print(f"Main results saved to: {OUTPUT_CSV_PATH}")
    print(f"Integrated responses saved to: {INTEGRATED_OUTPUT_CSV_PATH}")
    print(f"All plots saved to: {PLOTS_DIR}")

if __name__ == "__main__":
    main()