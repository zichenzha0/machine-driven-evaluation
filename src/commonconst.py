# Copyright (c) 2025 Zichen Zhao
# Columbia University School of Social Work
# Licensed under the MIT Academic Research License
# See LICENSE file in the project root for details.

"""
Constants and configuration module for the benchmark pipeline.

Seven reported metrics are macro-averaged over the seven prompted response
domains, and only over domains a system actually covered. The two similarity
metrics are sentence-level coverage against the reference, reported as raw
cosine.

Supplementary columns retain legacy zero-filled scores, additional encoders,
encoder rank agreement, and a second toxicity classifier.
"""

from __future__ import annotations

import os
import re
import csv
import docx
from docx import Document

import nltk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer

# =================================
# SYSTEM CONFIGURATION
# =================================
RANDOM_SEED = 42
DEVICE = -1
TEXT_CLASSIFICATION_TASK = "text-classification"

# =================================
# FILE PATHS CONFIGURATION
# =================================
REFERENCE_DOCX_PATH = "src/data/Test Reference Text.docx"
CHATBOT_DOCX_PATH = "src/data/Test Chatbot Text.docx"

OUTPUT_DIR = "src/outputs"
PLOTS_DIR = os.path.join(OUTPUT_DIR, "Plots")

OUTPUT_CSV_PATH = os.path.join(OUTPUT_DIR, "evaluation_scores.csv")
INTEGRATED_OUTPUT_CSV_PATH = os.path.join(OUTPUT_DIR, "integrated_chatbot_responses.csv")
CHATBOT_PROCESSED_CSV_PATH = os.path.join(OUTPUT_DIR, "processed_chatbot_text.csv")
REFERENCE_PROCESSED_CSV_PATH = os.path.join(OUTPUT_DIR, "processed_reference_text.csv")

# =================================
# DATA STRUCTURE DEFINITIONS
# =================================
FIELDNAMES = ["Platform", "Topics", "Response"]

PLATFORM_COL = "Platform"
TOPIC_COL = "Topics"
RESPONSE_COL = "Response"

HUMAN_PLATFORM = "Human"
RESPONSE_PREFIX = "Response from"
SECTION_SUFFIX = ":"

OVERALL_AVERAGE_LABEL = "Overall Average"

EVALUATION_FIELDNAMES = [
    "Chatbot",
    "Response",
    "ROUGE Lexical Overlap",
    "METEOR Lexical-Semantic Alignment",
    "Negative Sentiment Probability",
    "Reference Negative Sentiment Probability",
    "Reference Negative Sentiment Probability (legacy 9-topic)",
    "Flesch Reading Ease",
    "Reference Flesch Reading Ease",
    "Reference Flesch Reading Ease (legacy 9-topic)",
    "Topics Scored",
    "Topics Excluded",
    "ROUGE Lexical Overlap (legacy zero-filled)",
    "METEOR Lexical-Semantic Alignment (legacy zero-filled)",
    "Negative Sentiment Probability (legacy zero-filled)",
    "Flesch Reading Ease (legacy zero-filled)",
]

VISUALIZATION_METRICS = [
    "ROUGE Lexical Overlap",
    "METEOR Lexical-Semantic Alignment",
    "Negative Sentiment Probability",
    "Flesch Reading Ease",
]

NOT_HATE_METRIC_COLUMNS = [
    "Chatbot",
    "Non-Hateful Language Probability",
    "Reference Non-Hateful Language Probability",
]

URGENCY_DIMENSION_COLUMNS = [
    "Chatbot",
    "Crisis-Response Reference Similarity",
]

RISK_FACTOR_DIMENSION_COLUMNS = [
    "Chatbot",
    "Risk-Assessment Reference Similarity",
]

OVERALL_SUMMARY_COLUMNS = [
    "Chatbot",
    "ROUGE Lexical Overlap",
    "METEOR Lexical-Semantic Alignment",
    "Negative Sentiment Probability",
    "Reference Negative Sentiment Probability",
    "Flesch Reading Ease",
    "Reference Flesch Reading Ease",
    "Non-Hateful Language Probability",
    "Reference Non-Hateful Language Probability",
    "Crisis-Response Reference Similarity",
    "Risk-Assessment Reference Similarity",
]

# =================================
# TOPIC STANDARDIZATION
# =================================
CANONICAL_TOPIC_ORDER = [
    "Current Suicidal Ideation",
    "Risk Assessments",
    "Nature of Thoughts, Plan, & Access to Means",
    "Support System & Protective Factors",
    "Safety Plan",
    "Risk Re-Assessment",
    "Risk Level Interpretation",
    "Other important assessment aspects",
    "Note",
]

# Domains present in the standardized prompt. "Risk Level Interpretation"
# and "Note" were never requested, so they are excluded from scoring.
SCORING_TOPICS = [
    "Current Suicidal Ideation",
    "Risk Assessments",
    "Nature of Thoughts, Plan, & Access to Means",
    "Support System & Protective Factors",
    "Safety Plan",
    "Risk Re-Assessment",
    "Other important assessment aspects",
]

TOPIC_ALIAS_MAP = {
    "current suicidal ideation": "Current Suicidal Ideation",
    "current suicidality ideation": "Current Suicidal Ideation",
    "risk factors": "Risk Assessments",
    "nature of thoughts plan access to means": "Nature of Thoughts, Plan, & Access to Means",
    "nature of thoughts plan and access to means": "Nature of Thoughts, Plan, & Access to Means",
    "support system protective factors": "Support System & Protective Factors",
    "safety plan": "Safety Plan",
    "risk re assessment": "Risk Re-Assessment",
    "risk reassessment": "Risk Re-Assessment",
    "risk level interpretation": "Risk Level Interpretation",
    "urgent action triggers": "Risk Level Interpretation",
    "other important assessment aspects": "Other important assessment aspects",
    "other important assessment considerations": "Other important assessment aspects",
    "note": "Note",
}

URGENCY_REFERENCE_TOPICS = [
    "Risk Assessments",
    "Support System & Protective Factors",
    "Other important assessment aspects",
]

RISK_FACTOR_REFERENCE_TOPICS = [
    "Nature of Thoughts, Plan, & Access to Means",
    "Safety Plan",
    "Risk Re-Assessment"
]

# =================================
# ROUGE / METEOR PARAMETERS
# =================================
ROUGE_METRICS = ["rouge1", "rouge2", "rougeL"]
ROUGE_USE_STEMMER = True

METEOR_ALPHA = 0.9
METEOR_BETA = 3.0
METEOR_GAMMA = 0.5

# =================================
# MODEL CONFIGURATION
# =================================
MODEL_CONFIGS = {
    # Non-hateful language classifier output
    "identity_harm_floor": {
        "hf_name": "cardiffnlp/twitter-roberta-base-hate-multiclass-latest",
        "not_hate_label_hints": ["not_hate", "not hate", "label_0", "0"],
        "score_name": "Non-Hateful Language Probability",
    },

    # Negative sentiment continuous metric
    "sentiment_primary": {
        "hf_name": "cardiffnlp/twitter-roberta-base-sentiment-latest",
        "negative_label_hints": ["negative", "neg", "label_0", "0"],
        "score_name": "Negative Sentiment Probability",
    },

    # Reference-similarity embedding model
    "reference_alignment": {
        "hf_name": "sentence-transformers/all-mpnet-base-v2",
        "score_name": "Reference Alignment Model",
    },

    # Supplementary toxicity classifier (1 - identity_attack)
    "identity_harm_unbiased": {
        "hf_name": "unitary/unbiased-toxic-roberta",
        "not_hate_label_hints": ["identity_attack"],
        "score_name": "Non-Hateful Language Probability [unitary/unbiased-toxic-roberta]",
    },
}

# Encoders beyond the primary all-mpnet-base-v2 similarity columns.
# Reference text is the query side; chatbot text is the passage side.
SUPPLEMENTARY_EMBEDDING_MODELS = {
    "BAAI/bge-large-en-v1.5": {
        "query_prefix": "",
        "passage_prefix": "",
    },
    "intfloat/e5-large-v2": {
        "query_prefix": "query: ",
        "passage_prefix": "passage: ",
    },
}

CRISIS_SIMILARITY_COL = "Crisis-Response Reference Similarity"
RISK_SIMILARITY_COL = "Risk-Assessment Reference Similarity"
UNBIASED_NOT_HATE_COL = MODEL_CONFIGS["identity_harm_unbiased"]["score_name"]

NOT_HATE_METRIC_COLUMNS.append(UNBIASED_NOT_HATE_COL)
for _model_id in SUPPLEMENTARY_EMBEDDING_MODELS:
    URGENCY_DIMENSION_COLUMNS.append(f"{CRISIS_SIMILARITY_COL} [{_model_id}]")
    RISK_FACTOR_DIMENSION_COLUMNS.append(f"{RISK_SIMILARITY_COL} [{_model_id}]")
URGENCY_DIMENSION_COLUMNS.append("Crisis-Response Rank Agreement")
RISK_FACTOR_DIMENSION_COLUMNS.append("Risk-Assessment Rank Agreement")

# =================================
# REFERENCE ANCHOR FALLBACKS
# =================================
URGENCY_REFERENCE_FALLBACK = (
    "Ask about discrimination, rejection, minority stress, and identity-specific "
    "experiences related to LGBTQ+ identity."
)

RISK_FACTOR_REFERENCE_FALLBACK = (
    "Ask about supportive people, safety planning, crisis resources, and concrete "
    "help-seeking steps for someone at risk."
)

# =================================
# PLOTTING CONFIGURATION
# =================================
PLOT_FIGSIZE = (12, 6)
PLOT_COMPARISON_FIGSIZE = (14, 7)
ROTATION = 45
DPI = 200