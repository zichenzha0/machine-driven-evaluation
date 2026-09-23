# Copyright (c) 2025 Zichen Zhao
# Columbia University School of Social Work
# Licensed under the MIT Academic Research License
# See LICENSE file in the project root for details.

from src.commonconst import *

def extract_text_from_docx(doc_path):
    doc = docx.Document(doc_path)
    text = [paragraph.text.strip() for paragraph in doc.paragraphs if paragraph.text.strip() != ""]
    return text

def process_reference_text(reference_text):
    data = []
    current_section = None
    for line in reference_text:
        line = line.strip()
        if not line:
            continue
        if line.endswith(SECTION_SUFFIX):
            current_section = line[:-1].strip()
            continue
        if current_section is not None:
            data.append({
                "Platform": HUMAN_PLATFORM,
                "Topics": current_section,
                "Response": line
            })
    return data

def process_chatbot_responses(chatbot_text):
    data = []
    current_chatbot = None
    current_section = None
    for line in chatbot_text:
        line = line.strip()
        if not line:
            continue
        if RESPONSE_PREFIX in line:
            current_chatbot = line.split(RESPONSE_PREFIX)[-1].strip()
            current_section = None
            continue
        elif line.endswith(SECTION_SUFFIX):
            current_section = line[:-1].strip()
            continue
        if current_chatbot is not None and current_section is not None:
            data.append({
                "Platform": current_chatbot,
                "Topics": current_section,
                "Response": line
            })
    return data

def save_processed_files(chatbot_text, reference_text, chatbot_output_path, reference_output_path, integrated_output_path):
    def save_to_csv(file_path, fieldnames, data):
        with open(file_path, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)
    chatbot_data = process_chatbot_responses(chatbot_text)
    save_to_csv(chatbot_output_path, ['Platform', 'Topics', 'Response'], chatbot_data)
    reference_data = process_reference_text(reference_text)
    save_to_csv(reference_output_path, ['Platform', 'Topics', 'Response'], reference_data)
    chatbot_text_df = pd.DataFrame(chatbot_data)
    reference_text_df = pd.DataFrame(reference_data)
    chatbot_aggregated = (
        chatbot_text_df
        .groupby(['Platform', 'Topics'], dropna=False)['Response']
        .apply(' '.join)
        .reset_index()
    )
    reference_aggregated = (
        reference_text_df
        .groupby(['Platform', 'Topics'], dropna=False)['Response']
        .apply(' '.join)
        .reset_index()
    )
    integrated_df = pd.concat([chatbot_aggregated, reference_aggregated], ignore_index=True)
    integrated_df.to_csv(integrated_output_path, index=False)