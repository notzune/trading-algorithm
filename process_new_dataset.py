import torch
import pandas as pd
import json
import logging


def process_new_dataset(data, tokenizer, label_map=None, file_format='txt', max_length=512):
    """
    Enhanced function to process a new dataset for BERT-like models.
    ...
    """
    sentences, labels = [], []
    label_set = set()

    try:
        if isinstance(data, str):  # If 'data' is a file path
            if file_format == 'txt':
                with open(data, 'r', encoding='utf-8') as file:
                    for line in file:
                        sentence, label = extract_sentence_and_label(line, file_format)
                        sentences.append(sentence)
                        labels.append(label)
                        label_set.add(label)
            elif file_format == 'csv':
                df = pd.read_csv(data)
                # Assuming the CSV has columns 'sentence' and 'label'
                sentences, labels = list(df['sentence']), list(df['label'])
            elif file_format == 'json':
                with open(data, 'r') as file:
                    json_data = json.load(file)
                    # Implement JSON parsing logic based on your JSON structure
                    # Example: sentences = [item['sentence'] for item in json_data]
            else:
                logging.warning(f"Unsupported file format: {file_format}")

        elif isinstance(data, list):  # If 'data' is a list of sentences
            sentences = data
            labels = [0] * len(data)  # Placeholder labels

        if not label_map:
            label_map = {label: idx for idx, label in enumerate(sorted(label_set))}

        # Convert string labels to integers based on label_map
        labels = [label_map[label] for label in labels]

        inputs = tokenizer(sentences, padding='max_length', truncation=True, max_length=max_length, return_tensors='pt')
        labels_tensor = torch.tensor(labels)

        return inputs, labels_tensor

    except Exception as e:
        logging.error(f"Error processing dataset: {str(e)}")
        return {}, torch.tensor([])


def extract_sentence_and_label(line, file_format):
    """
    Enhanced function to extract sentences and labels from different file formats.
    ...
    """
    # Implement extraction logic for each file format
    if file_format == 'txt':
        sentence, label = line.strip().split('@')
        return sentence.strip(), label.strip()
    elif file_format == 'csv':
        # Assuming the CSV has columns 'sentence' and 'label'
        return line['sentence'], line['label']

    elif file_format == 'json':
        # Assuming each JSON record has keys 'sentence' and 'label'
        return line['sentence'], line['label']

    return line, 'unknown'  # Default case if format not matched

# Example usage
# data = 'path/to/your/dataset.txt'
# tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
# inputs, labels = process_new_dataset(data, tokenizer, file_format='txt')
