from torch.optim import AdamW
from tqdm import tqdm
import json
import logging
import pandas as pd
import torch

def fine_tune_model(new_train_loader, new_val_loader, model, tokenizer, epochs=3):
    optimizer = AdamW(model.parameters(), lr=5e-5)

    for epoch in range(epochs):
        print(f"Starting epoch {epoch + 1}/{epochs}")

        # Training phase
        model.train()
        total_train_loss = 0
        for batch in tqdm(new_train_loader, desc='Training'):
            b_input_ids, b_input_mask, b_labels = batch
            model.zero_grad()
            outputs = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)
            loss = outputs.loss
            total_train_loss += loss.item()
            loss.backward()
            optimizer.step()

        avg_train_loss = total_train_loss / len(new_train_loader)
        print(f"Average training loss: {avg_train_loss}")

        # Validation phase
        model.eval()
        total_eval_loss = 0
        for batch in tqdm(new_val_loader, desc='Validation'):
            b_input_ids, b_input_mask, b_labels = batch
            with torch.no_grad():
                outputs = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)
                loss = outputs.loss
                total_eval_loss += loss.item()

        avg_val_loss = total_eval_loss / len(new_val_loader)
        print(f"Validation loss: {avg_val_loss}")

        # Save a checkpoint after each epoch
        checkpoint_path = f"./finetuned-finbert-checkpoint-epoch-{epoch + 1}"
        model.save_pretrained(checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

    # Save the final fine-tuned model
    model_path = "../finetuned-finbert"
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    print(f"Model and tokenizer saved to {model_path}")


def process_new_dataset(data, tokenizer, file_format='txt', max_length=512):
    """
    Enhanced function to process a new dataset for BERT-like models.

    Parameters:
    - data (str or list): File path to the dataset or a list of sentences.
    - tokenizer (AutoTokenizer): Tokenizer for encoding the sentences.
    - file_format (str): Format of the dataset file ('txt', 'csv', etc.).
    - max_length (int): Maximum length for tokenization.

    Returns:
    - inputs (dict): Tokenized sentences.
    - labels (torch.Tensor): Tensor of labels.
    """

    sentences, labels = [], []
    label_map = {}

    try:
        if isinstance(data, str):  # If 'data' is a file path
            if file_format == 'txt':
                with open(data, 'r', encoding='utf-8') as file:
                    for line in file:
                        sentence, label = extract_sentence_and_label(line, file_format)
                        if label not in label_map:
                            label_map[label] = len(label_map)
                        sentences.append(sentence)
                        labels.append(label_map[label])
            elif file_format == 'csv':
                df = pd.read_csv(data)
                # Logic for processing CSV files
            elif file_format == 'json':
                with open(data, 'r') as file:
                    json_data = json.load(file)
                    # Logic for processing JSON files
            else:
                logging.warning(f"Unsupported file format: {file_format}")
        elif isinstance(data, list):  # If 'data' is a list of sentences
            sentences = data
            labels = [0] * len(data)  # Placeholder labels

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
