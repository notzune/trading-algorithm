import re

import torch
from datasets import load_dataset, DatasetDict
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Load the dataset with a specific configuration
dataset = load_dataset("financial_phrasebank", "sentences_allagree", download_mode="force_redownload")
# dataset = load_dataset("nickmuchi/financial-classification")

# Load model and tokenizer
model_name = "ProsusAI/finbert"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)


def encode_data(examples, index=None):  # Add index parameter to accept the index when with_indices=True
    """
    Tokenizes and encodes the dataset using the specified tokenizer.

    Args:
        examples (dict): A batch from the dataset.
        index (int, optional): Index of the batch. Defaults to None.

    Returns:
        dict: A dictionary with tokenized and encoded data ready for model input.
    """
    encoded = tokenizer(examples['sentence'], padding='max_length', truncation=True, max_length=512)
    encoded_tensors = {key: torch.tensor(val, dtype=torch.long) for key, val in encoded.items()}
    # Debugging print statement
    print(f"Type of input_ids in encode_data: {type(encoded_tensors['input_ids'])}")
    return encoded_tensors


encoded_dataset = dataset.map(encode_data, batched=True, with_indices=True)

# Test the output of the encode_data function
test_encoded = encode_data({'sentence': ["Test sentence"]})
print(f"Test encoded input_ids type: {type(test_encoded['input_ids'])}, shape: {test_encoded['input_ids'].shape}")
sample_after_encode = next(iter(encoded_dataset['train']))
print(
    f"After encode_data - input_ids: {sample_after_encode['input_ids']}, type: {type(sample_after_encode['input_ids'])}")


def format_labels(example, index=None):  # Add index parameter here as well
    """
    Converts various label formats of a single example in the dataset to a tensor.
    Handles labels that are integers, strings, or a combination of both.

    Args:
        example (dict): A single example from the dataset.
        index (int, optional): Index of the batch. Defaults to None.

    Returns:
        dict: A dictionary with the label converted to a tensor.
    """
    label_map = {'negative': 0, 'neutral': 1, 'positive': 2}
    label_str = str(example['label']).lower().strip()

    # Extract integer from string if present, or map string directly
    match = re.match(r'(\d+|\w+)', label_str)
    if match:
        extracted_label = match.group(1)
        if extracted_label.isdigit():
            label = int(extracted_label)
        elif extracted_label in label_map:
            label = label_map[extracted_label]
        else:
            raise ValueError(f"Unrecognized label format: {label_str}")
    else:
        raise ValueError(f"Invalid label format: {label_str}")

    # Convert the label to a tensor
    return {'labels': torch.tensor(label, dtype=torch.long)}


# Apply encoding and label formatting
processed_dataset = encoded_dataset.map(format_labels, with_indices=True)

# After applying format_labels
sample_after_label_format = next(iter(processed_dataset['train']))
print(
    f"After format_labels - labels: {sample_after_label_format['labels']}, type: {type(sample_after_label_format['labels'])}")

# Split the processed dataset
train_test_split = processed_dataset["train"].train_test_split(test_size=0.1)
dataset_dict = DatasetDict({
    "train": train_test_split["train"],
    "validation": train_test_split["test"]
})

# Before feeding into DataLoader
sample_before_loader = next(iter(dataset_dict['train']))
print(
    f"Before DataLoader - input_ids: {sample_before_loader['input_ids']}, type: {type(sample_before_loader['input_ids'])}")


# Custom collate function to handle lists of tensors
def collate_fn(batch):
    # Debugging print statements
    for i, item in enumerate(batch):
        print(f"Item {i} - input_ids type: {type(item['input_ids'])}, labels type: {type(item['labels'])}")
        if isinstance(item['input_ids'], list):
            print(f"Item {i} - input_ids is a list, which should be converted to a tensor.")
        if isinstance(item['labels'], int):
            print(f"Item {i} - labels is an int, which should be converted to a tensor.")

    input_ids = torch.stack([torch.tensor(item['input_ids'], dtype=torch.long) for item in batch])
    attention_mask = torch.stack([torch.tensor(item['attention_mask'], dtype=torch.long) for item in batch])
    labels = torch.stack([torch.tensor(item['labels'], dtype=torch.long) for item in batch])

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }


# DataLoaders with custom collate function
train_loader = DataLoader(dataset_dict['train'], batch_size=16, shuffle=True, collate_fn=collate_fn)
valid_loader = DataLoader(dataset_dict['validation'], batch_size=16, collate_fn=collate_fn)

# Get a batch from the DataLoader
batch = next(iter(train_loader))

# Check the batch types and shapes
print(f"DataLoader batch - input_ids type: {type(batch['input_ids'])}, shape: {batch['input_ids'].shape}")
print(f"DataLoader batch - labels type: {type(batch['labels'])}, shape: {batch['labels'].shape}")

# Optimizer
optimizer = AdamW(model.parameters(), lr=5e-5)

# Training Loop
epochs = 3
for epoch in range(epochs):
    model.train()
    for batch in train_loader:
        print(f"Training batch - input_ids type: {type(batch['input_ids'])}, shape: {batch['input_ids'].shape}")
        print(f"Training batch - labels type: {type(batch['labels'])}, shape: {batch['labels'].shape}")

        input_ids = batch['input_ids'].to(model.device)
        attention_mask = batch['attention_mask'].to(model.device)
        labels = batch['labels'].to(model.device)

        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    model.eval()
    total_eval_loss = 0
    for batch in valid_loader:
        batch = {k: v.to(model.device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
            loss = outputs.loss
            total_eval_loss += loss.item()

    avg_val_loss = total_eval_loss / len(valid_loader)
    print(f'Validation loss after epoch {epoch}: {avg_val_loss}')

# Save the fine-tuned model
model.save_pretrained("./finetuned-finbert")
