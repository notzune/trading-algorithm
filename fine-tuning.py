import torch
from datasets import load_dataset, DatasetDict
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AdamW
from torch.utils.data import DataLoader

# Load the dataset with a specific configuration
dataset = load_dataset("financial_phrasebank", "sentences_allagree", download_mode="force_redownload")

# Load model and tokenizer
model_name = "ProsusAI/finbert"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)


def encode_data(examples):
    """
    Tokenizes and encodes the dataset using the specified tokenizer.

    Args:
        examples (dict): A batch from the dataset.

    Returns:
        dict: A dictionary with tokenized and encoded data ready for model input.
    """
    return tokenizer(examples['sentence'], padding='max_length', truncation=True, max_length=512)


# Encode the entire dataset
encoded_dataset = dataset.map(encode_data, batched=True)


def format_labels(example):
    """
    Converts the label of a single example in the dataset to a tensor.

    Args:
        example (dict): A single example from the dataset.

    Returns:
        dict: A dictionary with the label converted to a tensor.
    """
    # Assuming the label is already an integer
    return {'labels': torch.tensor(example['label'])}


# Encode the dataset and then apply the label formatting
encoded_dataset = dataset.map(encode_data, batched=True)
processed_dataset = encoded_dataset.map(format_labels, batched=False)

# Split the processed dataset
train_test_split = processed_dataset["train"].train_test_split(test_size=0.1)
dataset_dict = DatasetDict({
    "train": train_test_split["train"],
    "validation": train_test_split["test"]
})

# DataLoader
train_loader = DataLoader(dataset_dict['train'], batch_size=16, shuffle=True)
valid_loader = DataLoader(dataset_dict['validation'], batch_size=16)

# Optimizer
optimizer = AdamW(model.parameters(), lr=5e-5)

# Training Loop
epochs = 3
for epoch in range(epochs):
    model.train()
    for batch in train_loader:
        # Ensure batch data is in tensor format
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
