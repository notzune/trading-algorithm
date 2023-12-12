import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from utils import fine_tune_model, process_new_dataset

model_dir = "/finetuned-finbert"  # Adjust as necessary
model = AutoModelForSequenceClassification.from_pretrained(model_dir)
tokenizer = AutoTokenizer.from_pretrained(model_dir)

dataset_dir = ""
# Assuming 'new_data' is either a file path or a list of sentences
new_inputs, new_labels_tensor = process_new_dataset(dataset_dir, tokenizer)

new_dataset = TensorDataset(new_inputs['input_ids'], new_inputs['attention_mask'], new_labels_tensor)

train_size = int(0.8 * len(new_dataset))
val_size = len(new_dataset) - train_size
new_train_dataset, new_val_dataset = random_split(new_dataset, [train_size, val_size])

new_train_loader = DataLoader(new_train_dataset, batch_size=16, shuffle=True)
new_val_loader = DataLoader(new_val_dataset, batch_size=16)

# Ensure fine_tune_model is defined to handle the training and validation
fine_tune_model(new_train_loader, new_val_loader, model, tokenizer)