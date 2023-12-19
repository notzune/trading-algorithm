from datasets import load_dataset
from torch.utils.data import DataLoader, TensorDataset, random_split, ConcatDataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from train_utils import fine_tune_model, process_new_dataset

model_dir = "../../finetuned-finbert"
model = AutoModelForSequenceClassification.from_pretrained(model_dir)
tokenizer = AutoTokenizer.from_pretrained(model_dir)

# Load multiple datasets
dataset1 = load_dataset("FinanceInc/auditor_sentiment")['train']
dataset2 = load_dataset("TimKoornstra/financial-tweets-sentiment")['train']

# Process each dataset
# Adjust the column names based on the actual dataset structure
inputs1, labels1 = process_new_dataset(dataset1['sentence'], tokenizer)
inputs2, labels2 = process_new_dataset(dataset2['tweet'], tokenizer)

# Create TensorDatasets for each
dataset1_tensor = TensorDataset(inputs1['input_ids'], inputs1['attention_mask'], labels1)
dataset2_tensor = TensorDataset(inputs2['input_ids'], inputs2['attention_mask'], labels2)

# Combine datasets
combined_dataset = ConcatDataset([dataset1_tensor, dataset2_tensor])

# Split into training and validation sets
train_size = int(0.8 * len(combined_dataset))
val_size = len(combined_dataset) - train_size
train_dataset, val_dataset = random_split(combined_dataset, [train_size, val_size])

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

# Fine-tune the model on the combined dataset
fine_tune_model(train_loader, val_loader, model, tokenizer)
