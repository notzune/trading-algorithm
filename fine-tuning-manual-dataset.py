import torch
from torch.optim import AdamW
from torch.utils.data import TensorDataset, DataLoader, random_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Set the model name
model_name = "ProsusAI/finbert"

# Load pre-trained model and tokenizer from Huggingface
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)


# Function to read a text file and return the lines
def read_txt_file(file_path):
    """
    Reads a text file and returns the lines as a list.

    Parameters:
    file_path (str): Path to the text file.

    Returns:
    list: Lines of the file.
    """
    with open(file_path, 'r', encoding='utf-16') as file:
        lines = file.readlines()
        print(f"Read {len(lines)} lines from the file.")
    return lines


# Function to process the raw lines and extract sentences and labels
def preprocess_data(lines):
    """
    Processes raw lines from the dataset and separates sentences and labels.

    Parameters:
    lines (list): List of raw strings from the dataset.

    Returns:
    tuple: A tuple containing lists of sentences and corresponding labels.
    """
    sentences = []
    labels = []
    for line in lines:
        sentence, label_str = line.strip().split('@')
        sentences.append(sentence.strip())
        if label_str == 'positive':
            labels.append(2)
        elif label_str == 'neutral':
            labels.append(1)
        elif label_str == 'negative':
            labels.append(0)
        else:
            raise ValueError(f"Unknown label: {label_str}")
    print(f"Processed {len(sentences)} sentences.")
    return sentences, labels


# Function to tokenize the sentences using the loaded tokenizer
def tokenize_sentences(sentences):
    """
    Tokenizes sentences using the pre-trained tokenizer.

    Parameters:
    sentences (list): List of sentences to tokenize.

    Returns:
    BatchEncoding: Tokenized and encoded sentences ready for model input.
    """
    print(f"Tokenizing {len(sentences)} sentences.")
    return tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')


# Example usage of the above functions
file_path = './datasets/fin-phrase-bank/Sentences_AllAgree.txt'
lines = read_txt_file(file_path)
sentences, labels = preprocess_data(lines)
encoded_inputs = tokenize_sentences(sentences)

# Create tensors for the labels
labels_tensor = torch.tensor(labels)

# Create a dataset from the encoded inputs and labels
dataset = TensorDataset(encoded_inputs['input_ids'], encoded_inputs['attention_mask'], labels_tensor)
print("Created TensorDataset.")

# Split the dataset into training and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
print(f"Dataset split into {train_size} training samples and {val_size} validation samples.")

# Create DataLoaders for training and validation sets
batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
print("Created DataLoaders.")

optimizer = AdamW(model.parameters(), lr=5e-5)

# Training loop with debug messages
epochs = 3
for epoch in range(epochs):
    print(f"Starting epoch {epoch + 1}/{epochs}")
    # Training phase
    model.train()
    total_train_loss = 0
    for batch in train_loader:
        b_input_ids, b_input_mask, b_labels = batch
        model.zero_grad()
        outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
        loss = outputs.loss
        total_train_loss += loss.item()
        loss.backward()
        optimizer.step()
    avg_train_loss = total_train_loss / len(train_loader)
    print(f"Average training loss: {avg_train_loss}")

    # Validation phase
    model.eval()
    total_eval_loss = 0
    for batch in val_loader:
        b_input_ids, b_input_mask, b_labels = batch
        with torch.no_grad():
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
            loss = outputs.loss
            total_eval_loss += loss.item()
    avg_val_loss = total_eval_loss / len(val_loader)
    print(f"Validation loss: {avg_val_loss}")

# Save the fine-tuned model
model_path = "./finetuned-finbert"
model.save_pretrained(model_path)
print(f"Model saved to {model_path}")
