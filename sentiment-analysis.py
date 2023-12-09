import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch.nn.functional as F

# Load FinBERT model and tokenizer
model_name = "ProsusAI/finbert"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)


def preprocess(text):
    """
    Preprocesses the given text for sentiment analysis.

    Args:
    text (str): The text to be analyzed.

    Returns:
    torch.Tensor: The processed inputs as tensors.
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    return inputs


def analyze_sentiment(text):
    """
    Analyzes the sentiment of the given text.

    Args:
    text (str): The text to be analyzed.

    Returns:
    torch.Tensor: The logits representing sentiment scores.
    """
    inputs = preprocess(text)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.logits


def get_sentiment(probabilities):
    """
    Determines the sentiment class from the given probabilities.

    Args:
    probabilities (torch.Tensor): The probabilities tensor.

    Returns:
    int: The sentiment class (0: Negative, 1: Neutral, 2: Positive).
    """
    probabilities = F.softmax(probabilities, dim=1)
    sentiment = torch.argmax(probabilities, dim=1)
    return sentiment.item()


# Example text for analysis
example_text = "The company's stock price surged following the announcement of record profits."

# Analyze sentiment
logits = analyze_sentiment(example_text)
sentiment_class = get_sentiment(logits)

# Output sentiment class
sentiment_labels = ["Negative", "Neutral", "Positive"]
print(f"Sentiment: {sentiment_labels[sentiment_class]}")
