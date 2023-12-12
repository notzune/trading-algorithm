import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def sentiment_analysis(text, verbose=False, model_path="./finetuned-finbert", tokenizer_path="./finetuned-finbert"):
    """
    Performs sentiment analysis on the given text.

    Args:
    text (str): The text to be analyzed.
    verbose (bool): Flag to output detailed information including probabilities.
    model_path (str): Path to the fine-tuned model directory.
    tokenizer_path (str): Path to the tokenizer directory.

    Returns:
    str: The sentiment class (Negative, Neutral, Positive).
    Optional: Prints detailed probabilities if verbose is True.
    """
    # Load model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    # Preprocess the text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)

    # Analyze sentiment
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    probabilities = F.softmax(logits, dim=1)

    # Determine sentiment class
    sentiment_class = torch.argmax(probabilities, dim=1).item()
    sentiment_labels = ["Negative", "Neutral", "Positive"]
    sentiment = sentiment_labels[sentiment_class]

    # Verbose output
    if verbose:
        print(f"[[Negative, Neutral, Positive]]")
        print(f"Probabilities: {probabilities.numpy()[0]}")
        print(f"Sentiment: {sentiment}")

    return sentiment


# Example usage
example_text = "The company's stock price surged following the announcement of record profits."
sentiment = sentiment_analysis(example_text, verbose=True)

# Output sentiment class
print(f"Sentiment: {sentiment}")
