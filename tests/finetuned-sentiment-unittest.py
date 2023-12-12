import torch
import unittest
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class FinBERTTest(unittest.TestCase):
    def setUp(self):
        model_dir = "/Users/zeyad/Documents/GitHub/trading-algorithm/finetuned-finbert"
        # Ensure the tokenizer is saved and loaded from the correct directory
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)

    def test_on_unseen_data(self):
        test_sentences = [
            "The company had a massive growth in profits this quarter.",
            "Market fluctuation has led to a considerable loss in revenue.",
            # Add more test sentences as needed
        ]

        inputs = self.tokenizer(test_sentences, padding=True, truncation=True, return_tensors='pt')
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

        # Output the probabilities for each class for each test sentence
        for i, prob in enumerate(probs):
            print(f"Sentence: '{test_sentences[i]}'")
            print(f"Probabilities: {prob}")
            print(f"Predicted sentiment: {prob.argmax().item()}")
            print("-" * 80)

        # If you have expected labels, compare them
        expected_labels = [2, 0]  # Replace with your expected labels for the test sentences
        predicted_labels = probs.argmax(dim=1).tolist()
        self.assertEqual(predicted_labels, expected_labels)

        # Additional verbosity
        print(f"Expected labels: {expected_labels}")
        print(f"Predicted labels: {predicted_labels}")


if __name__ == '__main__':
    unittest.main(verbosity=2)
