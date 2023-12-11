import torch
import unittest
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class FinBERTTest(unittest.TestCase):
    def setUp(self):
        model_dir = "/Users/zeyad/Documents/GitHub/trading-algorithm/finetuned-finbert"
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)

    def test_on_unseen_data(self):
        # Here you would load your unseen test data
        test_sentences = [
            "The company had a massive growth in profits this quarter.",
            "Market fluctuation has led to a considerable loss in revenue.",
            # Add more test sentences as needed
        ]

        # Tokenize the test data
        inputs = self.tokenizer(test_sentences, padding=True, truncation=True, return_tensors='pt')

        # Move the inputs to the same device as the model
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        # Perform inference
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Convert outputs to probabilities
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

        # Assertions to check if probabilities make sense
        # This is where you would write your own assertions
        for prob in probs:
            self.assertTrue(torch.is_tensor(prob))
            self.assertTrue(prob.argmax().item() in [0, 1, 2])  # Assuming you have three classes

        # If you have expected labels, compare them
        expected_labels = [2, 0]  # Replace with your expected labels for the test sentences
        predicted_labels = probs.argmax(dim=1).tolist()
        self.assertEqual(predicted_labels, expected_labels)


if __name__ == '__main__':
    unittest.main()
