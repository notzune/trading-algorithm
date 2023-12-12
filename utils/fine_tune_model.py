from torch.optim import AdamW
from tqdm import tqdm


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
