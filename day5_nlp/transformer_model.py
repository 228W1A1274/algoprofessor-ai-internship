import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_files
import os
import shutil
import numpy as np

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == 'cpu':
    print("WARNING: You are training BERT on a CPU. This will be very slow (Hours).")

class IMDBDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {"accuracy": (predictions == labels).mean()}

def train_bert_model():
    print("--- 1. Loading FULL Data (25,000 Reviews) ---")
    
    if not os.path.exists("aclImdb/train"):
        print("Downloading dataset...")
        url = "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
        if not os.path.exists("aclImdb_v1.tar.gz"):
            os.system(f"wget {url}")
        os.system("tar -xzf aclImdb_v1.tar.gz")
    
    data_dir = "aclImdb/train"
    data = load_files(data_dir, categories=['pos', 'neg'], encoding='utf-8', decode_error='replace')
    texts = data.data
    labels = data.target

    # --- REMOVED THE LIMIT HERE ---
    # Now using all 25,000 samples
    print(f"Training on {len(texts)} samples.")

    X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

    print("--- 2. Tokenization ---")
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    
    train_encodings = tokenizer(X_train, truncation=True, padding=True, max_length=128)
    test_encodings = tokenizer(X_test, truncation=True, padding=True, max_length=128)

    train_dataset = IMDBDataset(train_encodings, y_train)
    test_dataset = IMDBDataset(test_encodings, y_test)

    print("--- 3. Training with DistilBERT ---")
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')
    model.to(device)

    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,              
        per_device_train_batch_size=8,   
        per_device_eval_batch_size=16,
        logging_dir='./logs',
        logging_steps=100,               # Log less frequently
        eval_strategy="epoch",           
        save_strategy="epoch",
        load_best_model_at_end=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics
    )

    trainer.train()

    print("--- 4. Saving Model ---")
    model.save_pretrained("./models/bert_sentiment")
    tokenizer.save_pretrained("./models/bert_sentiment")
    print("Model saved to ./models/bert_sentiment")

if __name__ == "__main__":
    train_bert_model()
