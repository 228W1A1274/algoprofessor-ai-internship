import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
from text_preprocessing import clean_text
import tensorflow as tf
import os
from sklearn.datasets import load_files

def train_classic_sentiment():
    print("--- 1. Loading FULL IMDB Dataset ---")
    
    # Step 1: Download and Extract (Keras handles this automatically)
    dataset_path = tf.keras.utils.get_file(
        fname="aclImdb_v1.tar.gz", 
        origin="http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz", 
        extract=True
    )
    
    # Step 2: Find the folder
    # Keras usually downloads to ~/.keras/datasets/
    # The extracted folder is named 'aclImdb'
    base_dir = os.path.dirname(dataset_path)
    data_dir = os.path.join(base_dir, 'aclImdb', 'train')
    
    # Check if the folder exists
    if not os.path.exists(data_dir):
        # Fallback: Sometimes Keras returns the directory path itself
        data_dir = os.path.join(dataset_path, 'aclImdb', 'train')
        
    print(f"Data directory located at: {data_dir}")
    
    # Step 3: Load Data
    print("Loading data files (This takes a moment)...")
    try:
        data = load_files(data_dir, categories=['pos', 'neg'], encoding='utf-8', decode_error='replace')
    except FileNotFoundError:
        print(f"ERROR: Could not find 'pos' and 'neg' folders inside {data_dir}")
        return

    df = pd.DataFrame({'text': data.data, 'label': data.target})
    print(f"Dataset Shape: {df.shape} (Full Training Data)")
    
    print("--- 2. Preprocessing (Cleaning 25,000 reviews...) ---")
    df['clean_text'] = df['text'].apply(clean_text)
    
    X = df['clean_text']
    y = df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("--- 3. Vectorization (TF-IDF) ---")
    vectorizer = TfidfVectorizer(max_features=10000) 
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    print("--- 4. Training Logistic Regression ---")
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, y_train)
    
    # Evaluate
    preds = model.predict(X_test_vec)
    acc = accuracy_score(y_test, preds)
    print(f"\nClassic Model Accuracy: {acc*100:.2f}%")
    
    joblib.dump(model, 'models/classic_sentiment_model.pkl')
    joblib.dump(vectorizer, 'models/tfidf_vectorizer.pkl')
    print("Model saved to models/")

if __name__ == "__main__":
    train_classic_sentiment()
