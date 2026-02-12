from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
import joblib

def train_topic_classifier():
    print("--- 1. Loading 20 Newsgroups Dataset ---")
    # We pick 3 specific categories to make it clear
    categories = ['sci.space', 'sci.med', 'comp.graphics']
    
    train_data = fetch_20newsgroups(subset='train', categories=categories)
    test_data = fetch_20newsgroups(subset='test', categories=categories)
    
    print(f"Categories: {train_data.target_names}")
    print(f"Training samples: {len(train_data.data)}")
    
    print("--- 2. Building Pipeline (TF-IDF + Naive Bayes) ---")
    # Naive Bayes is excellent for text classification
    model = make_pipeline(TfidfVectorizer(), MultinomialNB())
    
    print("--- 3. Training ---")
    model.fit(train_data.data, train_data.target)
    
    print("--- 4. Evaluation ---")
    preds = model.predict(test_data.data)
    acc = accuracy_score(test_data.target, preds)
    print(f"Multi-Class Accuracy: {acc*100:.2f}%")
    
    # Save
    joblib.dump(model, 'models/topic_classifier.pkl')
    print("Topic Classifier saved.")

if __name__ == "__main__":
    train_topic_classifier()
