import joblib
from text_preprocessing import clean_text
import sys

def predict_sentiment(text):
    try:
        model = joblib.load('models/classic_sentiment_model.pkl')
        vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
        
        # Preprocess
        clean = clean_text(text)
        # Vectorize
        vec = vectorizer.transform([clean])
        # Predict
        prob = model.predict_proba(vec)[0]
        pred = model.predict(vec)[0]
        
        label = "POSITIVE" if pred == 1 else "NEGATIVE"
        confidence = prob[pred]
        
        return f"Sentiment: {label} ({confidence*100:.1f}%)"
    except Exception as e:
        return "Model not found. Run sentiment_analyzer.py first."

def predict_topic(text):
    try:
        model = joblib.load('models/topic_classifier.pkl')
        categories = ['Graphics', 'Medicine', 'Space'] # The order matches target_names
        
        pred_idx = model.predict([text])[0]
        return f"Topic: {categories[pred_idx]}"
    except:
        return "Topic Model not found."

if __name__ == "__main__":
    print("--- AI Text Analyst ---")
    sample_text = "The doctor said the surgery was successful."
    print(f"Input: {sample_text}")
    print(predict_sentiment(sample_text))
    print(predict_topic(sample_text))
