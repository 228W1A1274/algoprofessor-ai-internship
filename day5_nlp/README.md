# Day 5: NLP & Text Analysis

## Project Overview
This project implements Natural Language Processing (NLP) techniques to analyze text data. It includes a classic TF-IDF model, a state-of-the-art BERT model, and a multi-class topic classifier.

## Deliverables
1. **text_preprocessing.py**: Pipeline for cleaning and lemmatizing text.
2. **sentiment_analyzer.py**: Logistic Regression model achieving ~85% on IMDB data.
3. **transformer_model.py**: DistilBERT model for deep context understanding.
4. **text_classifier.py**: Classification system for News topics (Space, Med, Graphics).
5. **inference_api.py**: Simple script to test the models with new inputs.

## How to Run
1. Install dependencies: `pip install -r requirements.txt`
2. Train Sentiment Model: `python sentiment_analyzer.py`
3. Train BERT Model: `python transformer_model.py`
4. Train Topic Classifier: `python text_classifier.py`
5. Test Predictions: `python inference_api.py`
