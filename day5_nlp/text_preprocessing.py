import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import string

# Initialize the tools
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    """
    Takes a raw string and cleans it:
    1. Lowercase
    2. Remove punctuation
    3. Remove stopwords
    4. Lemmatize (run -> running)
    """
    if not isinstance(text, str):
        return ""

    # 1. Lowercase
    text = text.lower()
    
    # 2. Remove Punctuation (replace with space)
    text = re.sub(f"[{string.punctuation}]", " ", text)
    
    # 3. Tokenize (Split into words)
    tokens = text.split()
    
    # 4. Remove Stopwords & Lemmatize
    clean_tokens = [
        lemmatizer.lemmatize(word) 
        for word in tokens 
        if word not in stop_words and len(word) > 2
    ]
    
    return " ".join(clean_tokens)

if __name__ == "__main__":
    # Test it
    sample = "The movie was absolutely amazing! I loved the acting and the plot was running fast."
    cleaned = clean_text(sample)
    print(f"Original: {sample}")
    print(f"Cleaned:  {cleaned}")
