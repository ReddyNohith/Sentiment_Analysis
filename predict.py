"""
predict.py — Standalone Sentiment Analysis Inference Script
Task 3: Sentiment Analysis
Usage: python predict.py "Your review text here"
"""

import sys
import re
import joblib
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords', quiet=True)
STOP_WORDS = set(stopwords.words('english'))


def clean_text(text: str) -> str:
    """Clean a review string for inference."""
    text = text.lower()
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r"[^a-z\s']", '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = [w for w in text.split() if w not in STOP_WORDS and len(w) > 1]
    return ' '.join(tokens)


def predict_sentiment(text: str,
                      model_path: str = 'lr_sentiment_model.pkl',
                      vec_path: str = 'tfidf_vectorizer.pkl') -> dict:
    """
    Predict sentiment for a single review.

    Args:
        text: Raw review text
        model_path: Path to saved LogisticRegression model
        vec_path: Path to saved TF-IDF vectorizer

    Returns:
        dict with keys: sentiment, confidence, prob_neg, prob_pos
    """
    model = joblib.load(model_path)
    vec   = joblib.load(vec_path)

    cleaned = clean_text(text)
    X = vec.transform([cleaned])

    label = model.predict(X)[0]
    proba = model.predict_proba(X)[0]

    return {
        'text':       text,
        'sentiment':  'Positive' if label == 1 else 'Negative',
        'confidence': round(float(proba.max()), 4),
        'prob_neg':   round(float(proba[0]), 4),
        'prob_pos':   round(float(proba[1]), 4),
    }


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python predict.py "Your review text here"')
        print()
        print('Running demo examples...')
        demos = [
            'One of the greatest films I have ever seen. Truly moving.',
            'Complete garbage. The director had no idea what he was doing.',
            'It was decent. Not great, not terrible — just okay.',
        ]
        for d in demos:
            r = predict_sentiment(d)
            print(f'  [{r["sentiment"]:8s} {r["confidence"]:.3f}]  {d}')
    else:
        text = ' '.join(sys.argv[1:])
        result = predict_sentiment(text)
        print(f'Text       : {result["text"]}')
        print(f'Sentiment  : {result["sentiment"]}')
        print(f'Confidence : {result["confidence"]}')
        print(f'Prob Neg   : {result["prob_neg"]}')
        print(f'Prob Pos   : {result["prob_pos"]}')
