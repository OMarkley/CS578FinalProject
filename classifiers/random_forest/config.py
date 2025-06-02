import os
from datetime import datetime

BASE_DIR = os.path.join("classifiers", "random_forest")

# Results Storage
RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = os.path.join(BASE_DIR, f"run_{RUN_TIMESTAMP}")
TRAIN_MODEL_PATH = os.path.join(OUTPUT_DIR, 'rf_model.pkl')
TRAIN_VECTORIZER_PATH = os.path.join(OUTPUT_DIR, 'vectorizer.pkl')
RESULTS_PATH = os.path.join(OUTPUT_DIR, 'spam_predictions.csv')

VECTORIZER_TYPE = 'tfidf'  # 'tfidf' or 'word2vec'
USE_PRETRAINED_W2V = True # Use Google News Word2Vec

# IF mode is 'eval' and USE_PRETRAINED_W2V is False
EVAL_MODEL_PATH = os.path.join(BASE_DIR, "run_20250601_221059", "rf_model.pkl")
EVAL_VECTORIZER_PATH = os.path.join(BASE_DIR, "run_20250601_221059", "vectorizer.pkl")

# Training Set Options
EXCLUDE_HARD_HAM_FROM_TRAINING = True
PHISHING_EMAIL_PATH = os.path.join("data", "raw_data", "phishing-2024_plaintext_output.txt")