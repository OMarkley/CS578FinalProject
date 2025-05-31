import os
from datetime import datetime

BASE_DIR = os.path.join("classifiers", "random_forest")
RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = os.path.join(BASE_DIR, f"run_{RUN_TIMESTAMP}")

VECTORIZER_TYPE = 'tfidf'  # or 'word2vec'
MODEL_PATH = os.path.join(OUTPUT_DIR, 'rf_model.pkl')
VECTORIZER_PATH = os.path.join(OUTPUT_DIR, 'vectorizer.pkl')
RESULTS_PATH = os.path.join(OUTPUT_DIR, 'spam_predictions.csv')
EXCLUDE_HARD_HAM_FROM_TRAINING = True
SAVE_OUTPUT = True
