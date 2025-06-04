import os
from datetime import datetime

BASE_DIR = os.path.join("classifiers", "random_forest")

# Results Storage
RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = os.path.join(BASE_DIR, f"model_{RUN_TIMESTAMP}")
TRAIN_MODEL_PATH = os.path.join(OUTPUT_DIR, 'rf_model.pkl')
TRAIN_VECTORIZER_PATH = os.path.join(OUTPUT_DIR, 'vectorizer.pkl')
RESULTS_PATH = os.path.join(OUTPUT_DIR, 'spam_predictions.csv')