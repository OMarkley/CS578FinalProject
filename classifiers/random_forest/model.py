import joblib
import os
from config import TRAIN_MODEL_PATH, TRAIN_VECTORIZER_PATH, EVAL_MODEL_PATH, EVAL_VECTORIZER_PATH, USE_PRETRAINED_W2V, VECTORIZER_TYPE

def train_model(x_train_vec, y_train):
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(x_train_vec, y_train)
    return model

def save_model(model, vectorizer):
    os.makedirs(os.path.dirname(TRAIN_MODEL_PATH), exist_ok=True)
    joblib.dump(model, TRAIN_MODEL_PATH)
    if (not USE_PRETRAINED_W2V) or (VECTORIZER_TYPE == 'tfidf'):
        joblib.dump(vectorizer, TRAIN_VECTORIZER_PATH)
    else:
        print("[INFO] Skipping vectorizer save: using pre-trained Word2Vec")

def load_eval_model():
    model = joblib.load(EVAL_MODEL_PATH)
    vectorizer = joblib.load(EVAL_VECTORIZER_PATH)
    return model, vectorizer
