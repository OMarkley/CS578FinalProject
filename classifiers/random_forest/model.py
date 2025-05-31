from sklearn.ensemble import RandomForestClassifier
import joblib
from config import MODEL_PATH, VECTORIZER_PATH
import os

def train_model(x_train_vec, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(x_train_vec, y_train)
    return model

def save_model(model, vectorizer):
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)

def load_model():
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer