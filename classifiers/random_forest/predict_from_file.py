import re
import os
import pandas as pd
import joblib
from preprocess import preprocess_text
from model import load_model
from config import SAVE_OUTPUT, RESULTS_PATH


def load_extracted_emails(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        raw = f.read()

    emails = re.split(r"--- Email #[0-9]+ ---", raw)
    emails = [email.strip() for email in emails if email.strip()]
    return emails


def predict_emails(emails):
    # Load model and vectorizer
    model, vectorizer = load_model()

    # Preprocess
    preprocessed = [preprocess_text(email) for email in emails]

    # Vectorize
    if hasattr(vectorizer, "transform"):
        X = vectorizer.transform(preprocessed)
    else:
        def vectorize(tokens):
            vectors = [vectorizer.wv[w] for w in tokens if w in vectorizer.wv]
            return sum(vectors) / len(vectors) if vectors else [0.0] * vectorizer.vector_size
        import numpy as np
        X = np.array([vectorize(t) for t in preprocessed])

    # Predict
    predictions = model.predict(X)

    results_df = pd.DataFrame({
        "text": emails,
        "predicted_label": predictions
    })
    results_df["spam_label"] = results_df["predicted_label"].map({0: "Not Spam", 1: "Spam"})

    # Save if needed
    if SAVE_OUTPUT:
        os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
        results_df.to_csv(RESULTS_PATH, index=False)
        print(f"Saved predictions to: {RESULTS_PATH}")

    # Display
    print("\nSummary:")
    print(results_df["predicted_label"].value_counts().rename(index={0: "Not Spam", 1: "Spam"}))

    print("\nSample Spam:")
    print(results_df[results_df["predicted_label"] == 1][["spam_label", "text"]].sample(min(5, sum(predictions)), random_state=42))

    print("\nSample Not Spam:")
    print(results_df[results_df["predicted_label"] == 0][["spam_label", "text"]].sample(min(5, len(results_df) - sum(predictions)), random_state=42))


if __name__ == "__main__":
    input_path = "data/raw_data/phishing-2024_plaintext_output.txt"
    emails = load_extracted_emails(input_path)
    predict_emails(emails)
