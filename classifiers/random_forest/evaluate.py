### evaluate.py

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from config import RESULTS_PATH, SAVE_OUTPUT
import os

def evaluate_model(model, x_test_vec, y_test, x_test, df):
    y_pred = model.predict(x_test_vec)
    print("\nAccuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    results_df = pd.DataFrame({
        "text": x_test,
        "true_label": y_test,
        "predicted_label": y_pred
    })

    # Use .map to avoid duplicates during join
    group_map = df.drop_duplicates("text").set_index("text")["group"].to_dict()
    results_df["group"] = results_df["text"].map(group_map)

    return results_df

def save_and_report(results_df):
    if SAVE_OUTPUT:
        os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
        results_df.to_csv(RESULTS_PATH, index=False)
        print(f"\nSaved predictions to '{RESULTS_PATH}'")

    results_df["correct"] = results_df["true_label"] == results_df["predicted_label"]

    group_accuracy = results_df.groupby("group")["correct"].mean().reset_index(name="accuracy")
    print("\nPer-Group Accuracy:")
    print(group_accuracy)

    false_positives = results_df[(results_df["true_label"] == 0) & (results_df["predicted_label"] == 1)]
    false_negatives = results_df[(results_df["true_label"] == 1) & (results_df["predicted_label"] == 0)]

    print("\nFalse Positives:")
    print(false_positives.sample(min(5, len(false_positives)), random_state=42))

    print("\nFalse Negatives:")
    print(false_negatives.sample(min(5, len(false_negatives)), random_state=42))
