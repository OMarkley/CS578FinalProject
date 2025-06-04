import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
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

def save_summary_and_predictions(results_df, vectorizer_type, eval_dir, save_all=False):
    os.makedirs(eval_dir, exist_ok=True)
    summary_path = os.path.join(eval_dir, "summary.csv")
    prediction_path = os.path.join(eval_dir, "predictions.csv")

    results_df["correct"] = results_df["true_label"] == results_df["predicted_label"]

    # Per-group accuracy
    group_accuracy = results_df.groupby("group")["correct"].mean().reset_index(name="accuracy")
    
    overall_accuracy = accuracy_score(results_df["true_label"], results_df["predicted_label"])


    # Classification report
    report = classification_report(results_df["true_label"], results_df["predicted_label"], output_dict=True)
    report_df = pd.DataFrame(report).transpose().reset_index().rename(columns={"index": "class"})

    # Save summary with vectorizer info
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"Vectorizer Type: {vectorizer_type}\n\n")
        f.write(f"Overall Accuracy: {overall_accuracy:.4f}\n\n")
        f.write("Per-Group Accuracy:\n")
        group_accuracy.to_csv(f, index=False)
        f.write("\nClassification Report:\n")
        report_df.to_csv(f, index=False)

    print(f"\nSaved summary to '{summary_path}'")

    if save_all:
        results_df.to_csv(prediction_path, index=False)
        print(f"Saved all predictions to '{prediction_path}'")

    # Sample insights
    false_pos = results_df[(results_df["true_label"] == 0) & (results_df["predicted_label"] == 1)]
    false_neg = results_df[(results_df["true_label"] == 1) & (results_df["predicted_label"] == 0)]
    correct = results_df[results_df["true_label"] == results_df["predicted_label"]]

    print("\nFalse Positives:")
    print(false_pos.sample(min(5, len(false_pos)), random_state=42))

    print("\nFalse Negatives:")
    print(false_neg.sample(min(5, len(false_neg)), random_state=42))

    print("\nCorrect Predictions:")
    print(correct.sample(min(5, len(correct)), random_state=42))