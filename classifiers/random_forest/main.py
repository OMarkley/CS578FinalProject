import argparse
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from vectorize import preprocess_data
from model import train_model, save_model, load_eval_model
from evaluate import evaluate_model, save_and_report
from utils import load_phishing_emails_as_dataframe
from config import (
    PHISHING_EMAIL_PATH, EXCLUDE_HARD_HAM_FROM_TRAINING
)

def load_and_prepare_data():
    # Comment out phish_df if not needed/wanted
    base_df = pd.DataFrame(load_dataset("talby/spamassassin", "text")['train'])
    phish_df = load_phishing_emails_as_dataframe(PHISHING_EMAIL_PATH)
    df = pd.concat([base_df, phish_df], ignore_index=True)
    return df

def split_data(df):
    hard_ham_df = df[df["group"] == "hard_ham"]
    rest_df = df[df["group"] != "hard_ham"] if EXCLUDE_HARD_HAM_FROM_TRAINING else df

    x = rest_df["text"]
    y = rest_df["label"]
    group = rest_df["group"]

    x_train, x_test_part, y_train, y_test_part, group_train, group_test_part = train_test_split(
        x, y, group, test_size=0.2, stratify=y, random_state=17
    )

    if EXCLUDE_HARD_HAM_FROM_TRAINING:
        x_test = pd.concat([x_test_part, hard_ham_df["text"]])
        y_test = pd.concat([y_test_part, hard_ham_df["label"]])
        group_test = pd.concat([group_test_part, hard_ham_df["group"]])
    else:
        x_test = x_test_part
        y_test = y_test_part
        group_test = group_test_part

    test_df = pd.DataFrame({"text": x_test, "label": y_test, "group": group_test})
    
    print(f"[INFO] x_train size: {len(x_train)}")
    print(f"[INFO] x_test size: {len(x_test)}")

    train_groups = group_train.value_counts()
    test_groups = group_test.value_counts()

    print("[INFO] Train group breakdown:")
    print(train_groups.to_string())

    print("[INFO] Test group breakdown:")
    print(test_groups.to_string())
    return x_train, y_train, x_test, y_test, test_df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["train", "eval"], help="Train or evaluate the model")
    parser.add_argument("--save-output", action="store_true", default=False, help="Save evaluation results to CSV")
    args = parser.parse_args()

    df = load_and_prepare_data()
    
    print(f"[INFO] Total dataset size: {len(df)}")
    print("[INFO] Group distribution:\n", df["group"].value_counts())
    
    x_train, y_train, x_test, y_test, test_df = split_data(df)
    
    # test_df.to_json("test_set.json", orient="records", lines=True)

    if args.mode == "train":
        x_train_vec, x_test_vec, vectorizer = preprocess_data(x_train, x_test)
        model = train_model(x_train_vec, y_train)
        save_model(model, vectorizer)
    else:  # eval
        model, vectorizer = load_eval_model()
        _, x_test_vec, _ = preprocess_data([], x_test)

    results_df = evaluate_model(model, x_test_vec, y_test, x_test, test_df)
    if args.save_output:
        save_and_report(results_df, save_output=args.save_output)

if __name__ == "__main__":
    main()
