import argparse
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from vectorize import preprocess_data
from model import train_model, save_model, load_model
from evaluate import evaluate_model, save_and_report
from config import EXCLUDE_HARD_HAM_FROM_TRAINING

def load_and_prepare_data():
    data = load_dataset("talby/spamassassin", "text")
    df = pd.DataFrame(data['train'])
    return df

def split_data(df):
    hard_ham_df = df[df["group"] == "hard_ham"]
    standard_df = df[df["group"] != "hard_ham"]

    if EXCLUDE_HARD_HAM_FROM_TRAINING:
        x = standard_df["text"]
        y = standard_df["label"]
    else:
        x = df["text"]
        y = df["label"]

    x_train, x_test_part, y_train, y_test_part = train_test_split(x, y, test_size=0.2, stratify=y, random_state=17)

    if EXCLUDE_HARD_HAM_FROM_TRAINING:
        x_test = pd.concat([x_test_part, hard_ham_df["text"]])
        y_test = pd.concat([y_test_part, hard_ham_df["label"]])
    else:
        x_test = x_test_part
        y_test = y_test_part
        
    print("x_train:", len(x_train))
    print("x_test_part:", len(x_test_part))
    print("hard_ham_df:", len(hard_ham_df))
    print("x_test (final):", len(x_test))

    return x_train, y_train, x_test, y_test, df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "predict"], default="train", help="Run mode")
    args = parser.parse_args()

    df = load_and_prepare_data()
    print("Total rows in dataset:", len(df))
    print("Hard ham:", len(df[df['group'] == 'hard_ham']))
    print("Standard:", len(df[df['group'] != 'hard_ham']))

    x_train, y_train, x_test, y_test, df = split_data(df)


    if args.mode == "train":
        x_train_vec, x_test_vec, vectorizer = preprocess_data(x_train, x_test)
        model = train_model(x_train_vec, y_train)
        save_model(model, vectorizer)
    else:
        model, vectorizer = load_model()
        _, x_test_vec, _ = preprocess_data([], x_test)

    results_df = evaluate_model(model, x_test_vec, y_test, x_test, df)
    save_and_report(results_df)

if __name__ == "__main__":
    main()