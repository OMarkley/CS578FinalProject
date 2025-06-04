import argparse
import os
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from vectorize import preprocess_data
from model import train_model, save_model, load_eval_model
from evaluate import evaluate_model, save_summary_and_predictions
from datetime import datetime

# Fine-grained dataset and group roles
DATASETS = [
    {
        "path": "data/parsed_data/benign_realprogrammersusevim.csv",
        "groups": {
            "benign_realprogrammersusevim": "eval"
        }
    },
    {
        "path": "data/parsed_data/spam_realprogrammersusevim.csv",
        "groups": {
            "spam_realprogrammersusevim": "eval"
        }
    },
    {
        "path": "data/parsed_data/benign_realprogrammersusevim_train.csv",
        "groups": {
            "benign_realprogrammersusevim_train": "train"
        }
    },
    {
        "path": "data/parsed_data/spam_realprogrammersusevim_train.csv",
        "groups": {
            "spam_realprogrammersusevim_train": "train"
        }
    },
    {
        "path": "data/parsed_data/monkey_phishing_2024.csv",
        "groups": {
            "monkey_phishing_2024": "train"
        }
    },
    {
        "source": "huggingface",
        "dataset": "talby/spamassassin",
        "subset": "text",
        "groups": {
            "spam": "train",
            "spam_2": "train",
            "easy_ham": "train",
            "easy_ham_2": "train",
            "hard_ham": "train"
        }
    }
]

def load_and_prepare_data(mode):
    train_rows = []
    test_rows = []

    for entry in DATASETS:
        if entry.get("source") == "huggingface":
            dataset = load_dataset(entry["dataset"], entry.get("subset", None))
            df = pd.DataFrame(dataset['train'])
        else:
            df = pd.read_csv(entry["path"])

        group_roles = entry["groups"]
        for group_name, role in group_roles.items():
            group_df = df[df["group"] == group_name]
            if mode == "train" and role == "train":
                x_train, x_test = train_test_split(group_df, test_size=0.2, stratify=group_df["label"], random_state=17)
                train_rows.append(x_train)
                test_rows.append(x_test)
            elif mode == "eval" and role == "eval":
                test_rows.append(group_df)

    train_df = pd.concat(train_rows, ignore_index=True) if train_rows else pd.DataFrame(columns=["text", "label", "group"])
    test_df = pd.concat(test_rows, ignore_index=True) if test_rows else pd.DataFrame(columns=["text", "label", "group"])

    return train_df, test_df

def split_data(train_df, test_df):
    x_train = train_df["text"]
    y_train = train_df["label"]
    x_test = test_df["text"]
    y_test = test_df["label"]

    print(f"[INFO] x_train size: {len(x_train)}")
    print(f"[INFO] x_test size: {len(x_test)}")

    if not train_df.empty:
        print("[INFO] Train group breakdown:")
        print(train_df["group"].value_counts().to_string())

    print("[INFO] Test group breakdown:")
    print(test_df["group"].value_counts().to_string())

    return x_train, y_train, x_test, y_test, test_df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["train", "eval"], help="Train or evaluate the model")
    parser.add_argument("--eval-dir", help="Directory of model/vectorizer from a previous run (required for eval)")
    parser.add_argument("--vectorizer-type", choices=["tfidf", "word2vec"], default="tfidf",
                        help="Type of vectorizer to use: 'tfidf' or 'word2vec'")
    parser.add_argument("--use-pretrained-w2v", action="store_true", default=False,
                        help="Use pre-trained Google News Word2Vec (only for word2vec vectorizer)")
    parser.add_argument("--save-output", action="store_true", default=False, help="Save evaluation results to CSV")
    parser.add_argument("--grid-search", action="store_true", default=False, help="Enable GridSearchCV hyperparameter tuning")

    args = parser.parse_args()

    if args.mode == "eval" and not args.eval_dir:
        parser.error("--eval-dir is required in eval mode")

    train_df, test_df = load_and_prepare_data(args.mode)
    print(f"[INFO] Total dataset size: {len(train_df) + len(test_df)}")

    x_train, y_train, x_test, y_test, test_df = split_data(train_df, test_df)

    if args.mode == "train":
        run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_dir = os.path.join("classifiers", "random_forest", f"model_{run_timestamp}")

        x_train_vec, x_test_vec, vectorizer = preprocess_data(
            x_train, x_test,
            vectorizer_type=args.vectorizer_type,
            use_pretrained_w2v=args.use_pretrained_w2v
        )
        model, grid_df = train_model(x_train_vec, y_train, use_grid_search=args.grid_search)

        os.makedirs(model_dir, exist_ok=True)
        save_model(model, vectorizer, model_dir, args.vectorizer_type, args.use_pretrained_w2v)

        training_eval_dir = os.path.join(model_dir, "training_eval")
        os.makedirs(training_eval_dir, exist_ok=True)
        if args.grid_search and grid_df is not None:
            grid_path = os.path.join(training_eval_dir, "grid_search_results.csv")
            grid_df.to_csv(grid_path, index=False)
            print(f"[INFO] Saved grid search results to {grid_path}")
        results_df = evaluate_model(model, x_test_vec, y_test, x_test, test_df)
        save_summary_and_predictions(
            results_df,
            vectorizer_type=args.vectorizer_type,
            eval_dir=training_eval_dir,
            save_all=args.save_output
        )

    else:  # eval mode
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        test_eval_dir = os.path.join(args.eval_dir, f"test_eval_{timestamp}")

        model, vectorizer = load_eval_model(args.eval_dir)
        _, x_test_vec, _ = preprocess_data(
            [], x_test,
            eval_dir=args.eval_dir,
            vectorizer_type=args.vectorizer_type,
            use_pretrained_w2v=args.use_pretrained_w2v
        )
        results_df = evaluate_model(model, x_test_vec, y_test, x_test, test_df)
        save_summary_and_predictions(
            results_df,
            vectorizer_type=args.vectorizer_type,
            eval_dir=test_eval_dir,
            save_all=args.save_output
        )

if __name__ == "__main__":
    main()
