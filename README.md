# Random Forest Spam/Phishing Classifier

This project implements a spam and phishing email classifier using a Random Forest model. It supports TF-IDF and Word2Vec vectorization and allows fine-grained control over which dataset groups are used for training, evaluation, or both.

---

## 📦 Directory Structure

```
project-root/
├── main.py
├── config.py
├── model.py
├── vectorize.py
├── preprocess.py
├── evaluate.py
├── utils.py
├── classifiers/
│   └── random_forest/
│       └── run_<timestamp>/
│           ├── rf_model.pkl
│           ├── vectorizer.pkl (if saved)
│           └── predictions.csv
└── data/
    └── extractors/
        └── parsed_data/
```

---

## 🚀 Usage

### 1. Train a New Model

```bash
python main.py train \
  --vectorizer-type tfidf \
  [--use-pretrained-w2v]
```

**Options:**
- `--vectorizer-type`: `"tfidf"` or `"word2vec"`
- `--use-pretrained-w2v`: Flag to use Google News Word2Vec (only for `word2vec`)

The trained model and vectorizer will be saved in a timestamped directory inside `classifiers/random_forest/`.

---

### 2. Evaluate an Existing Model

```bash
python main.py eval \
  --eval-dir classifiers/random_forest/run_<timestamp> \
  --vectorizer-type tfidf \
  [--use-pretrained-w2v] \
  [--save-output]
```

**Required:**
- `--eval-dir`: Path to previously saved model directory

**Optional:**
- `--save-output`: Saves a CSV of predictions + group accuracy stats

**Evaluation results include:**
- Overall accuracy
- Classification report
- Per-group accuracy
- False positives and false negatives

---

## 📙 Dataset Configuration

Datasets and group behavior are defined directly in `main.py` via `DATASETS`:

```python
DATASETS = [
    {
        "path": "data/extractors/parsed_data/benign.csv",
        "groups": {
            "benign_group": "mixed"
        }
    },
    {
        "source": "huggingface",
        "dataset": "talby/spamassassin",
        "subset": "text",
        "groups": {
            "spam": "train",
            "hard_ham": "eval"
        }
    }
]
```

**Group Roles:**
- `train`: Only used for training
- `eval`: Only used for evaluation
- `mixed`: Randomly split into train/test (80/20 default)

---

## 📝 Notes

- All CSVs must contain columns: `text`, `label`, and `group`
- Word2Vec support includes training from scratch or loading pre-trained embeddings
- Outputs from evaluation are saved in the same directory as the model, unless overridden

---

## ✅ Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
```

Make sure the following libraries are included:
- `nltk`
- `scikit-learn`
- `gensim`
- `beautifulsoup4`
- `pandas`
- `datasets`

---

## 📫 Contact

For questions or contributions, open an issue or contact the repo maintainer.