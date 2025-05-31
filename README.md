Datasets

Training: huggingface talby/spamassassin
Testing (Phishing): Jose Monkey https://monkey.org/~jose/phishing/
- Extraction code from https://github.com/liakoyras/thesis-phishing-email-detection/blob/main/Import%20Text%20Data.ipynb

# 📧 Spam Classifier (Random Forest + TF-IDF/Word2Vec)

This project classifies email messages as **Spam** or **Not Spam** using a trained Random Forest model. The pipeline supports both training/testing on labeled datasets and batch predictions on unlabeled email text extracted from `.mbox`-style archives.

---

## 🔧 Configuration (`config.py`)

| Variable                         | Description                                                                       |
| -------------------------------- | --------------------------------------------------------------------------------- |
| `VECTORIZER_TYPE`                | `"tfidf"` or `"word2vec"` – determines which text vectorization method to use.    |
| `EXCLUDE_HARD_HAM_FROM_TRAINING` | If `True`, excludes `hard_ham` from the training set and adds it to the test set. |
| `SAVE_OUTPUT`                    | If `True`, prediction results will be saved to a timestamped CSV.                 |
| `MODEL_PATH` / `VECTORIZER_PATH` | Where the trained model/vectorizer are saved. Automatically updated each run.     |
| `RESULTS_PATH`                   | Location of the saved predictions file. Also updated each run.                    |

---

## 🚀 Training and Testing

To train the model on the SpamAssassin dataset and evaluate it:

```bash
python main.py --mode train
```

To test a previously trained model on the same data:

```bash
python main.py --mode predict
```

Output includes:

* Overall accuracy and classification report
* Group-based accuracy breakdown
* Sample false positives and false negatives
* Results saved to a CSV file if `SAVE_OUTPUT` is `True`

---

## 📄 Predict on Extracted Email Data

If you've extracted plain text emails using a tool or script (e.g., from `.mbox`), you can run predictions like this:

```bash
python classifiers/random_forest/predict_from_file.py
```

This script reads from:

```
data/raw_data/phishing-2024_plaintext_output.txt
```

Format of this file must be:

```
--- Email #1 ---
Text of email 1

--- Email #2 ---
Text of email 2
```

Predictions will:

* Be displayed in the console
* Optionally saved to a CSV (if `SAVE_OUTPUT` is enabled)
* Show 5 samples each of predicted spam and non-spam emails

---

## 📦 Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
```

Includes:

* `pandas`
* `scikit-learn`
* `nltk`
* `gensim`
* `bs4`

---

## 📁 Project Structure

```
classifiers/random_forest/
├── main.py                  # Train/test logic
├── config.py                # Global configuration
├── preprocess.py            # Text cleaning and lemmatization
├── vectorize.py             # TF-IDF / Word2Vec vectorization
├── model.py                 # Training and model I/O
├── evaluate.py              # Evaluation and reporting
├── predict_from_file.py     # Run predictions on new plain text emails
├── run_<timestamp>/         # Output folder for each run
```
