# Random Forest Phishing Classifier

Command-line interface for training and evaluating a Random Forest classifier for phishing email detection.

## Usage

python main.py <train|eval> [options]


### Training

python main.py train --vectorizer-type tfidf|word2vec [--use-pretrained-w2v] [--grid-search]

- `--vectorizer-type` (required): `tfidf` or `word2vec`
- `--use-pretrained-w2v` (optional): Use pre-trained Google News Word2Vec (only with `word2vec`)
- `--grid-search` (optional): Enable hyperparameter tuning

Trained models are saved to:

classifiers/random_forest/model_<timestamp>/

### Evaluation

python main.py eval --eval-dir <model_dir> --vectorizer-type tfidf|word2vec [--use-pretrained-w2v] [--save-output]


- `--eval-dir` (required): Path to saved model
- `--vectorizer-type` (required): Same as used for training
- `--use-pretrained-w2v` (optional): If pre-trained Word2Vec was used
- `--save-output` (optional): Save evaluation results to CSV

## Setup

pip install -r requirements.txt
