import os
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
import gensim.downloader as api
import numpy as np
from preprocess import preprocess_text
import joblib


def identity(x):
    return x

def preprocess_data(x_train, x_test, eval_dir=None, vectorizer_type="tfidf", use_pretrained_w2v=False):
    print("[INFO] Vectorizer mode:", "training" if len(x_train) > 0 else "evaluation")
    x_test_tokens = [preprocess_text(text) for text in x_test]

    if len(x_train) > 0:
        x_train_tokens = [preprocess_text(text) for text in x_train]

        if vectorizer_type == 'tfidf':
            vectorizer = TfidfVectorizer(tokenizer=identity, preprocessor=identity, lowercase=False, token_pattern=None)
            x_train_vec = vectorizer.fit_transform(x_train_tokens)
            x_test_vec = vectorizer.transform(x_test_tokens)
            return x_train_vec, x_test_vec, vectorizer

        elif vectorizer_type == 'word2vec':
            if use_pretrained_w2v:
                print("[INFO] Downloading Google News Word2Vec from Gensim...")
                model = api.load("word2vec-google-news-300")
            else:
                print("[INFO] Training new Word2Vec model")
                model = Word2Vec(sentences=x_train_tokens, vector_size=100, window=5, min_count=1, workers=4).wv

            def vectorize(tokens):
                vectors = [model[w] for w in tokens if w in model]
                return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)

            x_train_vec = np.array([vectorize(tokens) for tokens in x_train_tokens])
            x_test_vec = np.array([vectorize(tokens) for tokens in x_test_tokens])
            return x_train_vec, x_test_vec, model

        else:
            raise ValueError(f"Unsupported vectorizer type: {vectorizer_type}")

    else:
        if vectorizer_type == 'word2vec':
            if eval_dir:
                vectorizer_path = os.path.join(eval_dir, "vectorizer.pkl")
                if os.path.exists(vectorizer_path):
                    vectorizer = joblib.load(vectorizer_path)
                else:
                    print("[INFO] Loading Google News Word2Vec for evaluation...")
                    vectorizer = api.load("word2vec-google-news-300")
            else:
                raise ValueError("eval_dir must be specified in eval mode")
        else:
            if eval_dir:
                vectorizer_path = os.path.join(eval_dir, "vectorizer.pkl")
                if not os.path.exists(vectorizer_path):
                    raise FileNotFoundError(f"Expected vectorizer at {vectorizer_path}")
                vectorizer = joblib.load(vectorizer_path)
            else:
                raise ValueError("eval_dir must be specified in eval mode")

        if vectorizer_type == 'tfidf':
            x_test_vec = vectorizer.transform(x_test_tokens)
            return None, x_test_vec, None

        elif vectorizer_type == 'word2vec':
            def vectorize(tokens):
                vectors = [vectorizer[w] for w in tokens if w in vectorizer]
                return np.mean(vectors, axis=0) if vectors else np.zeros(vectorizer.vector_size)

            x_test_vec = np.array([vectorize(tokens) for tokens in x_test_tokens])
            return None, x_test_vec, vectorizer

        else:
            raise ValueError(f"Unsupported vectorizer type: {vectorizer_type}")