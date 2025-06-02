from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec, KeyedVectors
import gensim.downloader as api
import numpy as np
from preprocess import preprocess_text
from config import VECTORIZER_TYPE, EVAL_VECTORIZER_PATH, USE_PRETRAINED_W2V
import joblib

def identity(x):
    return x

def preprocess_data(x_train, x_test):
    print("[INFO] Vectorizer mode:", "training" if len(x_train) > 0 else "evaluation")
    x_test_tokens = [preprocess_text(text) for text in x_test]

    if len(x_train) > 0:  # training mode
        x_train_tokens = [preprocess_text(text) for text in x_train]

        if VECTORIZER_TYPE == 'tfidf':
            vectorizer = TfidfVectorizer(
                tokenizer=identity,
                preprocessor=identity,
                lowercase=False
            )
            x_train_vec = vectorizer.fit_transform(x_train_tokens)
            x_test_vec = vectorizer.transform(x_test_tokens)
            return x_train_vec, x_test_vec, vectorizer

        elif VECTORIZER_TYPE == 'word2vec':
            if USE_PRETRAINED_W2V:
                print("[INFO] Downloading Google News Word2Vec from Gensim... this may take a while on first run.")
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
            raise ValueError(f"Unsupported vectorizer type: {VECTORIZER_TYPE}")

    else:  # eval mode
        if VECTORIZER_TYPE == 'word2vec':
            if USE_PRETRAINED_W2V:
                print("[INFO] Loading Google News Word2Vec for evaluation...")
                vectorizer = api.load("word2vec-google-news-300")
            else:
                vectorizer = joblib.load(EVAL_VECTORIZER_PATH)
        else:
            vectorizer = joblib.load(EVAL_VECTORIZER_PATH)

        if VECTORIZER_TYPE == 'tfidf':
            x_test_vec = vectorizer.transform(x_test_tokens)
            return None, x_test_vec, None

        elif VECTORIZER_TYPE == 'word2vec':
            print("[INFO] Vectorizing test data with Word2Vec")
            def vectorize(tokens):
                vectors = [vectorizer.wv[w] for w in tokens if w in vectorizer.wv]
                return np.mean(vectors, axis=0) if vectors else np.zeros(vectorizer.vector_size)

            x_test_vec = np.array([vectorize(tokens) for tokens in x_test_tokens])
            return None, x_test_vec, vectorizer

        else:
            raise ValueError(f"Unsupported vectorizer type: {VECTORIZER_TYPE}")
