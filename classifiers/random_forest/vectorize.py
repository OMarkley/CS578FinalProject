from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
import numpy as np
from preprocess import preprocess_text
from config import VECTORIZER_TYPE

def identity(x):
    return x

def preprocess_data(x_train, x_test):
    x_train_tokens = [preprocess_text(text) for text in x_train]
    x_test_tokens = [preprocess_text(text) for text in x_test]

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
        model = Word2Vec(sentences=x_train_tokens, vector_size=100, window=5, min_count=1, workers=4)

        def vectorize(tokens):
            vectors = [model.wv[w] for w in tokens if w in model.wv]
            return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)

        x_train_vec = np.array([vectorize(tokens) for tokens in x_train_tokens])
        x_test_vec = np.array([vectorize(tokens) for tokens in x_test_tokens])
        return x_train_vec, x_test_vec, model

    else:
        raise ValueError(f"Unsupported vectorizer type: {VECTORIZER_TYPE}")