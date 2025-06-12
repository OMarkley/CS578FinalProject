from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import pandas as pd
import joblib
import os

def train_model(x_train_vec, y_train, use_grid_search=False):
    if not use_grid_search:
        model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        model.fit(x_train_vec, y_train)
        return model, None  # No grid search results

    param_grid = {
        'n_estimators': [50, 100, 200, 300],
        'max_depth': [5, 10, 15, 20, None],
        # 'min_samples_split': [2, 5, 10],
        # 'min_samples_leaf': [1, 2, 5]
    }

    base_model = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        scoring='accuracy',
        cv=3,
        n_jobs=-1,
        verbose=2,
        return_train_score=True
    )
    grid_search.fit(x_train_vec, y_train)

    print("[INFO] Best parameters:", grid_search.best_params_)
    print("[INFO] Best cross-validation score:", grid_search.best_score_)
    
    # Save full cv results
    full_results = pd.DataFrame(grid_search.cv_results_)
    full_results.to_csv("full_grid_search_results.csv", index=False)
    print("[INFO] Saved full grid search results.")

    return grid_search.best_estimator_, pd.DataFrame(grid_search.cv_results_)

def save_model(model, vectorizer, model_dir, vectorizer_type, use_pretrained_w2v):
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "rf_model.pkl")
    vectorizer_path = os.path.join(model_dir, "vectorizer.pkl")

    joblib.dump(model, model_path)
    if (not use_pretrained_w2v) or (vectorizer_type == 'tfidf'):
        joblib.dump(vectorizer, vectorizer_path)
    else:
        print("[INFO] Skipping vectorizer save: using pre-trained Word2Vec")


def load_eval_model(eval_dir):
    model_path = os.path.join(eval_dir, "rf_model.pkl")
    vectorizer_path = os.path.join(eval_dir, "vectorizer.pkl")

    model = joblib.load(model_path)
    if os.path.exists(vectorizer_path):
        vectorizer = joblib.load(vectorizer_path)
    else:
        print("[INFO] No vectorizer found in eval directory. Will use Google News Word2Vec.")
        vectorizer = None
    return model, vectorizer
