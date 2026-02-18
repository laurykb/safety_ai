from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from lightgbm import LGBMClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
import pickle
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from cyberbullying.config import PROCESSED_DATA_DIR, REPORTS_DIR, TRAINED_MODELS_DIR


def get_df_split (
    embedder : str,
) :
    df_path = PROCESSED_DATA_DIR / f"df_{embedder}.csv"
    df = pd.read_csv(df_path)
    X = df[[col for col in df.columns if col.startswith(f'text_{embedder}')]]  # embeddings
    y = df['type']  # target column
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


def train_model (
    model_name : str,
    X_train,
    y_train,
) :
    if model_name == 'logistic_regression' :
        model = LogisticRegression(max_iter=1000)
    elif model_name == 'random_forest' :
        model = RandomForestClassifier()
    elif model_name == 'svm' :
        model = SVC()
    elif model_name == 'gbm' :
        model = LGBMClassifier(n_estimators=300, learning_rate=0.05)
    elif model_name == 'neural_network' :
        model = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=200, verbose=False)
    model.fit(X_train, y_train)
    TRAINED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(TRAINED_MODELS_DIR / f"{model_name}.pkl", "wb") as file:
        pickle.dump(model, file)
    return model

def write_model_report (
    model_name : str,
    embedder : str,
    y_pred,
    y_test,
    suffix="",    
) :
    report = classification_report(y_test, y_pred, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / f"{model_name}_{embedder}_{suffix}_report.csv"
    df_report.to_csv(report_path, index=True)
    print(report)

def test_model (
    model,
    X_test,
    y_test,
    suffix="",    
) :
    y_pred = model.predict(X_test)
    print('Classification report for test set:')
    write_model_report(
        model_name=model.__class__.__name__,
        embedder=X_test.columns[0].split('_')[1],
        y_pred=y_pred,
        y_test=y_test,
        suffix=suffix
    )


def hyperparam_search(X_train, y_train):

    param_grid = {
    'num_leaves': [31, 63, 127],
    'max_depth': [-1, 10, 20],
    'learning_rate': [0.1, 0.05, 0.01],
    'n_estimators': [500, 1000, 1500],
    'min_data_in_leaf': [20, 50, 100],
    'feature_fraction': [0.8, 1.0],
    'bagging_fraction': [0.8, 1.0],
    'lambda_l1': [0, 0.5, 1],
    'lambda_l2': [0, 0.5, 1]
    }

    clf = LGBMClassifier(is_unbalance=True, random_state=42, n_jobs=-1)
    grid = GridSearchCV(
        estimator=clf,
        param_grid=param_grid,
        scoring='f1_macro',  # or 'accuracy'
        cv=3,
        verbose=2
    )
    grid.fit(X_train, y_train)

    print("Best parameters:", grid.best_params_)
    print("Best macro F1:", grid.best_score_)


