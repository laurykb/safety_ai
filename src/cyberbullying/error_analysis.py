import pandas as pd
import numpy as np
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from lightgbm import LGBMClassifier
from sklearn.neural_network import MLPClassifier

from cyberbullying.config import ERROR_ANALYSIS_DIR, RAW_DATA_DIR
from cyberbullying.embedder import embed_texts
from cyberbullying.feature_engineering import apply_feature_engineering
from cyberbullying.loading import binary_load_data, merge_datasets

OUTPUT_FILE = ERROR_ANALYSIS_DIR / "error_analysis_results.csv"
N_SAMPLES = 1000
RANDOM_SEED = 42

EMBEDDING_CONFIGS = {
    'tfidf': {
        'method': 'tfidf',
        'prefix': 'text_tfidf'
    },
    'bow': {
        'method': 'bow',
        'prefix': 'text_bow'
    },
    'word2vec': {
        'method': 'word2vec',
        'prefix': 'text_word2vec'
    },
    'glove': {
        'method': 'glove',
        'prefix': 'text_glove'
    },
    'bert': {
        'method': 'bert',
        'prefix': 'text_bert'
    }
}

# Modeles a tester
MODELS = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED),
    "SVM": SVC(kernel='rbf', random_state=RANDOM_SEED),
    "LightGBM": LGBMClassifier(verbose=-1, random_state=RANDOM_SEED),
    "MLP": MLPClassifier(max_iter=500, random_state=RANDOM_SEED)
}

def load_base_data():
    print("Chargement des donnees...")
    df_cyber = binary_load_data(
        str(RAW_DATA_DIR / "cyberbullying_tweets.csv"),
        "tweet_text",
        "cyberbullying_type",
        "not_cyberbullying",
        n_samples=N_SAMPLES,
    )
    df_tox = binary_load_data(
        str(RAW_DATA_DIR / "toxicity_parsed_dataset.csv"),
        "Text",
        "oh_label",
        0,
        n_samples=N_SAMPLES,
    )
    df_agg = binary_load_data(
        str(RAW_DATA_DIR / "aggression_parsed_dataset.csv"),
        "Text",
        "oh_label",
        0,
        n_samples=N_SAMPLES,
    )
    
    df = merge_datasets([df_cyber, df_tox, df_agg])
    df = apply_feature_engineering(df, column_name='text')
    return df

def get_features_and_labels(df, embedding_name, prefix):
    """
    Extrait X et y des donnees embeddes
    """
    feature_cols = [c for c in df.columns if c.startswith(prefix)]
    X = df[feature_cols]
    y = df['type']
    return X, y

def run_error_analysis():
    df_base = load_base_data()
    all_errors = []
    
    for embed_name, config in EMBEDDING_CONFIGS.items():
        print(f"\nAnalyse Embedding: {embed_name.upper()}...")
        
        try:
            df_current = df_base.copy()
            
            matrix = embed_texts(
                df_current['text'].tolist(),
                config['method']
            )
            
            embedding_df = pd.DataFrame(
                matrix,
                columns=[f"{config['prefix']}_{i}" for i in range(matrix.shape[1])]
            )
            df_current = pd.concat([df_current.reset_index(drop=True), embedding_df], axis=1)
            
            X, y = get_features_and_labels(df_current, embed_name, config['prefix'])
            texts = df_current['text']
            
            X_train, X_test, y_train, y_test, txt_train, txt_test = train_test_split(
                X, y, texts, test_size=0.2, random_state=RANDOM_SEED
            )
            
            for model_name, model in MODELS.items():
                print(f"  {model_name}...")
                
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                df_res = pd.DataFrame({
                    'text': txt_test.reset_index(drop=True),
                    'true_label': y_test.reset_index(drop=True),
                    'pred_label': y_pred
                })
                
                errors = df_res[df_res['true_label'] != df_res['pred_label']].copy()
                
                errors['model'] = model_name
                errors['embedding'] = embed_name
                
                conditions = [
                    (errors['true_label'] == 0) & (errors['pred_label'] == 1),
                    (errors['true_label'] == 1) & (errors['pred_label'] == 0)
                ]
                choices = ['Faux Positif', 'Faux Negatif']
                errors['error_type'] = np.select(conditions, choices, default='Autre')
                
                all_errors.append(errors)
                
        except Exception as e:
            print(f"  Erreur sur {embed_name}: {e}")

    if all_errors:
        final_df = pd.concat(all_errors, ignore_index=True)
        os.makedirs(OUTPUT_FILE.parent, exist_ok=True)
        final_df.to_csv(OUTPUT_FILE, index=False)
        print(f"\nAnalyse terminee: {len(final_df)} erreurs exportees")
        print(f"Fichier: {OUTPUT_FILE}")
        return final_df
    else:
        print("Aucune erreur trouvee")
        return None

if __name__ == "__main__":
    run_error_analysis()