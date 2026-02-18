#ETAPE 0 : Test de chargement des données

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd
from cyberbullying import binary_load_data, merge_datasets, DATA_PATHS
from cyberbullying.config import EXPERIMENTS_DIR, PRETRAINED_BERT_DIR

print("Chargement des 3 datasets...")
dfs = []
for path in DATA_PATHS:
    try:
        df = binary_load_data(path, n_samples=100)  # Petit subset pour test
        print(f"  ✓ {path.name}: {df.shape[0]} lignes")
        dfs.append(df)
    except Exception as e:
        print(f"  ✗ {path.name}: {e}")

print("\n Fusion des datasets...")
df_merged = merge_datasets(dfs)
print(f"  Total: {df_merged.shape[0]} lignes")
print(f"  Colonnes: {list(df_merged.columns)}")
print(f"  Classes: {df_merged['type'].value_counts().to_dict()}")

#Etape 1 : Test des fonctions de nettoyage et d'ingénierie des caractéristiques
from cyberbullying import apply_feature_engineering

print("\n Feature Engineering...")
df_clean = apply_feature_engineering(df_merged, column_name='text')
print(f"  Colonnes créées: {[c for c in df_clean.columns if 'text_' in c]}")
print(f"\n Exemple de texte nettoyé:")
print(f"  Avant: {df_merged['text'].iloc[0][:80]}...")
print(f"  Après: {df_clean['text'].iloc[0][:80]}...")

#Etape 2 : Embeddings BERT
from cyberbullying import embed_texts
import numpy as np
from pathlib import Path

print("\n Embedding BERT...")
print("  Chargement du modèle BERT fine-tuné (peut prendre 1-2min)...")
try:
    # Chemin vers le modèle BERT fine-tuné
    bert_path = PRETRAINED_BERT_DIR
    
    # embed_texts retourne une matrice numpy, pas un DataFrame
    matrix = embed_texts(df_clean['text'].tolist(), 'bert', save_path=str(bert_path))
    
    # Ajouter les embeddings au DataFrame
    embedding_df = pd.DataFrame(
        matrix,
        columns=[f"text_bert_{i}" for i in range(matrix.shape[1])]
    )
    df_embedded = pd.concat([df_clean.reset_index(drop=True), embedding_df], axis=1)
    
    print(f" Nouvelles colonnes: {df_embedded.shape[1]}")
    print(f" Shape: {df_embedded.shape}")

    # Sauvegarde pour plus tard
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = EXPERIMENTS_DIR / "bert_test.csv"
    df_embedded.to_csv(output_path, index=False)
    print(f" Sauvegardé: {output_path}")

    #Etape 3 : Entraînement et évaluation de 5 modèles
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from lightgbm import LGBMClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.metrics import classification_report
    import pandas as pd

    print("\n Entraînement des modèles...")

    # Split (une seule fois)
    X = df_embedded[[col for col in df_embedded.columns if col.startswith('text_bert')]]
    y = df_embedded['type']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Dictionnaire des modèles
    models = {
        'LogisticRegression': LogisticRegression(max_iter=1000),
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(kernel='rbf', random_state=42),
        'LightGBM': LGBMClassifier(n_estimators=300, learning_rate=0.05, verbose=-1),
        'MLP': MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=200, random_state=42)
    }

    # Entraîner et évaluer chaque modèle
    for model_name, model in models.items():
        print(f"\n  ⏳ {model_name}...")
        
        # Entraînement
        model.fit(X_train, y_train)
        
        # Prédiction
        y_pred = model.predict(X_test)
        
        # Rapport de classification
        report = classification_report(y_test, y_pred, output_dict=True)
        df_report = pd.DataFrame(report).transpose()
        
        # Sauvegarde du rapport
        results_dir = EXPERIMENTS_DIR / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        report_filename = results_dir / f"{model_name}_bert_report.csv"
        df_report.to_csv(report_filename, index=True)
        print(f"    Rapport sauvegardé: {report_filename}")
        
        # Affichage du rapport
        print(f"\n Classification Report - {model_name}:")
        print(classification_report(y_test, y_pred))

    print("\n Tous les modèles ont été testés et les résultats sauvegardés!")

except Exception as e:
    print(f"  ✗ Erreur BERT: {e}")
    print("  Note: BERT nécessite un modèle fine-tuné dans models/pretrained/my_finetuned_bert/")
