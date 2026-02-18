import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import glob
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cyberbullying.config import ANALYSIS_DIR, EXPERIMENTS_DIR

# --- CONFIGURATION ---
# Dossier où tes scripts de test ont sauvegardé les CSV (ex: ./results)
RESULTS_DIR = EXPERIMENTS_DIR / "results"
# Dossier où on va sauvegarder les graphiques générés
OUTPUT_DIR = ANALYSIS_DIR / "analysis_output"

# Liste des embeddings à analyser (noms utilisés dans tes fichiers CSV)
EMBEDDINGS = ['tfidf', 'bow', 'word2vec', 'glove', 'bert']

# Configuration esthétique des graphiques
sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams['figure.figsize'] = (12, 7)

# Création du dossier de sortie si inexistant
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_embedding_results(embedding_name):
    """
    Charge tous les rapports CSV associés à un embedding spécifique.
    Retourne un DataFrame consolidé.
    """
    # Pattern de recherche: ex: ./results/*_tfidf_report.csv
    pattern = str(RESULTS_DIR / f"*_{embedding_name}_report.csv")
    files = glob.glob(pattern)
    
    if not files:
        print(f"Aucun fichier trouvé pour l'embedding : {embedding_name}")
        return None

    data = []
    print(f"\n Chargement des résultats pour : {embedding_name.upper()}")

    for filepath in files:
        try:
            # Nom du fichier : "LightGBM_tfidf_report.csv"
            filename = Path(filepath).name
            # Extraction du nom du modèle (tout ce qui est avant _embedding)
            model_name = filename.split(f"_{embedding_name}")[0]
            
            df = pd.read_csv(filepath, index_col=0)
            
            # ON CIBLE LA CLASSE 1 (CYBERHARCÈLEMENT)
            # C'est souvent la ligne '1' ou '1.0' dans le rapport scikit-learn
            # On cherche une ligne dont l'index contient '1'
            class_1_idx = [idx for idx in df.index if '1' in str(idx)]
            
            if class_1_idx:
                row_1 = df.loc[class_1_idx[0]]
                precision = row_1['precision']
                recall = row_1['recall']
                f1 = row_1['f1-score']
            else:
                # Fallback si pas trouvé (cas rare)
                precision, recall, f1 = 0, 0, 0
            
            # On récupère aussi l'accuracy globale
            accuracy = df.loc['accuracy', 'precision'] if 'accuracy' in df.index else 0
            
            # Temps d'inférence (si disponible dans le CSV, sinon 0)
            # Adapte si tes CSV n'ont pas de colonne temps
            time = df['time'].iloc[0] if 'time' in df.columns else 0

            data.append({
                "Modèle": model_name,
                "Embedding": embedding_name,
                "Précision (Class 1)": precision,
                "Rappel (Class 1)": recall,
                "F1-Score (Class 1)": f1,
                "Accuracy Globale": accuracy
            })
            print(f" {model_name} chargé (F1: {f1:.3f})")

        except Exception as e:
            print(f" Erreur lecture {filename}: {e}")

    if not data:
        return None
        
    return pd.DataFrame(data).sort_values("F1-Score (Class 1)", ascending=False)

def plot_embedding_analysis(df, embedding_name):
    """
    Génère les graphiques pour un embedding donné.
    """
    if df is None:
        return

    # 1. Barplot F1-Score (Comparaison directe)
    plt.figure()
    ax = sns.barplot(data=df, x="Modèle", y="F1-Score (Class 1)", palette="viridis")
    plt.title(f"Performance par Modèle - Embedding : {embedding_name.upper()}")
    plt.ylim(0, 1.0)
    plt.ylabel("F1-Score (Classe Harcèlement)")
    plt.xticks(rotation=45)
    
    # Ajout des valeurs sur les barres
    for i in ax.containers:
        ax.bar_label(i, fmt='%.2f')
        
    save_path = OUTPUT_DIR / f"analysis_{embedding_name}_f1.png"
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"  Graphique F1 sauvegardé : {save_path}")
    plt.close()

    # 2. Scatter Plot Précision vs Rappel (Trade-off)
    plt.figure()
    sns.scatterplot(
        data=df, 
        x="Rappel (Class 1)", 
        y="Précision (Class 1)", 
        hue="Modèle", 
        style="Modèle", 
        s=200
    )
    # Zone idéale
    plt.axhline(0.8, color='green', linestyle='--', alpha=0.3)
    plt.axvline(0.8, color='green', linestyle='--', alpha=0.3)
    plt.title(f"Précision vs Rappel - Embedding : {embedding_name.upper()}")
    plt.xlim(0, 1.05)
    plt.ylim(0, 1.05)
    
    save_path = OUTPUT_DIR / f"analysis_{embedding_name}_tradeoff.png"
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"  Graphique Trade-off sauvegardé : {save_path}")
    plt.close()

# --- MAIN EXECUTION ---
print("Démarrage de l'analyse méthodique des résultats...")

all_results = []

# 1. Analyse par Embedding
for embed in EMBEDDINGS:
    df_res = load_embedding_results(embed)
    if df_res is not None:
        plot_embedding_analysis(df_res, embed)
        all_results.append(df_res)

# 2. Synthèse Globale (Si on a des données)
if all_results:
    df_global = pd.concat(all_results)
    
    # Heatmap comparative : Modèles vs Embeddings
    plt.figure(figsize=(10, 6))
    pivot = df_global.pivot(index="Modèle", columns="Embedding", values="F1-Score (Class 1)")
    sns.heatmap(pivot, annot=True, cmap="RdYlGn", fmt=".2f", vmin=0, vmax=1)
    plt.title("SYNTHÈSE GLOBALE : F1-Score (Classe Harcèlement)")
    
    save_path = OUTPUT_DIR / "global_heatmap_f1.png"
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"\n Synthèse globale générée : {save_path}")
    
    # Sauvegarde CSV global
    csv_path = OUTPUT_DIR / "global_results_table.csv"
    df_global.to_csv(csv_path, index=False)
    print(f" Tableau complet sauvegardé : {csv_path}")

print("\n Analyse terminée ! Vérifie le dossier 'analysis_output'.")