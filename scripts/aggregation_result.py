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
RESULTS_DIR = EXPERIMENTS_DIR / "results"  # Assure-toi que tes CSV sont là
OUTPUT_DIR = ANALYSIS_DIR / "analysis_output_full"
EMBEDDINGS = ['tfidf', 'bow', 'word2vec', 'glove', 'bert']

# Esthétique "Rapport Académique"
sns.set_theme(style="whitegrid", context="talk")
plt.rcParams['figure.figsize'] = (16, 9)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_all_results():
    """
    Charge TOUS les fichiers CSV et crée un DataFrame géant unique
    contenant toutes les métriques pour la Classe 1 (Harcèlement).
    """
    all_data = []
    
    # On cherche tous les fichiers CSV dans le dossier
    pattern = str(RESULTS_DIR / "*_report.csv")
    files = glob.glob(pattern)
    
    if not files:
        print("Aucun fichier trouvé. Vérifie le chemin RESULTS_DIR.")
        return pd.DataFrame()

    print(f"Traitement de {len(files)} fichiers de résultats...")

    for filepath in files:
        try:
            filename = Path(filepath).name
            # Nom typique: "LightGBM_bert_report.csv"
            clean_name = filename.replace("_report.csv", "")
            parts = clean_name.split("_")
            model_name = parts[0]
            # Si le fichier s'appelle "Model_embedding_report.csv", parts[1] est l'embedding
            # Sinon, on essaie de deviner
            embed_name = "Inconnu"
            for embed in EMBEDDINGS:
                if embed in filename.lower():
                    embed_name = embed
                    break
            
            df = pd.read_csv(filepath, index_col=0)
            
            # --- EXTRACTION MÉTHODIQUE ---
            
            # 1. Accuracy Globale (Juste sociale)
            acc = df.loc['accuracy', 'precision'] if 'accuracy' in df.index else 0
            
            # 2. Métriques CLASSE 1 (Celle qui compte : LE HARCÈLEMENT)
            # On cherche l'index qui contient '1' (parfois int, parfois str)
            idx_1 = [x for x in df.index if '1' in str(x)]
            if idx_1:
                row_1 = df.loc[idx_1[0]]
                prec = row_1['precision']
                rec = row_1['recall']
                f1 = row_1['f1-score']
            else:
                prec, rec, f1 = 0, 0, 0

            all_data.append({
                "Modèle": model_name,
                "Embedding": embed_name,
                "Accuracy": acc,
                "Précision (Harcèlement)": prec,
                "Rappel (Harcèlement)": rec,
                "F1-Score (Harcèlement)": f1
            })

        except Exception as e:
            print(f"Erreur sur {filename}: {e}")

    return pd.DataFrame(all_data)

def generate_comparative_charts(df):
    """Génère 4 graphiques, un pour chaque métrique."""
    
    metrics_to_plot = [
        ("Accuracy", "Comparaison de l'Accuracy Globale"),
        ("Précision (Harcèlement)", "Comparaison de la Précision"),
        ("Rappel (Harcèlement)", "Comparaison du Rappel"),
        ("F1-Score (Harcèlement)", "Comparaison du F1-Score")
    ]

    for metric, title in metrics_to_plot:
        plt.figure(figsize=(14, 7))
        
        # Barplot groupé : X=Modèle, Hue=Embedding (pour voir l'impact de la tech)
        # On peut inverser (X=Embedding) si tu préfères comparer les techs entre elles
        ax = sns.barplot(
            data=df, 
            x="Modèle", 
            y=metric, 
            hue="Embedding", 
            palette="viridis",
            edgecolor="black"
        )
        
        plt.title(title, fontsize=18, pad=20)
        plt.ylim(0, 1.05)
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', title="Embedding")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Ajout des valeurs sur les barres pour la précision
        for container in ax.containers:
            ax.bar_label(container, fmt='%.2f', padding=3, fontsize=10)

        # Sauvegarde
        safe_name = metric.split(" ")[0] # Garde juste "Accuracy", "Précision"...
        save_path = OUTPUT_DIR / f"Comparaison_{safe_name}.png"
        plt.tight_layout()
        plt.savefig(save_path)
        print(f"  📊 Graphique généré : {save_path}")
        plt.close()

# --- EXÉCUTION ---
print("Lancement de l'analyse complète multi-métriques...")
df_full = load_all_results()

if not df_full.empty:
    # Sauvegarde du tableau complet pour ton rapport écrit (Excel/CSV)
    csv_path = OUTPUT_DIR / "Tableau_Recapitulatif_Complet.csv"
    df_full.to_csv(csv_path, index=False)
    print(f" Données brutes sauvegardées : {csv_path}")
    
    # Génération des visuels
    generate_comparative_charts(df_full)
    
else:
    print("Aucune donnée chargée.")