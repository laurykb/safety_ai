import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import glob
from pathlib import Path

from cyberbullying.config import ANALYSIS_PROJECT_DIR, REPORTS_DIR

RESULTS_DIR = REPORTS_DIR
OUTPUT_DIR = ANALYSIS_PROJECT_DIR
EMBEDDINGS = ['tfidf', 'bow', 'word2vec', 'glove', 'bert']

sns.set_theme(style="whitegrid", context="talk")
plt.rcParams['figure.figsize'] = (16, 9)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_project_reports():
    all_data = []
    
    pattern = str(RESULTS_DIR / "*_report.csv")
    files = glob.glob(pattern)
    
    if not files:
        print(f"Erreur: Aucun fichier dans {RESULTS_DIR}")
        return pd.DataFrame()

    print(f"Analyse de {len(files)} rapports...")

    for filepath in files:
        try:
            filename = Path(filepath).name
            clean_name = filename.replace("_report.csv", "")
            
            parts = clean_name.split("_")
            model_name = parts[0]
            
            embed_name = "Unknown"
            for embed in EMBEDDINGS:
                if embed in filename.lower():
                    embed_name = embed
                    break
            
            df = pd.read_csv(filepath, index_col=0)
            
            acc = df.loc['accuracy', 'precision'] if 'accuracy' in df.index else 0
            
            idx_1 = [x for x in df.index if '1' in str(x)]
            
            if idx_1:
                row_1 = df.loc[idx_1[0]]
                prec, rec, f1 = row_1['precision'], row_1['recall'], row_1['f1-score']
            else:
                prec, rec, f1 = 0, 0, 0

            all_data.append({
                "Model": model_name,
                "Embedding": embed_name,
                "Accuracy": acc,
                "Precision": prec,
                "Recall": rec,
                "F1-Score": f1
            })

        except Exception as e:
            print(f"Erreur lecture {filename}: {e}")

    return pd.DataFrame(all_data)

def generate_project_charts(df):
    metrics = [
        ("F1-Score", "Performance globale (F1-Score)"),
        ("Recall", "Detection (Recall): Detecter tout"),
        ("Precision", "Reliability (Precision): Eviter fausses alertes")
    ]

    for metric, title in metrics:
        plt.figure(figsize=(14, 8))
        
        ax = sns.barplot(
            data=df, 
            x="Model", 
            y=metric, 
            hue="Embedding", 
            palette="viridis"
        )
        
        plt.title(title, fontsize=20, pad=20)
        plt.ylim(0, 1.05)
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', title="Embedding")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        for container in ax.containers:
            ax.bar_label(container, fmt='%.2f', padding=3, fontsize=10)

        safe_name = metric.split(" ")[0]
        save_path = OUTPUT_DIR / f"Comparison_{safe_name}.png"
        plt.tight_layout()
        plt.savefig(save_path)
        print(f"Generated: {save_path}")
        plt.close()

print("Analyse des resultats...")
df_project = load_project_reports()

if not df_project.empty:
    csv_path = OUTPUT_DIR / "Results_Consolidated.csv"
    df_project.to_csv(csv_path, index=False)
    print(f"Donnees sauvegardees: {csv_path}")
    
    generate_project_charts(df_project)
    print("Analyse terminee")
else:
    print("Aucune donnee trouvee")
