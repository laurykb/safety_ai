import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import glob
from pathlib import Path

from cyberbullying.config import ANALYSIS_BY_EMBEDDING_DIR, REPORTS_DIR

RESULTS_DIR = REPORTS_DIR
OUTPUT_DIR = ANALYSIS_BY_EMBEDDING_DIR
EMBEDDINGS = ['tfidf', 'bow', 'word2vec', 'glove', 'bert']

sns.set_theme(style="whitegrid", context="talk")
plt.rcParams['figure.figsize'] = (14, 8)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def analyze_embedding_performance(embedding_name):
    pattern = str(RESULTS_DIR / "*_report.csv")
    all_files = glob.glob(pattern)
    
    target_files = [f for f in all_files if embedding_name.lower() in Path(f).name.lower()]

    if not target_files:
        print(f"Pas de fichiers trouves pour: {embedding_name}")
        return None

    print(f"Analyse de {embedding_name.upper()} ({len(target_files)} fichiers)")
    
    data = []
    for filepath in target_files:
        try:
            filename = Path(filepath).name
            clean_name = filename.replace("_report.csv", "").replace(f"_{embedding_name}", "").replace("_", " ")
            
            df = pd.read_csv(filepath, index_col=0)
            
            idx_1 = [x for x in df.index if '1' in str(x)]
            
            if idx_1:
                row_1 = df.loc[idx_1[0]]
                prec, rec, f1 = row_1['precision'], row_1['recall'], row_1['f1-score']
            else:
                prec, rec, f1 = 0, 0, 0
                
            data.append({
                "Modele": clean_name,
                "Embedding": embedding_name,
                "Precision": prec,
                "Rappel": rec,
                "F1-Score": f1
            })
            
        except Exception as e:
            print(f"Erreur sur {filename}: {e}")

    return pd.DataFrame(data).sort_values("F1-Score", ascending=False)

def plot_specific_charts(df, embedding_name):
    if df is None or df.empty:
        return

    plt.figure()
    ax = sns.barplot(data=df, x="Modele", y="F1-Score", palette="viridis")
    plt.title(f"Performance {embedding_name.upper()} (F1-Score Harassment)", pad=20)
    plt.ylim(0, 1.05)
    plt.ylabel("F1-Score")
    plt.xticks(rotation=45)
    
    for i in ax.containers:
        ax.bar_label(i, fmt='%.2f', padding=3)
        
    save_path = OUTPUT_DIR / f"{embedding_name}_1_Leaderboard.png"
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    plt.figure()
    sns.scatterplot(data=df, x="Rappel", y="Precision", hue="Modele", style="Modele", s=300, alpha=0.9)
    
    plt.axhline(0.8, color='green', linestyle='--', alpha=0.3)
    plt.axvline(0.8, color='green', linestyle='--', alpha=0.3)
    plt.text(0.82, 0.95, "Zone Excellence", color='green', fontsize=12)

    plt.title(f"Precision vs Rappel - {embedding_name.upper()}", pad=20)
    plt.xlim(0, 1.05)
    plt.ylim(0, 1.05)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.grid(True, linestyle='--')
    
    save_path = OUTPUT_DIR / f"{embedding_name}_2_TradeOff.png"
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    print(f"Graphiques generes pour {embedding_name}")

print("Analyse par embedding (dossier Reports)...")

all_results = []

for embed in EMBEDDINGS:
    df_embed = analyze_embedding_performance(embed)
    if df_embed is not None:
        plot_specific_charts(df_embed, embed)
        all_results.append(df_embed)

if all_results:
    df_global = pd.concat(all_results)
    
    plt.figure(figsize=(12, 8))
    pivot = df_global.pivot(index="Modele", columns="Embedding", values="F1-Score")
    sns.heatmap(pivot, annot=True, cmap="RdYlGn", fmt=".2f", vmin=0, vmax=1)
    plt.title("Synthese globale: F1-Score")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "Global_Heatmap.png")
    print("Heatmap globale generee")

print(f"Analyse terminee. Resultats dans: {OUTPUT_DIR}/")
