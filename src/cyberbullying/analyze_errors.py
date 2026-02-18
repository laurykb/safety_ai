import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from cyberbullying.config import ERROR_ANALYSIS_DIR, ERROR_REPORTS_DIR

input_file = ERROR_ANALYSIS_DIR / "error_analysis_results.csv"
output_dir = ERROR_REPORTS_DIR

output_dir.mkdir(parents=True, exist_ok=True)

print("Chargement des donnees...")
df = pd.read_csv(input_file)

print(f"Total d'erreurs: {len(df)}\n")

sns.set_theme(style="whitegrid")
plt.rcParams['figure.facecolor'] = 'white'

faux_negatif = df[df['error_type'] == 'Faux Negatif'].copy()
faux_negatif = faux_negatif.sort_values(['embedding', 'model'])
faux_negatif.to_csv(output_dir / 'faux_negatifs.csv', index=False)
print(f"Faux negatifs: {len(faux_negatif)} erreurs")

faux_positif = df[df['error_type'] == 'Faux Positif'].copy()
faux_positif = faux_positif.sort_values(['embedding', 'model'])
faux_positif.to_csv(output_dir / 'faux_positifs.csv', index=False)
print(f"Faux positifs: {len(faux_positif)} erreurs\n")

embedding_summary = df.groupby('embedding').agg({
    'text': 'count',
    'error_type': lambda x: (x == 'Faux Negatif').sum()
}).rename(columns={'text': 'Total Erreurs', 'error_type': 'Faux Negatifs'})
embedding_summary['Faux Positifs'] = df.groupby('embedding')['error_type'].apply(lambda x: (x == 'Faux Positif').sum())
embedding_summary['% Faux Negatifs'] = (embedding_summary['Faux Negatifs'] / embedding_summary['Total Erreurs'] * 100).round(2)
embedding_summary.to_csv(output_dir / 'summary_by_embedding.csv')

model_summary = df.groupby('model').agg({
    'text': 'count',
    'error_type': lambda x: (x == 'Faux Negatif').sum()
}).rename(columns={'text': 'Total Erreurs', 'error_type': 'Faux Negatifs'})
model_summary['Faux Positifs'] = df.groupby('model')['error_type'].apply(lambda x: (x == 'Faux Positif').sum())
model_summary['% Faux Negatifs'] = (model_summary['Faux Negatifs'] / model_summary['Total Erreurs'] * 100).round(2)
model_summary.to_csv(output_dir / 'summary_by_model.csv')

print("1/4 - Graphique Resume par Embedding")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

x_labels = embedding_summary.index
x_pos = np.arange(len(x_labels))
width = 0.35

ax = axes[0, 0]
bars1 = ax.bar(x_pos - width/2, embedding_summary['Faux Negatifs'], width, label='Faux Negatifs', color='#d62728')
bars2 = ax.bar(x_pos + width/2, embedding_summary['Faux Positifs'], width, label='Faux Positifs', color='#ff7f0e')
ax.set_xlabel('Embedding')
ax.set_ylabel('Nombre d\'erreurs')
ax.set_title('Types d\'Erreurs par Embedding', fontsize=12, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(x_labels, rotation=45)
ax.legend()
ax.grid(axis='y', alpha=0.3)

for bar in bars1:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height)}', ha='center', va='bottom', fontsize=9)
for bar in bars2:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height)}', ha='center', va='bottom', fontsize=9)

ax = axes[0, 1]
colors_embedding = plt.cm.Set2(np.linspace(0, 1, len(embedding_summary)))
bars = ax.bar(embedding_summary.index, embedding_summary['Total Erreurs'], color=colors_embedding)
ax.set_xlabel('Embedding')
ax.set_ylabel('Nombre total d\'erreurs')
ax.set_title('Total Erreurs par Embedding', fontsize=12, fontweight='bold')
ax.tick_params(axis='x', rotation=45)
ax.grid(axis='y', alpha=0.3)
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height)}', ha='center', va='bottom', fontsize=9)

ax = axes[1, 0]
ax.barh(embedding_summary.index, embedding_summary['% Faux Negatifs'], color='#2ca02c')
ax.set_xlabel('Pourcentage (%)')
ax.set_title('% Faux Negatifs par Embedding', fontsize=12, fontweight='bold')
ax.grid(axis='x', alpha=0.3)
for i, v in enumerate(embedding_summary['% Faux Negatifs']):
    ax.text(v + 1, i, f'{v:.1f}%', va='center', fontsize=9)

ax = axes[1, 1]
ax.axis('off')
summary_text = "Resume par Embedding:\n\n"
for idx, row in embedding_summary.iterrows():
    summary_text += f"{idx.upper()}:\n"
    summary_text += f"  Total: {int(row['Total Erreurs'])}\n"
    summary_text += f"  Faux Negatifs: {int(row['Faux Negatifs'])} ({row['% Faux Negatifs']:.1f}%)\n"
    summary_text += f"  Faux Positifs: {int(row['Faux Positifs'])}\n\n"
ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig(output_dir / '1_resume_embeddings.png', dpi=300, bbox_inches='tight')
print(f"Sauvegarde: {output_dir / '1_resume_embeddings.png'}")
plt.close()

print("2/4 - Graphique Resume par Modele")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

x_labels_model = model_summary.index
x_pos_model = np.arange(len(x_labels_model))

ax = axes[0, 0]
bars1 = ax.bar(x_pos_model - width/2, model_summary['Faux Negatifs'], width, label='Faux Negatifs', color='#d62728')
bars2 = ax.bar(x_pos_model + width/2, model_summary['Faux Positifs'], width, label='Faux Positifs', color='#ff7f0e')
ax.set_xlabel('Modele')
ax.set_ylabel('Nombre d\'erreurs')
ax.set_title('Types d\'Erreurs par Modele', fontsize=12, fontweight='bold')
ax.set_xticks(x_pos_model)
ax.set_xticklabels(x_labels_model, rotation=45)
ax.legend()
ax.grid(axis='y', alpha=0.3)

for bar in bars1:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height)}', ha='center', va='bottom', fontsize=9)
for bar in bars2:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height)}', ha='center', va='bottom', fontsize=9)

ax = axes[0, 1]
colors_model = plt.cm.Pastel1(np.linspace(0, 1, len(model_summary)))
bars = ax.bar(model_summary.index, model_summary['Total Erreurs'], color=colors_model)
ax.set_xlabel('Modele')
ax.set_ylabel('Nombre total d\'erreurs')
ax.set_title('Total Erreurs par Modele', fontsize=12, fontweight='bold')
ax.tick_params(axis='x', rotation=45)
ax.grid(axis='y', alpha=0.3)
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height)}', ha='center', va='bottom', fontsize=9)

ax = axes[1, 0]
ax.barh(model_summary.index, model_summary['% Faux Negatifs'], color='#1f77b4')
ax.set_xlabel('Pourcentage (%)')
ax.set_title('% Faux Negatifs par Modele', fontsize=12, fontweight='bold')
ax.grid(axis='x', alpha=0.3)
for i, v in enumerate(model_summary['% Faux Negatifs']):
    ax.text(v + 1, i, f'{v:.1f}%', va='center', fontsize=9)

ax = axes[1, 1]
ax.axis('off')
summary_text = "Resume par Modele:\n\n"
for idx, row in model_summary.iterrows():
    summary_text += f"{idx}:\n"
    summary_text += f"  Total: {int(row['Total Erreurs'])}\n"
    summary_text += f"  Faux Negatifs: {int(row['Faux Negatifs'])} ({row['% Faux Negatifs']:.1f}%)\n"
    summary_text += f"  Faux Positifs: {int(row['Faux Positifs'])}\n\n"
ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, fontsize=9,
        verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

plt.tight_layout()
plt.savefig(output_dir / '2_resume_modeles.png', dpi=300, bbox_inches='tight')
print(f"Sauvegarde: {output_dir / '2_resume_modeles.png'}")
plt.close()

print("3/4 - Graphiques Detailles par Embedding x Modele")

embeddings = df['embedding'].unique()
for embedding in sorted(embeddings):
    print(f"  {embedding.upper()}...")
    df_emb = df[df['embedding'] == embedding]
    
    model_counts = df_emb.groupby('model')['error_type'].value_counts().unstack(fill_value=0)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    ax = axes[0]
    model_counts.plot(kind='bar', ax=ax, color=['#d62728', '#ff7f0e'])
    ax.set_title(f'Distribution des Erreurs - Embedding: {embedding.upper()}', fontsize=12, fontweight='bold')
    ax.set_ylabel('Nombre d\'erreurs')
    ax.set_xlabel('Modele')
    ax.tick_params(axis='x', rotation=45)
    ax.legend(title='Type d\'erreur')
    ax.grid(axis='y', alpha=0.3)
    
    ax = axes[1]
    error_pct = df_emb.groupby('model')['error_type'].apply(
        lambda x: (x == 'Faux Negatif').sum() / len(x) * 100
    ).sort_values(ascending=False)
    colors_bar = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(error_pct)))
    bars = ax.barh(error_pct.index, error_pct.values, color=colors_bar)
    ax.set_xlabel('% Faux Negatifs')
    ax.set_title(f'% Faux Negatifs par Modele - {embedding.upper()}', fontsize=12, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    for i, (idx, v) in enumerate(error_pct.items()):
        ax.text(v + 1, i, f'{v:.1f}%', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / f'3_details_{embedding}.png', dpi=300, bbox_inches='tight')
    plt.close()

print("4/4 - Heatmap Embedding x Modele")

pivot_errors = pd.crosstab(df['embedding'], df['model'])
pivot_faux_neg = df[df['error_type'] == 'Faux Negatif'].groupby(['embedding', 'model']).size().unstack(fill_value=0)
pivot_faux_pos = df[df['error_type'] == 'Faux Positif'].groupby(['embedding', 'model']).size().unstack(fill_value=0)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

sns.heatmap(pivot_errors, annot=True, fmt='d', cmap='YlOrRd', ax=axes[0], cbar_kws={'label': 'Total Erreurs'})
axes[0].set_title('Total Erreurs - Embedding x Modele', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Modele')
axes[0].set_ylabel('Embedding')

sns.heatmap(pivot_faux_neg, annot=True, fmt='d', cmap='Reds', ax=axes[1], cbar_kws={'label': 'Faux Negatifs'})
axes[1].set_title('Faux Negatifs - Embedding x Modele', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Modele')
axes[1].set_ylabel('Embedding')

sns.heatmap(pivot_faux_pos, annot=True, fmt='d', cmap='Oranges', ax=axes[2], cbar_kws={'label': 'Faux Positifs'})
axes[2].set_title('Faux Positifs - Embedding x Modele', fontsize=12, fontweight='bold')
axes[2].set_xlabel('Modele')
axes[2].set_ylabel('Embedding')

plt.tight_layout()
plt.savefig(output_dir / '4_heatmaps.png', dpi=300, bbox_inches='tight')
print(f"Sauvegarde: {output_dir / '4_heatmaps.png'}")
plt.close()

crosstab_embedding = pd.crosstab(df['embedding'], df['error_type'])
crosstab_embedding.to_csv(output_dir / 'embedding_error_crosstab.csv')

crosstab_model = pd.crosstab(df['model'], df['error_type'])
crosstab_model.to_csv(output_dir / 'model_error_crosstab.csv')

print(f"\nAnalyse terminee!")
print(f"Rapports generes dans: {output_dir}")
print(f"  1_resume_embeddings.png")
print(f"  2_resume_modeles.png")
print(f"  3_details_*.png (1 par embedding)")
print(f"  4_heatmaps.png")
