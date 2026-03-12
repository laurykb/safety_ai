# SafetyAI

**Detection de cyberharcelement par apprentissage automatique.** Classification binaire de textes (cyberbullying / normal). Les donnees sarcasme sont traitees comme les autres CSV (input standard).

## En bref

SafetyAI combine plusieurs approches de detection de cyberbullying :

- **Embeddings** : TF-IDF, BoW, Word2Vec, GloVe, BERT, RoBERTa
- **Modeles** : Logistic Regression, Random Forest, SVM, LightGBM, MLP
- **Fine-tuning** : Transformers (Hugging Face) pour classification
- **Explainabilite** : LIME pour interpreter les predictions

L'interface Streamlit permet de charger des donnees, configurer les embeddings, lancer l'entrainement, faire des predictions et consulter les metriques MLflow.

---

## Demarrage rapide

```bash
git clone <votre-repo>
cd projet_IA_cyber
pip install -r requirements.txt
python run.py
```

Ouvrir [http://localhost:8501](http://localhost:8501). Authentification par defaut : `admin` / `admin123`. Pour desactiver : `AUTH_DISABLED=1`.

---

## Installation

### Pre-requis

- Python 3.11 ou superieur
- Windows 10/11, Linux ou macOS

### Environnement virtuel

**Windows (PowerShell) :**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python run.py
```

**Linux / macOS :**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run.py
```

### GPU (CUDA)

Par defaut, PyTorch utilise le CPU. Pour activer le GPU NVIDIA :

```bash
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Docker

```bash
docker build -t safety-ai .
docker run -p 8501:8501 -e AUTH_DISABLED=1 safety-ai
```

---

## Utilisation

### Interface Streamlit

| Onglet | Description |
|--------|-------------|
| Datasets | Charger et visualiser les donnees (CSV/XLSX) |
| Embeddings | Vue des modeles pretraines et entraines |
| Operations | Etape 1 : telecharger datasets (cb1, cb2, sarcasm, wiki_toxic, hatexplain). Etapes 2-4 : doublons, embed, train. Pipeline complet. Selection des CSV pour chaque etape. |
| Fine-tuning | Entrainer des modeles Transformers |
| Inference | Evaluer un modele sur donnees deja traitees (df_*.csv) |
| Prediction | Tester des textes en direct |
| Resultats | Metriques et comparaisons |
| MLflow | Suivi des runs (F1, accuracy, CPU, GPU, RAM) |
| Parametres | Configuration globale |

### Ligne de commande

```bash
python run.py                    # Lancer l'app (port 8501)
python run.py --port 8502        # Port personnalise
python run.py test               # Tests unitaires
python run.py pipeline           # Pipeline end-to-end (charge, embed, train)
python run.py pipeline 5000      # Pipeline avec 5000 echantillons
python run.py research-download  # Telecharger tous les datasets
python run.py embed tfidf 1000   # Embedding TF-IDF + train rapide
python run.py aggregate          # Agreger rapports et graphiques
python run.py check-duplicates   # Verifier doublons dans les datasets
```

Pour telecharger des datasets specifiques en CLI : `python scripts/download_research_datasets.py --datasets cb1 cb2 sarcasm`.

---

## Donnees

Format attendu : CSV ou XLSX avec colonnes `text` et `type` (0 = normal, 1 = cyberbullying).

La commande `research-download` ou l'etape 1 Operations genere :

- `research/cyberbullying_cb1/train.csv`, `test.csv`
- `research/sarcasm_twitter/train.csv`
- `cb2.csv`, `wiki_toxic.csv`, `hatexplain.csv`

Placez vos fichiers dans `data/raw/` ou utilisez les chemins configurables dans l'onglet Parametres.

---

## Architecture

```
Donnees (CSV/XLSX)
       |
       v
  Preprocessing
       |
       v
  Embeddings (TF-IDF | BERT | RoBERTa | ...)
       |
       v
  Modele (sklearn | Transformer)
       |
       v
  Prediction + Explainabilite (LIME)
```

## Pipeline vs Inference

| | Pipeline (Operations) | Inference (onglet Test) |
|---|----------------------|-------------------------|
| **Role** | Cree les donnees traitees | Evalue un modele sur donnees existantes |
| **Entree** | CSV bruts (data/raw/) | df_*.csv (data/processed/) |
| **Sortie** | df_*.csv, rapports, modeles | Metriques affichees |

Le **pipeline** charge les CSV, applique l'embedding, entraine les modeles et genere df_*.csv. L'**inference** utilise ces fichiers pour evaluer un modele (train/test rapide). Lancer le pipeline en premier.

---

## Structure du projet

```
projet_IA_cyber/
├── run.py                 # Point d'entree (python run.py)
├── streamlit_app.py       # Application SafetyAI
├── requirements.txt
├── app/
│   ├── main.py            # Lancement Streamlit
│   └── ui/                # Onglets (tab_datasets, tab_operations, ...)
├── configs/
│   ├── train.yaml         # Config entrainement
│   ├── preprocessing.yaml
│   └── research.yaml
├── src/cyberbullying/     # loading, embedder, models, finetune, inference
├── scripts/               # run_pipeline, download_research_datasets, check_duplicates, etc.
├── tests/
└── docs/
```

---

## Compatibilite

- **Python** : 3.11+
- **OS** : Windows, Linux, macOS
- **Chemins** : `pathlib.Path` pour compatibilite cross-platform
- **Encodage** : UTF-8 recommande

---

## Documentation

- [docs/INTEGRATION_UI.md](docs/INTEGRATION_UI.md) - Pipeline vs Inference
- [docs/AUDIT_FICHIERS.md](docs/AUDIT_FICHIERS.md) - Audit des fichiers
- [docs/EVALUATION_MLOPS.md](docs/EVALUATION_MLOPS.md) - Evaluation MLOps (roadmap.sh)
- [docs/EVALUATION_AI_ENGINEER.md](docs/EVALUATION_AI_ENGINEER.md) - Evaluation AI Engineer (roadmap.sh)

---
