# SafetyAI

**Projet scolaire — Telecom Paris**
Détection de cyberharcèlement par apprentissage automatique, avec support du sarcasme et suivi MLflow.

---

## Contexte

Ce projet a été réalisé dans le cadre d'un cours à Telecom Paris. L'objectif était de construire un système capable de détecter si une phrase s'apparente à du cyberharcèlement ou non, en s'appuyant sur des patterns appris lors d'un pré-entraînement.

Le point de départ était un ensemble de datasets fournis par l'enseignant. J'ai ensuite étendu le projet dans deux directions :

- **Multiplication des approches** : plutôt que de choisir un seul pipeline, j'ai combiné plusieurs méthodes d'embedding et de classification. Chaque combinaison (embedding × classifieur) constitue une route indépendante, ce qui permet de comparer les résultats et d'exploiter les forces de chaque approche.
- **Extension au sarcasme** : les datasets initiaux ne couvraient pas le sarcasme. J'ai intégré des données spécifiques pour que le modèle puisse mieux distinguer une phrase sarcastique d'une insulte directe.

Enfin, j'ai intégré **MLflow** pour suivre les métriques hardware (CPU, GPU, RAM) et les performances des modèles lors des entraînements.

---

## Ce que fait le projet

- Classifie un texte en deux catégories : **cyberbullying** ou **normal**
- Supporte plusieurs méthodes d'**embedding** : TF-IDF, BoW, Word2Vec, GloVe, BERT, RoBERTa
- Supporte plusieurs **classifieurs** : Logistic Regression, Random Forest, SVM, LightGBM, MLP
- Permet le **fine-tuning** de Transformers (Hugging Face) pour la classification
- Intègre **LIME** pour l'explicabilité des prédictions
- Suit les métriques via **MLflow** (F1, accuracy, CPU, GPU, RAM)
- Propose une interface **Streamlit** pour tout piloter sans ligne de commande

---

## Démarrage rapide

```bash
git clone <votre-repo>
cd projet_IA_cyber
pip install -r requirements.txt
python run.py
```

Ouvrir [http://localhost:8501](http://localhost:8501). Authentification par défaut : `admin` / `admin123`. Pour désactiver : `AUTH_DISABLED=1`.

---

## Installation

### Pré-requis
- Python 3.11 ou supérieur
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
Par défaut, PyTorch utilise le CPU. Pour activer le GPU NVIDIA :
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
| Datasets | Charger et visualiser les données (CSV/XLSX) |
| Embeddings | Vue des modèles pré-entraînés et entraînés |
| Operations | Étape 1 : télécharger datasets (cb1, cb2, sarcasm, wiki_toxic, hatexplain). Étapes 2-4 : doublons, embed, train. Pipeline complet. |
| Fine-tuning | Entraîner des modèles Transformers |
| Inference | Évaluer un modèle sur données déjà traitées (df_*.csv) |
| Prediction | Tester des textes en direct |
| Résultats | Métriques et comparaisons |
| MLflow | Suivi des runs (F1, accuracy, CPU, GPU, RAM) |
| Paramètres | Configuration globale |

### Ligne de commande

```bash
python run.py                        # Lancer l'app (port 8501)
python run.py --port 8502            # Port personnalisé
python run.py test                   # Tests unitaires
python run.py pipeline               # Pipeline end-to-end (charge, embed, train)
python run.py pipeline 5000          # Pipeline avec 5000 échantillons
python run.py research-download      # Télécharger tous les datasets
python run.py embed tfidf 1000       # Embedding TF-IDF + train rapide
python run.py aggregate              # Agréger rapports et graphiques
python run.py check-duplicates       # Vérifier doublons dans les datasets
```

Pour télécharger des datasets spécifiques en CLI : `python scripts/download_research_datasets.py --datasets cb1 cb2 sarcasm`.

---

## Données

Format attendu : CSV ou XLSX avec colonnes `text` et `type` (0 = normal, 1 = cyberbullying).

La commande `research-download` ou l'étape 1 Operations génère :
- `research/cyberbullying_cb1/train.csv`, `test.csv`
- `research/sarcasm_twitter/train.csv`
- `cb2.csv`, `wiki_toxic.csv`, `hatexplain.csv`

Placez vos fichiers dans `data/raw/` ou utilisez les chemins configurables dans l'onglet Paramètres.

---

## Architecture

```
Données (CSV/XLSX)
        |
        v
Preprocessing
        |
        v
Embeddings (TF-IDF | BERT | RoBERTa | ...)
        |
        v
Modèle (sklearn | Transformer)
        |
        v
Prediction + Explicabilité (LIME)
```

## Pipeline vs Inference

| | Pipeline (Operations) | Inference (onglet Test) |
|---|----------------------|-------------------------|
| **Rôle** | Crée les données traitées | Évalue un modèle sur données existantes |
| **Entrée** | CSV bruts (data/raw/) | df_*.csv (data/processed/) |
| **Sortie** | df_*.csv, rapports, modèles | Métriques affichées |

Le **pipeline** charge les CSV, applique l'embedding, entraîne les modèles et génère df_*.csv. L'**inference** utilise ces fichiers pour évaluer un modèle (train/test rapide). Lancer le pipeline en premier.

---

## Structure du projet

```
projet_IA_cyber/
├── run.py                  # Point d'entrée (python run.py)
├── streamlit_app.py        # Application SafetyAI
├── requirements.txt
├── app/
│   ├── main.py             # Lancement Streamlit
│   └── ui/                 # Onglets (tab_datasets, tab_operations, ...)
├── configs/
│   ├── train.yaml          # Config entraînement
│   ├── preprocessing.yaml
│   └── research.yaml
├── src/cyberbullying/      # loading, embedder, models, finetune, inference
├── scripts/                # run_pipeline, download_research_datasets, check_duplicates, etc.
├── tests/
└── docs/
```

---

## Compatibilité

- **Python** : 3.11+
- **OS** : Windows, Linux, macOS
- **Chemins** : `pathlib.Path` pour compatibilité cross-platform
- **Encodage** : UTF-8 recommandé

---

## Documentation

- [docs/INTEGRATION_UI.md](docs/INTEGRATION_UI.md) — Pipeline vs Inference
- [docs/AUDIT_FICHIERS.md](docs/AUDIT_FICHIERS.md) — Audit des fichiers
- [docs/EVALUATION_MLOPS.md](docs/EVALUATION_MLOPS.md) — Évaluation MLOps
- [docs/EVALUATION_AI_ENGINEER.md](docs/EVALUATION_AI_ENGINEER.md) — Évaluation AI Engineer
