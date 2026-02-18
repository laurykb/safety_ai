# 🚀 Nouvelles Fonctionnalités - Cyberbullying Detection Dashboard

Ce document détaille les 6 pistes d'amélioration implémentées dans le projet.

---

## 📋 Vue d'ensemble

### ✅ Fonctionnalités implémentées

1. **🔥 Fine-tuning BERT/RoBERTa** - Entraînement de transformers sur mesure
2. **🔮 Prédiction en temps réel** - Interface de prédiction interactive
3. **🔍 Explainability (LIME/SHAP)** - Compréhension des prédictions
4. **⚙️ Optimisation automatique (Optuna)** - Recherche d'hyperparamètres
5. **🎯 Ensemble Methods** - Combinaison de modèles (Voting/Stacking)
6. **💾 Gestion de modèles** - Registry et versioning

---

## 1. 🔥 Fine-tuning BERT/RoBERTa

### Module: `src/cyberbullying/finetune.py`

**Fonctionnalités:**
- Fine-tuning de BERT ou RoBERTa sur vos données
- Configuration complète des hyperparamètres
- Support de datasets custom avec colonnes configurables
- Métriques d'évaluation complètes (F1, Accuracy, Precision, Recall)
- Sauvegarde automatique du modèle et du tokenizer

**Utilisation dans Streamlit:**
```python
# Onglet: "🔥 Fine-tuning Transformers"
- Sélectionner BERT ou RoBERTa
- Choisir le dataset d'entraînement
- Configurer les hyperparamètres:
  * Époques, batch size, learning rate
  * Warmup steps, weight decay
  * Validation split
- Cliquer sur "🚀 Lancer le fine-tuning"
```

**Paramètres configurables:**
- `num_epochs`: Nombre d'époques (1-20)
- `batch_size`: Taille du batch (4-64)
- `learning_rate`: Taux d'apprentissage (0.00001-0.001)
- `max_length`: Longueur max des tokens (64-512)
- `warmup_steps`: Steps de warmup (0-1000)
- `weight_decay`: Régularisation L2 (0.0-0.1)

**Sortie:**
- Modèle fine-tuné sauvegardé dans `models/trained/finetuned_*`
- Métriques d'évaluation affichées
- Logs d'entraînement dans `models/trained/finetuned_*/logs`

---

## 2. 🔮 Prédiction en temps réel

### Nouveau tab: "🔮 Prédiction en temps réel"

**Fonctionnalités:**
- 2 modes de prédiction:
  * **Modèles classiques** (sklearn): LogReg, RF, SVM, LightGBM, MLP
  * **Transformers fine-tunés**: BERT/RoBERTa
- Interface interactive avec zone de texte
- Exemples prédéfinis (Cyberbullying, Neutre, Ambiguë)
- Seuil de confiance ajustable
- Visualisation des probabilités par classe
- Explainability optionnelle (LIME)

**Utilisation:**
```python
# 1. Choisir le mode (sklearn ou Transformers)
# 2. Sélectionner l'embedding et le modèle
# 3. Entrer le texte à analyser
# 4. Activer l'explainability (optionnel)
# 5. Cliquer sur "🚀 Lancer la prédiction"
```

**Sortie:**
- Classe prédite (Cyberbullying / Normal)
- Confiance (probabilité)
- Graphique des probabilités par classe
- Explication LIME des mots influents (si activé)

---

## 3. 🔍 Explainability (LIME/SHAP)

### Module: `src/cyberbullying/explainability.py`

**Fonctionnalités:**
- **LIME** (Local Interpretable Model-agnostic Explanations):
  * Identifie les mots influençant la prédiction
  * Visualisation avec poids positifs/négatifs
  * Highlighting du texte original
  
- **SHAP** (SHapley Additive exPlanations):
  * Valeurs SHAP pour chaque feature
  * Top features les plus importantes
  * Compatible avec sklearn et transformers

**Utilisation:**
```python
# Dans l'onglet "🔮 Prédiction en temps réel"
1. Cocher "🔍 Activer l'explainability (LIME)"
2. Lancer une prédiction
3. Observer les mots surlignés:
   - Rouge = contribue au cyberbullying
   - Vert = contribue à la normalité
```

**Installation:**
```bash
pip install lime shap
```

**Exemple de sortie LIME:**
```
Mots influents:
➕ loser: +0.65   (contribue au cyberbullying)
➕ stupid: +0.52
➖ friend: -0.23  (contribue à la normalité)
```

---

## 4. ⚙️ Optimisation automatique (Optuna)

### Module: `src/cyberbullying/hyperopt.py`

**Fonctionnalités:**
- Recherche automatique des meilleurs hyperparamètres
- Support de tous les classifiers:
  * Logistic Regression
  * Random Forest
  * SVM
  * LightGBM
  * MLP
- Cross-validation K-Fold
- Visualisation de l'évolution de l'optimisation
- Sauvegarde automatique des meilleurs paramètres

**Utilisation dans Streamlit:**
```python
# Onglet: "🔧 Settings" > "🤖 Optimisation automatique"
1. Sélectionner l'embedding
2. Choisir le modèle à optimiser
3. Configurer:
   - Nombre d'essais (10-200)
   - K-Fold CV (3-10)
   - Limite d'échantillons
4. Cliquer sur "🚀 Lancer l'optimisation"
5. Sauvegarder les meilleurs paramètres trouvés
```

**Espaces de recherche:**

**Logistic Regression:**
- C: [0.001, 100] (log scale)
- solver: [lbfgs, liblinear, saga]
- max_iter: [100, 2000]

**Random Forest:**
- n_estimators: [50, 500]
- max_depth: [5, 50]
- min_samples_split: [2, 20]
- max_features: [sqrt, log2]

**LightGBM:**
- n_estimators: [50, 500]
- learning_rate: [0.001, 0.3]
- max_depth: [3, 15]
- num_leaves: [10, 100]
- subsample: [0.5, 1.0]

**Sortie:**
- Meilleurs hyperparamètres trouvés
- Meilleur F1-Score (cross-validation)
- Historique des essais
- Graphique d'évolution

---

## 5. 🎯 Ensemble Methods

### Module: `src/cyberbullying/ensemble.py`

**Fonctionnalités:**
- **Voting Ensemble**:
  * Hard voting: vote majoritaire
  * Soft voting: moyenne des probabilités
  
- **Stacking Ensemble**:
  * Modèles de base + meta-learner
  * Cross-validation pour éviter l'overfitting
  
- Comparaison automatique des modèles individuels
- Analyse de la diversité des prédictions

**Utilisation dans Streamlit:**
```python
# Onglet: "🤖 Inférence & Tests" > "🎯 Ensemble de modèles"
1. Sélectionner l'embedding
2. Choisir le type d'ensemble (Voting Soft/Hard, Stacking)
3. Sélectionner 2+ modèles à combiner
4. Pour Stacking: choisir le meta-learner
5. Cliquer sur "🚀 Créer et évaluer l'ensemble"
```

**Exemple de configuration:**
```python
# Voting Ensemble
Modèles: [LogisticRegression, RandomForest, LightGBM]
Type: Soft Voting

# Stacking Ensemble
Base models: [RandomForest, SVM, LightGBM]
Meta-learner: LogisticRegression
```

**Sortie:**
- Métriques de l'ensemble (F1, Accuracy, Precision, Recall)
- Comparaison avec modèles individuels
- Graphique montrant les gains de performance
- Sauvegarde dans le model registry

---

## 6. 💾 Gestion de modèles

### Module: `src/cyberbullying/model_manager.py`

**Fonctionnalités:**
- **ModelRegistry**: Registry centralisé des modèles
- Sauvegarde automatique avec métadonnées
- Versioning par timestamp
- Chargement de modèles sauvegardés
- Exportation en package standalone
- Suppression de modèles
- Recherche du meilleur modèle par métrique

**Utilisation dans Streamlit:**

### Sauvegarder un modèle:
```python
# Onglet: "🤖 Inférence & Tests" > "🔹 Modèle unique"
1. Entraîner et évaluer un modèle
2. Remplir le nom du modèle
3. Cliquer sur "💾 Sauvegarder"
```

### Gérer les modèles:
```python
# Onglet: "💾 Gestion des modèles"

## 📋 Liste des modèles:
- Voir tous les modèles sauvegardés
- Filtrer par embedding, nom
- Trier par F1, accuracy, date
- Visualiser les performances
- Supprimer des modèles

## 🔄 Charger un modèle:
- Sélectionner un modèle
- Charger pour utilisation
- Accès aux métadonnées

## 📦 Exporter:
- Créer un package standalone (.zip)
- Inclut modèle + vectorizer + métadonnées + README
- Téléchargement direct
```

**Structure du registry:**
```
models/trained/registry/
├── model_registry.json          # Index de tous les modèles
├── logistic_regression_tfidf_20260118_143022/
│   ├── model.pkl
│   ├── vectorizer.pkl
│   └── metadata.json
└── ensemble_voting_20260118_150500/
    ├── model.pkl
    └── metadata.json
```

**Métadonnées sauvegardées:**
```json
{
  "model_id": "lightgbm_bert_20260118_143022",
  "name": "lightgbm_bert",
  "model_type": "lightgbm",
  "embedding": "bert",
  "dataset": "cyberbullying_tweets.csv",
  "accuracy": 0.89,
  "precision": 0.87,
  "recall": 0.91,
  "f1": 0.89,
  "samples": 5000,
  "test_size": 1000,
  "timestamp": "20260118_143022",
  "params": {...}
}
```

---

## 🔧 Installation des dépendances

```bash
# Nouvelles dépendances
pip install optuna lime shap lightgbm

# Ou via requirements.txt
pip install -r requirements.txt
```

**Ajouts au requirements.txt:**
```
optuna==4.1.0
lime==0.2.0.1
shap==0.47.0
lightgbm==4.5.0
```

---

## 📊 Architecture mise à jour

```
projet_IA_cyber/
├── src/cyberbullying/
│   ├── __init__.py
│   ├── finetune.py           # ✨ NOUVEAU: Fine-tuning transformers
│   ├── explainability.py     # ✨ NOUVEAU: LIME/SHAP
│   ├── hyperopt.py           # ✨ NOUVEAU: Optuna
│   ├── ensemble.py           # ✨ NOUVEAU: Ensemble methods
│   ├── model_manager.py      # ✨ NOUVEAU: Registry de modèles
│   ├── embedder.py
│   ├── train.py
│   ├── evaluate.py
│   └── ...
├── streamlit_app.py          # ✨ MODIFIÉ: 8 onglets au lieu de 5
├── models/
│   ├── trained/
│   │   ├── registry/         # ✨ NOUVEAU: Registry des modèles
│   │   └── finetuned_*/      # ✨ NOUVEAU: Modèles fine-tunés
│   └── pretrained/
└── requirements.txt          # ✨ MODIFIÉ: nouvelles dépendances
```

---

## 🎯 Workflows recommandés

### Workflow 1: Entraîner le meilleur modèle

```python
1. Onglet "🔧 Settings" > "🤖 Optimisation automatique"
   - Optimiser les hyperparamètres pour plusieurs modèles
   - Sauvegarder les meilleurs paramètres

2. Onglet "🤖 Inférence & Tests" > "🎯 Ensemble de modèles"
   - Créer un ensemble avec les 3 meilleurs modèles
   - Évaluer les performances

3. Sauvegarder l'ensemble dans le registry

4. Onglet "💾 Gestion des modèles"
   - Exporter le modèle en package
```

### Workflow 2: Fine-tuner un transformer

```python
1. Onglet "🔥 Fine-tuning Transformers"
   - Choisir BERT ou RoBERTa
   - Configurer les hyperparamètres
   - Entraîner (peut prendre du temps)

2. Onglet "🔮 Prédiction en temps réel"
   - Tester le modèle fine-tuné
   - Activer l'explainability

3. Comparer avec les modèles classiques
```

### Workflow 3: Analyser une prédiction

```python
1. Onglet "🔮 Prédiction en temps réel"
   - Entrer un texte suspect
   - Activer l'explainability (LIME)
   - Lancer la prédiction

2. Observer:
   - Probabilité de cyberbullying
   - Mots les plus influents
   - Explication visuelle
```

---

## 📈 Améliorations futures possibles

1. **API REST** (FastAPI/Flask)
2. **Data augmentation** (back-translation, paraphrasing)
3. **Multi-task learning** (type de cyberbullying)
4. **Active learning** (échantillons informatifs)
5. **MLflow integration** (tracking expérimental)
6. **Docker containerization**
7. **CI/CD pipeline**
8. **Tests unitaires** (pytest)
9. **Monitoring** (drift detection)
10. **Attention visualization** (pour transformers)

---

## 🐛 Troubleshooting

### LIME ne fonctionne pas
```bash
pip install lime==0.2.0.1
```

### Optuna échoue
```bash
pip install optuna==4.1.0
# Vérifier que scikit-learn >= 1.0
```

### Fine-tuning BERT out of memory
- Réduire le `batch_size` (essayer 8 ou 4)
- Réduire `max_length` (essayer 64)
- Limiter le nombre d'échantillons

### ModèLE Registry vide
- Les modèles doivent être sauvegardés via le bouton "💾 Sauvegarder"
- Vérifier `models/trained/registry/model_registry.json`

---

## 📝 Changelog

### Version 2.0 (Janvier 2026)

**Ajouts majeurs:**
- ✨ Fine-tuning BERT/RoBERTa
- ✨ Prédiction en temps réel
- ✨ Explainability LIME/SHAP
- ✨ Optimisation Optuna
- ✨ Ensemble methods
- ✨ Model registry

**Améliorations:**
- Dashboard Streamlit: 5 → 8 onglets
- Interface utilisateur plus intuitive
- Métriques enrichies
- Visualisations améliorées

**Modules créés:**
- `src/cyberbullying/finetune.py`
- `src/cyberbullying/explainability.py`
- `src/cyberbullying/hyperopt.py`
- `src/cyberbullying/ensemble.py`
- `src/cyberbullying/model_manager.py`

---

## 📚 Références

- **LIME**: https://github.com/marcotcr/lime
- **SHAP**: https://github.com/slundberg/shap
- **Optuna**: https://optuna.org/
- **Transformers**: https://huggingface.co/docs/transformers/
- **Ensemble Methods**: https://scikit-learn.org/stable/modules/ensemble.html

---

✅ **Projet prêt pour la production !**
