# 📚 Guide des scripts

## 🎯 Vue d'ensemble

Les scripts Python sont regroupés dans le dossier scripts/ et servent à tester les embeddings, entraîner des modèles et agréger les résultats.

## 📋 Scripts disponibles

### Tests par embedding
- scripts/test_tfidf.py
- scripts/test_bow.py
- scripts/test_word2vec.py
- scripts/test_glove.py
- scripts/test_bert.py

Ces scripts génèrent :
- des CSV d’échantillons dans outputs/experiments
- des rapports dans outputs/experiments/results

### Agrégation des résultats
- scripts/aggregation_result_tfidf.py
- scripts/aggregation_result.py

Ces scripts lisent les rapports de outputs/experiments/results et produisent des visuels dans outputs/analysis.

## 📂 Chemins utiles

- Données brutes : data/raw
- Données traitées : data/processed
- Modèles entraînés : models/trained
- Modèles pré‑entraînés : models/pretrained
- Sorties (rapports/analyses) : outputs/
│
└── data_clean/
    ├── df_tfidf.csv
    ├── df_bow.csv
    ├── df_bert.csv
    └── ...
```

---

## ❓ Dépannage

### Erreur : "cannot import name 'X' from utils"
→ Assure-toi que l'environnement virtuel est activé :
```powershell
.\.venv\Scripts\Activate.ps1
```

### Erreur : "python: command not found"
→ Installe Python 3.9+ ou utilise `python3`

### Mémoire insuffisante
→ Réduis le nombre de samples ou d'embeddings :
```powershell
.\run_full_pipeline.ps1 -samples 500
```

---

## 📝 Notes

- Les scripts utilisent des valeurs de test (petit nombre de samples)
- Pour la production, augmente le nombre de samples et d'épochs
- Les résultats sont sauvegardés dans `models/` et `utils/reports/`
