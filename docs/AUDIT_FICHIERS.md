# Audit des fichiers

Notes sur l'organisation du projet — à jour au moment de la soutenance.

## Structure principale

- `run.py` : point d'entrée unique. Lance Streamlit, les tests ou le pipeline selon les args.
- `streamlit_app.py` : app principale, charge les onglets depuis `app/ui/`.
- `app/ui/` : un fichier par onglet (tab_operations, tab_inference, tab_predict, etc.)
- `src/cyberbullying/` : toute la logique métier (loading, embedder, models, finetune, inference, explainability)
- `scripts/` : scripts CLI autonomes (run_pipeline, download_research_datasets, check_duplicates, aggregate_results)
- `configs/` : YAML de configuration (train, preprocessing, research)
- `tests/` : tests unitaires pytest (loading, feature_engineering, validation)
- `docs/` : documentation interne

## Ce qui est stable

Tout ce qui est dans `src/cyberbullying/` et `scripts/` est stable et testé.
Les onglets Streamlit les plus robustes sont Operations, Inference et Prediction.

## Ce qui pourrait être amélioré

- MLflow : l'onglet fonctionne mais le rendu des graphiques dans Streamlit
  est parfois lent sur CPU. Sur GPU c'est correct.
- GloVe (Mittens) : l'entraînement from scratch est très lent sur CPU
  (>100 epochs sur un gros corpus). Préférer Word2Vec ou BERT dans ce cas.
- Le sarcasme reste difficile à distinguer du cyberbullying dans certains cas limites —
  c'est une limite connue du dataset, pas du modèle.
