# Projet ML – Detection de cyberharcelement

## Objectif
Pipeline de preparation, embeddings et entrainement de modeles pour la detection de cyberharcelement.

## Structure du projet

- src/cyberbullying : code source (chargement, features, embeddings, entrainement)
- scripts : scripts de tests et d'agregation
- notebooks : notebooks Jupyter
- data/raw : datasets bruts
- data/processed : datasets transformes
- models/trained : modeles entraines
- models/pretrained : modeles pre-entraines (BERT/RoBERTa/GloVe)
- outputs : rapports, analyses et artefacts
- docs : documentation

## Datasets

- https://research.cs.wisc.edu/bullying/data.html
- https://www.kaggle.com/datasets/saurabhshahane/cyberbullying-dataset

## Embeddings et modeles

Embeddings classiques : TF-IDF, Word2Vec, GloVe, BoW.
Embeddings avances : BERT, RoBERTa, etc.

## Interface web (Streamlit)

L'interface web permet de tester les modeles, lancer des analyses et faire des predictions en temps reel.

```bash
streamlit run streamlit_app.py
```

## Installation rapide

```bash
pip install -r requirements.txt
```

