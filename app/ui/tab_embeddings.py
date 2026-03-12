"""Embeddings tab - pretrained and trained models overview."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from cyberbullying.config import PRETRAINED_BERT_DIR, PRETRAINED_MODELS_DIR, PRETRAINED_ROBERTA_DIR, TRAINED_MODELS_DIR


def render(tab):
    with tab:
        st.subheader("Embeddings utilises")

        st.info(
            "**Embeddings** = vue des modeles pretraines (BERT, RoBERTa) et entraines. "
            "Consultation uniquement. Pour generer des embeddings, utiliser l'onglet Operations."
        )

        col1, col2 = st.columns(2)
        with col1:
            st.write("**Pré‑entraînés**")
            st.write(f"Chemin: {PRETRAINED_MODELS_DIR}")
            if PRETRAINED_MODELS_DIR.exists():
                st.dataframe(
                    pd.DataFrame({"fichier": [p.name for p in PRETRAINED_MODELS_DIR.iterdir()]})
                )
            else:
                st.info("Aucun modèle pré‑entraîné détecté.")

        with col2:
            st.write("**Entraînés**")
            st.write(f"Chemin: {TRAINED_MODELS_DIR}")
            if TRAINED_MODELS_DIR.exists():
                st.dataframe(
                    pd.DataFrame({"fichier": [p.name for p in TRAINED_MODELS_DIR.iterdir()]})
                )
            else:
                st.info("Aucun modèle entraîné détecté.")

        st.divider()
        st.write("**BERT / RoBERTa – détails**")
        bert_info = PRETRAINED_BERT_DIR / "config.json"
        roberta_info = PRETRAINED_ROBERTA_DIR / "config.json"

        col3, col4 = st.columns(2)
        with col3:
            st.write("BERT")
            if bert_info.exists():
                st.json(pd.read_json(bert_info).to_dict())
            else:
                st.info("Config BERT introuvable.")

        with col4:
            st.write("RoBERTa")
            if roberta_info.exists():
                st.json(pd.read_json(roberta_info).to_dict())
            else:
                st.info("Config RoBERTa introuvable.")
