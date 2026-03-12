"""Datasets tab - visualization and filtering."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from cyberbullying.loading import binary_load_data, merge_datasets

from .commons import data_file_label, list_data_files, load_binary_dataset


def render(tab):
    with tab:
        st.subheader("Visualisation et tri des datasets")

        st.info(
            "**Datasets** = visualiser et filtrer les CSV/XLSX dans data/raw/. "
            "Selectionnez des fichiers, fusionnez-les, filtrez par label. Aucune modification des donnees."
        )

        files = list_data_files()
        if not files:
            st.warning("Aucun dataset trouvé dans data/raw.")
        else:
            dataset_labels = [data_file_label(p) for p in files]
            selected = st.multiselect("Choisir un ou plusieurs datasets", dataset_labels, default=dataset_labels[:2])
            n_samples = st.number_input("Nombre d'échantillons (0 = tous)", min_value=0, value=500, step=100)
            label_filter = st.selectbox("Filtre label", ["Tous", "0", "1"])
            search_text = st.text_input("Recherche texte")
            do_merge = st.checkbox("Fusionner les datasets sélectionnés", value=False)

            dfs: list[pd.DataFrame] = []
            for label in selected:
                path = next(p for p in files if data_file_label(p) == label)
                try:
                    df = binary_load_data(path, n_samples=None if n_samples == 0 else int(n_samples))
                except Exception:
                    df = load_binary_dataset(path, n_samples=None if n_samples == 0 else int(n_samples))
                if "type" in df.columns and label_filter in {"0", "1"}:
                    df = df[df["type"] == int(label_filter)].copy()
                if search_text and "text" in df.columns:
                    df = df[df["text"].str.contains(search_text, case=False, na=False)].copy()
                dfs.append(df)

            if dfs:
                if do_merge and all("text" in d.columns and "type" in d.columns for d in dfs):
                    merged = merge_datasets(dfs)
                    st.write("Dataset fusionné")
                    st.dataframe(merged, use_container_width=True)
                    if "type" in merged.columns and len(merged) > 0:
                        counts = merged["type"].value_counts().reset_index()
                        counts.columns = ["label", "count"]
                        fig = px.bar(counts, x="label", y="count", title="Répartition des classes")
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    for label, df in zip(selected, dfs):
                        st.write(f"Dataset: {label}")
                        st.dataframe(df, use_container_width=True)
                        if "type" in df.columns and len(df) > 0:
                            counts = df["type"].value_counts().reset_index()
                            counts.columns = ["label", "count"]
                            fig = px.bar(counts, x="label", y="count", title=f"Répartition des classes – {label}")
                            st.plotly_chart(fig, use_container_width=True)
