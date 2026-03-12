"""Exploitation des résultats tab - saved reports and session tests."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from .commons import get_report_paths, parse_report_file


def render(tab):
    with tab:
        st.subheader("Analyse des rapports et metriques")

        st.info(
            "**Resultats** = consulter les rapports de classification (F1, accuracy, etc.) "
            "generes par Operations. Lancer Operations > Agreger rapports pour consolider."
        )

        tab_saved, tab_session = st.tabs(["Resultats sauvegardes", "Tests de session"])

        with tab_saved:
            st.write("**Analyse des rapports de classification sauvegardés**")

            report_paths = get_report_paths()

            if not report_paths:
                st.warning("Aucun rapport trouvé dans outputs/reports ou outputs/experiments/results.")
            else:
                rows = []
                for path in report_paths:
                    parsed = parse_report_file(path)
                    if parsed:
                        rows.append(parsed)

                df_reports = pd.DataFrame(rows)

                col_f1, col_f2, col_f3 = st.columns(3)
                with col_f1:
                    models_filter = st.multiselect("Filtrer par modèle", df_reports["model"].unique(), default=df_reports["model"].unique())
                with col_f2:
                    embeddings_filter = st.multiselect("Filtrer par embedding", df_reports["embedding"].unique(), default=df_reports["embedding"].unique())
                with col_f3:
                    metric_sort = st.selectbox("Trier par", ["f1", "accuracy", "precision", "recall"])

                df_filtered = df_reports[
                    (df_reports["model"].isin(models_filter)) &
                    (df_reports["embedding"].isin(embeddings_filter))
                ].sort_values(metric_sort, ascending=False)

                st.dataframe(df_filtered, use_container_width=True, hide_index=True)

                if not df_filtered.empty:
                    st.divider()

                    col_g1, col_g2 = st.columns(2)

                    with col_g1:
                        st.write("**F1-Score par modèle et embedding**")
                        fig1 = px.bar(
                            df_filtered,
                            x="model",
                            y="f1",
                            color="embedding",
                            barmode="group",
                            title="Comparaison F1-Score",
                            labels={'model': 'Modèle', 'f1': 'F1-Score', 'embedding': 'Embedding'}
                        )
                        fig1.update_layout(height=400)
                        st.plotly_chart(fig1, use_container_width=True)

                    with col_g2:
                        st.write("**Accuracy par modèle et embedding**")
                        fig2 = px.bar(
                            df_filtered,
                            x="model",
                            y="accuracy",
                            color="embedding",
                            barmode="group",
                            title="Comparaison Accuracy",
                            labels={'model': 'Modèle', 'accuracy': 'Accuracy', 'embedding': 'Embedding'}
                        )
                        fig2.update_layout(height=400)
                        st.plotly_chart(fig2, use_container_width=True)

                    st.write("**Heatmaps des métriques**")
                    col_h1, col_h2 = st.columns(2)

                    with col_h1:
                        pivot_f1 = df_filtered.pivot_table(index="model", columns="embedding", values="f1")
                        fig_h1 = px.imshow(
                            pivot_f1,
                            text_auto=".3f",
                            color_continuous_scale="RdYlGn",
                            title="Heatmap F1-Score",
                            labels={'x': 'Embedding', 'y': 'Modèle', 'color': 'F1'}
                        )
                        fig_h1.update_layout(height=400)
                        st.plotly_chart(fig_h1, use_container_width=True)

                    with col_h2:
                        pivot_acc = df_filtered.pivot_table(index="model", columns="embedding", values="accuracy")
                        fig_h2 = px.imshow(
                            pivot_acc,
                            text_auto=".3f",
                            color_continuous_scale="RdYlGn",
                            title="Heatmap Accuracy",
                            labels={'x': 'Embedding', 'y': 'Modèle', 'color': 'Accuracy'}
                        )
                        fig_h2.update_layout(height=400)
                        st.plotly_chart(fig_h2, use_container_width=True)

        with tab_session:
            st.write("**Résultats des tests d'inférence de cette session**")

            if "inference_results" not in st.session_state or not st.session_state.inference_results:
                st.info("Aucun test d'inférence effectué dans cette session. Allez dans l'onglet 'Inférence & Tests' pour lancer des tests.")
            else:
                df_session = pd.DataFrame(st.session_state.inference_results)

                col_rh1, col_rh2 = st.columns([3, 1])
                with col_rh1:
                    st.write(f"**{len(df_session)} test(s) effectué(s)**")
                with col_rh2:
                    if st.button("Rafraîchir", key="refresh_session"):
                        st.rerun()
                st.dataframe(df_session, use_container_width=True, hide_index=True)

                st.divider()

                col_s1, col_s2 = st.columns(2)

                with col_s1:
                    st.write("**Évolution du F1-Score par test**")
                    fig_s1 = px.line(
                        df_session.reset_index(),
                        x='index',
                        y='f1',
                        markers=True,
                        title='Évolution du F1-Score',
                        labels={'index': 'Numéro de test', 'f1': 'F1-Score'}
                    )
                    fig_s1.add_scatter(
                        x=df_session.reset_index()['index'],
                        y=df_session['f1'],
                        mode='markers',
                        marker=dict(size=10, color=df_session['f1'], colorscale='RdYlGn', showscale=True),
                        showlegend=False
                    )
                    st.plotly_chart(fig_s1, use_container_width=True)

                with col_s2:
                    st.write("**Comparaison des métriques par dataset**")
                    if len(df_session['dataset'].unique()) > 1:
                        fig_s2 = px.box(
                            df_session.melt(
                                id_vars=['dataset', 'embedding', 'model'],
                                value_vars=['accuracy', 'precision', 'recall', 'f1'],
                                var_name='Métrique',
                                value_name='Score'
                            ),
                            x='dataset',
                            y='Score',
                            color='Métrique',
                            title='Métriques par dataset',
                        )
                        st.plotly_chart(fig_s2, use_container_width=True)
                    else:
                        fig_s2 = px.bar(
                            df_session.iloc[-1:].melt(
                                id_vars=['dataset', 'embedding', 'model'],
                                value_vars=['accuracy', 'precision', 'recall', 'f1'],
                                var_name='Métrique',
                                value_name='Score'
                            ),
                            x='Métrique',
                            y='Score',
                            title=f'Métriques - Dernier test ({df_session.iloc[-1]["dataset"]})',
                            color='Métrique'
                        )
                        st.plotly_chart(fig_s2, use_container_width=True)

                st.write("**Récapitulatif par dataset**")
                summary_by_dataset = df_session.groupby('dataset').agg({
                    'accuracy': ['mean', 'max', 'min'],
                    'f1': ['mean', 'max', 'min'],
                    'samples': 'first'
                }).round(3)
                st.dataframe(summary_by_dataset, use_container_width=True)

                if st.button("Effacer les résultats de session", type="secondary"):
                    st.session_state.inference_results = []
                    st.rerun()
