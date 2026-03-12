"""Settings tab - hyperparameters and optimization."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from cyberbullying.config import PROCESSED_DATA_DIR

from .commons import get_model_params, load_processed_df

try:
    from cyberbullying.hyperopt import (
        auto_optimize_model,
        get_optimization_history,
        OPTUNA_AVAILABLE,
    )
except ImportError:
    OPTUNA_AVAILABLE = False


def render(tab):
    with tab:
        st.subheader("Hyperparametres des classifieurs")

        st.info(
            "**Parametres** = configurer les hyperparametres des modeles sklearn (LR, RF, SVM, etc.) "
            "et lancer l'optimisation automatique (Optuna). Utilise par l'onglet Inference."
        )

        tab_manual, tab_auto = st.tabs(["Configuration manuelle", "Optimisation automatique"])

        with tab_manual:
            params = get_model_params()

            col_lr, col_rf = st.columns(2)
            with col_lr:
                st.write("**Logistic Regression**")
                params["logistic_regression"]["C"] = st.number_input("C", value=float(params["logistic_regression"]["C"]))
                params["logistic_regression"]["max_iter"] = st.number_input(
                    "max_iter",
                    value=int(params["logistic_regression"]["max_iter"]),
                    step=100,
                )

            with col_rf:
                st.write("**Random Forest**")
                params["random_forest"]["n_estimators"] = st.number_input(
                    "n_estimators",
                    value=int(params["random_forest"]["n_estimators"]),
                    step=50,
                )
                max_depth = st.text_input("max_depth (vide = None)", value="")
                params["random_forest"]["max_depth"] = int(max_depth) if max_depth.strip() else None

            col_svm, col_lgbm = st.columns(2)
            with col_svm:
                st.write("**SVM**")
                params["svm"]["C"] = st.number_input("C_svm", value=float(params["svm"]["C"]))
                params["svm"]["kernel"] = st.selectbox("kernel", ["rbf", "linear", "poly"], index=0)

            with col_lgbm:
                st.write("**LightGBM**")
                params["lightgbm"]["n_estimators"] = st.number_input(
                    "n_estimators_lgbm",
                    value=int(params["lightgbm"]["n_estimators"]),
                    step=50,
                )
                params["lightgbm"]["learning_rate"] = st.number_input(
                    "learning_rate",
                    value=float(params["lightgbm"]["learning_rate"]),
                    step=0.01,
                    format="%.2f",
                )

            st.write("**MLP**")
            hidden_str = st.text_input("hidden_layer_sizes (ex: 128,64)", value="128,64")
            try:
                sizes = tuple(int(x.strip()) for x in hidden_str.split(",") if x.strip())
            except ValueError:
                sizes = (128, 64)
            params["mlp"]["hidden_layer_sizes"] = sizes
            params["mlp"]["max_iter"] = st.number_input(
                "max_iter_mlp",
                value=int(params["mlp"]["max_iter"]),
                step=50,
            )

            st.session_state.model_params = params
            st.success("Paramètres manuels mis à jour")

        with tab_auto:
            st.write("**Optimisation automatique d'hyperparamètres avec Optuna**")

            if not OPTUNA_AVAILABLE:
                st.error("Optuna n'est pas installé. Installez avec: pip install optuna")
            else:
                st.info("Optuna va automatiquement rechercher les meilleurs hyperparamètres pour maximiser le F1-Score.")

                col_opt1, col_opt2 = st.columns(2)

                with col_opt1:
                    processed_files = sorted([p for p in PROCESSED_DATA_DIR.glob("df_*.csv")])
                    available_embeddings = [p.stem.replace("df_", "") for p in processed_files] if processed_files else []

                    if not available_embeddings:
                        st.warning("Aucun dataset traité disponible.")
                    else:
                        opt_embedding = st.selectbox("Embedding", available_embeddings, key="opt_emb")
                        opt_model = st.selectbox(
                            "Modèle à optimiser",
                            ["logistic_regression", "random_forest", "svm", "lightgbm", "mlp"],
                            key="opt_model"
                        )

                with col_opt2:
                    n_trials = st.number_input("Nombre d'essais", min_value=10, max_value=200, value=50, step=10)
                    cv_folds = st.number_input("K-Fold CV", min_value=3, max_value=10, value=5, step=1)
                    sample_limit_opt = st.number_input("Limite échantillons (0=tous)", min_value=0, value=5000, step=1000, key="opt_sample")

                if st.button("Lancer l'optimisation", type="primary"):
                    if not available_embeddings:
                        st.error("Aucun dataset traité disponible.")
                    else:
                        try:
                            with st.spinner("Chargement des données..."):
                                df = load_processed_df(opt_embedding)
                                if df is None:
                                    st.error("Dataset introuvable.")
                                    st.stop()

                                feature_cols = [c for c in df.columns if c.startswith(f"text_{opt_embedding}")]
                                if not feature_cols:
                                    st.error("Colonnes d'embedding introuvables.")
                                    st.stop()

                                if sample_limit_opt > 0 and len(df) > sample_limit_opt:
                                    df_sample = df.sample(n=sample_limit_opt, random_state=42)
                                else:
                                    df_sample = df

                                X = df_sample[feature_cols]
                                y = df_sample["type"]

                                st.success(f"Données chargées: {len(X)} échantillons, {len(feature_cols)} features")

                            st.info(f"Recherche des meilleurs hyperparamètres pour {opt_model}...")

                            with st.spinner(f"Optimisation en cours ({n_trials} essais avec {cv_folds}-Fold CV)..."):
                                result = auto_optimize_model(
                                    opt_model,
                                    X,
                                    y,
                                    n_trials=int(n_trials),
                                    cv=int(cv_folds)
                                )

                            st.success("Optimisation terminée!")

                            col_r1, col_r2 = st.columns(2)

                            with col_r1:
                                st.write("**Meilleurs hyperparamètres:**")
                                st.json(result["best_params"])

                                if st.button("Sauvegarder ces paramètres"):
                                    st.session_state.model_params[opt_model] = result["best_params"]
                                    st.success("Paramètres sauvegardés!")
                                    st.rerun()

                            with col_r2:
                                st.metric("Meilleur F1-Score (CV)", f"{result['best_score']:.3f}")

                                history_df = get_optimization_history(result["study"])
                                st.write(f"**Historique** ({len(history_df)} essais):")
                                st.dataframe(history_df.head(10), use_container_width=True)

                            st.write("**Évolution de l'optimisation:**")
                            history_df_clean = history_df[history_df["state"] == "COMPLETE"].copy()

                            fig_opt = px.line(
                                history_df_clean,
                                x="trial",
                                y="value",
                                title="Évolution du F1-Score par essai",
                                labels={"trial": "Essai", "value": "F1-Score (CV)"}
                            )

                            fig_opt.add_hline(
                                y=result["best_score"],
                                line_dash="dash",
                                line_color="red",
                                annotation_text=f"Meilleur: {result['best_score']:.3f}"
                            )

                            st.plotly_chart(fig_opt, use_container_width=True)

                        except Exception as e:
                            st.error(f"Erreur: {e}")
                            import traceback
                            st.code(traceback.format_exc())
