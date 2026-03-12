"""Inférence & Tests tab - single model and ensemble."""
from __future__ import annotations

import time

import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

from cyberbullying.config import PROCESSED_DATA_DIR, TRAINED_MODELS_DIR, get_mlflow_tracking_uri
from cyberbullying.mlflow_utils import log_system_metrics_manual

from .commons import (
    build_model,
    data_file_label,
    fmt_duration,
    get_model_params,
    list_data_files,
    load_processed_df,
    supports_predict_proba,
)

try:
    from cyberbullying.ensemble import (
        create_voting_ensemble,
        create_stacking_ensemble,
        train_and_evaluate_ensemble,
        compare_models_for_ensemble,
    )
    ENSEMBLE_AVAILABLE = True
except ImportError:
    ENSEMBLE_AVAILABLE = False

try:
    from cyberbullying.model_manager import ModelRegistry
    MODEL_MANAGER_AVAILABLE = True
except ImportError:
    MODEL_MANAGER_AVAILABLE = False


def _render_single(tab_single):
    with tab_single:
        processed_files = sorted([p for p in PROCESSED_DATA_DIR.glob("df_*.csv")])
        available_embeddings = [p.stem.replace("df_", "") for p in processed_files]

        if not available_embeddings:
            st.warning("Aucun dataset traité trouvé dans data/processed.")
        else:
            col_inf1, col_inf2 = st.columns(2)

            with col_inf1:
                embedding = st.selectbox("Embedding", available_embeddings)
                model_key = st.selectbox(
                    "Modèle de classification",
                    ["logistic_regression", "random_forest", "svm", "lightgbm", "mlp"],
                )

            with col_inf2:
                raw_files = list_data_files()
                if raw_files:
                    dataset_source = st.selectbox("Dataset source", [data_file_label(p) for p in raw_files], help="Dataset utilisé pour les tests")
                else:
                    dataset_source = "Dataset inconnu"

                test_size = st.slider("Taille du test set (%)", min_value=10, max_value=50, value=20, step=5)
                sample_limit = st.number_input("Limite d'échantillons (0 = tous)", min_value=0, value=2000, step=500)

            if st.button("Lancer l'inférence", type="primary"):
                df = load_processed_df(embedding)
                if df is None:
                    st.error("Dataset traité introuvable.")
                else:
                    feature_cols = [c for c in df.columns if c.startswith(f"text_{embedding}")]
                    if not feature_cols:
                        st.error("Colonnes d'embedding introuvables.")
                    else:
                        if sample_limit > 0 and len(df) > sample_limit:
                            df_sample = df.sample(n=sample_limit, random_state=42)
                        else:
                            df_sample = df

                        if len(df_sample) < 500:
                            st.info(f"Dataset de {len(df_sample)} échantillons – l'inférence sera rapide. Pour des tests plus longs, augmentez la limite ou exécutez le pipeline pour générer plus de données.")

                        X = df_sample[feature_cols]
                        y = df_sample["type"]

                        params = get_model_params()[model_key]
                        model = build_model(model_key, params)

                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=test_size/100, random_state=42
                        )

                        t0 = time.perf_counter()
                        with st.spinner(f"Entraînement sur {len(X_train)} échantillons..."):
                            model.fit(X_train, y_train)
                        train_time = time.perf_counter() - t0

                        t1 = time.perf_counter()
                        with st.spinner(f"Prédiction sur {len(X_test)} échantillons..."):
                            y_pred = model.predict(X_test)
                        pred_time = time.perf_counter() - t1

                        st.success(f"Inférence terminée! (entraînement: {fmt_duration(train_time)}, prédiction: {fmt_duration(pred_time)})")

                        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                        acc = accuracy_score(y_test, y_pred)
                        report_dict = classification_report(y_test, y_pred, output_dict=True)
                        f1 = report_dict.get("1", {}).get("f1-score", 0)
                        prec = report_dict.get("1", {}).get("precision", 0)
                        rec = report_dict.get("1", {}).get("recall", 0)

                        try:
                            import mlflow
                            mlflow.set_tracking_uri(get_mlflow_tracking_uri())
                            mlflow.set_experiment("safety-ai")
                            with mlflow.start_run(run_name=f"inference_{model_key}_{embedding}", log_system_metrics=True):
                                mlflow.log_params({"embedding": embedding, "model": model_key, "dataset": dataset_source, "n_samples": len(df_sample)})
                                mlflow.log_metrics({"accuracy": acc, "f1": f1, "precision": prec, "recall": rec})
                                log_system_metrics_manual()
                        except Exception:
                            pass

                        with col_m1:
                            st.metric("Accuracy", f"{acc:.3f}")
                        with col_m2:
                            st.metric("Precision (classe 1)", f"{prec:.3f}")
                        with col_m3:
                            st.metric("Recall (classe 1)", f"{rec:.3f}")
                        with col_m4:
                            st.metric("F1-Score (classe 1)", f"{f1:.3f}")

                        st.divider()
                        st.write("**Informations sur le test:**")
                        info_df = pd.DataFrame({
                            "Paramètre": ["Dataset source", "Embedding", "Modèle", "Échantillons total", "Train set", "Test set", "Temps entraînement", "Temps prédiction"],
                            "Valeur": [
                                dataset_source,
                                embedding,
                                model_key,
                                len(df_sample),
                                len(X_train),
                                len(X_test),
                                fmt_duration(train_time),
                                fmt_duration(pred_time),
                            ]
                        })
                        st.dataframe(info_df, use_container_width=True, hide_index=True)

                        st.divider()
                        col_v1, col_v2 = st.columns(2)

                        with col_v1:
                            st.write("**Matrice de confusion**")
                            cm = confusion_matrix(y_test, y_pred)
                            fig_cm = px.imshow(
                                cm,
                                text_auto=True,
                                color_continuous_scale="Blues",
                                labels=dict(x="Prédit", y="Réel", color="Count"),
                                x=["Non-Cyber (0)", "Cyber (1)"],
                                y=["Non-Cyber (0)", "Cyber (1)"],
                            )
                            fig_cm.update_layout(height=400)
                            st.plotly_chart(fig_cm, use_container_width=True)

                        with col_v2:
                            st.write("**Métriques par classe**")
                            report_df = pd.DataFrame(report_dict).transpose()
                            metrics_df = report_df[report_df.index.isin(['0', '1'])][['precision', 'recall', 'f1-score']]
                            metrics_df.index = ['Non-Cyber (0)', 'Cyber (1)']
                            fig_metrics = px.bar(
                                metrics_df.reset_index().melt(id_vars='index'),
                                x='variable',
                                y='value',
                                color='index',
                                barmode='group',
                                title="Comparaison des métriques",
                                labels={'variable': 'Métrique', 'value': 'Score', 'index': 'Classe'}
                            )
                            fig_metrics.update_layout(height=400)
                            st.plotly_chart(fig_metrics, use_container_width=True)

                        if MODEL_MANAGER_AVAILABLE:
                            st.divider()
                            st.write("**Sauvegarder ce modèle**")
                            col_save1, col_save2 = st.columns([3, 1])
                            with col_save1:
                                model_name_save = st.text_input(
                                    "Nom du modèle",
                                    value=f"{model_key}_{embedding}",
                                    key="model_save_name"
                                )
                            with col_save2:
                                if st.button("Sauvegarder", key="save_model_btn"):
                                    try:
                                        registry = ModelRegistry(TRAINED_MODELS_DIR / "registry")
                                        metadata = {
                                            "name": model_name_save,
                                            "model_type": model_key,
                                            "embedding": embedding,
                                            "dataset": dataset_source,
                                            "accuracy": acc,
                                            "precision": report_dict.get('1', {}).get('precision', 0),
                                            "recall": report_dict.get('1', {}).get('recall', 0),
                                            "f1": report_dict.get('1', {}).get('f1-score', 0),
                                            "samples": len(df_sample),
                                            "test_size": len(X_test),
                                            "params": params,
                                        }
                                        model_path = registry.save_model(
                                            model=model,
                                            name=model_name_save,
                                            metadata=metadata,
                                            vectorizer=None
                                        )
                                        st.success(f"Modèle sauvegardé: {model_path.parent.name}")
                                    except Exception as e:
                                        st.error(f"Erreur lors de la sauvegarde: {e}")

                        st.write("**Rapport de classification détaillé**")
                        st.dataframe(pd.DataFrame(report_dict).transpose(), use_container_width=True)

                        st.write("**Distribution des prédictions**")
                        pred_df = pd.DataFrame({
                            'Classe réelle': y_test.reset_index(drop=True).map({0: 'Non-Cyber', 1: 'Cyber'}),
                            'Classe prédite': pd.Series(y_pred).map({0: 'Non-Cyber', 1: 'Cyber'}),
                            'Correct': (y_test.reset_index(drop=True).values == y_pred)
                        })
                        fig_dist = px.histogram(
                            pred_df,
                            x='Classe réelle',
                            color='Correct',
                            barmode='group',
                            title='Distribution des prédictions correctes/incorrectes',
                            labels={'Correct': 'Prédiction correcte'}
                        )
                        st.plotly_chart(fig_dist, use_container_width=True)

                        if "inference_results" not in st.session_state:
                            st.session_state.inference_results = []
                        st.session_state.inference_results.append({
                            "dataset": dataset_source,
                            "embedding": embedding,
                            "model": model_key,
                            "accuracy": acc,
                            "precision": report_dict.get('1', {}).get('precision', 0),
                            "recall": report_dict.get('1', {}).get('recall', 0),
                            "f1": report_dict.get('1', {}).get('f1-score', 0),
                            "samples": len(df_sample),
                            "test_size": len(X_test)
                        })
                        st.caption("Résultat ajouté aux Tests de session (onglet Exploitation).")


def _render_ensemble(tab_ensemble):
    with tab_ensemble:
        st.write("**Créer un ensemble de modèles pour améliorer les performances**")

        if not ENSEMBLE_AVAILABLE:
            st.error("Module ensemble non disponible.")
        else:
            processed_files = sorted([p for p in PROCESSED_DATA_DIR.glob("df_*.csv")])
            available_embeddings = [p.stem.replace("df_", "") for p in processed_files]

            if not available_embeddings:
                st.warning("Aucun dataset traité disponible.")
            else:
                st.caption("Sélectionnez un embedding, puis plusieurs modèles de classification à combiner.")
                ens_embedding = st.selectbox("Embedding", available_embeddings, key="ens_emb")
                ens_type = st.selectbox("Type d'ensemble", ["Voting (Soft)", "Voting (Hard)", "Stacking"])
                ens_models = st.multiselect(
                    "Modèles de classification à combiner",
                    ["logistic_regression", "random_forest", "svm", "lightgbm", "mlp"],
                    default=["logistic_regression", "random_forest", "lightgbm"]
                )
                col_ens1, col_ens2 = st.columns(2)
                with col_ens1:
                    ens_test_size = st.slider("Taille du test set (%)", min_value=10, max_value=50, value=20, step=5, key="ens_test")
                    ens_sample_limit = st.number_input("Limite échantillons (0=tous)", min_value=0, value=5000, step=500, key="ens_sample")
                with col_ens2:
                    meta_model_type = st.selectbox("Meta-learner (Stacking)", ["logistic_regression", "random_forest", "lightgbm"]) if "Stacking" in ens_type else "logistic_regression"

                if st.button("Créer et évaluer l'ensemble", type="primary"):
                    if len(ens_models) < 2:
                        st.error("Sélectionnez au moins 2 modèles pour créer un ensemble.")
                    else:
                        try:
                            with st.spinner("Chargement des données..."):
                                df = load_processed_df(ens_embedding)
                                if df is None:
                                    st.error("Dataset introuvable.")
                                    st.stop()

                                feature_cols = [c for c in df.columns if c.startswith(f"text_{ens_embedding}")]
                                if not feature_cols:
                                    st.error("Colonnes d'embedding introuvables.")
                                    st.stop()

                                if ens_sample_limit > 0 and len(df) > ens_sample_limit:
                                    df_sample = df.sample(n=ens_sample_limit, random_state=42)
                                else:
                                    df_sample = df

                                X = df_sample[feature_cols]
                                y = df_sample["type"]

                                X_train, X_test, y_train, y_test = train_test_split(
                                    X, y, test_size=ens_test_size/100, random_state=42
                                )

                            with st.spinner("Création des modèles de base..."):
                                params = get_model_params()
                                base_models = {}

                                soft_voting = "Soft" in ens_type
                                for model_name in ens_models:
                                    model = build_model(model_name, params[model_name], enable_proba=soft_voting)
                                    model.fit(X_train, y_train)
                                    base_models[model_name] = model

                                st.success(f"{len(base_models)} modèles de base entraînés")

                            if "Stacking" not in ens_type and soft_voting:
                                unsupported = [
                                    name for name, model in base_models.items()
                                    if not supports_predict_proba(model)
                                ]
                                if unsupported:
                                    st.warning(
                                        "Soft voting indisponible pour: "
                                        + ", ".join(unsupported)
                                        + ". Passage automatique en Voting (Hard)."
                                    )
                                    soft_voting = False

                            st.write("**Performance des modèles individuels:**")
                            comparison_df = compare_models_for_ensemble(base_models, X_train, y_train, cv=3)
                            st.dataframe(comparison_df, use_container_width=True)

                            with st.spinner("Création de l'ensemble..."):
                                if "Stacking" in ens_type:
                                    meta_params = params[meta_model_type]
                                    meta_model = build_model(meta_model_type, meta_params)
                                    ensemble = create_stacking_ensemble(base_models, meta_model, cv=3)
                                else:
                                    voting_type = "soft" if soft_voting else "hard"
                                    ensemble = create_voting_ensemble(base_models, voting=voting_type)

                            with st.spinner("Évaluation de l'ensemble..."):
                                results = train_and_evaluate_ensemble(
                                    ensemble, X_train, y_train, X_test, y_test
                                )

                            st.success("Ensemble créé et évalué!")

                            try:
                                import mlflow
                                mlflow.set_tracking_uri(get_mlflow_tracking_uri())
                                mlflow.set_experiment("safety-ai")
                                with mlflow.start_run(run_name=f"ensemble_{ens_type}_{ens_embedding}", log_system_metrics=True):
                                    mlflow.log_params({
                                        "embedding": ens_embedding,
                                        "ensemble_type": ens_type,
                                        "models": ", ".join(ens_models),
                                        "n_samples": len(df_sample),
                                    })
                                    log_system_metrics_manual()
                            except Exception:
                                pass

                            st.divider()

                            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                            with col_m1:
                                st.metric("Accuracy", f"{results['accuracy']:.3f}")
                            with col_m2:
                                st.metric("Precision", f"{results['precision']:.3f}")
                            with col_m3:
                                st.metric("Recall", f"{results['recall']:.3f}")
                            with col_m4:
                                st.metric("F1-Score", f"{results['f1']:.3f}")

                            st.write("**Configuration de l'ensemble:**")
                            info_df = pd.DataFrame({
                                "Paramètre": ["Dataset", "Embedding", "Type", "Modèles", "Test set"],
                                "Valeur": [
                                    f"df_{ens_embedding}.csv",
                                    ens_embedding,
                                    ens_type,
                                    ", ".join(ens_models),
                                    f"{len(X_test)} échantillons"
                                ]
                            })
                            st.dataframe(info_df, use_container_width=True, hide_index=True)

                            st.write("**Comparaison ensemble vs modèles individuels:**")

                            individual_results = []
                            for name, model in base_models.items():
                                y_pred_ind = model.predict(X_test)
                                acc = accuracy_score(y_test, y_pred_ind)
                                prec, rec, f1, _ = precision_recall_fscore_support(
                                    y_test, y_pred_ind, average="binary", zero_division=0
                                )

                                individual_results.append({
                                    "Modèle": name,
                                    "Type": "Individuel",
                                    "F1-Score": f1,
                                    "Accuracy": acc
                                })

                            individual_results.append({
                                "Modèle": f"Ensemble ({ens_type})",
                                "Type": "Ensemble",
                                "F1-Score": results["f1"],
                                "Accuracy": results["accuracy"]
                            })

                            comp_df = pd.DataFrame(individual_results).sort_values("F1-Score", ascending=False)

                            fig_comp = px.bar(
                                comp_df,
                                x="Modèle",
                                y="F1-Score",
                                color="Type",
                                title="Comparaison F1-Score: Ensemble vs Modèles individuels",
                                color_discrete_map={"Individuel": "lightblue", "Ensemble": "green"}
                            )
                            st.plotly_chart(fig_comp, use_container_width=True)

                            if "inference_results" not in st.session_state:
                                st.session_state.inference_results = []

                            st.session_state.inference_results.append({
                                "dataset": f"df_{ens_embedding}.csv",
                                "embedding": ens_embedding,
                                "model": f"Ensemble_{ens_type}",
                                "accuracy": results["accuracy"],
                                "precision": results["precision"],
                                "recall": results["recall"],
                                "f1": results["f1"],
                                "samples": len(df_sample),
                                "test_size": len(X_test)
                            })

                        except Exception as e:
                            st.error(f"Erreur: {e}")
                            import traceback
                            st.code(traceback.format_exc())


def render(tab):
    with tab:
        st.subheader("Inference et tests des modeles")

        st.info(
            "**Inference** = evaluer des modeles sklearn sur df_*.csv deja crees par Operations. "
            "Ne cree rien. Utile pour comparer modeles et creer des ensembles."
        )
        with st.expander("Details", expanded=False):
            st.markdown("""
**Inference** = evaluer et comparer des modeles sklearn sur des donnees **deja traitees**.

| Entree | Sortie |
|--------|--------|
| df_*.csv (data/processed/) | Metriques affichees (accuracy, F1...) |
| | Aucun fichier cree |

**Ne cree rien.** Utilise les df_*.csv produits par **Operations**.

- **Operations** : cree df_*.csv et entraine les modeles (pipeline complet)
- **Inference** : charge df_*.csv, fait un train/test rapide pour evaluer un modele ou creer un ensemble. Utile pour comparer modeles sans relancer tout le pipeline.
            """)

        tab_single, tab_ensemble = st.tabs(["Modele unique", "Ensemble de modeles"])
        _render_single(tab_single)
        _render_ensemble(tab_ensemble)
