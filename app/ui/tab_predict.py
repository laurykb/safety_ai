"""Prédiction en temps réel tab."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from cyberbullying.config import PROCESSED_DATA_DIR, PRETRAINED_MODELS_DIR, TRAINED_MODELS_DIR, get_mlflow_tracking_uri
from cyberbullying.mlflow_utils import log_system_metrics_manual

from .commons import get_model_params, load_processed_df

try:
    from cyberbullying.finetune import predict_with_finetuned
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    from cyberbullying.explainability import LIME_AVAILABLE
except ImportError:
    LIME_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier
except Exception:
    LGBMClassifier = None


def _render_gpu_info():
    try:
        import torch
        gpu_available = torch.cuda.is_available()
        if gpu_available:
            gpu_name = torch.cuda.get_device_name(0)
            st.success(f"GPU disponible: {gpu_name}")
        else:
            st.info("GPU non disponible - utilisation CPU (plus lent)")
            with st.expander("Activer CUDA (GPU)"):
                st.write("**1. Vérifiez vos drivers NVIDIA:**")
                st.code("nvidia-smi", language="bash")
                st.write("**2. Désinstallez PyTorch CPU puis installez avec CUDA:**")
                st.code(
                    "pip uninstall torch torchvision torchaudio -y\n"
                    "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121",
                    language="bash",
                )
                st.caption("Pour CUDA 12.4: remplacer cu121 par cu124")
                st.write("**3. Redémarrez l'application.**")
    except ImportError:
        st.warning("PyTorch non installe. Installez avec: pip install torch")


def _render_sklearn_prediction(input_text, pred_embedding, pred_model, confidence_threshold, use_explainability):
    from cyberbullying.embedder import embed_texts

    cache_key = f"model_{pred_model}_{pred_embedding}"

    if cache_key in st.session_state:
        model = st.session_state[cache_key]
        st.info(f"Utilisation du modèle en cache: {pred_model} + {pred_embedding}")
    else:
        with st.spinner(f"Entraînement du modèle {pred_model} avec {pred_embedding}..."):
            df_processed = load_processed_df(pred_embedding)
            if df_processed is None:
                st.error(f"Dataset processed pour {pred_embedding} introuvable.")
                return

            feature_cols = [c for c in df_processed.columns if c.startswith(f"text_{pred_embedding}")]
            if not feature_cols:
                st.error(f"Colonnes d'embedding {pred_embedding} introuvables.")
                return

            X = df_processed[feature_cols].values
            y = df_processed["type"].values

            params = get_model_params()

            if pred_model == "logistic_regression":
                model = LogisticRegression(**params["logistic_regression"])
            elif pred_model == "random_forest":
                model = RandomForestClassifier(**params["random_forest"])
            elif pred_model == "svm":
                model = SVC(**params["svm"], probability=True)
            elif pred_model == "mlp":
                model = MLPClassifier(**params["mlp"])
            elif pred_model == "lightgbm" and LGBMClassifier is not None:
                model = LGBMClassifier(**params["lightgbm"])
            else:
                st.error(f"Modèle {pred_model} non disponible.")
                return

            model.fit(X, y)
            st.session_state[cache_key] = model
            st.success(f"Modèle {pred_model} entraîné avec {pred_embedding}")
            try:
                import mlflow
                mlflow.set_tracking_uri(get_mlflow_tracking_uri())
                mlflow.set_experiment("safety-ai")
                with mlflow.start_run(run_name=f"prediction_setup_{pred_model}_{pred_embedding}", log_system_metrics=True):
                    mlflow.log_params({
                        "embedding": pred_embedding,
                        "model": pred_model,
                        "n_samples": len(df_processed),
                    })
                    log_system_metrics_manual()
            except Exception:
                pass

    X_input = embed_texts([input_text], method=pred_embedding)
    prediction = model.predict(X_input)[0]

    if hasattr(model, "predict_proba"):
        probas = model.predict_proba(X_input)[0]
        prob_cyber = probas[1]
    else:
        prob_cyber = float(prediction)

    st.divider()

    if prediction == 1:
        st.error("**CYBERBULLYING DÉTECTÉ**")
    else:
        st.success("**PAS DE CYBERBULLYING DÉTECTÉ**")

    col_r1, col_r2, col_r3 = st.columns(3)
    with col_r1:
        st.metric("Prédiction", "Cyberbullying" if prediction == 1 else "Normal")
    with col_r2:
        st.metric("Confiance", f"{prob_cyber:.1%}")
    with col_r3:
        st.metric("Seuil", f"{confidence_threshold:.1%}")

    st.write("**Distribution de probabilité:**")
    if hasattr(model, "predict_proba"):
        prob_df = pd.DataFrame({
            "Classe": ["Normal", "Cyberbullying"],
            "Probabilité": probas
        })
    else:
        prob_df = pd.DataFrame({
            "Classe": ["Normal", "Cyberbullying"],
            "Probabilité": [1 - prob_cyber, prob_cyber]
        })

    fig = px.bar(
        prob_df,
        x="Classe",
        y="Probabilité",
        color="Classe",
        color_discrete_map={"Normal": "green", "Cyberbullying": "red"},
        title="Probabilités par classe"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.info(f"**Modèle utilisé:** {pred_model} avec embedding {pred_embedding}")

    if use_explainability and LIME_AVAILABLE:
        st.divider()
        st.write("### Explication de la prédiction (LIME)")
        st.info("**Mots en rouge**: contribuent au cyberbullying | **Mots en vert**: contribuent à la normalité")

        try:
            from lime.lime_text import LimeTextExplainer

            explainer = LimeTextExplainer(class_names=["Normal", "Cyberbullying"])

            def predict_fn(texts):
                X_batch = embed_texts(texts, method=pred_embedding)
                if hasattr(model, "predict_proba"):
                    return model.predict_proba(X_batch)
                else:
                    predictions = model.predict(X_batch)
                    probas = np.zeros((len(predictions), 2))
                    for i, pred in enumerate(predictions):
                        probas[i] = [1-pred, pred]
                    return probas

            exp = explainer.explain_instance(
                input_text,
                predict_fn,
                num_features=10,
                num_samples=100
            )

            st.write("**Mots les plus influents:**")

            lime_data = exp.as_list()
            if lime_data:
                words = [item[0] for item in lime_data]
                weights = [item[1] for item in lime_data]

                lime_df = pd.DataFrame({
                    "Mot": words,
                    "Impact": weights,
                    "Direction": ["Cyberbullying" if w > 0 else "Normal" for w in weights]
                }).sort_values("Impact", key=abs, ascending=False)

                fig_lime = px.bar(
                    lime_df,
                    x="Impact",
                    y="Mot",
                    orientation="h",
                    color="Direction",
                    color_discrete_map={"Cyberbullying": "red", "Normal": "green"},
                    title="Impact des mots sur la prédiction"
                )
                st.plotly_chart(fig_lime, use_container_width=True)
            else:
                st.info("Aucun mot particulièrement influent détecté.")

        except Exception as e:
            st.error(f"Erreur LIME: {e}")


def _render_transformer_prediction(input_text, selected_model_name, model_paths, confidence_threshold):
    model_path = model_paths[selected_model_name]

    if "[PRÉ-ENTRAÎNÉ]" in selected_model_name:
        st.info(f"Utilisation du modèle pré-entraîné: {model_path.name}")
    else:
        st.info(f"Utilisation du modèle fine-tuné: {model_path.name}")

    predictions, probabilities = predict_with_finetuned(
        model_path,
        [input_text],
        batch_size=1,
        max_length=128
    )

    prediction = predictions[0]
    prob_cyber = probabilities[0][1]

    st.divider()

    if prediction == 1:
        st.error("**CYBERBULLYING DÉTECTÉ**")
    else:
        st.success("**PAS DE CYBERBULLYING DÉTECTÉ**")

    col_r1, col_r2, col_r3 = st.columns(3)
    with col_r1:
        st.metric("Prédiction", "Cyberbullying" if prediction == 1 else "Normal")
    with col_r2:
        st.metric("Confiance", f"{prob_cyber:.1%}")
    with col_r3:
        st.metric("Seuil", f"{confidence_threshold:.1%}")

    prob_df = pd.DataFrame({
        "Classe": ["Normal", "Cyberbullying"],
        "Probabilité": probabilities[0]
    })

    fig = px.bar(
        prob_df,
        x="Classe",
        y="Probabilité",
        color="Classe",
        color_discrete_map={"Normal": "green", "Cyberbullying": "red"},
        title="Probabilités par classe"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.info(f"**Modèle utilisé:** {model_path.name}")


def render(tab):
    with tab:
        st.subheader("Prediction en temps reel")

        st.info(
            "**Prediction** = tester un texte avec un modele (sklearn ou Transformer fine-tune). "
            "Utilise les df_*.csv ou modeles fine-tunes. Option LIME pour expliquer la prediction."
        )

        _render_gpu_info()

        use_explainability = st.checkbox(
            "Activer l'explainability (LIME)",
            value=False,
            help="Explique quels mots influencent la prédiction (nécessite LIME installé)"
        )

        if use_explainability and not LIME_AVAILABLE:
            st.warning("LIME n'est pas installé. Installez avec: pip install lime")
            use_explainability = False

        col_pred1, col_pred2 = st.columns([2, 1])

        with col_pred1:
            pred_mode = st.radio(
                "Mode de prédiction",
                ["Modèles classiques (sklearn)", "Transformers fine-tunés"],
                horizontal=True
            )

        with col_pred2:
            confidence_threshold = st.slider("Seuil de confiance", 0.0, 1.0, 0.5, 0.05)

        input_text = st.text_area(
            "Texte à analyser",
            height=150,
            placeholder="Entrez le texte à analyser...",
            help="Entrez le texte que vous souhaitez analyser"
        )

        st.divider()

        if pred_mode == "Modèles classiques (sklearn)":
            processed_files = sorted([p for p in PROCESSED_DATA_DIR.glob("df_*.csv")])
            available_embeddings_raw = [p.stem.replace("df_", "") for p in processed_files]
            supported_embeddings = ["word2vec", "glove", "bert", "roberta"]
            available_embeddings = [e for e in available_embeddings_raw if e in supported_embeddings]

            if not available_embeddings:
                st.warning("Aucun embedding traité disponible.")
                st.info("Les embeddings supportes sont: word2vec, glove, bert, roberta")
                st.info("Note: TF-IDF et BoW sont exclus car ils ne fonctionnent pas correctement sur un seul texte.")
            else:
                col_model1, col_model2 = st.columns(2)

                with col_model1:
                    pred_embedding = st.selectbox("Embedding", available_embeddings, key="pred_emb")

                with col_model2:
                    pred_model = st.selectbox(
                        "Modèle",
                        ["logistic_regression", "random_forest", "svm", "lightgbm", "mlp"],
                        key="pred_model"
                    )

                if st.button("Lancer la prédiction", type="primary", disabled=not input_text.strip(), key="pred_btn1"):
                    if not input_text.strip():
                        st.error("Veuillez entrer un texte.")
                    else:
                        with st.spinner("Prédiction en cours..."):
                            try:
                                _render_sklearn_prediction(
                                    input_text, pred_embedding, pred_model,
                                    confidence_threshold, use_explainability
                                )
                            except Exception as e:
                                st.error(f"Erreur: {e}")

        else:
            if not TRANSFORMERS_AVAILABLE:
                st.error("Transformers non disponible. Installez avec: pip install transformers torch")
            else:
                pretrained_models = sorted([p for p in PRETRAINED_MODELS_DIR.glob("my_finetuned_*") if p.is_dir()])
                finetuned_models = sorted([p for p in TRAINED_MODELS_DIR.glob("finetuned_*") if p.is_dir()])

                all_models = []
                model_paths = {}

                for model in pretrained_models:
                    name = f"[PRÉ-ENTRAÎNÉ] {model.name}"
                    all_models.append(name)
                    model_paths[name] = model

                for model in finetuned_models:
                    name = f"[FINE-TUNÉ] {model.name}"
                    all_models.append(name)
                    model_paths[name] = model

                if not all_models:
                    st.warning("Aucun modèle Transformer trouvé.")
                    st.info("Les modèles pré-entraînés BERT/RoBERTa devraient être dans models/pretrained/")
                    st.info("Vous pouvez créer des modèles fine-tunés dans l'onglet 'Fine-tuning Transformers'")
                else:
                    selected_model_name = st.selectbox(
                        "Modèle Transformer",
                        all_models,
                        key="pred_transformer"
                    )

                    if st.button("Lancer la prédiction", type="primary", disabled=not input_text.strip(), key="pred_btn2"):
                        if not input_text.strip():
                            st.error("Veuillez entrer un texte.")
                        else:
                            with st.spinner("Prédiction en cours..."):
                                try:
                                    _render_transformer_prediction(
                                        input_text, selected_model_name,
                                        model_paths, confidence_threshold
                                    )
                                except Exception as e:
                                    st.error(f"Erreur: {e}")
                                    import traceback
                                    st.code(traceback.format_exc())
