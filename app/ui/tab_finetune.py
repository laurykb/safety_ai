"""Fine-tuning Transformers tab."""
from __future__ import annotations

import time

import pandas as pd
import streamlit as st

from cyberbullying.config import TRAINED_MODELS_DIR, get_mlflow_tracking_uri
from cyberbullying.mlflow_utils import log_system_metrics_manual

from .commons import data_file_label, list_data_files, load_binary_dataset

try:
    from transformers import AutoTokenizer
    from cyberbullying.finetune import (
        prepare_datasets,
        finetune_transformer,
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


def render(tab):
    with tab:
        st.subheader("Fine-tuning des modeles Transformers")

        st.info(
            "**Fine-tuning** = entrainer BERT ou RoBERTa sur vos donnees. "
            "Entree : CSV brut. Sortie : modele Transformer sauvegarde. Paradigme different de Operations (sklearn)."
        )
        with st.expander("Details", expanded=False):
            st.markdown("""
**Fine-tuning** = entrainer BERT ou RoBERTa (modeles Transformers) sur vos donnees.

| Entree | Sortie |
|--------|--------|
| CSV brut (text, type) | Modele Transformer sauvegarde (models/trained/) |

**Paradigme different** de Operations/Inference :
- Operations/Inference : TF-IDF ou BERT embedding + classifieur sklearn (LR, RF, SVM...)
- Fine-tuning : le Transformer entier est entraine pour la classification

Utilise les donnees brutes (pas df_*.csv). Le modele fine-tune est ensuite utilisable dans l'onglet Prediction.
            """)

        if not TRANSFORMERS_AVAILABLE:
            st.error("Le module transformers n'est pas disponible. Installez-le avec: pip install transformers torch")
        else:
            st.info("Cette section permet de fine-tuner les modèles BERT et RoBERTa sur vos données de cyberbullying.")

            col_ft1, col_ft2 = st.columns(2)

            with col_ft1:
                st.write("**Configuration du fine-tuning**")
                transformer_type = st.selectbox("Modèle Transformer", ["BERT (bert-base-uncased)", "RoBERTa (roberta-base)"])

                files = list_data_files()
                if files:
                    dataset_for_ft = st.selectbox("Dataset d'entraînement", [data_file_label(p) for p in files])
                else:
                    st.warning("Aucun dataset disponible.")
                    dataset_for_ft = None

                text_col = st.text_input("Colonne texte", value="text")
                label_col = st.text_input("Colonne label", value="type")

                epochs = st.number_input("Nombre d'époques", min_value=1, max_value=20, value=3, step=1)
                batch_size = st.number_input("Batch size", min_value=4, max_value=64, value=16, step=4)
                learning_rate = st.number_input("Learning rate", min_value=0.00001, max_value=0.001, value=0.00002, format="%.5f", step=0.00001)
                max_length = st.number_input("Max length (tokens)", min_value=64, max_value=512, value=128, step=64)

            with col_ft2:
                st.write("**Paramètres avancés**")
                warmup_steps = st.number_input("Warmup steps", min_value=0, max_value=1000, value=100, step=50)
                weight_decay = st.number_input("Weight decay", min_value=0.0, max_value=0.1, value=0.01, format="%.3f", step=0.01)
                save_steps = st.number_input("Save steps", min_value=100, max_value=2000, value=500, step=100)
                eval_steps = st.number_input("Eval steps", min_value=100, max_value=2000, value=500, step=100)

                eval_split = st.slider("Pourcentage validation", min_value=0.1, max_value=0.4, value=0.2, step=0.05)
                sample_limit = st.number_input("Limite échantillons (0=tous)", min_value=0, value=0, step=1000, help="Pour tests rapides")

            st.divider()

            if st.button("Lancer le fine-tuning", type="primary"):
                if dataset_for_ft is None:
                    st.error("Veuillez sélectionner un dataset.")
                else:
                    try:
                        with st.spinner("Chargement du dataset..."):
                            df_path = next(p for p in list_data_files() if data_file_label(p) == dataset_for_ft)
                            df = load_binary_dataset(df_path, n_samples=None)

                            if text_col not in df.columns or label_col not in df.columns:
                                st.error(f"Colonnes manquantes. Colonnes disponibles: {list(df.columns)}")
                                st.stop()

                            if sample_limit > 0 and len(df) > sample_limit:
                                df = df.sample(n=sample_limit, random_state=42)

                            st.success(f"Dataset chargé: {len(df)} échantillons")

                        model_name = "bert-base-uncased" if "BERT" in transformer_type else "roberta-base"
                        output_dir = TRAINED_MODELS_DIR / f"finetuned_{model_name.split('-')[0]}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
                        output_dir.mkdir(parents=True, exist_ok=True)

                        with st.spinner("Initialisation du tokenizer..."):
                            tokenizer = AutoTokenizer.from_pretrained(model_name)

                        with st.spinner("Préparation des datasets..."):
                            train_dataset, val_dataset = prepare_datasets(
                                df,
                                text_column=text_col,
                                label_column=label_col,
                                tokenizer=tokenizer,
                                max_length=int(max_length),
                                test_size=eval_split,
                            )
                            st.success(f"Train: {len(train_dataset)} | Validation: {len(val_dataset)}")
                            if len(train_dataset) < 200:
                                st.warning(f"Dataset petit ({len(train_dataset)} ex.). Le fine-tuning sera rapide. Pour des résultats plus robustes, utilisez plus de données.")

                        st.info(f"""
                        **Configuration d'entraînement:**
                        - Modèle: {model_name}
                        - Dataset: {dataset_for_ft}
                        - Échantillons: {len(df)} (Train: {len(train_dataset)}, Val: {len(val_dataset)})
                        - Époques: {epochs}
                        - Batch size: {batch_size}
                        - Learning rate: {learning_rate}
                        - Output: {output_dir.name}
                        """)

                        st.info(f"Entraînement de {model_name} sur {len(train_dataset)} ex. train, {len(val_dataset)} ex. val ({int(epochs)} époques). Cela peut prendre plusieurs minutes.")
                        progress_bar = st.progress(0, text="Entraînement en cours...")

                        t_ft_start = time.perf_counter()
                        trainer, metrics = finetune_transformer(
                            model_name=model_name,
                            train_dataset=train_dataset,
                            val_dataset=val_dataset,
                            output_dir=output_dir,
                            num_epochs=int(epochs),
                            batch_size=int(batch_size),
                            learning_rate=float(learning_rate),
                            warmup_steps=int(warmup_steps),
                            weight_decay=float(weight_decay),
                            save_steps=int(save_steps),
                            eval_steps=int(eval_steps),
                        )
                        t_ft_elapsed = time.perf_counter() - t_ft_start

                        progress_bar.progress(100, text=f"Entraînement terminé en {t_ft_elapsed:.1f}s")

                        tokenizer.save_pretrained(str(output_dir))

                        st.success(f"Fine-tuning terminé en {t_ft_elapsed:.1f}s ({t_ft_elapsed/60:.1f} min)")

                        try:
                            import mlflow
                            mlflow.set_tracking_uri(get_mlflow_tracking_uri())
                            mlflow.set_experiment("safety-ai")
                            with mlflow.start_run(run_name=f"finetune_{model_name.split('-')[0]}_{pd.Timestamp.now().strftime('%H%M%S')}", log_system_metrics=True):
                                mlflow.log_params({
                                    "model": model_name,
                                    "dataset": dataset_for_ft,
                                    "epochs": epochs,
                                    "batch_size": batch_size,
                                    "learning_rate": learning_rate,
                                })
                                mlflow.log_metrics({
                                    "accuracy": metrics.get("eval_accuracy", 0),
                                    "f1": metrics.get("eval_f1", 0),
                                    "precision": metrics.get("eval_precision", 0),
                                    "recall": metrics.get("eval_recall", 0),
                                })
                                log_system_metrics_manual()
                                mlflow.log_artifact(str(output_dir))
                        except Exception:
                            pass

                        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                        with col_m1:
                            st.metric("Accuracy", f"{metrics.get('eval_accuracy', 0):.3f}")
                        with col_m2:
                            st.metric("Precision", f"{metrics.get('eval_precision', 0):.3f}")
                        with col_m3:
                            st.metric("Recall", f"{metrics.get('eval_recall', 0):.3f}")
                        with col_m4:
                            st.metric("F1-Score", f"{metrics.get('eval_f1', 0):.3f}")

                        st.write("**Métriques complètes:**")
                        st.json(metrics)

                        st.success(f"Modèle sauvegardé dans: `{output_dir}`")

                    except Exception as e:
                        st.error(f"Erreur durant le fine-tuning: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())
