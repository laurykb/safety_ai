from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cyberbullying.config import (
    ANALYSIS_DIR,
    ERROR_ANALYSIS_DIR,
    EXPERIMENTS_DIR,
    PRETRAINED_BERT_DIR,
    PRETRAINED_ROBERTA_DIR,
    PRETRAINED_MODELS_DIR,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    REPORTS_DIR,
    TRAINED_MODELS_DIR,
)
from cyberbullying.loading import binary_load_data, merge_datasets

try:
    from transformers import AutoTokenizer
    from cyberbullying.finetune import (
        prepare_datasets,
        finetune_transformer,
        predict_with_finetuned,
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    from cyberbullying.explainability import (
        explain_with_lime,
        explain_with_shap,
        highlight_text_lime,
        format_lime_explanation,
        LIME_AVAILABLE,
        SHAP_AVAILABLE,
    )
except ImportError:
    LIME_AVAILABLE = False
    SHAP_AVAILABLE = False

try:
    from cyberbullying.hyperopt import (
        auto_optimize_model,
        get_optimization_history,
        OPTUNA_AVAILABLE,
    )
except ImportError:
    OPTUNA_AVAILABLE = False

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

try:
    from lightgbm import LGBMClassifier
except Exception:  # pragma: no cover - optional dependency
    LGBMClassifier = None


st.set_page_config(
    page_title="Cyberbullying Dashboard",
    page_icon="chart_with_upwards_trend",
    layout="wide",
)


# Fonctions utilitaires pour le chargement de données et le caching

@st.cache_data(show_spinner=False)
def list_data_files() -> list[Path]:
    return sorted([p for p in RAW_DATA_DIR.iterdir() if p.suffix.lower() in {".csv", ".xlsx"}])


@st.cache_data(show_spinner=False)
def load_binary_dataset(path: Path, n_samples: int | None = None) -> pd.DataFrame:
    """Charge un dataset brut (CSV ou Excel) sans validation stricte."""
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    elif path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    else:
        df = pd.DataFrame()
    
    if n_samples and len(df) > n_samples:
        df = df.sample(n=n_samples, random_state=42)
    
    return df


@st.cache_data(show_spinner=False)
def load_processed_df(embedding: str) -> pd.DataFrame | None:
    file_path = PROCESSED_DATA_DIR / f"df_{embedding}.csv"
    if not file_path.exists():
        return None
    return pd.read_csv(file_path)


def parse_report_file(path: Path) -> dict[str, Any] | None:
    try:
        df = pd.read_csv(path, index_col=0)
    except Exception:
        return None

    name = path.stem.replace("_report", "")
    parts = name.split("_")
    model_name = parts[0] if parts else "Unknown"
    embed_name = "Unknown"
    for embed in ["tfidf", "bow", "word2vec", "glove", "bert", "roberta"]:
        if embed in name.lower():
            embed_name = embed
            break

    acc = df.loc["accuracy", "precision"] if "accuracy" in df.index else np.nan
    class_1_idx = [idx for idx in df.index if "1" in str(idx)]
    if class_1_idx:
        row = df.loc[class_1_idx[0]]
        prec = row.get("precision", np.nan)
        rec = row.get("recall", np.nan)
        f1 = row.get("f1-score", np.nan)
    else:
        prec = rec = f1 = np.nan

    return {
        "model": model_name,
        "embedding": embed_name,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "path": str(path),
    }


def get_model_params() -> dict[str, dict[str, Any]]:
    defaults = {
        "logistic_regression": {"C": 1.0, "max_iter": 1000},
        "random_forest": {"n_estimators": 200, "max_depth": None},
        "svm": {"C": 1.0, "kernel": "rbf"},
        "lightgbm": {"n_estimators": 300, "learning_rate": 0.05},
        "mlp": {"hidden_layer_sizes": (128, 64), "max_iter": 300},
    }
    if "model_params" not in st.session_state:
        st.session_state.model_params = defaults
    return st.session_state.model_params


def build_model(model_key: str, params: dict[str, Any], enable_proba: bool = False):
    if model_key == "logistic_regression":
        return LogisticRegression(C=params["C"], max_iter=params["max_iter"])
    if model_key == "random_forest":
        return RandomForestClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            random_state=42,
        )
    if model_key == "svm":
        return SVC(C=params["C"], kernel=params["kernel"], probability=enable_proba)
    if model_key == "lightgbm":
        if LGBMClassifier is None:
            raise RuntimeError("LightGBM n'est pas disponible dans l'environnement.")
        return LGBMClassifier(
            n_estimators=params["n_estimators"],
            learning_rate=params["learning_rate"],
            random_state=42,
            verbose=-1,
        )
    if model_key == "mlp":
        return MLPClassifier(
            hidden_layer_sizes=params["hidden_layer_sizes"],
            max_iter=params["max_iter"],
            random_state=42,
        )
    raise ValueError("Modèle inconnu.")


def supports_predict_proba(model) -> bool:
    try:
        _ = model.predict_proba
        return True
    except Exception:
        return False


# Initialisation de l'interface

st.title("Cyberbullying – Dashboard des métriques")

with st.sidebar:
    st.header("Navigation")
    st.write("Tableau de bord Streamlit pour visualiser les données et les performances.")
    st.write(f"Données: {RAW_DATA_DIR}")
    st.write(f"Modèles entraînés: {TRAINED_MODELS_DIR}")


# Tabs

tab_data, tab_embeddings, tab_finetune, tab_inference, tab_predict, tab_results, tab_settings = st.tabs(
    [
        "Datasets",
        "Embeddings",
        "Fine-tuning Transformers",
        "Inférence & Tests",
        "Prédiction en temps réel",
        "Exploitation des résultats",
        "Settings",
    ]
)


# Onglet: Datasets

with tab_data:
    st.subheader("Visualisation et tri des datasets")

    files = list_data_files()
    if not files:
        st.warning("Aucun dataset trouvé dans data/raw.")
    else:
        dataset_names = [p.name for p in files]
        selected = st.multiselect("Choisir un ou plusieurs datasets", dataset_names, default=dataset_names[:2])
        n_samples = st.number_input("Nombre d'échantillons (0 = tous)", min_value=0, value=500, step=100)
        label_filter = st.selectbox("Filtre label", ["Tous", "0", "1"])
        search_text = st.text_input("Recherche texte")
        do_merge = st.checkbox("Fusionner les datasets sélectionnés", value=False)

        dfs: list[pd.DataFrame] = []
        for name in selected:
            path = RAW_DATA_DIR / name
            df = load_binary_dataset(path, n_samples=None if n_samples == 0 else int(n_samples))
            if "type" in df.columns:
                if label_filter in {"0", "1"}:
                    df = df[df["type"] == int(label_filter)]
            if search_text:
                if "text" in df.columns:
                    df = df[df["text"].str.contains(search_text, case=False, na=False)]
            dfs.append(df)

        if dfs:
            if do_merge and all("text" in d.columns and "type" in d.columns for d in dfs):
                merged = merge_datasets(dfs)
                st.write("Dataset fusionné")
                st.dataframe(merged, use_container_width=True)
                if "type" in merged.columns:
                    counts = merged["type"].value_counts().reset_index()
                    counts.columns = ["label", "count"]
                    fig = px.bar(counts, x="label", y="count", title="Répartition des classes")
                    st.plotly_chart(fig, use_container_width=True)
            else:
                for name, df in zip(selected, dfs):
                    st.write(f"Dataset: {name}")
                    st.dataframe(df, use_container_width=True)
                    if "type" in df.columns:
                        counts = df["type"].value_counts().reset_index()
                        counts.columns = ["label", "count"]
                        fig = px.bar(counts, x="label", y="count", title=f"Répartition des classes – {name}")
                        st.plotly_chart(fig, use_container_width=True)


# Onglet: Embeddings

with tab_embeddings:
    st.subheader("Embeddings utilisés")

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


# Onglet: Fine-tuning Transformers

with tab_finetune:
    st.subheader("Fine-tuning des modèles Transformers")
    
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
                dataset_for_ft = st.selectbox("Dataset d'entraînement", [p.name for p in files])
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
                    # Charger les données
                    with st.spinner("Chargement du dataset..."):
                        df_path = RAW_DATA_DIR / dataset_for_ft
                        df = load_binary_dataset(df_path, n_samples=None)
                        
                        if text_col not in df.columns or label_col not in df.columns:
                            st.error(f"Colonnes manquantes. Colonnes disponibles: {list(df.columns)}")
                            st.stop()
                        
                        # Échantillonnage si nécessaire
                        if sample_limit > 0 and len(df) > sample_limit:
                            df = df.sample(n=sample_limit, random_state=42)
                        
                        st.success(f"Dataset chargé: {len(df)} échantillons")
                    
                    # Préparer le modèle
                    model_name = "bert-base-uncased" if "BERT" in transformer_type else "roberta-base"
                    output_dir = TRAINED_MODELS_DIR / f"finetuned_{model_name.split('-')[0]}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
                    output_dir.mkdir(parents=True, exist_ok=True)
                    
                    with st.spinner("Initialisation du tokenizer..."):
                        tokenizer = AutoTokenizer.from_pretrained(model_name)
                    
                    # Préparer les datasets
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
                    
                    # Afficher les infos d'entraînement
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
                    
                    # Fine-tuning
                    progress_bar = st.progress(0, text="Entraînement en cours...")
                    
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
                    
                    progress_bar.progress(100, text="Entraînement terminé!")
                    
                    # Sauvegarder le tokenizer
                    tokenizer.save_pretrained(str(output_dir))
                    
                    # Afficher les résultats
                    st.success("Fine-tuning terminé avec succès!")
                    
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


# Onglet: Inférence et Tests

with tab_inference:
    st.subheader("Inférence et tests des modèles de classification")
    
    # Sous-onglets
    tab_single, tab_ensemble = st.tabs(["Modèle unique", "Ensemble de modèles"])
    
    # === MODÈLE UNIQUE ===
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
            # Sélection du dataset source
            raw_files = list_data_files()
            if raw_files:
                dataset_source = st.selectbox("Dataset source", [p.name for p in raw_files], help="Dataset utilisé pour les tests")
            else:
                dataset_source = "Dataset inconnu"
            
            test_size = st.slider("Taille du test set (%)", min_value=10, max_value=50, value=20, step=5)
            sample_limit = st.number_input("Limite d'échantillons (0 = tous)", min_value=0, value=0, step=500)

        if st.button("Lancer l'inférence", type="primary"):
            df = load_processed_df(embedding)
            if df is None:
                st.error("Dataset traité introuvable.")
            else:
                feature_cols = [c for c in df.columns if c.startswith(f"text_{embedding}")]
                if not feature_cols:
                    st.error("Colonnes d'embedding introuvables.")
                else:
                    # Échantillonnage si nécessaire
                    if sample_limit > 0 and len(df) > sample_limit:
                        df_sample = df.sample(n=sample_limit, random_state=42)
                    else:
                        df_sample = df
                    
                    X = df_sample[feature_cols]
                    y = df_sample["type"]

                    params = get_model_params()[model_key]
                    model = build_model(model_key, params)

                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size/100, random_state=42
                    )
                    
                    with st.spinner("Entraînement en cours..."):
                        model.fit(X_train, y_train)
                    
                    with st.spinner("Prédiction en cours..."):
                        y_pred = model.predict(X_test)
                    
                    # Métriques principales
                    st.success("Inférence terminée!")
                    
                    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                    acc = accuracy_score(y_test, y_pred)
                    
                    report_dict = classification_report(y_test, y_pred, output_dict=True)
                    
                    with col_m1:
                        st.metric("Accuracy", f"{acc:.3f}")
                    with col_m2:
                        st.metric("Precision (classe 1)", f"{report_dict.get('1', {}).get('precision', 0):.3f}")
                    with col_m3:
                        st.metric("Recall (classe 1)", f"{report_dict.get('1', {}).get('recall', 0):.3f}")
                    with col_m4:
                        st.metric("F1-Score (classe 1)", f"{report_dict.get('1', {}).get('f1-score', 0):.3f}")
                    
                    st.divider()
                    
                    # Informations sur le test
                    st.write("**Informations sur le test:**")
                    info_df = pd.DataFrame({
                        "Paramètre": ["Dataset source", "Embedding", "Modèle", "Échantillons total", "Train set", "Test set"],
                        "Valeur": [
                            dataset_source,
                            embedding,
                            model_key,
                            len(df_sample),
                            len(X_train),
                            len(X_test)
                        ]
                    })
                    st.dataframe(info_df, use_container_width=True, hide_index=True)
                    
                    st.divider()
                    
                    # Visualisations avancées
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
                      # Sauvegarder le modèle
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
                                        vectorizer=None  # À ajouter si nécessaire
                                    )
                                    
                                    st.success(f"Modèle sauvegardé: {model_path.parent.name}")
                                    
                                except Exception as e:
                                    st.error(f"Erreur lors de la sauvegarde: {e}")
                    
                    #   st.plotly_chart(fig_metrics, use_container_width=True)
                    
                    # Rapport détaillé
                    st.write("**Rapport de classification détaillé**")
                    st.dataframe(pd.DataFrame(report_dict).transpose(), use_container_width=True)
                    
                    # Distribution des prédictions
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
                    
                    # Sauvegarder les résultats dans session_state
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
    
    # === ENSEMBLE DE MODÈLES ===
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
                col_ens1, col_ens2 = st.columns(2)
                
                with col_ens1:
                    ens_embedding = st.selectbox("Embedding", available_embeddings, key="ens_emb")
                    ens_type = st.selectbox("Type d'ensemble", ["Voting (Soft)", "Voting (Hard)", "Stacking"])
                    
                    ens_models = st.multiselect(
                        "Modèles à combiner",
                        ["logistic_regression", "random_forest", "svm", "lightgbm", "mlp"],
                        default=["logistic_regression", "random_forest", "lightgbm"]
                    )
                
                with col_ens2:
                    raw_files = list_data_files()
                    if raw_files:
                        ens_dataset = st.selectbox("Dataset source", [p.name for p in raw_files], key="ens_data")
                    else:
                        ens_dataset = "Dataset inconnu"
                    
                    ens_test_size = st.slider("Taille du test set (%)", min_value=10, max_value=50, value=20, step=5, key="ens_test")
                    ens_sample_limit = st.number_input("Limite échantillons (0=tous)", min_value=0, value=5000, step=500, key="ens_sample")
                    
                    if ens_type == "Stacking":
                        meta_model_type = st.selectbox("Meta-learner", ["logistic_regression", "random_forest", "lightgbm"])
                
                if st.button("Créer et évaluer l'ensemble", type="primary"):
                    if len(ens_models) < 2:
                        st.error("Sélectionnez au moins 2 modèles pour créer un ensemble.")
                    else:
                        try:
                            # Charger les données
                            with st.spinner("Chargement des données..."):
                                df = load_processed_df(ens_embedding)
                                if df is None:
                                    st.error("Dataset introuvable.")
                                    st.stop()
                                
                                feature_cols = [c for c in df.columns if c.startswith(f"text_{ens_embedding}")]
                                if not feature_cols:
                                    st.error("Colonnes d'embedding introuvables.")
                                    st.stop()
                                
                                # Échantillonnage
                                if ens_sample_limit > 0 and len(df) > ens_sample_limit:
                                    df_sample = df.sample(n=ens_sample_limit, random_state=42)
                                else:
                                    df_sample = df
                                
                                X = df_sample[feature_cols]
                                y = df_sample["type"]
                                
                                X_train, X_test, y_train, y_test = train_test_split(
                                    X, y, test_size=ens_test_size/100, random_state=42
                                )
                            
                            # Créer les modèles de base
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
                            
                            # Comparer les modèles individuels
                            st.write("**Performance des modèles individuels:**")
                            comparison_df = compare_models_for_ensemble(base_models, X_train, y_train, cv=3)
                            st.dataframe(comparison_df, use_container_width=True)
                            
                            # Créer l'ensemble
                            with st.spinner("Création de l'ensemble..."):
                                if "Stacking" in ens_type:
                                    meta_params = params[meta_model_type]
                                    meta_model = build_model(meta_model_type, meta_params)
                                    ensemble = create_stacking_ensemble(base_models, meta_model, cv=3)
                                else:
                                    voting_type = "soft" if soft_voting else "hard"
                                    ensemble = create_voting_ensemble(base_models, voting=voting_type)
                            
                            # Évaluer l'ensemble
                            with st.spinner("Évaluation de l'ensemble..."):
                                results = train_and_evaluate_ensemble(
                                    ensemble, X_train, y_train, X_test, y_test
                                )
                            
                            # Afficher les résultats
                            st.success("Ensemble créé et évalué!")
                            
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
                            
                            # Informations
                            st.write("**Configuration de l'ensemble:**")
                            info_df = pd.DataFrame({
                                "Paramètre": ["Dataset", "Embedding", "Type", "Modèles", "Test set"],
                                "Valeur": [
                                    ens_dataset,
                                    ens_embedding,
                                    ens_type,
                                    ", ".join(ens_models),
                                    f"{len(X_test)} échantillons"
                                ]
                            })
                            st.dataframe(info_df, use_container_width=True, hide_index=True)
                            
                            # Comparaison avec modèles individuels
                            st.write("**Comparaison ensemble vs modèles individuels:**")
                            
                            individual_results = []
                            for name, model in base_models.items():
                                y_pred_ind = model.predict(X_test)
                                from sklearn.metrics import accuracy_score, precision_recall_fscore_support
                                
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
                            
                            # Sauvegarder les résultats
                            if "inference_results" not in st.session_state:
                                st.session_state.inference_results = []
                            
                            st.session_state.inference_results.append({
                                "dataset": ens_dataset,
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


# Onglet: Prédiction en temps réel

with tab_predict:
    st.subheader("Prédiction en temps réel")
    
    st.info("Entrez un texte pour prédire s'il contient du cyberbullying.")
    
    # Note sur le GPU
    try:
        import torch
        gpu_available = torch.cuda.is_available()
        if gpu_available:
            gpu_name = torch.cuda.get_device_name(0)
            st.success(f"GPU disponible: {gpu_name}")
        else:
            st.info("GPU non disponible - utilisation CPU (plus lent)")
            # Debug info
            with st.expander("Info GPU (diagnostic)"):
                st.write(f"- PyTorch version: {torch.__version__}")
                st.write(f"- CUDA available: {torch.cuda.is_available()}")
                st.write(f"- CUDA version: {torch.version.cuda}")
                st.write("Note: Si vous avez un GPU mais CUDA n'est pas detecte:")
                st.write("  1. Installez PyTorch avec CUDA: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
                st.write("  2. Verifiez les drivers NVIDIA: nvidia-smi")
    except ImportError:
        st.warning("PyTorch non installe. Installez avec: pip install torch")
    
    # Activer l'explicabilité des prédictions avec LIME
    use_explainability = st.checkbox(
        "Activer l'explainability (LIME)",
        value=False,
        help="Explique quels mots influencent la prédiction (nécessite LIME installé)"
    )
    
    if use_explainability and not LIME_AVAILABLE:
        st.warning("LIME n'est pas installé. Installez avec: pip install lime")
        use_explainability = False
    
    # Sélection du modèle
    col_pred1, col_pred2 = st.columns([2, 1])
    
    with col_pred1:
        pred_mode = st.radio(
            "Mode de prédiction",
            ["Modèles classiques (sklearn)", "Transformers fine-tunés"],
            horizontal=True
        )
    
    with col_pred2:
        confidence_threshold = st.slider("Seuil de confiance", 0.0, 1.0, 0.5, 0.05)
    
    # Zone de texte
    input_text = st.text_area(
        "Texte à analyser",
        height=150,
        placeholder="Entrez le texte à analyser...",
        help="Entrez le texte que vous souhaitez analyser"
    )
    
    st.divider()
    
    # === MODE 1: Modèles classiques ===
    if pred_mode == "Modèles classiques (sklearn)":
        processed_files = sorted([p for p in PROCESSED_DATA_DIR.glob("df_*.csv")])
        available_embeddings_raw = [p.stem.replace("df_", "") for p in processed_files]
        
        # Filtrer les embeddings supportes par embed_texts()
        # TF-IDF et BoW ne fonctionnent pas correctement sur un seul texte
        # Les embeddings denses (word2vec, glove, bert, roberta) sont recommandes
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
            
            if st.button("Lancer la prédiction", type="primary", disabled=not input_text.strip()):
                if not input_text.strip():
                    st.error("Veuillez entrer un texte.")
                else:
                    with st.spinner("Prédiction en cours..."):
                        try:
                            # Clé de cache pour le modèle entraîné
                            cache_key = f"model_{pred_model}_{pred_embedding}"
                            
                            # Vérifier si le modèle est déjà en cache
                            if cache_key in st.session_state:
                                model = st.session_state[cache_key]
                                st.info(f"Utilisation du modèle en cache: {pred_model} + {pred_embedding}")
                            else:
                                # Charger les données processed pour entraîner le modèle
                                with st.spinner(f"Entraînement du modèle {pred_model} avec {pred_embedding}..."):
                                    df_processed = load_processed_df(pred_embedding)
                                    if df_processed is None:
                                        st.error(f"Dataset processed pour {pred_embedding} introuvable.")
                                        st.stop()
                                    
                                    feature_cols = [c for c in df_processed.columns if c.startswith(f"text_{pred_embedding}")]
                                    if not feature_cols:
                                        st.error(f"Colonnes d'embedding {pred_embedding} introuvables.")
                                        st.stop()
                                    
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
                                        st.stop()
                                    
                                    model.fit(X, y)
                                    st.session_state[cache_key] = model
                                    st.success(f"Modèle {pred_model} entraîné avec {pred_embedding}")
                            
                            # Vectoriser le texte d'entrée
                            from cyberbullying.embedder import embed_texts
                            
                            # embed_texts retourne directement une matrice numpy
                            X_input = embed_texts([input_text], method=pred_embedding)
                            
                            # Prédiction
                            prediction = model.predict(X_input)[0]
                            
                            # Probabilités
                            if hasattr(model, "predict_proba"):
                                probas = model.predict_proba(X_input)[0]
                                prob_cyber = probas[1]
                            else:
                                prob_cyber = float(prediction)
                            
                            st.divider()
                            
                            # Résultat
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
                            
                            # Distribution de probabilité
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
                            
                            # LIME Explainability
                            if use_explainability and LIME_AVAILABLE:
                                st.divider()
                                st.write("### Explication de la prédiction (LIME)")
                                st.info("**Mots en rouge**: contribuent au cyberbullying | **Mots en vert**: contribuent à la normalité")
                                
                                try:
                                    # Utiliser LIME pour expliquer la prédiction
                                    # Note: Pour TF-IDF/BoW, LIME fonctionne directement sur le texte
                                    # Pour les embeddings denses, c'est plus complexe
                                    from lime.lime_text import LimeTextExplainer
                                    
                                    explainer = LimeTextExplainer(class_names=["Normal", "Cyberbullying"])
                                    
                                    # Fonction de prédiction pour LIME
                                    def predict_fn(texts):
                                        # Vectoriser tous les textes en une seule fois
                                        X_batch = embed_texts(texts, method=pred_embedding)
                                        
                                        if hasattr(model, "predict_proba"):
                                            return model.predict_proba(X_batch)
                                        else:
                                            predictions = model.predict(X_batch)
                                            # Convertir en probabilités binaires
                                            probas = np.zeros((len(predictions), 2))
                                            for i, pred in enumerate(predictions):
                                                probas[i] = [1-pred, pred]
                                            return probas
                                    
                                    # Générer l'explication
                                    exp = explainer.explain_instance(
                                        input_text, 
                                        predict_fn, 
                                        num_features=10,
                                        num_samples=100
                                    )
                                    
                                    # Afficher les mots les plus influents
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
                            
                        except Exception as e:
                            st.error(f"Erreur: {e}")
    
    # === MODE 2: Transformers ===
    else:
        if not TRANSFORMERS_AVAILABLE:
            st.error("Transformers non disponible. Installez avec: pip install transformers torch")
        else:
            # Lister tous les modèles disponibles (pré-entraînés + fine-tunés)
            pretrained_models = sorted([p for p in PRETRAINED_MODELS_DIR.glob("my_finetuned_*") if p.is_dir()])
            finetuned_models = sorted([p for p in TRAINED_MODELS_DIR.glob("finetuned_*") if p.is_dir()])
            
            all_models = []
            model_paths = {}
            
            # Ajouter les modèles pré-entraînés (BERT/RoBERTa)
            for model in pretrained_models:
                name = f"[PRÉ-ENTRAÎNÉ] {model.name}"
                all_models.append(name)
                model_paths[name] = model
            
            # Ajouter les modèles fine-tunés personnalisés
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
                                # Récupérer le chemin du modèle sélectionné
                                model_path = model_paths[selected_model_name]
                                
                                # Afficher le type de modèle utilisé
                                if "[PRÉ-ENTRAÎNÉ]" in selected_model_name:
                                    st.info(f"Utilisation du modèle pré-entraîné: {model_path.name}")
                                else:
                                    st.info(f"Utilisation du modèle fine-tuné: {model_path.name}")
                                
                                # Prédiction avec le modèle Transformer
                                predictions, probabilities = predict_with_finetuned(
                                    model_path,
                                    [input_text],
                                    batch_size=1,
                                    max_length=128
                                )
                                
                                prediction = predictions[0]
                                prob_cyber = probabilities[0][1]
                                
                                st.divider()
                                
                                # Résultat
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
                                
                                # Graphique
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
                                
                            except Exception as e:
                                st.error(f"Erreur: {e}")
                                import traceback
                                st.code(traceback.format_exc())


# Onglet: Exploitation des résultats

with tab_results:
    st.subheader("Analyse des rapports et métriques")
    
    # Onglets pour séparer les résultats sauvegardés et les tests en session
    tab_saved, tab_session = st.tabs(["Résultats sauvegardés", "Tests de session"])
    
    # --- Résultats sauvegardés ---
    with tab_saved:
        st.write("**Analyse des rapports de classification sauvegardés**")
        
        report_paths = []
        report_paths += list(REPORTS_DIR.glob("*_report.csv"))
        report_paths += list((EXPERIMENTS_DIR / "results").glob("*_report.csv"))

        if not report_paths:
            st.warning("Aucun rapport trouvé dans outputs/reports ou outputs/experiments/results.")
        else:
            rows = []
            for path in report_paths:
                parsed = parse_report_file(path)
                if parsed:
                    rows.append(parsed)

            df_reports = pd.DataFrame(rows)
            
            # Filtres
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
                
                # Graphiques améliorés
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

                # Heatmaps pour toutes les métriques
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
                
                # Comparaison multi-métriques
                st.write("**Comparaison multi-métriques**")
                metrics_long = df_filtered.melt(
                    id_vars=['model', 'embedding'],
                    value_vars=['accuracy', 'precision', 'recall', 'f1'],
                    var_name='Métrique',
                    value_name='Score'
                )
                
                fig_multi = px.box(
                    metrics_long,
                    x='Métrique',
                    y='Score',
                    color='embedding',
                    title='Distribution des métriques par embedding',
                    labels={'Score': 'Score', 'Métrique': 'Métrique'}
                )
                st.plotly_chart(fig_multi, use_container_width=True)
    
    # --- Tests de session ---
    with tab_session:
        st.write("**Résultats des tests d'inférence de cette session**")
        
        if "inference_results" not in st.session_state or not st.session_state.inference_results:
            st.info("Aucun test d'inférence effectué dans cette session. Allez dans l'onglet 'Inférence & Tests' pour lancer des tests.")
        else:
            df_session = pd.DataFrame(st.session_state.inference_results)
            
            st.write(f"**{len(df_session)} test(s) effectué(s)**")
            st.dataframe(df_session, use_container_width=True, hide_index=True)
            
            st.divider()
            
            # Visualisations des tests de session
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
            
            # Tableau récapitulatif par dataset
            st.write("**Récapitulatif par dataset**")
            summary_by_dataset = df_session.groupby('dataset').agg({
                'accuracy': ['mean', 'max', 'min'],
                'f1': ['mean', 'max', 'min'],
                'samples': 'first'
            }).round(3)
            st.dataframe(summary_by_dataset, use_container_width=True)
            
            # Option pour effacer les résultats
            if st.button("Effacer les résultats de session"):
                st.session_state.inference_results = []
                st.rerun()


# Onglet: Paramètres et configuration

with tab_settings:
    st.subheader("Hyperparamètres des classifieurs")
    
    # Sous-onglets
    tab_manual, tab_auto = st.tabs(["Configuration manuelle", "Optimisation automatique"])
    
    # === Configuration manuelle ===
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
    
    # === Optimisation automatique ===
    with tab_auto:
        st.write("**Optimisation automatique d'hyperparamètres avec Optuna**")
        
        if not OPTUNA_AVAILABLE:
            st.error("Optuna n'est pas installé. Installez avec: pip install optuna")
        else:
            st.info("Optuna va automatiquement rechercher les meilleurs hyperparamètres pour maximiser le F1-Score.")
            
            col_opt1, col_opt2 = st.columns(2)
            
            with col_opt1:
                # Sélection de l'embedding
                processed_files = sorted([p for p in PROCESSED_DATA_DIR.glob("df_*.csv")])
                available_embeddings = [p.stem.replace("df_", "") for p in processed_files]
                
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
                        # Charger les données
                        with st.spinner("Chargement des données..."):
                            df = load_processed_df(opt_embedding)
                            if df is None:
                                st.error("Dataset introuvable.")
                                st.stop()
                            
                            feature_cols = [c for c in df.columns if c.startswith(f"text_{opt_embedding}")]
                            if not feature_cols:
                                st.error("Colonnes d'embedding introuvables.")
                                st.stop()
                            
                            # Échantillonnage si nécessaire
                            if sample_limit_opt > 0 and len(df) > sample_limit_opt:
                                df_sample = df.sample(n=sample_limit_opt, random_state=42)
                            else:
                                df_sample = df
                            
                            X = df_sample[feature_cols]
                            y = df_sample["type"]
                            
                            st.success(f"Données chargées: {len(X)} échantillons, {len(feature_cols)} features")
                        
                        # Optimisation des hyperparamètres
                        st.info(f"Recherche des meilleurs hyperparamètres pour {opt_model}...")
                        progress_placeholder = st.empty()
                        
                        with st.spinner(f"Optimisation en cours ({n_trials} essais avec {cv_folds}-Fold CV)..."):
                            result = auto_optimize_model(
                                opt_model,
                                X,
                                y,
                                n_trials=int(n_trials),
                                cv=int(cv_folds)
                            )
                        
                        # Résultats
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
                            
                            # Historique
                            history_df = get_optimization_history(result["study"])
                            st.write(f"**Historique** ({len(history_df)} essais):")
                            st.dataframe(history_df.head(10), use_container_width=True)
                        
                        # Graphique d'optimisation
                        st.write("**Évolution de l'optimisation:**")
                        history_df_clean = history_df[history_df["state"] == "COMPLETE"].copy()
                        
                        fig_opt = px.line(
                            history_df_clean,
                            x="trial",
                            y="value",
                            title="Évolution du F1-Score par essai",
                            labels={"trial": "Essai", "value": "F1-Score (CV)"}
                        )
                        
                        # Ajouter la meilleure valeur
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
