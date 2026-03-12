"""Operations tab - pipeline step by step."""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import streamlit as st
import yaml

from .commons import data_file_label, list_data_files

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

EMBEDDINGS = ["tfidf", "bow", "word2vec", "glove", "bert", "roberta"]
MODEL_KEYS = ["logistic_regression", "random_forest", "svm", "lightgbm", "mlp"]

# IDs des datasets pour le téléchargement (correspondent à download_research_datasets.py)
DATASET_IDS = ["cb1", "cb2", "sarcasm", "wiki_toxic", "hatexplain"]


def _run_script(script_name: str, args: list[str] | None = None) -> tuple[int, str]:
    """Execute a script with proper PYTHONPATH. Returns (exit_code, output)."""
    env = os.environ.copy()
    pp = os.pathsep.join([
        str(PROJECT_ROOT),
        str(PROJECT_ROOT / "src"),
        str(PROJECT_ROOT / "configs"),
        str(PROJECT_ROOT / "scripts"),
    ])
    env["PYTHONPATH"] = f"{pp}{os.pathsep}{env.get('PYTHONPATH', '')}"
    cmd = [sys.executable, str(SCRIPTS_DIR / script_name)] + (args or [])
    try:
        result = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=3600,
        )
        out = (result.stdout or "") + (result.stderr or "")
        return result.returncode, out.strip() or "(aucune sortie)"
    except subprocess.TimeoutExpired:
        return -1, "Timeout (1h depassee)."
    except Exception as e:
        return -1, str(e)


def _build_override(lr_c, lr_max_iter, rf_n, rf_depth, svm_c, svm_kernel, lgb_n, lgb_lr, mlp_hidden, mlp_iter) -> str | None:
    """Construit un fichier YAML temporaire pour les hyperparametres."""
    try:
        sizes = tuple(int(x.strip()) for x in mlp_hidden.split(",") if x.strip())
    except ValueError:
        sizes = (128, 64)
    override = {
        "models": {
            "logistic_regression": {"C": lr_c, "max_iter": int(lr_max_iter)},
            "random_forest": {"n_estimators": rf_n, "max_depth": None if rf_depth == 0 else rf_depth},
            "svm": {"C": svm_c, "kernel": svm_kernel},
            "lightgbm": {"n_estimators": lgb_n, "learning_rate": lgb_lr},
            "mlp": {"hidden_layer_sizes": list(sizes), "max_iter": int(mlp_iter)},
        }
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(override, f, default_flow_style=False, allow_unicode=True)
        return f.name


def _hyperparams_expander(key_prefix: str):
    """Affiche les champs hyperparametres dans un expander."""
    with st.expander("Hyperparametres modeles (optionnel)", expanded=False):
        col_lr, col_rf = st.columns(2)
        with col_lr:
            lr_c = st.number_input("LR C (regularisation)", 0.01, 10.0, 1.0, 0.1, key=f"{key_prefix}_lr_c")
            lr_max_iter = st.number_input("LR max_iter", 100, 5000, 1000, 100, key=f"{key_prefix}_lr_iter")
        with col_rf:
            rf_n = st.number_input("RF n_estimators", 50, 500, 200, 50, key=f"{key_prefix}_rf_n")
            rf_depth = st.number_input("RF max_depth (0=illimite)", 0, 50, 0, 1, key=f"{key_prefix}_rf_d")
        col_svm, col_lgb = st.columns(2)
        with col_svm:
            svm_c = st.number_input("SVM C", 0.01, 10.0, 1.0, 0.1, key=f"{key_prefix}_svm_c")
            svm_kernel = st.selectbox("SVM kernel", ["rbf", "linear", "poly"], key=f"{key_prefix}_svm_k")
        with col_lgb:
            lgb_n = st.number_input("LGB n_estimators", 50, 500, 300, 50, key=f"{key_prefix}_lgb_n")
            lgb_lr = st.number_input("LGB learning_rate", 0.01, 0.5, 0.05, 0.01, key=f"{key_prefix}_lgb_lr")
        mlp_hidden = st.text_input("MLP hidden_layer_sizes", "128,64", key=f"{key_prefix}_mlp_h")
        mlp_iter = st.number_input("MLP max_iter", 100, 1000, 300, 50, key=f"{key_prefix}_mlp_iter")
    return lr_c, lr_max_iter, rf_n, rf_depth, svm_c, svm_kernel, lgb_n, lgb_lr, mlp_hidden, mlp_iter


def render(tab):
    with tab:
        st.subheader("Operations pipeline")

        st.info(
            "**Operations** = pipeline qui cree les donnees et modeles. "
            "Entree : CSV bruts. Sortie : df_*.csv, rapports, modeles. "
            "Chaque etape a ses propres parametres. Selectionnez les donnees avant de lancer."
        )

        files = list_data_files()
        file_labels = [data_file_label(p) for p in files] if files else []

        # ========== Section Donnees (pour etapes 2, 3, 4, pipeline) ==========
        st.divider()
        st.write("### Donnees")

        csv_selected = st.multiselect(
            "Fichiers CSV a utiliser",
            file_labels,
            default=file_labels[:2] if len(file_labels) >= 2 else file_labels,
            help="Utilise pour doublons, embed, train et pipeline. Vide = tous les fichiers par defaut.",
            key="op_csv",
        )

        data_files_arg = ["--data-files"] + csv_selected if csv_selected else []

        # ========== Etape 1 : Telecharger ==========
        with st.expander("Etape 1 : Telecharger datasets", expanded=True):
            st.caption("Telecharge les datasets selectionnes depuis HuggingFace.")
            datasets_to_download = st.multiselect(
                "Datasets a telecharger",
                DATASET_IDS,
                default=DATASET_IDS[:3],
                help="cb1, cb2, sarcasm, wiki_toxic, hatexplain.",
                key="op_datasets_dl",
            )
            if st.button("Telecharger datasets selectionnes", key="op_dl"):
                if not datasets_to_download:
                    st.warning("Selectionnez au moins un dataset a telecharger.")
                else:
                    with st.spinner("Telechargement..."):
                        args_dl = ["--datasets"] + datasets_to_download
                        code, out = _run_script("download_research_datasets.py", args_dl)
                        st.code(out, language="text")
                        st.success("Termine.") if code == 0 else st.error(f"Erreur (code {code})")

        # ========== Etape 2 : Verifier doublons ==========
        with st.expander("Etape 2 : Verifier doublons", expanded=True):
            st.caption("Analyse les doublons (texte identique) et l'overlap train/test sur les CSV selectionnes.")
            if st.button("Verifier doublons sur les CSV selectionnes", key="op_dup"):
                with st.spinner("Analyse..."):
                    code, out = _run_script("check_duplicates.py", data_files_arg if csv_selected else [])
                    st.code(out, language="text")
                    st.success("Termine.") if code == 0 else st.error(f"Erreur (code {code})")

        # ========== Etape 3 : Embed ==========
        with st.expander("Etape 3 : Embed", expanded=True):
            st.caption("Charge les CSV selectionnes, applique l'embedding, sauvegarde df_*.csv dans data/processed/.")
            emb_embedding = st.selectbox("Embedding", EMBEDDINGS, key="op_emb")
            emb_n = st.slider(
                "Nombre d'echantillons par fichier",
                500, 50000, 2000, 500,
                help="Limite le nombre de lignes chargees par CSV.",
                key="op_emb_n",
            )
            emb_mf = st.slider(
                "Max features (TF-IDF / BoW)",
                100, 10000, 5000, 500,
                help="Nombre max de features pour TF-IDF et BoW.",
                key="op_emb_mf",
            )
            if st.button("Lancer embed uniquement", key="op_embed_btn"):
                args = ["--stage", "embed", "--embedding", emb_embedding, "-n", str(emb_n),
                        "--max-features", str(emb_mf)] + data_files_arg
                with st.spinner("Embed en cours..."):
                    code, out = _run_script("run_pipeline.py", args)
                    st.code(out, language="text")
                    st.success("Termine.") if code == 0 else st.error(f"Erreur (code {code})")

        # ========== Etape 4 : Train ==========
        with st.expander("Etape 4 : Train", expanded=True):
            st.caption("Entraine les modeles sklearn sur les df_*.csv deja crees par l'etape Embed.")
            train_embedding = st.selectbox("Embedding a utiliser", EMBEDDINGS, key="op_train_emb")
            train_n = st.slider("Nombre d'echantillons", 500, 50000, 2000, 500, key="op_train_n")
            train_mf = st.slider("Max features", 100, 10000, 5000, 500, key="op_train_mf")
            train_test_size = st.slider("Proportion test (%)", 10, 50, 20, 5, key="op_train_ts") / 100
            train_seed = st.number_input("Seed aleatoire", 0, 99999, 42, key="op_train_seed")
            train_save = st.selectbox(
                "Modele a entrainer et deployer",
                ["Aucun"] + MODEL_KEYS,
                help="Modele a entrainer et exporter vers models/trained/pipeline_deploy/. Aucun = entrainer logistic_regression et random_forest sans deploiement.",
                key="op_train_save",
            )

            hp = _hyperparams_expander("op_train")

            if st.button("Lancer train uniquement", key="op_train_btn"):
                override_path = _build_override(*hp) if hp else None
                try:
                    train_models = [train_save] if train_save != "Aucun" else ["logistic_regression", "random_forest"]
                    args = ["--stage", "train", "--embedding", train_embedding, "-n", str(train_n),
                            "--max-features", str(train_mf), "--test-size", str(train_test_size),
                            "--seed", str(train_seed), "--models", *train_models]
                    if train_save != "Aucun":
                        args.extend(["--save-model", train_save])
                    if override_path:
                        args.extend(["--override", override_path])
                    with st.spinner("Train en cours..."):
                        code, out = _run_script("run_pipeline.py", args)
                        st.code(out, language="text")
                        st.success("Termine.") if code == 0 else st.error(f"Erreur (code {code})")
                finally:
                    if override_path:
                        Path(override_path).unlink(missing_ok=True)

        # ========== Pipeline complet ==========
        with st.expander("Pipeline complet (load + embed + train)", expanded=True):
            st.caption("Lance toutes les etapes en une fois : chargement CSV, embedding, entrainement.")
            pipe_embedding = st.selectbox("Embedding", EMBEDDINGS, key="op_pipe_emb")
            pipe_n = st.slider("Nombre d'echantillons", 500, 50000, 2000, 500, key="op_pipe_n")
            pipe_mf = st.slider("Max features", 100, 10000, 5000, 500, key="op_pipe_mf")
            pipe_test_size = st.slider("Proportion test (%)", 10, 50, 20, 5, key="op_pipe_ts") / 100
            pipe_seed = st.number_input("Seed aleatoire", 0, 99999, 42, key="op_pipe_seed")
            pipe_save = st.selectbox(
                "Modele a entrainer et deployer",
                ["Aucun"] + MODEL_KEYS,
                help="Modele a entrainer et exporter vers models/trained/pipeline_deploy/. Aucun = entrainer logistic_regression et random_forest sans deploiement.",
                key="op_pipe_save",
            )

            hp_pipe = _hyperparams_expander("op_pipe")

            if st.button("Lancer pipeline complet", type="primary", key="op_full"):
                override_path = _build_override(*hp_pipe) if hp_pipe else None
                try:
                    pipe_models = [pipe_save] if pipe_save != "Aucun" else ["logistic_regression", "random_forest"]
                    args = ["--stage", "all", "--embedding", pipe_embedding, "-n", str(pipe_n),
                            "--max-features", str(pipe_mf), "--test-size", str(pipe_test_size),
                            "--seed", str(pipe_seed), "--models", *pipe_models] + data_files_arg
                    if pipe_save != "Aucun":
                        args.extend(["--save-model", pipe_save])
                    if override_path:
                        args.extend(["--override", override_path])
                    with st.spinner("Pipeline complet en cours..."):
                        code, out = _run_script("run_pipeline.py", args)
                        st.code(out, language="text")
                        st.success("Termine.") if code == 0 else st.error(f"Erreur (code {code})")
                finally:
                    if override_path:
                        Path(override_path).unlink(missing_ok=True)

        # ========== Agreger rapports ==========
        with st.expander("Agreger rapports", expanded=False):
            st.caption("Consolide les rapports et genere les graphiques.")
            per_emb = st.checkbox("Graphique par embedding", False, key="op_per_emb")
            if st.button("Agreger rapports", key="op_agg"):
                args = ["--per-embedding"] if per_emb else []
                with st.spinner("Agregation..."):
                    code, out = _run_script("aggregate_results.py", args)
                    st.code(out, language="text")
                    st.success("Termine.") if code == 0 else st.error(f"Erreur (code {code})")
