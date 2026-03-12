"""SafetyAI - Application Streamlit de detection de cyberbullying."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cyberbullying.config import RAW_DATA_DIR, TRAINED_MODELS_DIR
from app.ui.tab_datasets import render as render_datasets
from app.ui.tab_embeddings import render as render_embeddings
from app.ui.tab_finetune import render as render_finetune
from app.ui.tab_inference import render as render_inference
from app.ui.tab_predict import render as render_predict
from app.ui.tab_results import render as render_results
from app.ui.tab_mlflow import render as render_mlflow
from app.ui.tab_settings import render as render_settings
from app.ui.tab_operations import render as render_operations

# Auth
AUTH_DISABLED = os.environ.get("AUTH_DISABLED", "").lower() in ("1", "true", "yes")
if not AUTH_DISABLED:
    try:
        from auth.manager import authenticate, load_users, make_captcha, check_captcha
        _auth_available = True
    except ImportError:
        _auth_available = False
else:
    _auth_available = False

st.set_page_config(
    page_title="SafetyAI",
    page_icon="shield",
    layout="wide",
)

if _auth_available and not AUTH_DISABLED:
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "user_role" not in st.session_state:
        st.session_state.user_role = None

    if not st.session_state.authenticated:
        st.title("Connexion - SafetyAI")
        cfg = load_users()
        bot_cfg = cfg.get("bot_check", {})
        captcha_required = bot_cfg.get("captcha_required", False)

        if captcha_required and "_captcha_expected" not in st.session_state:
            a, b, question = make_captcha()
            st.session_state._captcha_expected = a + b
            st.session_state._captcha_question = question

        with st.form("login_form"):
            username = st.text_input("Identifiant")
            password = st.text_input("Mot de passe", type="password")
            captcha_ok = True
            if captcha_required:
                answer = st.number_input(
                    st.session_state.get("_captcha_question", "Captcha"),
                    min_value=0, value=0, step=1
                )
                captcha_ok = check_captcha(st.session_state.get("_captcha_expected"), answer)
            submitted = st.form_submit_button("Se connecter")

        if submitted:
            ok, msg = authenticate(username, password, captcha_ok=captcha_ok)
            if ok:
                st.session_state.authenticated = True
                st.session_state.user_role = msg
                for k in ("_captcha_expected", "_captcha_question"):
                    st.session_state.pop(k, None)
                st.success("Connexion réussie.")
                st.rerun()
            else:
                st.error(msg)
                for k in ("_captcha_expected", "_captcha_question"):
                    st.session_state.pop(k, None)
        st.stop()

st.title("SafetyAI")

with st.sidebar:
    if _auth_available and st.session_state.get("authenticated"):
        st.caption(f"Connecte ({st.session_state.get('user_role', 'user')})")
        if st.button("Deconnexion"):
            st.session_state.authenticated = False
            st.session_state.user_role = None
            st.rerun()

tab_data, tab_embeddings, tab_operations, tab_finetune, tab_inference, tab_predict, tab_results, tab_mlflow, tab_settings = st.tabs([
    "Datasets", "Embeddings", "Operations", "Fine-tuning", "Inference", "Prediction", "Resultats", "MLflow", "Parametres"
])

render_datasets(tab_data)
render_embeddings(tab_embeddings)
render_operations(tab_operations)
render_finetune(tab_finetune)
render_inference(tab_inference)
render_predict(tab_predict)
render_results(tab_results)
render_mlflow(tab_mlflow)
render_settings(tab_settings)
