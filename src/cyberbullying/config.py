"""Configuration des chemins et constantes du projet."""
from pathlib import Path

ROOT_DIR: Path = Path(__file__).resolve().parents[2]
DATA_DIR: Path = ROOT_DIR / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
EMBEDDING_CACHE_DIR: Path = PROCESSED_DATA_DIR / "embedding_cache"
MODELS_DIR: Path = ROOT_DIR / "models"
TRAINED_MODELS_DIR: Path = MODELS_DIR / "trained"
PRETRAINED_MODELS_DIR: Path = MODELS_DIR / "pretrained"
OUTPUTS_DIR: Path = ROOT_DIR / "outputs"
REPORTS_DIR: Path = OUTPUTS_DIR / "reports"
ANALYSIS_DIR: Path = OUTPUTS_DIR / "analysis"
EXPERIMENTS_DIR: Path = OUTPUTS_DIR / "experiments"
MLFLOW_DIR: Path = OUTPUTS_DIR / "mlruns"


def get_mlflow_tracking_uri() -> str:
    """URI MLflow pour tracking local (file://)."""
    return MLFLOW_DIR.resolve().as_uri()

# =============================================================================
# DATASETS
# =============================================================================
# Fichiers produits par: python run.py research-download
# Format attendu: colonnes [text, type] (type 0/1 binaire)
# Structure simplifiée: research/ pour CB1 et sarcasm (train/test), plats pour le reste
RESEARCH_CB1_DIR: Path = RAW_DATA_DIR / "research" / "cyberbullying_cb1"
RESEARCH_SARCASM_DIR: Path = RAW_DATA_DIR / "research" / "sarcasm_twitter"

DATA_PATHS: list[Path] = [
    RESEARCH_CB1_DIR / "train.csv",
    RAW_DATA_DIR / "cb2.csv",
    RESEARCH_SARCASM_DIR / "train.csv",
    RAW_DATA_DIR / "wiki_toxic.csv",
    RAW_DATA_DIR / "hatexplain.csv",
]

# Dataset unifié (fusion de tous les datasets téléchargés)
UNIFIED_TRAIN_PATH: Path = RAW_DATA_DIR / "unified" / "train.csv"

PRETRAINED_BERT_DIR: Path = PRETRAINED_MODELS_DIR / "my_finetuned_bert"
PRETRAINED_ROBERTA_DIR: Path = PRETRAINED_MODELS_DIR / "my_finetuned_roberta"
