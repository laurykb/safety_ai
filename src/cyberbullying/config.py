"""Configuration des chemins et constantes du projet."""
from pathlib import Path

# =============================================================================
# PATHS
# =============================================================================

ROOT_DIR: Path = Path(__file__).resolve().parents[2]
DATA_DIR: Path = ROOT_DIR / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
MODELS_DIR: Path = ROOT_DIR / "models"
TRAINED_MODELS_DIR: Path = MODELS_DIR / "trained"
PRETRAINED_MODELS_DIR: Path = MODELS_DIR / "pretrained"
OUTPUTS_DIR: Path = ROOT_DIR / "outputs"
REPORTS_DIR: Path = OUTPUTS_DIR / "reports"
ERROR_REPORTS_DIR: Path = OUTPUTS_DIR / "error_reports"
ERROR_ANALYSIS_DIR: Path = OUTPUTS_DIR / "error_analysis"
ANALYSIS_DIR: Path = OUTPUTS_DIR / "analysis"
ANALYSIS_BY_EMBEDDING_DIR: Path = OUTPUTS_DIR / "analysis_by_embedding"
ANALYSIS_PROJECT_DIR: Path = OUTPUTS_DIR / "analysis_project"
EXPERIMENTS_DIR: Path = OUTPUTS_DIR / "experiments"
CHECKPOINTS_DIR: Path = OUTPUTS_DIR / "checkpoints"
WANDB_DIR: Path = OUTPUTS_DIR / "wandb"

# =============================================================================
# DATASETS
# =============================================================================

DATASET_FILES: list[str] = [
    "cyberbullying_tweets.csv",
    "toxicity_parsed_dataset.csv",
    "aggression_parsed_dataset.csv",
]

DATA_PATHS: list[Path] = [RAW_DATA_DIR / f for f in DATASET_FILES]

PRETRAINED_BERT_DIR: Path = PRETRAINED_MODELS_DIR / "my_finetuned_bert"
PRETRAINED_ROBERTA_DIR: Path = PRETRAINED_MODELS_DIR / "my_finetuned_roberta"
