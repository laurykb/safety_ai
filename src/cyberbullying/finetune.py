"""Module pour le fine-tuning des modèles Transformers (BERT/RoBERTa)."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

warnings.filterwarnings("ignore")


class CyberbullyingDataset(Dataset):
    """Dataset PyTorch pour le fine-tuning."""

    def __init__(self, texts: list[str], labels: list[int], tokenizer, max_length: int = 128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long),
        }


def compute_metrics(eval_pred):
    """Calcule les métriques pour l'évaluation."""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="binary", zero_division=0
    )
    acc = accuracy_score(labels, predictions)

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def prepare_datasets(
    df: pd.DataFrame,
    text_column: str,
    label_column: str,
    tokenizer,
    max_length: int = 128,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[CyberbullyingDataset, CyberbullyingDataset]:
    """Prépare les datasets d'entraînement et de validation."""
    
    texts = df[text_column].astype(str).tolist()
    labels = df[label_column].astype(int).tolist()

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=test_size, random_state=random_state, stratify=labels
    )

    train_dataset = CyberbullyingDataset(train_texts, train_labels, tokenizer, max_length)
    val_dataset = CyberbullyingDataset(val_texts, val_labels, tokenizer, max_length)

    return train_dataset, val_dataset


def finetune_transformer(
    model_name: str,
    train_dataset: Dataset,
    val_dataset: Dataset,
    output_dir: Path,
    num_epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    warmup_steps: int = 100,
    weight_decay: float = 0.01,
    save_steps: int = 500,
    logging_steps: int = 100,
    eval_steps: int = 500,
) -> tuple[Trainer, dict[str, Any]]:
    """Fine-tune un modèle Transformer."""

    # Charger le modèle
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        ignore_mismatched_sizes=True,
    )

    # Configuration de l'entraînement
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        logging_dir=str(output_dir / "logs"),
        logging_steps=logging_steps,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_steps=save_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        report_to="none",
        seed=42,
    )

    # Créer le Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    # Entraîner le modèle
    trainer.train()

    # Évaluation finale
    final_metrics = trainer.evaluate()

    return trainer, final_metrics


def predict_with_finetuned(
    model_path: Path,
    texts: list[str],
    batch_size: int = 32,
    max_length: int = 128,
) -> tuple[np.ndarray, np.ndarray]:
    """Prédictions avec un modèle fine-tuné."""
    
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_path))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    all_predictions = []
    all_probabilities = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]

        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)

        predictions = torch.argmax(logits, dim=1).cpu().numpy()
        probabilities = probs.cpu().numpy()

        all_predictions.extend(predictions)
        all_probabilities.extend(probabilities)

    return np.array(all_predictions), np.array(all_probabilities)
