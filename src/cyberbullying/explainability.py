"""Module pour l'explainability des modèles (LIME et SHAP)."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    import lime
    import lime.lime_text
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


def explain_with_lime(
    text: str,
    predict_fn: callable,
    class_names: list[str] = None,
    num_features: int = 10,
    num_samples: int = 1000,
) -> dict[str, Any]:
    """
    Explique une prédiction avec LIME.
    
    Args:
        text: Texte à expliquer
        predict_fn: Fonction de prédiction (doit retourner des probabilités)
        class_names: Noms des classes
        num_features: Nombre de features à afficher
        num_samples: Nombre d'échantillons pour LIME
    
    Returns:
        Dict contenant l'explication et les visualisations
    """
    if not LIME_AVAILABLE:
        raise ImportError("LIME n'est pas installé. Installez avec: pip install lime")
    
    if class_names is None:
        class_names = ["Normal", "Cyberbullying"]
    
    # Créer l'explainer
    explainer = lime.lime_text.LimeTextExplainer(
        class_names=class_names,
        split_expression=r'\W+',
        random_state=42
    )
    
    # Générer l'explication
    explanation = explainer.explain_instance(
        text,
        predict_fn,
        num_features=num_features,
        num_samples=num_samples
    )
    
    # Extraire les informations
    result = {
        "explanation": explanation,
        "words": [],
        "weights": [],
        "prediction": explanation.predict_proba.argmax(),
        "confidence": explanation.predict_proba.max(),
    }
    
    # Pour la classe prédite
    for word, weight in explanation.as_list():
        result["words"].append(word)
        result["weights"].append(weight)
    
    return result


def explain_with_shap(
    texts: list[str],
    model,
    vectorizer=None,
    max_display: int = 20,
) -> dict[str, Any]:
    """
    Explique des prédictions avec SHAP.
    
    Args:
        texts: Liste de textes à expliquer
        model: Modèle sklearn
        vectorizer: Vectorizer (TF-IDF, etc.)
        max_display: Nombre de features à afficher
    
    Returns:
        Dict contenant les valeurs SHAP et les visualisations
    """
    if not SHAP_AVAILABLE:
        raise ImportError("SHAP n'est pas installé. Installez avec: pip install shap")
    
    # Vectoriser les textes si nécessaire
    if vectorizer is not None:
        X = vectorizer.transform(texts)
    else:
        X = texts
    
    # Créer l'explainer
    if hasattr(model, "predict_proba"):
        explainer = shap.Explainer(model.predict_proba, X)
    else:
        explainer = shap.Explainer(model.predict, X)
    
    # Calculer les valeurs SHAP
    shap_values = explainer(X)
    
    result = {
        "shap_values": shap_values,
        "explainer": explainer,
        "feature_names": vectorizer.get_feature_names_out() if vectorizer else None,
    }
    
    return result


def get_top_features_shap(
    shap_values,
    feature_names: list[str],
    top_k: int = 10,
    class_idx: int = 1,
) -> pd.DataFrame:
    """Retourne les top features selon SHAP."""
    
    if hasattr(shap_values, "values"):
        values = shap_values.values
    else:
        values = shap_values
    
    # Si c'est multi-class, prendre la classe spécifiée
    if len(values.shape) == 3:
        values = values[:, :, class_idx]
    
    # Moyenne absolue des valeurs SHAP
    mean_abs_shap = np.abs(values).mean(axis=0)
    
    # Top features
    top_indices = np.argsort(mean_abs_shap)[-top_k:][::-1]
    
    df = pd.DataFrame({
        "Feature": [feature_names[i] if feature_names else f"Feature_{i}" for i in top_indices],
        "Mean |SHAP|": mean_abs_shap[top_indices],
    })
    
    return df


def format_lime_explanation(explanation_dict: dict[str, Any]) -> str:
    """Formate une explication LIME en texte lisible."""
    
    lines = []
    lines.append("=== Explication LIME ===")
    lines.append(f"Prédiction: {explanation_dict['prediction']}")
    lines.append(f"Confiance: {explanation_dict['confidence']:.2%}")
    lines.append("\nMots influents:")
    
    for word, weight in zip(explanation_dict['words'], explanation_dict['weights']):
        direction = "➕" if weight > 0 else "➖"
        lines.append(f"  {direction} {word}: {weight:.3f}")
    
    return "\n".join(lines)


def highlight_text_lime(text: str, explanation_dict: dict[str, Any]) -> str:
    """
    Génère du HTML avec les mots importants surlignés.
    
    Vert = contribue à la classe Normal
    Rouge = contribue à la classe Cyberbullying
    """
    words = text.split()
    word_dict = dict(zip(explanation_dict['words'], explanation_dict['weights']))
    
    html_parts = []
    for word in words:
        clean_word = word.lower().strip('.,!?;:()[]{}"\'-')
        
        if clean_word in word_dict:
            weight = word_dict[clean_word]
            if weight > 0:
                # Contribue au cyberbullying (rouge)
                intensity = min(255, int(abs(weight) * 500))
                html_parts.append(f'<span style="background-color: rgba(255, 0, 0, {intensity/255:.2f}); padding: 2px 4px; border-radius: 3px;">{word}</span>')
            else:
                # Contribue au normal (vert)
                intensity = min(255, int(abs(weight) * 500))
                html_parts.append(f'<span style="background-color: rgba(0, 255, 0, {intensity/255:.2f}); padding: 2px 4px; border-radius: 3px;">{word}</span>')
        else:
            html_parts.append(word)
    
    return " ".join(html_parts)
