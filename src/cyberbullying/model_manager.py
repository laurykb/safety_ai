"""Module pour la sauvegarde et le chargement de modèles."""

from __future__ import annotations

import json
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

warnings.filterwarnings("ignore")


class ModelRegistry:
    """Registry pour gérer les modèles entraînés."""
    
    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.registry_file = self.base_dir / "model_registry.json"
        self.registry = self._load_registry()
    
    def _load_registry(self) -> dict:
        """Charge le registry depuis le fichier JSON."""
        if self.registry_file.exists():
            with open(self.registry_file, "r") as f:
                return json.load(f)
        return {}
    
    def _save_registry(self):
        """Sauvegarde le registry dans le fichier JSON."""
        with open(self.registry_file, "w") as f:
            json.dump(self.registry, f, indent=2)
    
    def save_model(
        self,
        model: Any,
        name: str,
        metadata: dict[str, Any] = None,
        vectorizer: Any = None,
    ) -> Path:
        """
        Sauvegarde un modèle avec ses métadonnées.
        
        Args:
            model: Modèle sklearn à sauvegarder
            name: Nom du modèle
            metadata: Métadonnées (embedding, dataset, metrics, etc.)
            vectorizer: Vectorizer optionnel (pour TF-IDF, BoW, etc.)
        
        Returns:
            Path du modèle sauvegardé
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_id = f"{name}_{timestamp}"
        model_dir = self.base_dir / model_id
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Sauvegarder le modèle
        model_path = model_dir / "model.pkl"
        joblib.dump(model, model_path)
        
        # Sauvegarder le vectorizer si fourni
        if vectorizer is not None:
            vectorizer_path = model_dir / "vectorizer.pkl"
            joblib.dump(vectorizer, vectorizer_path)
        
        # Créer les métadonnées
        if metadata is None:
            metadata = {}
        
        metadata.update({
            "model_id": model_id,
            "name": name,
            "timestamp": timestamp,
            "model_path": str(model_path),
            "vectorizer_path": str(model_dir / "vectorizer.pkl") if vectorizer else None,
        })
        
        # Sauvegarder les métadonnées
        metadata_path = model_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        
        # Enregistrer dans le registry
        self.registry[model_id] = metadata
        self._save_registry()
        
        return model_path
    
    def load_model(self, model_id: str) -> dict[str, Any]:
        """
        Charge un modèle et ses composants.
        
        Args:
            model_id: ID du modèle à charger
        
        Returns:
            Dict contenant model, vectorizer (si existe), metadata
        """
        if model_id not in self.registry:
            raise ValueError(f"Modèle {model_id} introuvable dans le registry.")
        
        metadata = self.registry[model_id]
        model_path = Path(metadata["model_path"])
        
        # Charger le modèle
        model = joblib.load(model_path)
        
        # Charger le vectorizer si existe
        vectorizer = None
        if metadata.get("vectorizer_path"):
            vectorizer_path = Path(metadata["vectorizer_path"])
            if vectorizer_path.exists():
                vectorizer = joblib.load(vectorizer_path)
        
        return {
            "model": model,
            "vectorizer": vectorizer,
            "metadata": metadata,
        }
    
    def list_models(self, filter_by: dict[str, Any] = None) -> pd.DataFrame:
        """
        Liste tous les modèles du registry.
        
        Args:
            filter_by: Dict pour filtrer (ex: {"embedding": "bert"})
        
        Returns:
            DataFrame avec les informations des modèles
        """
        if not self.registry:
            return pd.DataFrame()
        
        models_list = []
        for model_id, metadata in self.registry.items():
            # Appliquer les filtres
            if filter_by:
                skip = False
                for key, value in filter_by.items():
                    if metadata.get(key) != value:
                        skip = True
                        break
                if skip:
                    continue
            
            models_list.append({
                "model_id": model_id,
                "name": metadata.get("name", "Unknown"),
                "embedding": metadata.get("embedding", "Unknown"),
                "dataset": metadata.get("dataset", "Unknown"),
                "accuracy": metadata.get("accuracy", 0),
                "f1": metadata.get("f1", 0),
                "timestamp": metadata.get("timestamp", "Unknown"),
            })
        
        return pd.DataFrame(models_list).sort_values("timestamp", ascending=False)
    
    def delete_model(self, model_id: str):
        """Supprime un modèle du registry et du disque."""
        if model_id not in self.registry:
            raise ValueError(f"Modèle {model_id} introuvable.")
        
        metadata = self.registry[model_id]
        model_dir = Path(metadata["model_path"]).parent
        
        # Supprimer les fichiers
        if model_dir.exists():
            import shutil
            shutil.rmtree(model_dir)
        
        # Supprimer du registry
        del self.registry[model_id]
        self._save_registry()
    
    def get_best_model(self, metric: str = "f1", filter_by: dict[str, Any] = None) -> str:
        """
        Retourne l'ID du meilleur modèle selon une métrique.
        
        Args:
            metric: Métrique à optimiser (f1, accuracy, precision, recall)
            filter_by: Filtres optionnels
        
        Returns:
            model_id du meilleur modèle
        """
        df = self.list_models(filter_by=filter_by)
        
        if df.empty:
            raise ValueError("Aucun modèle disponible.")
        
        if metric not in df.columns:
            raise ValueError(f"Métrique {metric} non disponible.")
        
        best_idx = df[metric].idxmax()
        return df.loc[best_idx, "model_id"]


def export_model_package(
    model_id: str,
    registry: ModelRegistry,
    output_dir: Path,
) -> Path:
    """
    Exporte un modèle avec tous ses composants dans un package standalone.
    
    Args:
        model_id: ID du modèle à exporter
        registry: ModelRegistry
        output_dir: Répertoire de sortie
    
    Returns:
        Path du package créé
    """
    import shutil
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Charger le modèle
    components = registry.load_model(model_id)
    
    # Créer le répertoire du package
    package_dir = output_dir / f"{model_id}_package"
    package_dir.mkdir(exist_ok=True)
    
    # Copier le modèle
    joblib.dump(components["model"], package_dir / "model.pkl")
    
    # Copier le vectorizer si existe
    if components["vectorizer"] is not None:
        joblib.dump(components["vectorizer"], package_dir / "vectorizer.pkl")
    
    # Copier les métadonnées
    with open(package_dir / "metadata.json", "w") as f:
        json.dump(components["metadata"], f, indent=2)
    
    # Créer un README
    readme_content = f"""# Model Package: {model_id}

## Métadonnées
- Name: {components['metadata'].get('name', 'Unknown')}
- Embedding: {components['metadata'].get('embedding', 'Unknown')}
- Dataset: {components['metadata'].get('dataset', 'Unknown')}
- Accuracy: {components['metadata'].get('accuracy', 'N/A')}
- F1-Score: {components['metadata'].get('f1', 'N/A')}
- Created: {components['metadata'].get('timestamp', 'Unknown')}

## Utilisation

```python
import joblib

# Charger le modèle
model = joblib.load('model.pkl')

# Charger le vectorizer (si existe)
vectorizer = joblib.load('vectorizer.pkl')

# Prédiction
X = vectorizer.transform(["Your text here"])
prediction = model.predict(X)
```
"""
    
    with open(package_dir / "README.md", "w") as f:
        f.write(readme_content)
    
    # Créer une archive
    archive_path = output_dir / f"{model_id}_package"
    shutil.make_archive(str(archive_path), "zip", package_dir)
    
    return Path(str(archive_path) + ".zip")
