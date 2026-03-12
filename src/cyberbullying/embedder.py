"""Méthodes d'embedding pour transformer le texte en vecteurs."""

import hashlib
import logging
import multiprocessing
import pickle
from collections import Counter
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from joblib import Parallel, delayed


def _get_cache_dir() -> Path:
    try:
        from cyberbullying.config import EMBEDDING_CACHE_DIR
        return EMBEDDING_CACHE_DIR
    except ImportError:
        return Path(__file__).resolve().parents[2] / "data" / "processed" / "embedding_cache"


from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# =============================================================================
# 1. CONFIGURATION & TYPES
# =============================================================================

logger = logging.getLogger(__name__)

class EmbeddingError(Exception):
    """Erreur de base pour les embeddings."""

class MissingDependencyError(EmbeddingError):
    """Dépendance non installée."""

class GlovePathRequiredError(EmbeddingError):
    """Chemin GloVe requis quand retrain=False."""

class UnknownMethodError(EmbeddingError):
    """Méthode d'embedding inconnue."""

AggregationMethod = Literal["mean", "max", "min", "median", "sum", "mean_top_3"]
EmbeddingMethod = Literal["tfidf", "bow", "word2vec", "glove", "bert", "roberta"]


def _embedding_cache_key(texts: list[str], method: str, **params) -> str:
    """Clé de cache déterministe pour embeddings."""
    h = hashlib.sha256()
    for t in texts:
        h.update(str(t).encode("utf-8", errors="replace"))
    h.update(f"{method}_{params}".encode())
    return h.hexdigest()[:32]


# =============================================================================
# 2. HELPERS (AGRÉGATION & PARALLÉLISME)
# =============================================================================

def _aggregate_vectors(
    vectors: np.ndarray,
    method: AggregationMethod,
    vector_size: int
) -> np.ndarray:
    """
    Agrège les vecteurs de mots en un vecteur unique.

    Args:
        vectors: Matrice des vecteurs de mots.
        method: Méthode d'agrégation.
        vector_size: Dimension du vecteur de sortie.

    Returns:
        Vecteur agrégé (1D).
    """
    # Cas vide : aucun mot reconnu
    if vectors.size == 0:
        return np.zeros(vector_size, dtype=np.float32)

    if method == "mean":
        return np.mean(vectors, axis=0)

    if method == "sum":
        return np.sum(vectors, axis=0)

    if method == "max":
        return np.max(vectors, axis=0)

    if method == "min":
        return np.min(vectors, axis=0)

    if method == "median":
        return np.median(vectors, axis=0)

    if method == "mean_top_3":
        # Tri sur chaque dimension
        sorted_vecs = np.sort(vectors, axis=0)
        # Sécurité pour les tweets très courts (< 3 mots reconnus)
        n = min(3, len(vectors))
        # Moyenne des n plus grandes valeurs sur chaque dimension
        return np.mean(sorted_vecs[-n:], axis=0)

    raise ValueError(f"Méthode d'agrégation inconnue : {method}")


def _worker_word_embedding(
    text: str,
    model_dict: dict[str, np.ndarray] | Any,
    method: AggregationMethod,
    vector_size: int
) -> np.ndarray:
    """
    Transforme un texte en vecteur (exécuté par un worker).

    Args:
        text: Texte à vectoriser.
        model_dict: Dictionnaire {mot: vecteur} ou KeyedVectors.
        method: Méthode d'agrégation.
        vector_size: Dimension des vecteurs.

    Returns:
        Vecteur du document.
    """
    if not isinstance(text, str):
        return np.zeros(vector_size, dtype=np.float32)

    tokens = text.lower().split()

    # Récupération des vecteurs (compatible dict et Gensim KeyedVectors)
    vectors_list = []
    for word in tokens:
        try:
            # Pour un dict standard (GloVe) ou Gensim (Word2Vec)
            vec = model_dict[word]
            vectors_list.append(vec)
        except KeyError:
            continue

    vectors = np.array(vectors_list, dtype=np.float32)
    return _aggregate_vectors(vectors, method, vector_size)


def _concat_embedding(
    df: pd.DataFrame,
    matrix: np.ndarray,
    prefix: str
) -> pd.DataFrame:
    """
    Ajoute les colonnes d'embedding au DataFrame.

    Args:
        df: DataFrame source.
        matrix: Matrice d'embeddings à ajouter.
        prefix: Préfixe des nouvelles colonnes.

    Returns:
        DataFrame avec les nouvelles colonnes.
    """
    embedding_df = pd.DataFrame(
        matrix,
        columns=[f"{prefix}_{i}" for i in range(matrix.shape[1])]
    )
    return pd.concat([df.reset_index(drop=True), embedding_df], axis=1)


# =============================================================================
# 3. FREQUENCY EMBEDDINGS (TF-IDF, BOW)
# =============================================================================

def tfidf_embedding(
    texts: list[str],
    max_features: int,
    ngram_range: tuple[int, int]
) -> tuple[np.ndarray, TfidfVectorizer]:
    """
    Génère les vecteurs TF-IDF.

    Args:
        texts: Liste de textes.
        max_features: Nombre maximum de mots.
        ngram_range: Taille des n-grammes (ex: (1, 2)).

    Returns:
        Tuple (Matrice TF-IDF, Vectorizer entraîné).
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=2,
        max_df=0.95,
        stop_words='english'
    )
    matrix = vectorizer.fit_transform(texts)
    return matrix.toarray(), vectorizer


def apply_tfidf_embedding(
    df: pd.DataFrame,
    column_name: str,
    max_features: int = 5000,
    ngram_range: tuple[int, int] = (1, 2)
) -> pd.DataFrame:
    """
    Applique TF-IDF sur une colonne.

    Args:
        df: DataFrame contenant les données.
        column_name: Nom de la colonne texte.
        max_features: Nombre maximum de features.
        ngram_range: Taille des n-grammes.

    Returns:
        DataFrame enrichi avec les vecteurs TF-IDF.
    """
    matrix, _ = tfidf_embedding(df[column_name].tolist(), max_features, ngram_range)
    return _concat_embedding(df, matrix, f"{column_name}_tfidf")


def bow_embedding(
    texts: list[str],
    max_features: int,
    ngram_range: tuple[int, int]
) -> tuple[np.ndarray, CountVectorizer]:
    """
    Génère les vecteurs Bag of Words.

    Args:
        texts: Liste de textes.
        max_features: Nombre maximum de mots.
        ngram_range: Taille des n-grammes.

    Returns:
        Tuple (Matrice BoW, Vectorizer entraîné).
    """
    vectorizer = CountVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=2,
        max_df=0.95,
        stop_words='english'
    )
    matrix = vectorizer.fit_transform(texts)
    return matrix.toarray(), vectorizer


def apply_bow_embedding(
    df: pd.DataFrame,
    column_name: str,
    max_features: int = 5000,
    ngram_range: tuple[int, int] = (1, 2)
) -> pd.DataFrame:
    """
    Applique Bag of Words sur une colonne.

    Args:
        df: DataFrame contenant les données.
        column_name: Nom de la colonne texte.
        max_features: Nombre maximum de features.
        ngram_range: Taille des n-grammes.

    Returns:
        DataFrame enrichi avec les vecteurs BoW.
    """
    matrix, _ = bow_embedding(df[column_name].tolist(), max_features, ngram_range)
    return _concat_embedding(df, matrix, f"{column_name}_bow")


# =============================================================================
# 4. WORD EMBEDDINGS (WORD2VEC, GLOVE)
# =============================================================================

# --- WORD2VEC ---

def word2vec_embedding(
    texts: list[str],
    vector_size: int = 100,
    window: int = 5,
    min_count: int = 2,
    method: AggregationMethod = "mean"
) -> np.ndarray:
    """
    Entraîne et applique Word2Vec.

    Args:
        texts: Liste de textes.
        vector_size: Dimension des vecteurs.
        window: Fenêtre de contexte.
        min_count: Fréquence minimale des mots.
        method: Méthode d'agrégation.

    Returns:
        Matrice d'embeddings (n_samples, vector_size).
    """
    try:
        from gensim.models import Word2Vec
    except ImportError as e:
        raise MissingDependencyError("pip install gensim") from e

    tokenized = [text.split() for text in texts]

    # Entraînement
    model = Word2Vec(
        sentences=tokenized,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=multiprocessing.cpu_count(),
        epochs=10
    )

    # Inférence Parallélisée (Unifiée avec GloVe)
    results = Parallel(n_jobs=-1, backend="threading")(
        delayed(_worker_word_embedding)(text, model.wv, method, vector_size)
        for text in texts
    )

    return np.array(results)


def apply_word2vec_embedding(
    df: pd.DataFrame,
    column_name: str,
    vector_size: int = 100,
    window: int = 5,
    min_count: int = 2,
    method: AggregationMethod = "mean"
) -> pd.DataFrame:
    """
    Applique Word2Vec sur une colonne.

    Args:
        df: DataFrame source.
        column_name: Colonne texte.
        vector_size: Dimension des vecteurs.
        window: Fenêtre de contexte.
        min_count: Fréquence minimale.
        method: Agrégation.

    Returns:
        DataFrame enrichi.
    """
    matrix = word2vec_embedding(
        df[column_name].tolist(), vector_size, window, min_count, method
    )
    return _concat_embedding(df, matrix, f"{column_name}_w2v")


# --- GLOVE ---

def _build_cooccurrence_matrix_fast(texts: list[str]):
    """
    Construit la matrice de co-occurrence (optimisé X.T * X).

    Args:
        texts: Liste de textes.

    Returns:
        Tuple (Matrice co-occurrence, Vocabulaire).
    """
    vectorizer = CountVectorizer(
        min_df=2,
        max_df=0.95,
        token_pattern=r"(?u)\b\w+\b",
        stop_words='english'
    )
    X = vectorizer.fit_transform(texts)
    vocab = vectorizer.vocabulary_

    cooc_matrix = (X.T * X)
    cooc_matrix.setdiag(0)
    return cooc_matrix, vocab


def train_glove_model(
    texts: list[str],
    vector_size: int = 200,
    epochs: int = 100,
    learning_rate: float = 0.05,
    save_path: str | None = None
) -> dict[str, np.ndarray]:
    """
    Entraîne un modèle GloVe via Mittens.

    Args:
        texts: Corpus de textes.
        vector_size: Dimension des vecteurs.
        epochs: Nombre d'itérations.
        learning_rate: Taux d'apprentissage.
        save_path: Chemin de sauvegarde (optionnel).

    Returns:
        Dictionnaire {mot: vecteur}.
    """
    try:
        from mittens import GloVe
    except ImportError:
        raise ImportError("pip install mittens")

    logger.info("Construction rapide de la matrice de co-occurrence...")
    cooc_matrix, vocab = _build_cooccurrence_matrix_fast(texts)
    vocab_size = len(vocab)
    logger.info(f"Vocabulaire : {vocab_size} mots | Matrice : {cooc_matrix.shape}")

    logger.info("Entraînement GloVe (Mittens)...")
    glove = GloVe(n=vector_size, max_iter=epochs, learning_rate=learning_rate)
    embeddings_matrix = glove.fit(cooc_matrix.toarray())

    id2word = {i: w for w, i in vocab.items()}
    embeddings = {id2word[i]: embeddings_matrix[i] for i in range(vocab_size)}

    if save_path:
        with open(save_path, 'wb') as f:
            pickle.dump(embeddings, f)

    return embeddings


def load_glove_model(path: str) -> dict[str, np.ndarray]:
    """
    Charge un modèle GloVe.

    Args:
        path: Chemin du fichier (.pkl ou .txt).

    Returns:
        Dictionnaire {mot: vecteur}.
    """
    if path.endswith('.pkl'):
        with open(path, 'rb') as f:
            return pickle.load(f)

    embeddings = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            vals = line.split()
            embeddings[vals[0]] = np.array(vals[1:], dtype='float32')
    return embeddings


def glove_embedding(
    texts: list[str],
    glove_path: str | None = None,
    method: AggregationMethod = "mean",
    retrain: bool = False,
    vector_size: int = 100,
    save_path: str | None = None
) -> np.ndarray:
    """
    Pipeline complet GloVe (Train/Load + Transform).

    Args:
        texts: Liste de textes.
        glove_path: Chemin du modèle (si !retrain).
        method: Agrégation des vecteurs.
        retrain: Si True, entraîne un nouveau modèle.
        vector_size: Dimension (si retrain).
        save_path: Sauvegarde du modèle (si retrain).

    Returns:
        Matrice d'embeddings.
    """
    if retrain or not glove_path:
        # Pas de modèle pré-entraîné : entraînement sur le corpus (comportement Word2Vec)
        print(">>> Entraînement GloVe (glove_path absent ou retrain=True)...")
        model = train_glove_model(texts, vector_size, save_path=save_path)
    else:
        print(f">>> Chargement GloVe : {glove_path}...")
        model = load_glove_model(glove_path)
        vector_size = len(next(iter(model.values())))

    print(">>> Transformation (Parallèle)...")
    results = Parallel(n_jobs=-1, backend="threading")(
        delayed(_worker_word_embedding)(text, model, method, vector_size)
        for text in texts
    )
    return np.array(results)


def apply_glove_embedding(
    df: pd.DataFrame,
    column_name: str,
    glove_path: str | None = None,
    method: AggregationMethod = "mean",
    retrain: bool = False,
    vector_size: int = 100,
    save_path: str | None = None
) -> pd.DataFrame:
    """
    Applique GloVe sur une colonne.

    Args:
        df: DataFrame source.
        column_name: Colonne texte.
        glove_path: Chemin du modèle.
        method: Agrégation.
        retrain: Réentraîner le modèle ?
        vector_size: Dimension.
        save_path: Chemin de sauvegarde.

    Returns:
        DataFrame enrichi.
    """
    matrix = glove_embedding(
        df[column_name].tolist(),
        glove_path=glove_path,
        method=method,
        retrain=retrain,
        vector_size=vector_size,
        save_path=save_path
    )
    return _concat_embedding(df, matrix, f"{column_name}_glove")


# =============================================================================
# 5. SENTENCE EMBEDDINGS (BERT)
# =============================================================================

def train_bert_model(
    texts: list[str],
    model_name: str = "all-MiniLM-L6-v2",
    epochs: int = 1,
    batch_size: int = 32,
    save_path: str | None = None
):
    """
    Fine-tune un modèle BERT (TSDAE).

    Args:
        texts: Corpus de textes.
        model_name: Modèle HuggingFace de base.
        epochs: Nombre d'époques.
        batch_size: Taille du batch.
        save_path: Dossier de sauvegarde.

    Returns:
        Modèle SentenceTransformer entraîné.
    """
    try:
        from sentence_transformers import InputExample, SentenceTransformer, losses
        from torch.utils.data import DataLoader
    except ImportError:
        raise MissingDependencyError("pip install sentence-transformers torch")

    logger.info(f"Chargement BERT : {model_name}...")
    model = SentenceTransformer(model_name)

    # TSDAE : Denoising AutoEncoder.
    # Pour la compatibilité avec la v3 et Trainer, on doit fournir [source, target].
    # Comme c'est un auto-encodeur, source = target = text.
    # La loss va bruiter la source en interne, mais elle a besoin de 2 entrées.
    train_examples = [InputExample(texts=[t, t]) for t in texts if len(t) > 10]

    # DataLoader standard
    train_dataloader = DataLoader(train_examples, batch_size=batch_size, shuffle=True)

    # Configuration de la Loss TSDAE
    train_loss = losses.DenoisingAutoEncoderLoss(model, decoder_name_or_path=model_name, tie_encoder_decoder=True)

    logger.info(f"Fine-tuning BERT ({epochs} epochs)...")
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=epochs,
        weight_decay=0,
        scheduler='constantlr',
        optimizer_params={'lr': 3e-5},
        show_progress_bar=True
    )

    if save_path:
        model.save(save_path)
    return model


def bert_embedding(
    texts: list[str],
    model_name_or_path: str = "all-MiniLM-L6-v2",
    retrain: bool = False,
    save_path: str | None = None,
    batch_size: int = 128,
    device: str = "cuda"
) -> np.ndarray:
    """
    Génère des embeddings BERT avec support GPU.

    Args:
        texts: Liste de textes.
        model_name_or_path: Nom du modèle ou chemin.
        retrain: Fine-tuner le modèle ?
        save_path: Chemin de sauvegarde/chargement.
        batch_size: Taille du batch d'inférence (défaut 128 pour GPU).
        device: Appareil d'inférence ("cuda" pour GPU, "cpu" sinon).

    Returns:
        Matrice d'embeddings.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise MissingDependencyError("pip install sentence-transformers")

    if retrain:
        print(">>> Fine-tuning BERT...")
        model = train_bert_model(texts, model_name=model_name_or_path, save_path=save_path)
    else:
        load_path = save_path if (save_path and not retrain) else model_name_or_path
        print(f">>> Chargement BERT : {load_path}...")
        model = SentenceTransformer(load_path)

    # Vérifier la disponibilité du GPU et charger le modèle
    import torch
    if device == "cuda" and not torch.cuda.is_available():
        print("GPU non disponible. Utilisation du CPU.")
        device = "cpu"

    # Charger le modele sur le device (GPU ou CPU)
    model = model.to(device)

    print(f">>> Encodage BERT (Batch: {batch_size}, Device: {device})...")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
        device=device
    )
    return embeddings


def apply_bert_embedding(
    df: pd.DataFrame,
    column_name: str,
    model_name: str = "all-MiniLM-L6-v2",
    retrain: bool = False,
    save_path: str | None = None,
    batch_size: int = 128
) -> pd.DataFrame:
    """
    Applique BERT sur une colonne.

    Args:
        df: DataFrame source.
        column_name: Colonne texte.
        model_name: Modèle de base.
        retrain: Fine-tuner ?
        save_path: Chemin de sauvegarde.
        batch_size: Taille du batch (défaut 128 pour GPU).

    Returns:
        DataFrame enrichi.
    """
    matrix = bert_embedding(
        df[column_name].tolist(),
        model_name_or_path=model_name,
        retrain=retrain,
        save_path=save_path,
        batch_size=batch_size
    )
    return _concat_embedding(df, matrix, f"{column_name}_bert")


# =============================================================================
# 5b. ROBERTA EMBEDDINGS
# =============================================================================

def roberta_embedding(
    texts: list[str],
    model_name_or_path: str = "roberta-base",
    retrain: bool = False,
    save_path: str | None = None,
    batch_size: int = 128,
    device: str = "cuda"
) -> np.ndarray:
    """
    Génère des embeddings RoBERTa avec support GPU.

    Args:
        texts: Liste de textes.
        model_name_or_path: Nom du modèle ou chemin (par défaut roberta-base).
        retrain: Fine-tuner le modèle ?
        save_path: Chemin de sauvegarde/chargement.
        batch_size: Taille du batch d'inférence (défaut 128 pour GPU).
        device: Appareil d'inférence ("cuda" pour GPU, "cpu" sinon).

    Returns:
        Matrice d'embeddings.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise MissingDependencyError("pip install sentence-transformers")

    if retrain:
        print(">>> Fine-tuning RoBERTa...")
        # Réutilise le même processus que BERT
        model = train_bert_model(texts, model_name=model_name_or_path, save_path=save_path)
    else:
        load_path = save_path if (save_path and not retrain) else model_name_or_path
        print(f">>> Chargement RoBERTa : {load_path}...")
        model = SentenceTransformer(load_path)

    # Vérifier la disponibilité du GPU et charger le modèle
    import torch
    if device == "cuda" and not torch.cuda.is_available():
        print("GPU non disponible. Utilisation du CPU.")
        device = "cpu"

    # Charger le modele sur le device (GPU ou CPU)
    model = model.to(device)

    print(f">>> Encodage RoBERTa (Batch: {batch_size}, Device: {device})...")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
        device=device
    )
    return embeddings


def apply_roberta_embedding(
    df: pd.DataFrame,
    column_name: str,
    model_name: str = "roberta-base",
    retrain: bool = False,
    save_path: str | None = None,
    batch_size: int = 128
) -> pd.DataFrame:
    """
    Applique RoBERTa sur une colonne.

    Args:
        df: DataFrame source.
        column_name: Colonne texte.
        model_name: Modèle de base.
        retrain: Fine-tuner ?
        save_path: Chemin de sauvegarde.
        batch_size: Taille du batch (défaut 128 pour GPU).

    Returns:
        DataFrame enrichi.
    """
    matrix = roberta_embedding(
        df[column_name].tolist(),
        model_name_or_path=model_name,
        retrain=retrain,
        save_path=save_path,
        batch_size=batch_size
    )
    return _concat_embedding(df, matrix, f"{column_name}_roberta")


# =============================================================================
# 6. UNIFIED INTERFACE
# =============================================================================

def _embed_texts_impl(
    texts: list[str],
    method: EmbeddingMethod,
    max_features: int,
    ngram_range: tuple[int, int],
    glove_path: str | None,
    vector_size: int,
    aggregation: AggregationMethod,
    retrain: bool,
    save_path: str | None,
) -> np.ndarray:
    """Implémentation réelle des embeddings (sans cache)."""
    if method == "tfidf":
        mat, _ = tfidf_embedding(texts, max_features, ngram_range)
        return mat

    if method == "bow":
        mat, _ = bow_embedding(texts, max_features, ngram_range)
        return mat

    if method == "word2vec":
        return word2vec_embedding(texts, vector_size, method=aggregation)

    if method == "glove":
        return glove_embedding(
            texts, glove_path, aggregation, retrain, vector_size, save_path
        )

    if method == "bert":
        return bert_embedding(texts, retrain=retrain, save_path=save_path)

    if method == "roberta":
        return roberta_embedding(texts, retrain=retrain, save_path=save_path)

    raise UnknownMethodError(f"Méthode inconnue: {method}")


def embed_texts(
    texts: list[str],
    method: EmbeddingMethod,
    max_features: int = 5000,
    ngram_range: tuple[int, int] = (1, 2),
    glove_path: str | None = None,
    vector_size: int = 100,
    aggregation: AggregationMethod = "mean",
    retrain: bool = False,
    save_path: str | None = None,
) -> np.ndarray:
    """
    Interface unique pour toutes les méthodes d'embedding (avec cache disque).

    Args:
        texts: Liste de textes.
        method: Méthode ('tfidf', 'bow', 'word2vec', 'glove', 'bert', 'roberta').
        max_features: (TF-IDF/BoW) Nombre max de features.
        ngram_range: (TF-IDF/BoW) Taille des n-grammes.
        glove_path: (GloVe) Chemin du modèle pré-entraîné.
        vector_size: (W2V/GloVe) Dimension des vecteurs.
        aggregation: (W2V/GloVe) Méthode d'agrégation.
        retrain: (GloVe/BERT/RoBERTa) Réentraîner le modèle ? (désactive le cache)
        save_path: (GloVe/BERT/RoBERTa) Chemin de sauvegarde.

    Returns:
        Matrice d'embeddings.
    """
    params = {
        "max_features": max_features,
        "ngram_range": ngram_range,
        "glove_path": glove_path,
        "vector_size": vector_size,
        "aggregation": aggregation,
    }
    if not retrain:
        cache_dir = _get_cache_dir()
        key = _embedding_cache_key(texts, method, **params)
        cache_path = cache_dir / f"{key}.pkl"
        if cache_path.exists():
            try:
                with open(cache_path, "rb") as f:
                    return pickle.load(f)
            except Exception:
                pass

    result = _embed_texts_impl(
        texts, method, max_features, ngram_range,
        glove_path, vector_size, aggregation, retrain, save_path,
    )

    if not retrain:
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            with open(cache_path, "wb") as f:
                pickle.dump(result, f)
        except Exception:
            pass

    return result
