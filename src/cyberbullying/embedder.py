"""Méthodes d'embedding : transforme du texte en vecteurs numériques.

On supporte TF-IDF, BoW, Word2Vec, GloVe, BERT et RoBERTa.
Les méthodes sont exposées via embed_texts() qui gère aussi un cache disque
pour éviter de recalculer des embeddings coûteux (BERT surtout).
"""

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

logger = logging.getLogger(__name__)


# --- Exceptions custom ---

class EmbeddingError(Exception):
    pass

class MissingDependencyError(EmbeddingError):
    pass

class GlovePathRequiredError(EmbeddingError):
    pass

class UnknownMethodError(EmbeddingError):
    pass


AggregationMethod = Literal["mean", "max", "min", "median", "sum", "mean_top_3"]
EmbeddingMethod = Literal["tfidf", "bow", "word2vec", "glove", "bert", "roberta"]


def _embedding_cache_key(texts: list[str], method: str, **params) -> str:
    """Clé de cache déterministe."""
    h = hashlib.sha256()
    for t in texts:
        h.update(str(t).encode("utf-8", errors="replace"))
    h.update(f"{method}_{params}".encode())
    return h.hexdigest()[:32]


# --- Agrégation de vecteurs ---

def _aggregate_vectors(vectors: np.ndarray, method: AggregationMethod, vector_size: int) -> np.ndarray:
    """Agrège une liste de vecteurs de mots en un seul vecteur document."""
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
        sorted_vecs = np.sort(vectors, axis=0)
        n = min(3, len(vectors))  # sécurité pour les tweets courts
        return np.mean(sorted_vecs[-n:], axis=0)
    raise ValueError(f"Méthode d'agrégation inconnue : {method}")


def _worker_word_embedding(text: str, model_dict, method: AggregationMethod, vector_size: int) -> np.ndarray:
    """Worker pour paralléliser l'embedding de documents (Word2Vec/GloVe)."""
    if not isinstance(text, str):
        return np.zeros(vector_size, dtype=np.float32)
    tokens = text.lower().split()
    vectors_list = []
    for word in tokens:
        try:
            vec = model_dict[word]
            vectors_list.append(vec)
        except KeyError:
            continue
    vectors = np.array(vectors_list, dtype=np.float32)
    return _aggregate_vectors(vectors, method, vector_size)


def _concat_embedding(df: pd.DataFrame, matrix: np.ndarray, prefix: str) -> pd.DataFrame:
    """Ajoute les colonnes d'embedding au DataFrame source."""
    embedding_df = pd.DataFrame(
        matrix,
        columns=[f"{prefix}_{i}" for i in range(matrix.shape[1])]
    )
    return pd.concat([df.reset_index(drop=True), embedding_df], axis=1)


# --- TF-IDF et Bag of Words ---

def tfidf_embedding(texts: list[str], max_features: int, ngram_range: tuple[int, int]):
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=2,
        max_df=0.95,
        stop_words='english'
    )
    matrix = vectorizer.fit_transform(texts)
    return matrix.toarray(), vectorizer


def apply_tfidf_embedding(df: pd.DataFrame, column_name: str, max_features: int = 5000, ngram_range: tuple[int, int] = (1, 2)) -> pd.DataFrame:
    matrix, _ = tfidf_embedding(df[column_name].tolist(), max_features, ngram_range)
    return _concat_embedding(df, matrix, f"{column_name}_tfidf")


def bow_embedding(texts: list[str], max_features: int, ngram_range: tuple[int, int]):
    vectorizer = CountVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=2,
        max_df=0.95,
        stop_words='english'
    )
    matrix = vectorizer.fit_transform(texts)
    return matrix.toarray(), vectorizer


def apply_bow_embedding(df: pd.DataFrame, column_name: str, max_features: int = 5000, ngram_range: tuple[int, int] = (1, 2)) -> pd.DataFrame:
    matrix, _ = bow_embedding(df[column_name].tolist(), max_features, ngram_range)
    return _concat_embedding(df, matrix, f"{column_name}_bow")


# --- Word2Vec ---

def word2vec_embedding(
    texts: list[str],
    vector_size: int = 100,
    window: int = 5,
    min_count: int = 2,
    method: AggregationMethod = "mean"
) -> np.ndarray:
    try:
        from gensim.models import Word2Vec
    except ImportError as e:
        raise MissingDependencyError("pip install gensim") from e

    tokenized = [text.split() for text in texts]
    model = Word2Vec(
        sentences=tokenized,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=multiprocessing.cpu_count(),
        epochs=10
    )
    results = Parallel(n_jobs=-1, backend="threading")(
        delayed(_worker_word_embedding)(text, model.wv, method, vector_size)
        for text in texts
    )
    return np.array(results)


def apply_word2vec_embedding(df: pd.DataFrame, column_name: str, vector_size: int = 100, window: int = 5, min_count: int = 2, method: AggregationMethod = "mean") -> pd.DataFrame:
    matrix = word2vec_embedding(df[column_name].tolist(), vector_size, window, min_count, method)
    return _concat_embedding(df, matrix, f"{column_name}_w2v")


# --- GloVe ---

def _build_cooccurrence_matrix_fast(texts: list[str]):
    """Matrice de co-occurrence via X.T * X (plus rapide que la construction manuelle)."""
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
    """Entraîne un modèle GloVe from scratch via Mittens."""
    try:
        from mittens import GloVe
    except ImportError:
        raise ImportError("pip install mittens")

    logger.info("Construction de la matrice de co-occurrence...")
    cooc_matrix, vocab = _build_cooccurrence_matrix_fast(texts)
    vocab_size = len(vocab)
    logger.info(f"Vocab: {vocab_size} mots | Matrice: {cooc_matrix.shape}")

    glove = GloVe(n=vector_size, max_iter=epochs, learning_rate=learning_rate)
    embeddings_matrix = glove.fit(cooc_matrix.toarray())
    id2word = {i: w for w, i in vocab.items()}
    embeddings = {id2word[i]: embeddings_matrix[i] for i in range(vocab_size)}

    if save_path:
        with open(save_path, 'wb') as f:
            pickle.dump(embeddings, f)
    return embeddings


def load_glove_model(path: str) -> dict[str, np.ndarray]:
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
    if retrain or not glove_path:
        print(">>> Entraînement GloVe (pas de modèle pré-entraîné fourni)...")
        model = train_glove_model(texts, vector_size, save_path=save_path)
    else:
        print(f">>> Chargement GloVe : {glove_path}...")
        model = load_glove_model(glove_path)
        vector_size = len(next(iter(model.values())))

    print(">>> Transformation (parallèle)...")
    results = Parallel(n_jobs=-1, backend="threading")(
        delayed(_worker_word_embedding)(text, model, method, vector_size)
        for text in texts
    )
    return np.array(results)


def apply_glove_embedding(df: pd.DataFrame, column_name: str, glove_path: str | None = None, method: AggregationMethod = "mean", retrain: bool = False, vector_size: int = 100, save_path: str | None = None) -> pd.DataFrame:
    matrix = glove_embedding(df[column_name].tolist(), glove_path=glove_path, method=method, retrain=retrain, vector_size=vector_size, save_path=save_path)
    return _concat_embedding(df, matrix, f"{column_name}_glove")


# --- BERT ---

def train_bert_model(texts: list[str], model_name: str = "all-MiniLM-L6-v2", epochs: int = 1, batch_size: int = 32, save_path: str | None = None):
    """Fine-tune BERT via TSDAE (Denoising AutoEncoder)."""
    try:
        from sentence_transformers import InputExample, SentenceTransformer, losses
        from torch.utils.data import DataLoader
    except ImportError:
        raise MissingDependencyError("pip install sentence-transformers torch")

    logger.info(f"Chargement BERT : {model_name}...")
    model = SentenceTransformer(model_name)

    # TSDAE: source = target = text (auto-encodeur bruité)
    train_examples = [InputExample(texts=[t, t]) for t in texts if len(t) > 10]
    train_dataloader = DataLoader(train_examples, batch_size=batch_size, shuffle=True)
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
    """Génère des embeddings BERT. Supporte GPU (batch_size=128 recommandé)."""
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

    import torch
    if device == "cuda" and not torch.cuda.is_available():
        print("GPU non disponible, fallback CPU.")
        device = "cpu"
    model = model.to(device)

    print(f">>> Encodage BERT (batch={batch_size}, device={device})...")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
        device=device
    )
    return embeddings


def apply_bert_embedding(df: pd.DataFrame, column_name: str, model_name: str = "all-MiniLM-L6-v2", retrain: bool = False, save_path: str | None = None, batch_size: int = 128) -> pd.DataFrame:
    matrix = bert_embedding(df[column_name].tolist(), model_name_or_path=model_name, retrain=retrain, save_path=save_path, batch_size=batch_size)
    return _concat_embedding(df, matrix, f"{column_name}_bert")


# --- RoBERTa (réutilise la même logique que BERT) ---

def roberta_embedding(
    texts: list[str],
    model_name_or_path: str = "roberta-base",
    retrain: bool = False,
    save_path: str | None = None,
    batch_size: int = 128,
    device: str = "cuda"
) -> np.ndarray:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise MissingDependencyError("pip install sentence-transformers")

    if retrain:
        print(">>> Fine-tuning RoBERTa...")
        model = train_bert_model(texts, model_name=model_name_or_path, save_path=save_path)
    else:
        load_path = save_path if (save_path and not retrain) else model_name_or_path
        print(f">>> Chargement RoBERTa : {load_path}...")
        model = SentenceTransformer(load_path)

    import torch
    if device == "cuda" and not torch.cuda.is_available():
        print("GPU non disponible, fallback CPU.")
        device = "cpu"
    model = model.to(device)

    print(f">>> Encodage RoBERTa (batch={batch_size}, device={device})...")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
        device=device
    )
    return embeddings


def apply_roberta_embedding(df: pd.DataFrame, column_name: str, model_name: str = "roberta-base", retrain: bool = False, save_path: str | None = None, batch_size: int = 128) -> pd.DataFrame:
    matrix = roberta_embedding(df[column_name].tolist(), model_name_or_path=model_name, retrain=retrain, save_path=save_path, batch_size=batch_size)
    return _concat_embedding(df, matrix, f"{column_name}_roberta")


# --- Interface unifiée avec cache ---

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
    if method == "tfidf":
        mat, _ = tfidf_embedding(texts, max_features, ngram_range)
        return mat
    if method == "bow":
        mat, _ = bow_embedding(texts, max_features, ngram_range)
        return mat
    if method == "word2vec":
        return word2vec_embedding(texts, vector_size, method=aggregation)
    if method == "glove":
        return glove_embedding(texts, glove_path, aggregation, retrain, vector_size, save_path)
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
    Point d'entrée unique pour tous les embeddings.

    Gère un cache disque basé sur un hash du corpus + méthode + params.
    Le cache est désactivé si retrain=True (fine-tuning BERT/GloVe).

    Méthodes supportées : tfidf, bow, word2vec, glove, bert, roberta
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
                pass  # cache corrompu, on recalcule

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
