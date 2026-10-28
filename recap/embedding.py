"""
embedding utilities for plan retrieval.

supports two providers:
    - "local": sentence-transformers (runs on CPU, no API cost)
    - "openai": text-embedding-3-small via API (billed per token)

default: local with all-MiniLM-L6-v2 (same as config.EMBEDDING_MODEL)
"""

import logging
import numpy as np

from recap import config

logger = logging.getLogger(__name__)

_LOCAL_MODEL = None


def get_embedding(text, provider=None, model=None):
    """compute embedding vector for a text string.

    returns: numpy array of shape (dim,)
    """
    provider = provider or config.EMBEDDING_PROVIDER
    model = model or config.EMBEDDING_MODEL

    if provider == "local":
        return _get_local_embedding(text, model)
    elif provider == "openai":
        return _get_openai_embedding(text, model)
    else:
        raise ValueError(f"unknown embedding provider: {provider}")


def embed_texts(texts, provider=None, model=None):
    """embed a list of texts, returning a (n, dim) float32 array.

    uses the sentence-transformers batch encoder for the local backend (much
    faster than one call per text when indexing a seed split); falls back to
    per-text calls for the openai backend.
    """
    provider = provider or config.EMBEDDING_PROVIDER
    model = model or config.EMBEDDING_MODEL

    if not texts:
        return np.zeros((0, 0), dtype=np.float32)

    if provider == "local":
        st_model = _get_local_model(model)
        vecs = st_model.encode(list(texts), convert_to_numpy=True, normalize_embeddings=True)
        return np.asarray(vecs, dtype=np.float32)
    return np.asarray([get_embedding(t, provider=provider, model=model) for t in texts],
                      dtype=np.float32)


def cosine_similarity(a, b):
    """compute cosine similarity between two vectors."""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm == 0:
        return 0.0
    return float(dot / norm)


def cosine_similarity_matrix(query_vec, stored_vecs):
    """compute cosine similarity between a query vector and a matrix of stored vectors.

    args:
        query_vec: shape (dim,)
        stored_vecs: shape (n, dim)

    returns: array of shape (n,) with similarity scores
    """
    query_vec = np.asarray(query_vec, dtype=np.float32)
    stored_vecs = np.asarray(stored_vecs, dtype=np.float32)

    if stored_vecs.ndim == 1:
        stored_vecs = stored_vecs.reshape(1, -1)

    dot_products = stored_vecs @ query_vec
    query_norm = np.linalg.norm(query_vec)
    stored_norms = np.linalg.norm(stored_vecs, axis=1)
    norms = query_norm * stored_norms
    norms[norms == 0] = 1e-10
    return dot_products / norms


# ---- local (sentence-transformers) ----

def _get_local_model(model_name):
    """lazy-load the sentence-transformer model (cached globally)."""
    global _LOCAL_MODEL
    if _LOCAL_MODEL is None or _LOCAL_MODEL._model_name != model_name:
        from sentence_transformers import SentenceTransformer
        logger.info(f"loading sentence-transformer model: {model_name}")
        _LOCAL_MODEL = SentenceTransformer(model_name)
        _LOCAL_MODEL._model_name = model_name
    return _LOCAL_MODEL


def _get_local_embedding(text, model_name):
    """compute embedding using sentence-transformers."""
    model = _get_local_model(model_name)
    vec = model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
    return vec


# ---- openai ----

def _get_openai_embedding(text, model_name):
    """compute embedding using OpenAI API."""
    from openai import OpenAI
    client = OpenAI(api_key=config.OPENAI_API_KEY)
    resp = client.embeddings.create(model=model_name, input=text)
    return np.array(resp.data[0].embedding, dtype=np.float32)
