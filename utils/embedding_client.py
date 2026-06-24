# utils/embedding_client.py
# Local embedding client — sentence-transformers, CPU only, offline.
# Used ONLY in pre-computation (Phase A). Never called during ranking (Phase B).
#
# Model: all-mpnet-base-v2  (768-dim, good semantic quality)
# Install: pip install sentence-transformers

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    """Lazy-load the model on first call."""
    global _model
    if _model is None:
        _model = SentenceTransformer("all-mpnet-base-v2")
    return _model


def get_embedding(text: str) -> np.ndarray:
    """Encode a single string → normalized 768-dim float32 vector."""
    return _get_model().encode(text, normalize_embeddings=True)


def get_batch_embeddings(texts: list[str], batch_size: int = 64) -> np.ndarray:
    """
    Encode a list of strings → (N, 768) float32 array.
    Normalized embeddings (unit vectors) for cosine similarity via dot product.
    """
    return _get_model().encode(
        texts,
        normalize_embeddings=True,
        batch_size=batch_size,
        show_progress_bar=True,
    )
