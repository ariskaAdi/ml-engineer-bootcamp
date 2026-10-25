"""Kandidat opsional: sentence embedding (e5-small) sebagai fitur. Hanya dipakai dengan --with-embeddings."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class E5Vectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, model_name: str = "intfloat/multilingual-e5-small", batch_size: int = 64):
        self.model_name = model_name
        self.batch_size = batch_size

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if not hasattr(self, "_model"):
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        texts = ["query: " + t for t in list(X)]
        return np.asarray(self._model.encode(texts, batch_size=self.batch_size, normalize_embeddings=True))
