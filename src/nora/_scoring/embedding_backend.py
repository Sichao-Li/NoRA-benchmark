from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import hashlib
from pathlib import Path
import re
from typing import Sequence

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DEFAULT_EMBEDDING_CACHE_ROOT = ".reason_eval_embedding_cache"


def _clip01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def _slugify_cache_component(value: str) -> str:
    collapsed = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    normalized = collapsed.strip("._-")
    return normalized or "default"


class SimilarityBackend(ABC):
    """Abstract backend for text embeddings."""

    @abstractmethod
    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Encode texts into a 2D array of vectors."""

    def cache_namespace(self) -> str:
        """Stable namespace used to scope cache directories for this backend."""

        return _slugify_cache_component(type(self).__name__)

    def with_cache(
        self,
        cache_root: str | Path = DEFAULT_EMBEDDING_CACHE_ROOT,
        namespace: str | None = None,
    ) -> CachedSimilarityBackend:
        """Wrap this backend with a reusable on-disk embedding cache."""

        return CachedSimilarityBackend(base=self, cache_root=cache_root, namespace=namespace)

    def similarity_matrix(self, pred_texts: Sequence[str], gold_texts: Sequence[str]) -> np.ndarray:
        """Compute a cosine-based similarity matrix rescaled to [0, 1]."""

        if not pred_texts or not gold_texts:
            return np.zeros((len(pred_texts), len(gold_texts)), dtype=float)
        pred_vecs = self.encode(pred_texts)
        gold_vecs = self.encode(gold_texts)
        raw = cosine_similarity(pred_vecs, gold_vecs)
        return np.clip((raw + 1.0) / 2.0, 0.0, 1.0)

    def similarity(self, text_a: str, text_b: str) -> float:
        """Compute a single pairwise similarity in [0, 1]."""

        return _clip01(self.similarity_matrix([text_a], [text_b])[0, 0])


@dataclass(slots=True)
class _HashingFallbackBackend(SimilarityBackend):
    n_features: int = 512

    def __post_init__(self) -> None:
        self._vectorizer = HashingVectorizer(
            alternate_sign=False,
            n_features=self.n_features,
            norm="l2",
            ngram_range=(1, 2),
        )

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        matrix = self._vectorizer.transform(texts)
        return matrix.toarray().astype(float)

    def cache_namespace(self) -> str:
        return f"hashing-fallback__{self.n_features}"


@dataclass(slots=True)
class SentenceTransformerBackend(SimilarityBackend):
    """Sentence-transformers backend with deterministic local fallback."""

    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    local_files_only: bool = True
    allow_fallback: bool = True

    def __post_init__(self) -> None:
        self._model = None
        self._fallback = _HashingFallbackBackend()
        self.using_fallback = False
        self.using_cross_encoder = self.model_name.startswith("cross-encoder/")

    def cache_namespace(self) -> str:
        prefix = "cross-encoder" if self.using_cross_encoder else "sentence-transformer"
        if not self.allow_fallback:
            return f"{prefix}__{_slugify_cache_component(self.model_name)}"
        # Resolve the concrete operating mode before picking a cache namespace so
        # we never mix real model embeddings with fallback vectors.
        self._load_model()
        if self.using_fallback:
            return self._fallback.cache_namespace()
        return f"{prefix}__{_slugify_cache_component(self.model_name)}"

    def _load_model(self) -> object | None:
        if self._model is not None or self.using_fallback:
            return self._model
        try:
            if self.using_cross_encoder:
                from sentence_transformers import CrossEncoder

                self._model = CrossEncoder(self.model_name, local_files_only=self.local_files_only)
            else:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(self.model_name, local_files_only=self.local_files_only)
        except Exception:
            if not self.allow_fallback:
                raise
            self.using_fallback = True
            self._model = None
        return self._model

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        model = self._load_model()
        if model is None:
            return self._fallback.encode(texts)
        if self.using_cross_encoder:
            raise NotImplementedError(
                "CrossEncoder models do not expose reusable embeddings. "
                "Call similarity(...) or similarity_matrix(...) instead."
            )
        encoded = model.encode(list(texts), convert_to_numpy=True, normalize_embeddings=True)
        if encoded.ndim == 1:
            encoded = encoded.reshape(1, -1)
        return encoded.astype(float)

    def similarity_matrix(self, pred_texts: Sequence[str], gold_texts: Sequence[str]) -> np.ndarray:
        if not pred_texts or not gold_texts:
            return np.zeros((len(pred_texts), len(gold_texts)), dtype=float)
        if not self.using_cross_encoder:
            return SimilarityBackend.similarity_matrix(self, pred_texts, gold_texts)

        model = self._load_model()
        if model is None:
            return self._fallback.similarity_matrix(pred_texts, gold_texts)

        pairs = [(pred_text, gold_text) for pred_text in pred_texts for gold_text in gold_texts]
        raw_scores = np.asarray(model.predict(pairs), dtype=float).reshape(len(pred_texts), len(gold_texts))
        return self._normalize_cross_encoder_scores(raw_scores)

    def similarity(self, text_a: str, text_b: str) -> float:
        if not self.using_cross_encoder:
            return SimilarityBackend.similarity(self, text_a, text_b)
        return _clip01(self.similarity_matrix([text_a], [text_b])[0, 0])

    @staticmethod
    def _normalize_cross_encoder_scores(raw_scores: np.ndarray) -> np.ndarray:
        if raw_scores.size == 0:
            return raw_scores.astype(float)
        if np.all((0.0 <= raw_scores) & (raw_scores <= 1.0)):
            return np.clip(raw_scores, 0.0, 1.0)
        if np.all((0.0 <= raw_scores) & (raw_scores <= 5.0)):
            return np.clip(raw_scores / 5.0, 0.0, 1.0)
        return 1.0 / (1.0 + np.exp(-raw_scores))


@dataclass(slots=True)
class CachedSimilarityBackend(SimilarityBackend):
    """Backend wrapper that reuses embeddings across runs and comparisons.

    The cache root stores one subdirectory per backend/model namespace. For
    sentence-transformer embeddings this lets a single backend instance be
    shared across many prediction sets while keeping disk caches isolated per
    embedding model.
    """

    base: SimilarityBackend
    cache_root: str | Path = DEFAULT_EMBEDDING_CACHE_ROOT
    namespace: str | None = None
    persist_embeddings: bool = True
    _embedding_mem: dict[str, np.ndarray] = field(default_factory=dict, init=False, repr=False)
    _pair_mem: dict[tuple[str, str], float] = field(default_factory=dict, init=False, repr=False)
    _encode_supported: bool | None = field(default=None, init=False, repr=False)

    def cache_namespace(self) -> str:
        return _slugify_cache_component(self.namespace or self.base.cache_namespace())

    @property
    def cache_dir(self) -> Path:
        cache_dir = Path(self.cache_root) / self.cache_namespace()
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir

    def _embeddings_dir(self) -> Path:
        path = self.cache_dir / "embeddings"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def _text_key(text: str) -> str:
        return text

    def _embedding_path(self, text: str) -> Path:
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return self._embeddings_dir() / f"{digest}.npy"

    def _load_cached_embedding(self, text: str) -> np.ndarray | None:
        key = self._text_key(text)
        if key in self._embedding_mem:
            return self._embedding_mem[key]
        path = self._embedding_path(text)
        if not self.persist_embeddings or not path.exists():
            return None
        vector = np.load(path, allow_pickle=False).astype(float)
        self._embedding_mem[key] = vector
        return vector

    def _save_cached_embedding(self, text: str, vector: np.ndarray) -> None:
        key = self._text_key(text)
        normalized = np.asarray(vector, dtype=float)
        self._embedding_mem[key] = normalized
        if self.persist_embeddings:
            np.save(self._embedding_path(text), normalized)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        text_list = list(texts)
        if not text_list:
            return np.empty((0, 0), dtype=float)
        if self._encode_supported is False:
            raise NotImplementedError("Base backend does not expose reusable embeddings.")

        cached = [self._load_cached_embedding(text) for text in text_list]
        missing_texts = list(dict.fromkeys(text for text, vector in zip(text_list, cached) if vector is None))
        if missing_texts:
            try:
                encoded_missing = np.asarray(self.base.encode(missing_texts), dtype=float)
            except NotImplementedError:
                self._encode_supported = False
                raise
            if encoded_missing.ndim == 1:
                encoded_missing = encoded_missing.reshape(1, -1)
            for text, vector in zip(missing_texts, encoded_missing):
                self._save_cached_embedding(text, vector)
            cached = [self._load_cached_embedding(text) for text in text_list]

        self._encode_supported = True
        return np.vstack(cached)

    def similarity_matrix(self, pred_texts: Sequence[str], gold_texts: Sequence[str]) -> np.ndarray:
        if not pred_texts or not gold_texts:
            return np.zeros((len(pred_texts), len(gold_texts)), dtype=float)

        if self._encode_supported is not False:
            try:
                pred_vecs = self.encode(pred_texts)
                gold_vecs = self.encode(gold_texts)
            except NotImplementedError:
                self._encode_supported = False
            else:
                raw = cosine_similarity(pred_vecs, gold_vecs)
                return np.clip((raw + 1.0) / 2.0, 0.0, 1.0)

        matrix = np.zeros((len(pred_texts), len(gold_texts)), dtype=float)
        for row, pred_text in enumerate(pred_texts):
            for col, gold_text in enumerate(gold_texts):
                key = (pred_text, gold_text)
                if key not in self._pair_mem:
                    self._pair_mem[key] = _clip01(self.base.similarity(pred_text, gold_text))
                matrix[row, col] = self._pair_mem[key]
        return matrix

    def similarity(self, text_a: str, text_b: str) -> float:
        return _clip01(self.similarity_matrix([text_a], [text_b])[0, 0])


def build_sentence_transformer_backend(
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    *,
    cache_root: str | Path | None = None,
    local_files_only: bool = True,
    allow_fallback: bool = True,
) -> SimilarityBackend:
    """Build a sentence-transformer backend with optional model-scoped caching."""

    backend: SimilarityBackend = SentenceTransformerBackend(
        model_name=model_name,
        local_files_only=local_files_only,
        allow_fallback=allow_fallback,
    )
    if cache_root is None:
        return backend
    return CachedSimilarityBackend(base=backend, cache_root=cache_root)
