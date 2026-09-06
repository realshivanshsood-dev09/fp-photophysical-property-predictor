"""Swappable sequence-only feature sets."""

from .composition import CompositionFeaturizer
from .embeddings import ESM2Featurizer, EmbeddingError, EmbeddingSpec

__all__ = ["CompositionFeaturizer", "ESM2Featurizer", "EmbeddingError", "EmbeddingSpec"]
