"""Reserved Phase-2 interface; v1 deliberately does not ship an ESM model."""

from __future__ import annotations


class ESM2Featurizer:
    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError(
            "ESM2 embeddings are intentionally deferred to Phase 2; v1 uses composition features only."
        )
