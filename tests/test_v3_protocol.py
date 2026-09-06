from __future__ import annotations

import numpy as np
import pytest

from fp_predictor.config import DEFAULT_V3_CONFIG, validate_v3_config
from fp_predictor.features.embeddings import ESM2Featurizer, EmbeddingSpec


def _v3_config():
    return {
        **DEFAULT_V3_CONFIG,
        "features": dict(DEFAULT_V3_CONFIG["features"]),
        "estimator": dict(DEFAULT_V3_CONFIG["estimator"]),
        "split": dict(DEFAULT_V3_CONFIG["split"]),
        "frozen_v2b": dict(DEFAULT_V3_CONFIG["frozen_v2b"]),
        "models": list(DEFAULT_V3_CONFIG["models"]),
    }


def test_v3_configuration_is_pinned_to_one_representation_and_estimator():
    config = _v3_config()
    assert validate_v3_config(config)["features"]["embedding_dimension"] == 480
    config["features"]["pooling"] = "cls_token"
    with pytest.raises(ValueError, match="pinned ESM-2"):
        validate_v3_config(config)

    config = _v3_config()
    config["estimator"]["alpha"] = 10.0
    with pytest.raises(ValueError, match="Ridge"):
        validate_v3_config(config)


def test_v3_embedding_cache_is_content_addressed_and_dimension_checked(tmp_path):
    spec = EmbeddingSpec(
        model_name="facebook/esm2_t12_35M_UR50D",
        revision="6fbf070e65b0b7291e7bbcd451118c216cff79d8",
    )
    featurizer = ESM2Featurizer(spec, tmp_path)
    key = featurizer._cache_key("ACDE")
    featurizer._store_cached(key, np.ones(480, dtype=np.float32))
    loaded = featurizer._load_cached(key)
    assert loaded is not None and loaded.shape == (480,)
    assert featurizer._cache_key("ACDE") != featurizer._cache_key("ACDF")
    assert featurizer._cache_key("ACDE") != ESM2Featurizer(
        EmbeddingSpec(model_name=spec.model_name, revision=spec.revision, batch_size=8), tmp_path
    )._cache_key("ACDE")
