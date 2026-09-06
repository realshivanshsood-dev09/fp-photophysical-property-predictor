"""Frozen, content-addressed ESM-2 sequence embeddings for V3."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import platform
from typing import Iterable

import numpy as np

from fp_predictor.clean import normalize_sequence


class EmbeddingError(RuntimeError):
    """Frozen embedding generation cannot safely proceed."""


@dataclass(frozen=True)
class EmbeddingSpec:
    """The complete, immutable definition of one V3 embedding representation."""

    model_name: str
    revision: str
    pooling: str = "final_hidden_state_residue_mean"
    embedding_dimension: int = 480
    batch_size: int = 4
    device: str = "auto"


class ESM2Featurizer:
    """Generate frozen ESM-2 embeddings without observing any target values.

    Cache keys include both the normalized sequence and every representation
    setting. The cache can therefore be shared across CV folds without fitting
    any transform on held-out observations.
    """

    def __init__(self, spec: EmbeddingSpec, cache_dir: str | Path) -> None:
        if spec.pooling != "final_hidden_state_residue_mean":
            raise EmbeddingError("V3 permits final-layer residue-token mean pooling only.")
        if spec.embedding_dimension != 480:
            raise EmbeddingError("V3 requires the 480-dimensional ESM2 t12 representation.")
        if spec.batch_size < 1:
            raise EmbeddingError("Embedding batch_size must be positive.")
        self.spec = spec
        self.cache_dir = Path(cache_dir)
        self._cache_hits = 0
        self._cache_misses = 0
        self._unique_sequences: set[str] = set()
        self._model_metadata: dict | None = None

    @staticmethod
    def _sha256_bytes(value: bytes) -> str:
        return hashlib.sha256(value).hexdigest()

    @staticmethod
    def _hash_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def _cache_key(self, sequence: str) -> str:
        payload = {"sequence": sequence, **asdict(self.spec)}
        return self._sha256_bytes(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8"))

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / self.spec.revision / key[:2] / f"{key}.npy"

    def _load_cached(self, key: str) -> np.ndarray | None:
        path = self._cache_path(key)
        if not path.exists():
            return None
        try:
            vector = np.load(path, allow_pickle=False)
        except (OSError, ValueError) as error:
            raise EmbeddingError(f"Cannot read embedding cache entry {path}: {error}") from error
        if vector.shape != (self.spec.embedding_dimension,) or not np.isfinite(vector).all():
            raise EmbeddingError(f"Invalid embedding cache entry {path}.")
        self._cache_hits += 1
        return np.asarray(vector, dtype=np.float32)

    def _store_cached(self, key: str, vector: np.ndarray) -> None:
        path = self._cache_path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp.npy")
        np.save(temporary, np.asarray(vector, dtype=np.float32), allow_pickle=False)
        temporary.replace(path)

    def _load_model(self):
        try:
            import huggingface_hub
            import torch
            import transformers
        except ImportError as error:  # keeps V1/V2 usable without optional V3 dependencies
            raise EmbeddingError(
                "V3 requires optional dependencies torch, transformers, and huggingface_hub. "
                "Install fp-photophysics-predictor[v3]."
            ) from error

        snapshot = Path(huggingface_hub.snapshot_download(
            repo_id=self.spec.model_name, revision=self.spec.revision, repo_type="model",
        ))
        if snapshot.name != self.spec.revision:
            raise EmbeddingError(
                f"Downloaded model snapshot {snapshot.name!r} does not match pinned revision {self.spec.revision!r}."
            )
        tokenizer = transformers.AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
        model = transformers.AutoModel.from_pretrained(snapshot, local_files_only=True)
        hidden_size = int(getattr(model.config, "hidden_size", -1))
        if hidden_size != self.spec.embedding_dimension:
            raise EmbeddingError(
                f"Pinned model exposes hidden_size={hidden_size}, expected {self.spec.embedding_dimension}."
            )
        max_positions = int(getattr(model.config, "max_position_embeddings", 0))
        if max_positions <= 2:
            raise EmbeddingError("Pinned model does not expose a usable maximum sequence length.")
        requested_device = self.spec.device
        if requested_device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        elif requested_device in {"cpu", "cuda"}:
            if requested_device == "cuda" and not torch.cuda.is_available():
                raise EmbeddingError("V3 requested CUDA but no CUDA device is available.")
            device = requested_device
        else:
            raise EmbeddingError("V3 device must be auto, cpu, or cuda.")
        model.to(device)
        model.eval()
        files = {
            str(path.relative_to(snapshot)).replace("\\", "/"): self._hash_file(path)
            for path in sorted(snapshot.rglob("*")) if path.is_file()
        }
        self._model_metadata = {
            "model_name": self.spec.model_name,
            "revision": self.spec.revision,
            "snapshot_path": str(snapshot),
            "model_class": type(model).__name__,
            "tokenizer_class": type(tokenizer).__name__,
            "hidden_size": hidden_size,
            "max_position_embeddings": max_positions,
            "pooling": self.spec.pooling,
            "embedding_dimension": self.spec.embedding_dimension,
            "device": device,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "huggingface_hub": huggingface_hub.__version__,
            "model_file_sha256": files,
            "platform": platform.platform(),
        }
        return tokenizer, model, torch, device, max_positions

    def transform(self, sequences: Iterable[str]) -> np.ndarray:
        """Return one 480-dimensional frozen vector per input sequence."""
        normalized = [normalize_sequence(sequence) for sequence in sequences]
        self._unique_sequences.update(normalized)
        keys = [self._cache_key(sequence) for sequence in normalized]
        vectors: dict[str, np.ndarray] = {}
        missing: dict[str, str] = {}
        for sequence, key in zip(normalized, keys, strict=True):
            if key in vectors:
                continue
            cached = self._load_cached(key)
            if cached is None:
                missing[key] = sequence
            else:
                vectors[key] = cached

        if missing:
            tokenizer, model, torch, device, max_positions = self._load_model()
            for sequence in missing.values():
                if len(sequence) + 2 > max_positions:
                    raise EmbeddingError(
                        f"Sequence length {len(sequence)} exceeds the pinned model limit of {max_positions - 2} residues; "
                        "V3 never truncates sequences."
                    )
            missing_items = list(missing.items())
            with torch.inference_mode():
                for offset in range(0, len(missing_items), self.spec.batch_size):
                    batch = missing_items[offset:offset + self.spec.batch_size]
                    encoded = tokenizer(
                        [sequence for _, sequence in batch], add_special_tokens=True, padding=True,
                        truncation=False, return_tensors="pt", return_special_tokens_mask=True,
                    )
                    special_mask = encoded.pop("special_tokens_mask").bool()
                    encoded = {name: value.to(device) for name, value in encoded.items()}
                    output = model(**encoded).last_hidden_state
                    residue_mask = encoded["attention_mask"].bool() & ~special_mask.to(device)
                    counts = residue_mask.sum(dim=1)
                    if bool((counts == 0).any()):
                        raise EmbeddingError("Tokenizer produced an empty residue-token mask.")
                    pooled = (output * residue_mask.unsqueeze(-1)).sum(dim=1) / counts.unsqueeze(-1)
                    for (key, _), vector in zip(batch, pooled.detach().cpu().numpy(), strict=True):
                        vector = np.asarray(vector, dtype=np.float32)
                        if vector.shape != (self.spec.embedding_dimension,) or not np.isfinite(vector).all():
                            raise EmbeddingError("ESM-2 produced a non-finite or incorrectly sized embedding.")
                        self._store_cached(key, vector)
                        vectors[key] = vector
                        self._cache_misses += 1
        elif self._model_metadata is None:
            # Provenance must describe the actual pinned model even for a warm cache.
            tokenizer, model, torch, device, max_positions = self._load_model()
            del tokenizer, model, torch, device, max_positions

        return np.vstack([vectors[key] for key in keys]).astype(np.float32, copy=False)

    def provenance(self, sequences: Iterable[str]) -> dict:
        if self._model_metadata is None:
            raise EmbeddingError("Embedding provenance is unavailable before transform().")
        entries = []
        for sequence in sorted(set(normalize_sequence(item) for item in sequences)):
            key = self._cache_key(sequence)
            path = self._cache_path(key)
            entries.append({
                "sequence_sha256": self._sha256_bytes(sequence.encode("utf-8")),
                "cache_key_sha256": key,
                "embedding_sha256": self._hash_file(path),
                "cache_path": str(path),
            })
        return {
            "spec": asdict(self.spec),
            "model": self._model_metadata,
            "cache": {
                "cache_dir": str(self.cache_dir),
                "hits": self._cache_hits,
                "misses": self._cache_misses,
                "unique_sequences": len(self._unique_sequences),
                "entries": entries,
            },
        }
