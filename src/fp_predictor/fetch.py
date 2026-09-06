"""Supported FPbase GraphQL acquisition with immutable, reproducible caching."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import requests

FPBASE_GRAPHQL_URL = "https://www.fpbase.org/graphql/"
RAW_SCHEMA_VERSION = 1

# This query was checked against the public GraphQL introspection schema on
# 2026-09-06.  The state selected by `defaultState` is FPbase's server-defined
# default, not a brightness-based choice made by this project.
PROTEIN_QUERY = """
query FPProteins($after: String, $first: Int!) {
  allProteins(after: $after, first: $first) {
    pageInfo { hasNextPage endCursor }
    edges {
      cursor
      node {
        id uuid name slug baseName aliases chromophore seq seqValidated
        agg cofactor switchType
        parentOrganism { id scientificName }
        primaryReference { doi year }
        references { doi year }
        defaultState {
          id name slug brightness extCoeff qy exMax emMax lifetime maturation pka
          isDark sourceMap
        }
        states {
          id name slug brightness extCoeff qy exMax emMax lifetime maturation pka
          isDark sourceMap
        }
      }
    }
  }
}
""".strip()


class FPbaseFetchError(RuntimeError):
    """FPbase did not provide a complete, usable response."""


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _extract_nodes(page: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        connection = page["data"]["allProteins"]
        edges = connection["edges"]
    except (KeyError, TypeError) as error:
        raise FPbaseFetchError("Response lacks data.allProteins.edges; schema may have changed.") from error
    if not isinstance(edges, list) or any(not isinstance(edge.get("node"), dict) for edge in edges):
        raise FPbaseFetchError("FPbase returned malformed protein edges.")
    return [edge["node"] for edge in edges]


class FPbaseClient:
    """Tiny raw GraphQL client kept separate from downstream transformations."""

    def __init__(
        self,
        endpoint: str = FPBASE_GRAPHQL_URL,
        session: requests.Session | None = None,
        timeout_seconds: int = 45,
    ) -> None:
        self.endpoint = endpoint
        self.session = session or requests.Session()
        self.timeout_seconds = timeout_seconds

    def fetch_all(self, page_size: int = 100) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        if not 1 <= page_size <= 500:
            raise ValueError("page_size must be between 1 and 500.")
        after: str | None = None
        pages: list[dict[str, Any]] = []
        nodes: list[dict[str, Any]] = []
        seen_cursors: set[str] = set()
        seen_ids: set[str] = set()

        while True:
            try:
                response = self.session.post(
                    self.endpoint,
                    json={"query": PROTEIN_QUERY, "variables": {"after": after, "first": page_size}},
                    timeout=self.timeout_seconds,
                )
                response.raise_for_status()
                payload = response.json()
            except requests.RequestException as error:
                raise FPbaseFetchError(f"Could not retrieve FPbase GraphQL data: {error}") from error
            except ValueError as error:
                raise FPbaseFetchError("FPbase returned non-JSON data.") from error
            if payload.get("errors"):
                message = json.dumps(payload["errors"], ensure_ascii=False)
                raise FPbaseFetchError(f"FPbase GraphQL returned errors: {message}")

            page_nodes = _extract_nodes(payload)
            for node in page_nodes:
                identifier = str(node.get("id") or node.get("uuid") or "")
                if not identifier:
                    raise FPbaseFetchError("A FPbase protein record has no identifier.")
                if identifier in seen_ids:
                    raise FPbaseFetchError(f"Duplicate protein identifier across pages: {identifier}")
                seen_ids.add(identifier)
            pages.append(payload)
            nodes.extend(page_nodes)

            page_info = payload["data"]["allProteins"].get("pageInfo")
            if not isinstance(page_info, dict) or "hasNextPage" not in page_info:
                raise FPbaseFetchError("FPbase response lacks pageInfo.hasNextPage; refusing partial data.")
            if not page_info["hasNextPage"]:
                break
            next_cursor = page_info.get("endCursor")
            if not next_cursor or next_cursor in seen_cursors:
                raise FPbaseFetchError("FPbase pagination cursor is missing or repeated; refusing partial data.")
            seen_cursors.add(next_cursor)
            after = str(next_cursor)
        return nodes, pages


def find_latest_raw(raw_dir: str | Path) -> Path:
    candidates = sorted(Path(raw_dir).glob("fpbase_*.json"))
    if not candidates:
        raise FileNotFoundError(f"No cached FPbase raw file found in {raw_dir}.")
    return candidates[-1]


def fetch_to_cache(
    raw_dir: str | Path = "data/raw",
    force: bool = False,
    page_size: int = 100,
    client: FPbaseClient | None = None,
) -> Path:
    """Fetch every page once, then save it under a timestamped immutable filename."""
    destination = Path(raw_dir)
    destination.mkdir(parents=True, exist_ok=True)
    if not force:
        existing = sorted(destination.glob("fpbase_*.json"))
        if existing:
            return existing[-1]

    client = client or FPbaseClient()
    nodes, pages = client.fetch_all(page_size=page_size)
    if not nodes:
        raise FPbaseFetchError("FPbase returned zero proteins; refusing to cache an empty dataset.")
    retrieved_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    payload = {
        "raw_schema_version": RAW_SCHEMA_VERSION,
        "source": {
            "provider": "FPbase",
            "transport": "GraphQL",
            "endpoint": client.endpoint,
            "retrieved_at": retrieved_at,
            "page_size": page_size,
            "query_sha256": hashlib.sha256(PROTEIN_QUERY.encode("utf-8")).hexdigest(),
            "record_count": len(nodes),
        },
        "pages": pages,
    }
    # Hash a canonical representation before adding the hash field itself;
    # `load_raw_records` additionally records the hash of the complete file.
    payload["source"]["content_sha256_excluding_self"] = _sha256_json(payload)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = destination / f"fpbase_{stamp}.json"
    suffix = 1
    while output.exists():
        output = destination / f"fpbase_{stamp}_{suffix}.json"
        suffix += 1
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return output


def load_raw_records(path: str | Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Read an immutable cached payload while retaining source metadata."""
    source_path = Path(path)
    try:
        payload = json.loads(source_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise FPbaseFetchError(f"Cannot read raw FPbase payload {source_path}: {error}") from error
    if isinstance(payload, list):  # convenient, transparent support for small fixtures
        records = payload
        metadata: dict[str, Any] = {"source_path": str(source_path), "fixture_or_legacy": True}
    elif isinstance(payload, dict) and isinstance(payload.get("pages"), list):
        records = [node for page in payload["pages"] for node in _extract_nodes(page)]
        metadata = dict(payload.get("source", {}))
        # Snapshots written before schema version 1's hash-field rename used
        # this value for the same canonical, self-excluding content hash.
        if "payload_sha256" in metadata and "content_sha256_excluding_self" not in metadata:
            metadata["content_sha256_excluding_self"] = metadata.pop("payload_sha256")
        metadata["source_path"] = str(source_path)
        metadata["raw_payload_sha256"] = _sha256_json(payload)
    elif isinstance(payload, dict) and isinstance(payload.get("records"), list):
        records = payload["records"]
        metadata = dict(payload.get("source", {}))
        metadata["source_path"] = str(source_path)
    else:
        raise FPbaseFetchError("Raw payload has neither paginated pages nor a records list.")
    if not all(isinstance(record, dict) for record in records):
        raise FPbaseFetchError("Raw payload contains a non-object protein record.")
    return records, metadata
