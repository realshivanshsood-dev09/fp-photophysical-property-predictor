from __future__ import annotations

import pytest

from fp_predictor.fetch import FPbaseClient, FPbaseFetchError, load_raw_records


def _page(identifier: str, has_next: bool, cursor: str | None):
    return {"data": {"allProteins": {"pageInfo": {"hasNextPage": has_next, "endCursor": cursor}, "edges": [{"node": {"id": identifier}}]}}}


class Response:
    def __init__(self, payload): self.payload = payload
    def raise_for_status(self): pass
    def json(self): return self.payload


class Session:
    def __init__(self): self.calls = []
    def post(self, endpoint, json, timeout):
        self.calls.append(json)
        return Response(_page("one", True, "cursor-1") if len(self.calls) == 1 else _page("two", False, None))


def test_graphql_pagination_collects_all_pages():
    session = Session()
    records, pages = FPbaseClient(session=session).fetch_all(page_size=25)
    assert [record["id"] for record in records] == ["one", "two"]
    assert len(pages) == 2
    assert session.calls[1]["variables"]["after"] == "cursor-1"


def test_malformed_response_fails_instead_of_returning_partial_data():
    class BadSession:
        def post(self, *args, **kwargs):
            return Response({"data": {"allProteins": {"edges": []}}})

    with pytest.raises(FPbaseFetchError, match="pageInfo"):
        FPbaseClient(session=BadSession()).fetch_all()


def test_legacy_content_hash_is_exposed_with_accurate_name(tmp_path):
    path = tmp_path / "raw.json"
    path.write_text('{"source":{"payload_sha256":"abc"},"pages":[{"data":{"allProteins":{"edges":[]}}}]}')
    records, metadata = load_raw_records(path)
    assert records == []
    assert metadata["content_sha256_excluding_self"] == "abc"
    assert "payload_sha256" not in metadata
