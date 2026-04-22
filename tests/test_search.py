"""Tests for the search path: _run_search() and search() output modes."""

import json
import sys
import pytest

import gemini_search


def _make_mock_response(text="Answer text", search_queries=None, sources=None):
    """Build a minimal mock of the generate_content response."""
    search_queries = search_queries or ["test query"]
    sources = sources or [{"title": "Source One", "url": "https://example.com"}]

    # Use plain objects — no need to import pytest_mock types here.
    class FakeWeb:
        def __init__(self, title, uri):
            self.title = title
            self.uri = uri

    class FakeChunk:
        def __init__(self, title, uri):
            self.web = FakeWeb(title, uri)

    class FakeGroundingMeta:
        def __init__(self, queries, chunks):
            self.web_search_queries = queries
            self.grounding_chunks = chunks

    class FakeCandidate:
        def __init__(self, grounding_metadata):
            self.grounding_metadata = grounding_metadata

    class FakeResponse:
        def __init__(self, text, candidates):
            self.text = text
            self.candidates = candidates

    chunks = [FakeChunk(s["title"], s["url"]) for s in sources]
    grounding_meta = FakeGroundingMeta(search_queries, chunks)
    candidate = FakeCandidate(grounding_meta)
    return FakeResponse(text, [candidate])


def test_run_search_json_shape(mocker):
    """_run_search returns dict with expected keys."""
    mock_client = mocker.MagicMock()
    mock_client.models.generate_content.return_value = _make_mock_response(
        text="Deep answer",
        search_queries=["q1"],
        sources=[{"title": "T1", "url": "https://t1.com"}],
    )

    result = gemini_search._run_search("test query", "gemini-3-flash-preview", mock_client)

    assert result["query"] == "test query"
    assert result["model"] == "gemini-3-flash-preview"
    assert result["answer"] == "Deep answer"
    assert result["search_queries_used"] == ["q1"]
    assert len(result["sources"]) == 1
    assert result["sources"][0]["title"] == "T1"
    assert result["sources"][0]["url"] == "https://t1.com"


def test_run_search_empty_grounding(mocker):
    """_run_search handles response with no grounding metadata."""
    class FakeCandidate:
        pass  # no grounding_metadata attribute

    class FakeResponse:
        text = "Plain answer"
        candidates = [FakeCandidate()]

    mock_client = mocker.MagicMock()
    mock_client.models.generate_content.return_value = FakeResponse()

    result = gemini_search._run_search("query", "model", mock_client)

    assert result["answer"] == "Plain answer"
    assert result["search_queries_used"] == []
    assert result["sources"] == []


def test_search_json_output(mocker, capsys):
    """search() with as_json=True prints valid JSON with correct shape."""
    mocker.patch("gemini_search.get_api_key", return_value="fake-key")
    mock_client = mocker.MagicMock()
    mock_client.models.generate_content.return_value = _make_mock_response(
        text="Answer", search_queries=["q"], sources=[{"title": "S", "url": "https://s.com"}]
    )
    mocker.patch("gemini_search._make_client", return_value=mock_client)

    gemini_search.search("test", model="gemini-3-flash-preview", as_json=True)

    captured = capsys.readouterr()
    output = json.loads(captured.out)
    assert output["query"] == "test"
    assert output["model"] == "gemini-3-flash-preview"
    assert "search_queries_used" in output
    assert "answer" in output
    assert "sources" in output


def test_search_raw_urls_output(mocker, capsys):
    """search() with raw_urls=True prints sources without synthesized answer."""
    mocker.patch("gemini_search.get_api_key", return_value="fake-key")
    mock_client = mocker.MagicMock()
    mock_client.models.generate_content.return_value = _make_mock_response(
        text="Should not appear",
        sources=[{"title": "Src", "url": "https://src.com"}],
    )
    mocker.patch("gemini_search._make_client", return_value=mock_client)

    gemini_search.search("q", raw_urls=True)

    captured = capsys.readouterr()
    assert "=== SOURCES ===" in captured.out
    assert "https://src.com" in captured.out
    assert "ANSWER" not in captured.out


def test_search_full_output(mocker, capsys):
    """search() default text mode prints answer and sources."""
    mocker.patch("gemini_search.get_api_key", return_value="fake-key")
    mock_client = mocker.MagicMock()
    mock_client.models.generate_content.return_value = _make_mock_response(
        text="Full answer text",
        sources=[{"title": "FullSrc", "url": "https://full.com"}],
    )
    mocker.patch("gemini_search._make_client", return_value=mock_client)

    gemini_search.search("query")

    captured = capsys.readouterr()
    assert "=== ANSWER (grounded in Google Search) ===" in captured.out
    assert "Full answer text" in captured.out
    assert "=== SOURCES ===" in captured.out


def test_get_api_key_missing(monkeypatch):
    """get_api_key() calls sys.exit(1) when env var is absent."""
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(SystemExit) as exc:
        gemini_search.get_api_key()
    assert exc.value.code == 1
