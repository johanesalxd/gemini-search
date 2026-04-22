"""Tests for the search path: _run_search(), _build_search_contents(), and search() output modes."""

import json
import pathlib
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


# --- _build_search_contents tests ---

def test_build_search_contents_no_file():
    """_build_search_contents returns bare query string when no file given."""
    result = gemini_search._build_search_contents("my query", None)
    assert result == "my query"


def test_build_search_contents_text_file(tmp_path, mocker):
    """_build_search_contents returns [file_text, query] list for text files."""
    txt = tmp_path / "notes.txt"
    txt.write_text("This is note content.", encoding="utf-8")

    # Mock Part.from_bytes so we do not need the real SDK
    mocker.patch("google.genai.types.Part.from_bytes")

    result = gemini_search._build_search_contents("my query", str(txt))

    assert isinstance(result, list)
    assert len(result) == 2
    assert result[0] == "This is note content."
    assert result[1] == "my query"


def test_build_search_contents_markdown_file(tmp_path, mocker):
    """_build_search_contents handles .md files as text (inline string)."""
    md = tmp_path / "brief.md"
    md.write_text("# Brief\n\nContent here.", encoding="utf-8")

    mocker.patch("google.genai.types.Part.from_bytes")

    result = gemini_search._build_search_contents("summarize", str(md))

    assert isinstance(result, list)
    assert result[0] == "# Brief\n\nContent here."
    assert result[1] == "summarize"


def test_build_search_contents_pdf_file(tmp_path, mocker):
    """_build_search_contents uses Part.from_bytes for PDF files."""
    pdf = tmp_path / "report.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake pdf bytes")

    fake_part = object()
    mock_from_bytes = mocker.patch("google.genai.types.Part.from_bytes", return_value=fake_part)

    result = gemini_search._build_search_contents("summarize", str(pdf))

    mock_from_bytes.assert_called_once_with(
        data=b"%PDF-1.4 fake pdf bytes", mime_type="application/pdf"
    )
    assert isinstance(result, list)
    assert result[0] is fake_part
    assert result[1] == "summarize"


def test_detect_mime_unknown_file_defaults_to_octet_stream(tmp_path):
    """Unknown file types default to octet-stream, not text/plain."""
    blob = tmp_path / "payload.unknownbin"
    blob.write_bytes(b"\x00\x01\x02")

    assert gemini_search._detect_mime(blob) == "application/octet-stream"


def test_build_search_contents_missing_file():
    """_build_search_contents exits with error when file does not exist."""
    with pytest.raises(SystemExit) as exc:
        gemini_search._build_search_contents("query", "/nonexistent/path/file.txt")
    assert exc.value.code == 1


def test_run_search_passes_file_contents_to_api(tmp_path, mocker):
    """_run_search forwards --file contents to generate_content."""
    txt = tmp_path / "context.txt"
    txt.write_text("Context about dolphins.", encoding="utf-8")

    mock_client = mocker.MagicMock()
    mock_client.models.generate_content.return_value = _make_mock_response(
        text="Dolphin answer",
        search_queries=["dolphins"],
        sources=[],
    )
    # Stub Part.from_bytes (not needed for text files but defensive)
    mocker.patch("google.genai.types.Part.from_bytes")

    result = gemini_search._run_search("tell me about dolphins", "model", mock_client, file_path=str(txt))

    # The API was called with a list [file_text, query], not a bare string
    call_args = mock_client.models.generate_content.call_args
    assert call_args.kwargs["contents"] == ["Context about dolphins.", "tell me about dolphins"]
    assert result["query"] == "tell me about dolphins"
