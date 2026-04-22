"""Tests for _run_deep_research(), _build_dr_input(), and deep_research() output modes.

Mock objects reflect the real _interactions SDK discriminated-union output shape:
  - TextContent:               type="text", .text, .annotations
  - GoogleSearchResultContent: type="google_search_result", .result[]
Each item in interaction.outputs has a top-level `type` attribute; there is no
`.parts` nesting (that is the google.genai.types.Content shape, not the
interactions Content union).
"""

import json
import pathlib
import pytest

import gemini_search


def _make_mock_interaction(
    interaction_id="ia-123",
    status="completed",
    agent="deep-research-preview-04-2026",
    text="Research report text.",
    sources=None,
):
    """Build a minimal mock Interaction object matching the real SDK shape."""
    sources = sources or [{"title": "Src A", "url": "https://a.com"}]

    class FakeAnnotation:
        def __init__(self, source):
            self.source = source

    class FakeTextContent:
        type = "text"

        def __init__(self, text, annotations=None):
            self.text = text
            self.annotations = annotations or []

    class FakeGoogleSearchResult:
        def __init__(self, title, url):
            self.title = title
            self.url = url

    class FakeGoogleSearchResultContent:
        type = "google_search_result"

        def __init__(self, results):
            self.result = results

    text_content = FakeTextContent(text=text)
    search_results = FakeGoogleSearchResultContent(
        results=[FakeGoogleSearchResult(s["title"], s["url"]) for s in sources]
    )

    class FakeInteraction:
        pass

    obj = FakeInteraction()
    obj.id = interaction_id
    obj.status = status
    obj.outputs = [text_content, search_results]
    return obj


def test_run_deep_research_json_shape(mocker):
    """_run_deep_research returns dict with expected keys."""
    mock_client = mocker.MagicMock()
    mock_client.interactions.create.return_value = _make_mock_interaction(
        text="Research output",
        sources=[{"title": "T1", "url": "https://t1.com"}],
    )
    mock_client.interactions.get.return_value = mock_client.interactions.create.return_value

    result = gemini_search._run_deep_research(
        "deep query",
        "deep-research-preview-04-2026",
        mock_client,
    )

    assert result["query"] == "deep query"
    assert result["agent"] == "deep-research-preview-04-2026"
    assert result["interaction_id"] == "ia-123"
    assert result["status"] == "completed"
    assert result["answer"] == "Research output"
    assert len(result["sources"]) == 1
    assert result["sources"][0]["title"] == "T1"
    assert result["sources"][0]["url"] == "https://t1.com"


def test_run_deep_research_empty_outputs(mocker):
    """_run_deep_research handles interaction with no outputs."""
    class FakeInteraction:
        id = "ia-empty"
        status = "completed"
        outputs = None

    mock_client = mocker.MagicMock()
    mock_client.interactions.create.return_value = FakeInteraction()
    mock_client.interactions.get.return_value = FakeInteraction()

    result = gemini_search._run_deep_research("q", "agent-id", mock_client)

    assert result["answer"] == ""
    assert result["sources"] == []


def test_run_deep_research_text_only_no_search_results(mocker):
    """_run_deep_research handles TextContent-only outputs with no search results."""
    class FakeTextContent:
        type = "text"
        text = "Text-only report."
        annotations = []

    class FakeInteraction:
        id = "ia-textonly"
        status = "completed"
        outputs = [FakeTextContent()]

    mock_client = mocker.MagicMock()
    mock_client.interactions.create.return_value = FakeInteraction()
    mock_client.interactions.get.return_value = FakeInteraction()

    result = gemini_search._run_deep_research("q", "agent-id", mock_client)

    assert result["answer"] == "Text-only report."
    assert result["sources"] == []


def test_run_deep_research_extracts_url_citation_annotations(mocker):
    """_run_deep_research extracts url_citation annotations from TextContent.

    Docs: ai.google.dev/gemini-api/docs/interactions — TextContent.annotations,
    UrlCitation type (type="url_citation", .url, .title).
    """
    class FakeAnnotation:
        type = "url_citation"
        url = "https://cited.example.com"
        title = "Cited Page"

    class FakeTextContent:
        type = "text"
        text = "Research with inline citation."
        annotations = [FakeAnnotation()]

    class FakeInteraction:
        id = "ia-annot"
        status = "completed"
        outputs = [FakeTextContent()]

    mock_client = mocker.MagicMock()
    mock_client.interactions.create.return_value = FakeInteraction()
    mock_client.interactions.get.return_value = FakeInteraction()

    result = gemini_search._run_deep_research("q", "agent-id", mock_client)

    assert result["answer"] == "Research with inline citation."
    assert len(result["sources"]) == 1
    assert result["sources"][0]["url"] == "https://cited.example.com"
    assert result["sources"][0]["title"] == "Cited Page"


def test_run_deep_research_deduplicates_sources(mocker):
    """_run_deep_research deduplicates sources with identical URLs."""
    class FakeResult:
        def __init__(self, title, url):
            self.title = title
            self.url = url

    class FakeSearchContent:
        type = "google_search_result"
        result = [
            FakeResult("Page A", "https://a.com"),
            FakeResult("Page A duplicate", "https://a.com"),
            FakeResult("Page B", "https://b.com"),
        ]

    class FakeTextContent:
        type = "text"
        text = "report"
        annotations = []

    class FakeInteraction:
        id = "ia-dedup"
        status = "completed"
        outputs = [FakeTextContent(), FakeSearchContent()]

    mock_client = mocker.MagicMock()
    mock_client.interactions.create.return_value = FakeInteraction()
    mock_client.interactions.get.return_value = FakeInteraction()

    result = gemini_search._run_deep_research("q", "agent-id", mock_client)

    urls = [s["url"] for s in result["sources"]]
    assert urls.count("https://a.com") == 1
    assert "https://b.com" in urls
    assert len(result["sources"]) == 2


def test_deep_research_json_output(mocker, capsys):
    """deep_research() with as_json=True prints valid JSON with correct shape."""
    mocker.patch("gemini_search.get_api_key", return_value="fake-key")
    mock_client = mocker.MagicMock()
    mock_client.interactions.create.return_value = _make_mock_interaction(
        text="Report", sources=[{"title": "S", "url": "https://s.com"}]
    )
    mock_client.interactions.get.return_value = mock_client.interactions.create.return_value
    mocker.patch("gemini_search._make_client", return_value=mock_client)

    gemini_search.deep_research("topic", as_json=True)

    captured = capsys.readouterr()
    output = json.loads(captured.out)
    assert output["query"] == "topic"
    assert "agent" in output
    assert "interaction_id" in output
    assert "status" in output
    assert output["answer"] == "Report"
    assert len(output["sources"]) == 1
    assert output["sources"][0]["url"] == "https://s.com"
    # deep-research result must NOT contain search-specific key
    assert "search_queries_used" not in output


def test_deep_research_full_output(mocker, capsys):
    """deep_research() default text mode prints report header and sources."""
    mocker.patch("gemini_search.get_api_key", return_value="fake-key")
    mock_client = mocker.MagicMock()
    mock_client.interactions.create.return_value = _make_mock_interaction(
        text="Full research text",
        sources=[{"title": "Research Src", "url": "https://rsrc.com"}],
    )
    mock_client.interactions.get.return_value = mock_client.interactions.create.return_value
    mocker.patch("gemini_search._make_client", return_value=mock_client)

    gemini_search.deep_research("research topic")

    captured = capsys.readouterr()
    assert "=== DEEP RESEARCH REPORT ===" in captured.out
    assert "Full research text" in captured.out
    assert "=== SOURCES ===" in captured.out
    assert "Research Src" in captured.out


def test_deep_research_progress_notice(mocker, capsys):
    """deep_research() prints a progress notice to stderr before the call."""
    mocker.patch("gemini_search.get_api_key", return_value="fake-key")
    mock_client = mocker.MagicMock()
    mock_client.interactions.create.return_value = _make_mock_interaction()
    mock_client.interactions.get.return_value = mock_client.interactions.create.return_value
    mocker.patch("gemini_search._make_client", return_value=mock_client)

    gemini_search.deep_research("topic")

    captured = capsys.readouterr()
    assert "Deep Research in progress" in captured.err


def test_run_deep_research_extracts_url_citation_annotations_from_dicts(mocker):
    """_run_deep_research extracts sources from dict-form url_citation annotations.

    The Interactions API reference (ai.google.dev/api/interactions-api, TextContent)
    documents annotations as UrlCitation objects with type="url_citation", url, and
    title. The SDK returns them as dicts in practice (matching the canonical Python
    example in the URL context section of the interactions guide).
    """
    class FakeTextContent:
        type = "text"
        text = "Research report with cited sources."
        annotations = [
            {"type": "url_citation", "url": "https://cited.example.com", "title": "Cited Page"},
            {"type": "url_citation", "url": "https://second.example.com", "title": "Second Page"},
            # Non-url_citation annotations must be ignored
            {"type": "file_citation", "document_uri": "files/x", "file_name": "doc.pdf", "source": "snippet"},
        ]

    class FakeInteraction:
        id = "ia-annot"
        status = "completed"
        outputs = [FakeTextContent()]

    mock_client = mocker.MagicMock()
    mock_client.interactions.create.return_value = FakeInteraction()
    mock_client.interactions.get.return_value = FakeInteraction()

    result = gemini_search._run_deep_research("q", "agent-id", mock_client)

    assert result["answer"] == "Research report with cited sources."
    urls = [s["url"] for s in result["sources"]]
    assert "https://cited.example.com" in urls
    assert "https://second.example.com" in urls
    # file_citation must not appear (no url field)
    assert len(result["sources"]) == 2
    titles = {s["url"]: s["title"] for s in result["sources"]}
    assert titles["https://cited.example.com"] == "Cited Page"


def test_run_deep_research_deduplicates_annotation_and_search_result_urls(mocker):
    """Sources from url_citation annotations and google_search_result are deduplicated."""
    shared_url = "https://shared.example.com"

    class FakeTextContent:
        type = "text"
        text = "Report."
        annotations = [
            {"type": "url_citation", "url": shared_url, "title": "Shared via annotation"},
        ]

    class FakeSearchResult:
        def __init__(self, title, url):
            self.title = title
            self.url = url

    class FakeSearchContent:
        type = "google_search_result"
        result = [
            FakeSearchResult("Shared via search", shared_url),
            FakeSearchResult("Unique", "https://unique.example.com"),
        ]

    class FakeInteraction:
        id = "ia-dedup2"
        status = "completed"
        outputs = [FakeTextContent(), FakeSearchContent()]

    mock_client = mocker.MagicMock()
    mock_client.interactions.create.return_value = FakeInteraction()
    mock_client.interactions.get.return_value = FakeInteraction()

    result = gemini_search._run_deep_research("q", "agent-id", mock_client)

    urls = [s["url"] for s in result["sources"]]
    assert urls.count(shared_url) == 1
    assert "https://unique.example.com" in urls
    assert len(result["sources"]) == 2


# --- _build_dr_input tests ---

def test_build_dr_input_no_file():
    """_build_dr_input returns bare query string when no file given."""
    result = gemini_search._build_dr_input("my query", None)
    assert result == "my query"


def test_build_dr_input_text_file(tmp_path):
    """_build_dr_input prepends file content for text files."""
    txt = tmp_path / "brief.txt"
    txt.write_text("Research focus: quantum computing.", encoding="utf-8")

    result = gemini_search._build_dr_input("conduct research on this", str(txt))

    assert "[Document: brief.txt]" in result
    assert "Research focus: quantum computing." in result
    assert "conduct research on this" in result
    # File content comes before the query
    assert result.index("Research focus") < result.index("conduct research on this")


def test_build_dr_input_markdown_file(tmp_path):
    """_build_dr_input handles .md files as text."""
    md = tmp_path / "notes.md"
    md.write_text("# Notes\n\nFocus on CRISPR.", encoding="utf-8")

    result = gemini_search._build_dr_input("research this brief", str(md))

    assert "[Document: notes.md]" in result
    assert "# Notes" in result
    assert "research this brief" in result


def test_build_dr_input_binary_warns_and_falls_back(tmp_path, capsys):
    """_build_dr_input warns and returns bare query for unsupported binary types."""
    # Use a .bin extension; MIME will be unknown/application type, not text/*.
    # PDF and image/* are now handled via _build_dr_multimodal_input, not _build_dr_input.
    # This test covers the defensive fallback for truly unsupported types called directly.
    pdf = tmp_path / "report.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake bytes")

    result = gemini_search._build_dr_input("summarize", str(pdf))

    captured = capsys.readouterr()
    assert "WARNING" in captured.err
    assert "not supported" in captured.err
    assert result == "summarize"


def test_build_dr_input_missing_file():
    """_build_dr_input exits with error when file does not exist."""
    with pytest.raises(SystemExit) as exc:
        gemini_search._build_dr_input("query", "/nonexistent/path/file.txt")
    assert exc.value.code == 1


def test_run_deep_research_passes_file_content_to_agent(tmp_path, mocker):
    """_run_deep_research forwards text file content inline to agent input."""
    txt = tmp_path / "brief.txt"
    txt.write_text("Focus on fusion energy.", encoding="utf-8")

    interaction = _make_mock_interaction(text="Fusion report")
    mock_client = mocker.MagicMock()
    mock_client.interactions.create.return_value = interaction
    mock_client.interactions.get.return_value = interaction

    gemini_search._run_deep_research("research this", "agent-id", mock_client, file_path=str(txt))

    call_kwargs = mock_client.interactions.create.call_args.kwargs
    agent_input = call_kwargs["input"]
    assert "[Document: brief.txt]" in agent_input
    assert "Focus on fusion energy." in agent_input
    assert "research this" in agent_input


# --- _build_dr_multimodal_input tests ---

def _make_fake_file_obj(uri="https://files.example.com/abc123", name="files/abc123"):
    """Build a minimal mock Files API file object with ACTIVE state."""
    class FakeState:
        name = "ACTIVE"

    class FakeFile:
        pass

    obj = FakeFile()
    obj.uri = uri
    obj.name = name
    obj.state = FakeState()
    return obj


def test_build_dr_multimodal_input_pdf(tmp_path, mocker):
    """_build_dr_multimodal_input returns typed document input list for PDF."""
    pdf = tmp_path / "report.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake pdf")

    fake_file = _make_fake_file_obj()
    mock_client = mocker.MagicMock()
    mock_client.files.upload.return_value = fake_file
    mock_client.files.get.return_value = fake_file

    result = gemini_search._build_dr_multimodal_input(
        "summarize this", pdf, mock_client
    )

    assert isinstance(result, list)
    assert len(result) == 2
    assert result[0] == {"type": "text", "text": "summarize this"}
    assert result[1]["type"] == "document"
    assert result[1]["uri"] == fake_file.uri
    assert result[1]["mime_type"] == "application/pdf"
    mock_client.files.upload.assert_called_once_with(
        file=str(pdf), config={"mime_type": "application/pdf"}
    )


def test_build_dr_multimodal_input_image(tmp_path, mocker):
    """_build_dr_multimodal_input returns typed image input list for PNG."""
    png = tmp_path / "chart.png"
    png.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)  # minimal PNG header

    fake_file = _make_fake_file_obj()
    mock_client = mocker.MagicMock()
    mock_client.files.upload.return_value = fake_file
    mock_client.files.get.return_value = fake_file

    result = gemini_search._build_dr_multimodal_input(
        "describe this chart", png, mock_client
    )

    assert isinstance(result, list)
    assert result[0] == {"type": "text", "text": "describe this chart"}
    assert result[1]["type"] == "image"
    assert result[1]["uri"] == fake_file.uri
    assert result[1]["mime_type"] == "image/png"
    mock_client.files.upload.assert_called_once_with(
        file=str(png), config={"mime_type": "image/png"}
    )


def test_build_dr_multimodal_input_waits_for_active(tmp_path, mocker):
    """_build_dr_multimodal_input polls until file state is ACTIVE."""
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.4")

    class FakeStatePending:
        name = "PROCESSING"

    class FakeStateActive:
        name = "ACTIVE"

    class FakeFilePending:
        uri = "https://files.example.com/abc"
        name = "files/abc"
        state = FakeStatePending()

    class FakeFileActive:
        uri = "https://files.example.com/abc"
        name = "files/abc"
        state = FakeStateActive()

    mock_client = mocker.MagicMock()
    mock_client.files.upload.return_value = FakeFilePending()
    # First call returns PROCESSING, second returns ACTIVE
    mock_client.files.get.side_effect = [FakeFilePending(), FakeFileActive()]
    mocker.patch("time.sleep")  # avoid actual sleep

    result = gemini_search._build_dr_multimodal_input("q", pdf, mock_client)

    assert result[1]["type"] == "document"
    assert mock_client.files.get.call_count == 2


def test_build_dr_multimodal_input_upload_failure(tmp_path, mocker, capsys):
    """_build_dr_multimodal_input exits on upload failure."""
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.4")

    mock_client = mocker.MagicMock()
    mock_client.files.upload.side_effect = RuntimeError("upload failed")

    with pytest.raises(SystemExit) as exc:
        gemini_search._build_dr_multimodal_input("q", pdf, mock_client)

    assert exc.value.code == 1
    captured = capsys.readouterr()
    assert "ERROR" in captured.err


def test_build_dr_multimodal_input_timeout(tmp_path, mocker, capsys):
    """_build_dr_multimodal_input exits when file never becomes ACTIVE."""
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.4")

    class FakeStatePending:
        name = "PROCESSING"

    class FakeFilePending:
        uri = "https://files.example.com/abc"
        name = "files/abc"
        state = FakeStatePending()

    mock_client = mocker.MagicMock()
    mock_client.files.upload.return_value = FakeFilePending()
    mock_client.files.get.return_value = FakeFilePending()
    mocker.patch("time.sleep")

    with pytest.raises(SystemExit) as exc:
        gemini_search._build_dr_multimodal_input(
            "q", pdf, mock_client, _upload_wait_seconds=0
        )

    assert exc.value.code == 1


# --- _run_deep_research dispatch tests ---

def test_run_deep_research_dispatches_pdf_to_multimodal(tmp_path, mocker):
    """_run_deep_research calls _build_dr_multimodal_input for PDF files."""
    pdf = tmp_path / "report.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake pdf")

    interaction = _make_mock_interaction(text="PDF research")
    mock_client = mocker.MagicMock()
    mock_client.interactions.create.return_value = interaction
    mock_client.interactions.get.return_value = interaction

    fake_multimodal = [
        {"type": "text", "text": "summarize"},
        {"type": "document", "uri": "https://files.example.com/x", "mime_type": "application/pdf"},
    ]
    mock_build = mocker.patch(
        "gemini_search._build_dr_multimodal_input", return_value=fake_multimodal
    )

    gemini_search._run_deep_research("summarize", "agent-id", mock_client, file_path=str(pdf))

    mock_build.assert_called_once()
    call_kwargs = mock_client.interactions.create.call_args.kwargs
    assert call_kwargs["input"] == fake_multimodal


def test_run_deep_research_dispatches_image_to_multimodal(tmp_path, mocker):
    """_run_deep_research calls _build_dr_multimodal_input for image files."""
    png = tmp_path / "chart.png"
    png.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 8)

    interaction = _make_mock_interaction(text="Image research")
    mock_client = mocker.MagicMock()
    mock_client.interactions.create.return_value = interaction
    mock_client.interactions.get.return_value = interaction

    fake_multimodal = [
        {"type": "text", "text": "analyze this"},
        {"type": "image", "uri": "https://files.example.com/y", "mime_type": "image/png"},
    ]
    mock_build = mocker.patch(
        "gemini_search._build_dr_multimodal_input", return_value=fake_multimodal
    )

    gemini_search._run_deep_research("analyze this", "agent-id", mock_client, file_path=str(png))

    mock_build.assert_called_once()
    call_kwargs = mock_client.interactions.create.call_args.kwargs
    assert call_kwargs["input"] == fake_multimodal


def test_run_deep_research_warns_for_unsupported_binary(tmp_path, mocker, capsys):
    """_run_deep_research warns and uses bare query for unsupported binary types."""
    # .zip is not text, PDF, or image
    zipf = tmp_path / "archive.zip"
    zipf.write_bytes(b"PK\x03\x04 fake zip")

    interaction = _make_mock_interaction(text="Report")
    mock_client = mocker.MagicMock()
    mock_client.interactions.create.return_value = interaction
    mock_client.interactions.get.return_value = interaction

    gemini_search._run_deep_research("analyze", "agent-id", mock_client, file_path=str(zipf))

    captured = capsys.readouterr()
    assert "WARNING" in captured.err
    call_kwargs = mock_client.interactions.create.call_args.kwargs
    assert call_kwargs["input"] == "analyze"
