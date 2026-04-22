"""Tests for _run_deep_research(), deep_research(), and related helpers.

Mock objects reflect the real _interactions SDK discriminated-union output shape:
  - TextContent:               type="text", .text, .annotations
  - GoogleSearchResultContent: type="google_search_result", .result[]
  - ImageContent:              type="image", .data (base64)
Each item in interaction.outputs has a top-level `type` attribute; there is no
`.parts` nesting (that is the google.genai.types.Content shape, not the
interactions Content union).
"""

import base64
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

def test_build_dr_input_text_file(tmp_path):
    """_build_dr_input prepends file content for text files."""
    txt = tmp_path / "brief.txt"
    txt.write_text("Research focus: quantum computing.", encoding="utf-8")

    result = gemini_search._build_dr_input("conduct research on this", str(txt))

    assert "[Document: brief.txt]" in result
    assert "Research focus: quantum computing." in result
    assert "conduct research on this" in result
    assert result.index("Research focus") < result.index("conduct research on this")


def test_build_dr_input_markdown_file(tmp_path):
    """_build_dr_input handles .md files as text."""
    md = tmp_path / "notes.md"
    md.write_text("# Notes\n\nFocus on CRISPR.", encoding="utf-8")

    result = gemini_search._build_dr_input("research this brief", str(md))

    assert "[Document: notes.md]" in result
    assert "# Notes" in result
    assert "research this brief" in result


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


# --- continuation (previous_interaction_id) tests ---

def test_run_deep_research_followup_uses_model_not_agent(mocker):
    """_run_deep_research uses model-based interaction for post-report follow-up.

    Docs: ai.google.dev/gemini-api/docs/deep-research#follow-up-questions-and-interactions
    The post-report follow-up must use model= (not agent=) and must NOT send background=True.
    """
    interaction = _make_mock_interaction(interaction_id="ia-new-456")
    mock_client = mocker.MagicMock()
    mock_client.interactions.create.return_value = interaction

    result = gemini_search._run_deep_research(
        "follow-up question",
        "deep-research-preview-04-2026",
        mock_client,
        previous_interaction_id="ia-prior-abc123",
    )

    # API call shape: model-based, synchronous, no agent
    call_kwargs = mock_client.interactions.create.call_args.kwargs
    assert call_kwargs.get("model") == gemini_search._DEFAULT_FOLLOWUP_MODEL
    assert "agent" not in call_kwargs
    assert "background" not in call_kwargs
    assert call_kwargs["previous_interaction_id"] == "ia-prior-abc123"
    # Result shape includes follow-up fields
    assert result["query"] == "follow-up question"
    assert result["interaction_id"] == "ia-new-456"
    assert result["status"] == "completed"
    assert result["previous_interaction_id"] == "ia-prior-abc123"
    assert result["followup_model"] == gemini_search._DEFAULT_FOLLOWUP_MODEL


def test_run_deep_research_followup_uses_custom_model(mocker):
    """_run_deep_research uses the provided followup_model for follow-up."""
    interaction = _make_mock_interaction(interaction_id="ia-custom-model")
    mock_client = mocker.MagicMock()
    mock_client.interactions.create.return_value = interaction

    result = gemini_search._run_deep_research(
        "follow-up",
        "deep-research-preview-04-2026",
        mock_client,
        previous_interaction_id="ia-prior-abc",
        followup_model="gemini-3-flash-preview",
    )

    call_kwargs = mock_client.interactions.create.call_args.kwargs
    assert call_kwargs["model"] == "gemini-3-flash-preview"
    assert result["followup_model"] == "gemini-3-flash-preview"


def test_run_deep_research_followup_does_not_poll(mocker):
    """_run_deep_research follow-up path does not call interactions.get (synchronous)."""
    interaction = _make_mock_interaction(interaction_id="ia-sync-follow")
    mock_client = mocker.MagicMock()
    mock_client.interactions.create.return_value = interaction

    gemini_search._run_deep_research(
        "what does section 2 mean?",
        "deep-research-preview-04-2026",
        mock_client,
        previous_interaction_id="ia-prior-xyz",
    )

    # interactions.get must not be called for the follow-up path
    mock_client.interactions.get.assert_not_called()


def test_run_deep_research_fresh_uses_agent_and_background(mocker):
    """_run_deep_research fresh run uses agent + background, no follow-up fields."""
    interaction = _make_mock_interaction()
    mock_client = mocker.MagicMock()
    mock_client.interactions.create.return_value = interaction
    mock_client.interactions.get.return_value = interaction

    result = gemini_search._run_deep_research(
        "fresh query",
        "deep-research-preview-04-2026",
        mock_client,
    )

    call_kwargs = mock_client.interactions.create.call_args.kwargs
    assert call_kwargs.get("agent") == "deep-research-preview-04-2026"
    assert call_kwargs.get("background") is True
    assert "model" not in call_kwargs
    assert "previous_interaction_id" not in call_kwargs
    assert "previous_interaction_id" not in result
    assert "followup_model" not in result


def test_deep_research_json_output_includes_followup_fields(mocker, capsys):
    """deep_research() JSON output includes previous_interaction_id and followup_model when provided."""
    mocker.patch("gemini_search.get_api_key", return_value="fake-key")
    mock_client = mocker.MagicMock()
    mock_client.interactions.create.return_value = _make_mock_interaction(
        interaction_id="ia-follow-789",
        text="Follow-up answer",
        sources=[{"title": "S", "url": "https://s.com"}],
    )
    mock_client.interactions.get.return_value = mock_client.interactions.create.return_value
    mocker.patch("gemini_search._make_client", return_value=mock_client)

    gemini_search.deep_research(
        "follow-up topic",
        as_json=True,
        previous_interaction_id="ia-prior-123",
    )

    captured = capsys.readouterr()
    output = json.loads(captured.out)
    assert output["interaction_id"] == "ia-follow-789"
    assert output["previous_interaction_id"] == "ia-prior-123"
    assert output["followup_model"] == gemini_search._DEFAULT_FOLLOWUP_MODEL
    assert output["answer"] == "Follow-up answer"


def test_deep_research_followup_progress_notice(mocker, capsys):
    """deep_research() with previous_interaction_id prints a follow-up notice (not the research notice)."""
    mocker.patch("gemini_search.get_api_key", return_value="fake-key")
    mock_client = mocker.MagicMock()
    mock_client.interactions.create.return_value = _make_mock_interaction()
    mock_client.interactions.get.return_value = mock_client.interactions.create.return_value
    mocker.patch("gemini_search._make_client", return_value=mock_client)

    gemini_search.deep_research("follow-up q", previous_interaction_id="ia-prior-xxx")

    captured = capsys.readouterr()
    assert "Post-report follow-up" in captured.err
    assert "Deep Research in progress" not in captured.err


def test_deep_research_followup_text_output(mocker, capsys):
    """deep_research() follow-up text output shows Follow-up Model, not Agent."""
    mocker.patch("gemini_search.get_api_key", return_value="fake-key")
    mock_client = mocker.MagicMock()
    mock_client.interactions.create.return_value = _make_mock_interaction(
        text="Follow-up answer text",
    )
    mock_client.interactions.get.return_value = mock_client.interactions.create.return_value
    mocker.patch("gemini_search._make_client", return_value=mock_client)

    gemini_search.deep_research("elaborate on point 3", previous_interaction_id="ia-prior-zzz")

    captured = capsys.readouterr()
    assert "Follow-up Model:" in captured.out
    assert gemini_search._DEFAULT_FOLLOWUP_MODEL in captured.out
    assert "Prior Interaction ID: ia-prior-zzz" in captured.out


def test_deep_research_fresh_progress_notice(mocker, capsys):
    """deep_research() without previous_interaction_id prints the Deep Research notice."""
    mocker.patch("gemini_search.get_api_key", return_value="fake-key")
    mock_client = mocker.MagicMock()
    mock_client.interactions.create.return_value = _make_mock_interaction()
    mock_client.interactions.get.return_value = mock_client.interactions.create.return_value
    mocker.patch("gemini_search._make_client", return_value=mock_client)

    gemini_search.deep_research("fresh topic")

    captured = capsys.readouterr()
    assert "Deep Research in progress" in captured.err
    assert "Post-report follow-up" not in captured.err


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


# --- agent_config tests ---

def test_run_deep_research_passes_agent_config(mocker):
    """_run_deep_research passes agent_config with visualization='auto' for fresh runs."""
    interaction = _make_mock_interaction()
    mock_client = mocker.MagicMock()
    mock_client.interactions.create.return_value = interaction
    mock_client.interactions.get.return_value = interaction

    gemini_search._run_deep_research("q", "agent-id", mock_client)

    call_kwargs = mock_client.interactions.create.call_args.kwargs
    assert "agent_config" in call_kwargs
    assert call_kwargs["agent_config"]["type"] == "deep-research"
    assert call_kwargs["agent_config"]["visualization"] == "auto"


def test_run_deep_research_no_visualization(mocker):
    """_run_deep_research passes visualization='off' when visualization=False."""
    interaction = _make_mock_interaction()
    mock_client = mocker.MagicMock()
    mock_client.interactions.create.return_value = interaction
    mock_client.interactions.get.return_value = interaction

    gemini_search._run_deep_research(
        "q", "agent-id", mock_client, visualization=False
    )

    call_kwargs = mock_client.interactions.create.call_args.kwargs
    assert call_kwargs["agent_config"]["visualization"] == "off"


def test_run_deep_research_followup_omits_agent_config(mocker):
    """_run_deep_research follow-up path does not pass agent_config."""
    interaction = _make_mock_interaction()
    mock_client = mocker.MagicMock()
    mock_client.interactions.create.return_value = interaction

    gemini_search._run_deep_research(
        "follow-up", "agent-id", mock_client,
        previous_interaction_id="ia-prior",
    )

    call_kwargs = mock_client.interactions.create.call_args.kwargs
    assert "agent_config" not in call_kwargs


# --- image output tests ---

def _make_mock_interaction_with_images(
    interaction_id="ia-img-123",
    text="Report with charts.",
    image_data_list=None,
):
    """Build a mock Interaction with text + image outputs."""
    if image_data_list is None:
        image_data_list = [base64.b64encode(b"fake-png-bytes-1").decode()]

    class FakeTextContent:
        type = "text"

        def __init__(self, text):
            self.text = text
            self.annotations = []

    class FakeImageContent:
        type = "image"

        def __init__(self, data):
            self.data = data

    outputs = [FakeTextContent(text=text)]
    for img_data in image_data_list:
        outputs.append(FakeImageContent(data=img_data))

    class FakeInteraction:
        pass

    obj = FakeInteraction()
    obj.id = interaction_id
    obj.status = "completed"
    obj.outputs = outputs
    return obj


def test_run_deep_research_saves_images_to_disk(mocker, tmp_path):
    """_run_deep_research saves image outputs to /tmp and includes paths in result."""
    img_bytes = b"\x89PNG\r\n\x1a\nfake-image-data"
    img_b64 = base64.b64encode(img_bytes).decode()

    interaction = _make_mock_interaction_with_images(
        interaction_id="ia-viz-001",
        text="Report with chart.",
        image_data_list=[img_b64],
    )
    mock_client = mocker.MagicMock()
    mock_client.interactions.create.return_value = interaction
    mock_client.interactions.get.return_value = interaction

    result = gemini_search._run_deep_research("q", "agent-id", mock_client)

    assert len(result["images"]) == 1
    img_info = result["images"][0]
    assert img_info["index"] == 1
    assert "ia-viz-001" in img_info["path"]
    assert img_info["path"].endswith(".png")
    # Verify file was actually written
    saved = pathlib.Path(img_info["path"])
    assert saved.exists()
    assert saved.read_bytes() == img_bytes


def test_run_deep_research_saves_multiple_images(mocker):
    """_run_deep_research handles multiple image outputs."""
    img1 = base64.b64encode(b"image-1").decode()
    img2 = base64.b64encode(b"image-2").decode()

    interaction = _make_mock_interaction_with_images(
        interaction_id="ia-multi-img",
        text="Report.",
        image_data_list=[img1, img2],
    )
    mock_client = mocker.MagicMock()
    mock_client.interactions.create.return_value = interaction
    mock_client.interactions.get.return_value = interaction

    result = gemini_search._run_deep_research("q", "agent-id", mock_client)

    assert len(result["images"]) == 2
    assert result["images"][0]["index"] == 1
    assert result["images"][1]["index"] == 2
    # Verify both files exist
    for img_info in result["images"]:
        assert pathlib.Path(img_info["path"]).exists()


def test_run_deep_research_no_images_returns_empty_list(mocker):
    """_run_deep_research returns empty images list when no image outputs."""
    interaction = _make_mock_interaction(text="No images here.")
    mock_client = mocker.MagicMock()
    mock_client.interactions.create.return_value = interaction
    mock_client.interactions.get.return_value = interaction

    result = gemini_search._run_deep_research("q", "agent-id", mock_client)

    assert result["images"] == []


def test_run_deep_research_image_with_no_data_skipped(mocker):
    """_run_deep_research skips image outputs with no data."""

    class FakeImageNoData:
        type = "image"
        data = None

    class FakeTextContent:
        type = "text"
        text = "Report."
        annotations = []

    class FakeInteraction:
        id = "ia-no-data"
        status = "completed"
        outputs = [FakeTextContent(), FakeImageNoData()]

    mock_client = mocker.MagicMock()
    mock_client.interactions.create.return_value = FakeInteraction()
    mock_client.interactions.get.return_value = FakeInteraction()

    result = gemini_search._run_deep_research("q", "agent-id", mock_client)

    assert result["images"] == []


def test_deep_research_text_output_prints_images(mocker, capsys):
    """deep_research() text mode prints image paths when images are present."""
    img_b64 = base64.b64encode(b"test-image").decode()
    mocker.patch("gemini_search.get_api_key", return_value="fake-key")
    mock_client = mocker.MagicMock()
    mock_client.interactions.create.return_value = _make_mock_interaction_with_images(
        text="Report", image_data_list=[img_b64]
    )
    mock_client.interactions.get.return_value = mock_client.interactions.create.return_value
    mocker.patch("gemini_search._make_client", return_value=mock_client)

    gemini_search.deep_research("topic")

    captured = capsys.readouterr()
    assert "=== IMAGES ===" in captured.out
    assert "image_001.png" in captured.out


def test_deep_research_json_output_includes_images(mocker, capsys):
    """deep_research() JSON output includes images list."""
    img_b64 = base64.b64encode(b"chart-bytes").decode()
    mocker.patch("gemini_search.get_api_key", return_value="fake-key")
    mock_client = mocker.MagicMock()
    mock_client.interactions.create.return_value = _make_mock_interaction_with_images(
        interaction_id="ia-json-img",
        text="Report",
        image_data_list=[img_b64],
    )
    mock_client.interactions.get.return_value = mock_client.interactions.create.return_value
    mocker.patch("gemini_search._make_client", return_value=mock_client)

    gemini_search.deep_research("topic", as_json=True)

    captured = capsys.readouterr()
    output = json.loads(captured.out)
    assert "images" in output
    assert len(output["images"]) == 1
    assert output["images"][0]["index"] == 1
    assert "ia-json-img" in output["images"][0]["path"]


def test_save_image_creates_directory_and_file(tmp_path, mocker):
    """_save_image creates output directory and writes decoded image."""
    img_bytes = b"raw-image-data"
    img_b64 = base64.b64encode(img_bytes).decode()

    # Override prefix to use tmp_path
    mocker.patch(
        "gemini_search._IMAGE_OUTPUT_DIR_PREFIX",
        str(tmp_path / "gemini-search-"),
    )

    path = gemini_search._save_image(img_b64, "test-id", 1)

    assert pathlib.Path(path).exists()
    assert pathlib.Path(path).read_bytes() == img_bytes
    assert "test-id" in path
    assert path.endswith("image_001.png")
