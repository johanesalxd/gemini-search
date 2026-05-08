"""Tests for _run_deep_research(), deep_research(), and helpers.

Mock objects reflect the SDK v2 (google-genai>=2.0.0) steps structure:
  interaction.steps = [
      ThoughtStep(type="thought", summary=[TextContent(...)]),
      ModelOutputStep(type="model_output", content=[...]),
  ]
Sources come from TextContent.annotations (UrlCitation).
"""

import base64
import json
import pathlib

import pytest

import gemini_search

# ---------------------------------------------------------------------------
# Mock factories
# ---------------------------------------------------------------------------


class _FakeTextContent:
    type = "text"

    def __init__(self, text, annotations=None):
        self.text = text
        self.annotations = annotations or []


class _FakeImageContent:
    type = "image"

    def __init__(self, data):
        self.data = data


class _FakeAnnotation:
    def __init__(self, url, title=None, ann_type="url_citation"):
        self.type = ann_type
        self.url = url
        self.title = title


class _FakeThoughtStep:
    type = "thought"

    def __init__(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        self.summary = [_FakeTextContent(t) for t in texts]
        self.signature = None


class _FakeModelOutputStep:
    type = "model_output"

    def __init__(self, content):
        self.content = content


def _make_mock_interaction(
    interaction_id="ia-123",
    status="completed",
    text="Research report text.",
    annotations=None,
    image_data_list=None,
    thought_summaries=None,
):
    """Build a mock Interaction matching SDK v2 steps structure.

    Args:
        interaction_id: Fake interaction ID.
        status: Interaction status string.
        text: Report text for the ModelOutputStep TextContent.
        annotations: List of _FakeAnnotation for the TextContent.
        image_data_list: List of base64 image data strings.
        thought_summaries: List of thought summary text strings.
    """
    steps = []

    if thought_summaries:
        for summary_text in thought_summaries:
            steps.append(_FakeThoughtStep(summary_text))

    content = [_FakeTextContent(text=text, annotations=annotations)]
    for img_data in image_data_list or []:
        content.append(_FakeImageContent(data=img_data))
    steps.append(_FakeModelOutputStep(content=content))

    class FakeInteraction:
        pass

    obj = FakeInteraction()
    obj.id = interaction_id
    obj.status = status
    obj.steps = steps
    return obj


# ---------------------------------------------------------------------------
# Core parsing: _run_deep_research result shape
# ---------------------------------------------------------------------------


def test_run_deep_research_happy_path(mocker):
    """Full result dict includes text, sources from annotations, and images."""
    img_b64 = base64.b64encode(b"\x89PNGfake").decode()
    interaction = _make_mock_interaction(
        text="Full report.",
        annotations=[
            _FakeAnnotation("https://a.com", title="Source A"),
            _FakeAnnotation("https://b.com"),
        ],
        image_data_list=[img_b64],
    )
    mock_client = mocker.MagicMock()
    mock_client.interactions.create.return_value = interaction
    mock_client.interactions.get.return_value = interaction

    result = gemini_search._run_deep_research(
        "deep query", "deep-research-preview-04-2026", mock_client
    )

    assert result["query"] == "deep query"
    assert result["agent"] == "deep-research-preview-04-2026"
    assert result["interaction_id"] == "ia-123"
    assert result["status"] == "completed"
    assert result["answer"] == "Full report."
    assert len(result["sources"]) == 2
    assert result["sources"][0] == {"title": "Source A", "url": "https://a.com"}
    assert result["sources"][1]["url"] == "https://b.com"
    assert len(result["images"]) == 1
    assert result["images"][0]["index"] == 1
    assert "ia-123" in result["images"][0]["path"]
    assert "fallback_answer" not in result
    assert "empty_answer_diagnostic" not in result


def test_run_deep_research_empty_steps(mocker):
    """Handles interaction with steps=None gracefully."""

    class FakeInteraction:
        id = "ia-empty"
        status = "completed"
        steps = None

    mock_client = mocker.MagicMock()
    mock_client.interactions.create.return_value = FakeInteraction()
    mock_client.interactions.get.return_value = FakeInteraction()

    result = gemini_search._run_deep_research("q", "agent-id", mock_client)

    assert result["answer"] == ""
    assert result["sources"] == []
    assert result["images"] == []


def test_run_deep_research_extracts_url_citations(mocker):
    """Sources are extracted from url_citation annotations on TextContent."""
    interaction = _make_mock_interaction(
        text="Report with citations.",
        annotations=[
            _FakeAnnotation("https://cited.example.com", title="Cited Page"),
            # Non-url_citation annotations are ignored.
            _FakeAnnotation("files/x", ann_type="file_citation"),
        ],
    )
    mock_client = mocker.MagicMock()
    mock_client.interactions.create.return_value = interaction
    mock_client.interactions.get.return_value = interaction

    result = gemini_search._run_deep_research("q", "agent-id", mock_client)

    assert len(result["sources"]) == 1
    assert result["sources"][0]["url"] == "https://cited.example.com"
    assert result["sources"][0]["title"] == "Cited Page"


def test_run_deep_research_deduplicates_annotation_urls(mocker):
    """Duplicate URLs across annotations are deduplicated."""
    interaction = _make_mock_interaction(
        text="Report.",
        annotations=[
            _FakeAnnotation("https://a.com", title="First mention"),
            _FakeAnnotation("https://a.com", title="Second mention"),
            _FakeAnnotation("https://b.com"),
        ],
    )
    mock_client = mocker.MagicMock()
    mock_client.interactions.create.return_value = interaction
    mock_client.interactions.get.return_value = interaction

    result = gemini_search._run_deep_research("q", "agent-id", mock_client)

    urls = [s["url"] for s in result["sources"]]
    assert urls.count("https://a.com") == 1
    assert "https://b.com" in urls
    assert len(result["sources"]) == 2


def test_run_deep_research_saves_image_to_disk(mocker, tmp_path):
    """Image content in ModelOutputStep is decoded and saved to disk."""
    img_bytes = b"\x89PNG\r\n\x1a\nfake-image-data"
    img_b64 = base64.b64encode(img_bytes).decode()
    interaction = _make_mock_interaction(
        interaction_id="ia-viz-001",
        text="Report with chart.",
        image_data_list=[img_b64],
    )
    mock_client = mocker.MagicMock()
    mock_client.interactions.create.return_value = interaction
    mock_client.interactions.get.return_value = interaction

    result = gemini_search._run_deep_research("q", "agent-id", mock_client)

    assert len(result["images"]) == 1
    saved = pathlib.Path(result["images"][0]["path"])
    assert saved.exists()
    assert saved.read_bytes() == img_bytes


def test_run_deep_research_skips_image_without_data(mocker):
    """Image content with data=None is silently skipped."""
    interaction = _make_mock_interaction(text="Report.")
    # Inject an image with no data into the model_output step.
    model_step = interaction.steps[-1]
    model_step.content.append(_FakeImageContent(data=None))

    mock_client = mocker.MagicMock()
    mock_client.interactions.create.return_value = interaction
    mock_client.interactions.get.return_value = interaction

    result = gemini_search._run_deep_research("q", "agent-id", mock_client)

    assert result["images"] == []


# ---------------------------------------------------------------------------
# Fallback logic: empty report with thought summaries
# ---------------------------------------------------------------------------


def test_run_deep_research_empty_text_falls_back_to_thought_summary(mocker):
    """When report text is empty, ThoughtStep summaries are surfaced as fallback."""
    interaction = _make_mock_interaction(
        text=None,
        thought_summaries=[
            "Researching quantum computing.",
            "Found key papers on entanglement.",
        ],
    )
    mock_client = mocker.MagicMock()
    mock_client.interactions.create.return_value = interaction
    mock_client.interactions.get.return_value = interaction

    result = gemini_search._run_deep_research("q", "agent-id", mock_client)

    assert result["answer"] == ""
    assert "Researching quantum computing." in result["fallback_answer"]
    assert "Found key papers on entanglement." in result["fallback_answer"]
    assert result["fallback_answer_source"] == "thought_summary"
    assert "empty_answer_diagnostic" in result


def test_run_deep_research_populated_text_ignores_thought_fallback(mocker):
    """When report text is present, thought summaries do not trigger fallback."""
    interaction = _make_mock_interaction(
        text="Official final report.",
        thought_summaries=["Some planning thought."],
    )
    mock_client = mocker.MagicMock()
    mock_client.interactions.create.return_value = interaction
    mock_client.interactions.get.return_value = interaction

    result = gemini_search._run_deep_research("q", "agent-id", mock_client)

    assert result["answer"] == "Official final report."
    assert "fallback_answer" not in result
    assert "empty_answer_diagnostic" not in result


def test_run_deep_research_empty_text_no_thoughts_shows_diagnostic(mocker):
    """Empty report with no thoughts sets diagnostic without fallback."""
    interaction = _make_mock_interaction(text=None)
    mock_client = mocker.MagicMock()
    mock_client.interactions.create.return_value = interaction
    mock_client.interactions.get.return_value = interaction

    result = gemini_search._run_deep_research("q", "agent-id", mock_client)

    assert result["answer"] == ""
    assert "empty_answer_diagnostic" in result
    assert "fallback_answer" not in result


# ---------------------------------------------------------------------------
# API dispatch: agent_config, follow-up path, polling
# ---------------------------------------------------------------------------


def test_run_deep_research_fresh_sends_agent_config(mocker):
    """Fresh run sends agent_config with type, thinking_summaries, and visualization."""
    interaction = _make_mock_interaction()
    mock_client = mocker.MagicMock()
    mock_client.interactions.create.return_value = interaction
    mock_client.interactions.get.return_value = interaction

    gemini_search._run_deep_research("q", "agent-id", mock_client)

    call_kwargs = mock_client.interactions.create.call_args.kwargs
    assert call_kwargs["agent"] == "agent-id"
    assert call_kwargs["background"] is True
    assert "model" not in call_kwargs
    cfg = call_kwargs["agent_config"]
    assert cfg["type"] == "deep-research"
    assert cfg["thinking_summaries"] == "auto"
    assert cfg["visualization"] == "auto"


def test_run_deep_research_visualization_off(mocker):
    """visualization=False sends visualization='off' in agent_config."""
    interaction = _make_mock_interaction()
    mock_client = mocker.MagicMock()
    mock_client.interactions.create.return_value = interaction
    mock_client.interactions.get.return_value = interaction

    gemini_search._run_deep_research("q", "agent-id", mock_client, visualization=False)

    cfg = mock_client.interactions.create.call_args.kwargs["agent_config"]
    assert cfg["visualization"] == "off"


def test_run_deep_research_followup_uses_model(mocker):
    """Follow-up path uses model (not agent), includes previous_interaction_id."""
    interaction = _make_mock_interaction(interaction_id="ia-followup")
    mock_client = mocker.MagicMock()
    mock_client.interactions.create.return_value = interaction

    result = gemini_search._run_deep_research(
        "elaborate on point 2",
        "deep-research-preview-04-2026",
        mock_client,
        previous_interaction_id="ia-prior-abc",
    )

    call_kwargs = mock_client.interactions.create.call_args.kwargs
    assert call_kwargs["model"] == gemini_search._DEFAULT_FOLLOWUP_MODEL
    assert call_kwargs["previous_interaction_id"] == "ia-prior-abc"
    assert "agent" not in call_kwargs
    assert "background" not in call_kwargs
    assert "agent_config" not in call_kwargs
    assert result["previous_interaction_id"] == "ia-prior-abc"
    assert result["followup_model"] == gemini_search._DEFAULT_FOLLOWUP_MODEL


def test_run_deep_research_followup_skips_polling(mocker):
    """Follow-up path does not call interactions.get (synchronous)."""
    interaction = _make_mock_interaction()
    mock_client = mocker.MagicMock()
    mock_client.interactions.create.return_value = interaction

    gemini_search._run_deep_research(
        "q", "agent-id", mock_client, previous_interaction_id="ia-prior"
    )

    mock_client.interactions.get.assert_not_called()


# ---------------------------------------------------------------------------
# CLI output: deep_research() JSON and text modes
# ---------------------------------------------------------------------------


def test_deep_research_json_output(mocker, capsys):
    """deep_research(as_json=True) prints valid JSON with expected shape."""
    mocker.patch("gemini_search.get_api_key", return_value="fake-key")
    mock_client = mocker.MagicMock()
    mock_client.interactions.create.return_value = _make_mock_interaction(
        text="Report",
        annotations=[_FakeAnnotation("https://s.com", title="S")],
    )
    mock_client.interactions.get.return_value = (
        mock_client.interactions.create.return_value
    )
    mocker.patch("gemini_search._make_client", return_value=mock_client)

    gemini_search.deep_research("topic", as_json=True)

    output = json.loads(capsys.readouterr().out)
    assert output["query"] == "topic"
    assert output["answer"] == "Report"
    assert output["sources"][0]["url"] == "https://s.com"
    assert "search_queries_used" not in output


def test_deep_research_text_output(mocker, capsys):
    """deep_research() default text mode prints report header and sources."""
    mocker.patch("gemini_search.get_api_key", return_value="fake-key")
    mock_client = mocker.MagicMock()
    mock_client.interactions.create.return_value = _make_mock_interaction(
        text="Full research text",
        annotations=[_FakeAnnotation("https://rsrc.com", title="Research Src")],
    )
    mock_client.interactions.get.return_value = (
        mock_client.interactions.create.return_value
    )
    mocker.patch("gemini_search._make_client", return_value=mock_client)

    gemini_search.deep_research("research topic")

    captured = capsys.readouterr()
    assert "=== DEEP RESEARCH REPORT ===" in captured.out
    assert "Full research text" in captured.out
    assert "=== SOURCES ===" in captured.out
    assert "Research Src" in captured.out


def test_deep_research_empty_answer_prints_warning(mocker, capsys):
    """Empty answer prints warning to stderr and shows fallback section."""
    mocker.patch("gemini_search.get_api_key", return_value="fake-key")
    mock_client = mocker.MagicMock()
    mock_client.interactions.create.return_value = _make_mock_interaction(
        text=None,
        thought_summaries=["Planning: scoped the question."],
    )
    mock_client.interactions.get.return_value = (
        mock_client.interactions.create.return_value
    )
    mocker.patch("gemini_search._make_client", return_value=mock_client)

    gemini_search.deep_research("topic")

    captured = capsys.readouterr()
    assert "WARNING" in captured.err
    assert "=== DEEP RESEARCH REPORT: EMPTY ===" in captured.out
    assert "=== FALLBACK (thought summaries" in captured.out
    assert "Planning: scoped the question." in captured.out
    # Must NOT label thought summaries as the official report.
    assert "=== DEEP RESEARCH REPORT ===" not in captured.out


# ---------------------------------------------------------------------------
# File input dispatch
# ---------------------------------------------------------------------------


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


def test_run_deep_research_passes_file_content_to_agent(tmp_path, mocker):
    """Text file content is forwarded inline to the agent input."""
    txt = tmp_path / "brief.txt"
    txt.write_text("Focus on fusion energy.", encoding="utf-8")

    interaction = _make_mock_interaction(text="Fusion report")
    mock_client = mocker.MagicMock()
    mock_client.interactions.create.return_value = interaction
    mock_client.interactions.get.return_value = interaction

    gemini_search._run_deep_research(
        "research this", "agent-id", mock_client, file_path=str(txt)
    )

    agent_input = mock_client.interactions.create.call_args.kwargs["input"]
    assert "Focus on fusion energy." in agent_input


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
    """Returns typed document input list for PDF."""
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


def test_build_dr_multimodal_input_image(tmp_path, mocker):
    """Returns typed image input list for PNG."""
    png = tmp_path / "chart.png"
    png.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)

    fake_file = _make_fake_file_obj()
    mock_client = mocker.MagicMock()
    mock_client.files.upload.return_value = fake_file
    mock_client.files.get.return_value = fake_file

    result = gemini_search._build_dr_multimodal_input(
        "describe this chart", png, mock_client
    )

    assert result[1]["type"] == "image"
    assert result[1]["mime_type"] == "image/png"


def test_build_dr_multimodal_input_waits_for_active(tmp_path, mocker):
    """Polls until file state is ACTIVE."""
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
    mock_client.files.get.side_effect = [FakeFilePending(), FakeFileActive()]
    mocker.patch("time.sleep")

    result = gemini_search._build_dr_multimodal_input("q", pdf, mock_client)

    assert result[1]["type"] == "document"
    assert mock_client.files.get.call_count == 2


def test_build_dr_multimodal_input_upload_failure(tmp_path, mocker, capsys):
    """Exits on upload failure."""
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.4")

    mock_client = mocker.MagicMock()
    mock_client.files.upload.side_effect = RuntimeError("upload failed")

    with pytest.raises(SystemExit) as exc:
        gemini_search._build_dr_multimodal_input("q", pdf, mock_client)

    assert exc.value.code == 1
    assert "ERROR" in capsys.readouterr().err


def test_build_dr_multimodal_input_timeout(tmp_path, mocker):
    """Exits when file never becomes ACTIVE."""
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


# ---------------------------------------------------------------------------
# File dispatch routing
# ---------------------------------------------------------------------------


def test_run_deep_research_dispatches_pdf_to_multimodal(tmp_path, mocker):
    """PDF files are routed through _build_dr_multimodal_input."""
    pdf = tmp_path / "report.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake pdf")

    interaction = _make_mock_interaction(text="PDF research")
    mock_client = mocker.MagicMock()
    mock_client.interactions.create.return_value = interaction
    mock_client.interactions.get.return_value = interaction

    fake_multimodal = [
        {"type": "text", "text": "summarize"},
        {
            "type": "document",
            "uri": "https://files.example.com/x",
            "mime_type": "application/pdf",
        },
    ]
    mock_build = mocker.patch(
        "gemini_search._build_dr_multimodal_input",
        return_value=fake_multimodal,
    )

    gemini_search._run_deep_research(
        "summarize", "agent-id", mock_client, file_path=str(pdf)
    )

    mock_build.assert_called_once()
    assert mock_client.interactions.create.call_args.kwargs["input"] == fake_multimodal


def test_run_deep_research_warns_for_unsupported_binary(tmp_path, mocker, capsys):
    """Unsupported binary types emit a warning and use bare query."""
    zipf = tmp_path / "archive.zip"
    zipf.write_bytes(b"PK\x03\x04 fake zip")

    interaction = _make_mock_interaction(text="Report")
    mock_client = mocker.MagicMock()
    mock_client.interactions.create.return_value = interaction
    mock_client.interactions.get.return_value = interaction

    gemini_search._run_deep_research(
        "analyze", "agent-id", mock_client, file_path=str(zipf)
    )

    assert "WARNING" in capsys.readouterr().err
    assert mock_client.interactions.create.call_args.kwargs["input"] == "analyze"


# ---------------------------------------------------------------------------
# Helper: _save_image
# ---------------------------------------------------------------------------


def test_save_image_creates_directory_and_file(tmp_path, mocker):
    """_save_image creates output directory and writes decoded image."""
    img_bytes = b"raw-image-data"
    img_b64 = base64.b64encode(img_bytes).decode()

    mocker.patch(
        "gemini_search._IMAGE_OUTPUT_DIR_PREFIX",
        str(tmp_path / "gemini-search-"),
    )

    path = gemini_search._save_image(img_b64, "test-id", 1)

    assert pathlib.Path(path).exists()
    assert pathlib.Path(path).read_bytes() == img_bytes
    assert "test-id" in path
    assert path.endswith("image_001.png")
