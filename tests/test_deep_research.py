"""Tests for _run_deep_research() and deep_research() output modes.

Mock objects reflect the real _interactions SDK discriminated-union output shape:
  - TextContent:               type="text", .text, .annotations
  - GoogleSearchResultContent: type="google_search_result", .result[]
Each item in interaction.outputs has a top-level `type` attribute; there is no
`.parts` nesting (that is the google.genai.types.Content shape, not the
interactions Content union).
"""

import json
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
