"""Tests for _run_deep_research() and deep_research() output modes."""

import json
import pytest

import gemini_search


def _make_mock_interaction(
    interaction_id="ia-123",
    status="completed",
    agent="deep-research-pro-preview-12-2025",
    text="Research report text.",
    sources=None,
):
    """Build a minimal mock Interaction object."""
    sources = sources or [{"title": "Src A", "url": "https://a.com"}]

    class FakePart:
        def __init__(self, text=None, results=None):
            self.text = text
            if results is not None:
                self.result = results

    class FakeResultItem:
        def __init__(self, title, url):
            self.title = title
            self.url = url

    class FakeContent:
        def __init__(self, parts):
            self.parts = parts

    source_parts = [
        FakePart(results=[FakeResultItem(s["title"], s["url"]) for s in sources])
    ]
    text_part = FakePart(text=text)
    content = FakeContent([text_part] + source_parts)

    class FakeInteraction:
        pass

    obj = FakeInteraction()
    obj.id = interaction_id
    obj.status = status
    obj.outputs = [content]
    return obj


def test_run_deep_research_json_shape(mocker):
    """_run_deep_research returns dict with expected keys."""
    mock_client = mocker.MagicMock()
    mock_client.interactions.create.return_value = _make_mock_interaction(
        text="Research output",
        sources=[{"title": "T1", "url": "https://t1.com"}],
    )

    result = gemini_search._run_deep_research(
        "deep query",
        "deep-research-pro-preview-12-2025",
        mock_client,
    )

    assert result["query"] == "deep query"
    assert result["agent"] == "deep-research-pro-preview-12-2025"
    assert result["interaction_id"] == "ia-123"
    assert result["status"] == "completed"
    assert result["answer"] == "Research output"
    assert len(result["sources"]) == 1
    assert result["sources"][0]["title"] == "T1"


def test_run_deep_research_empty_outputs(mocker):
    """_run_deep_research handles interaction with no outputs."""
    class FakeInteraction:
        id = "ia-empty"
        status = "completed"
        outputs = None

    mock_client = mocker.MagicMock()
    mock_client.interactions.create.return_value = FakeInteraction()

    result = gemini_search._run_deep_research("q", "agent-id", mock_client)

    assert result["answer"] == ""
    assert result["sources"] == []


def test_deep_research_json_output(mocker, capsys):
    """deep_research() with as_json=True prints valid JSON with correct shape."""
    mocker.patch("gemini_search.get_api_key", return_value="fake-key")
    mock_client = mocker.MagicMock()
    mock_client.interactions.create.return_value = _make_mock_interaction(
        text="Report", sources=[{"title": "S", "url": "https://s.com"}]
    )
    mocker.patch("gemini_search._make_client", return_value=mock_client)

    gemini_search.deep_research("topic", as_json=True)

    captured = capsys.readouterr()
    output = json.loads(captured.out)
    assert output["query"] == "topic"
    assert "agent" in output
    assert "interaction_id" in output
    assert "status" in output
    assert "answer" in output
    assert "sources" in output
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
    mocker.patch("gemini_search._make_client", return_value=mock_client)

    gemini_search.deep_research("topic")

    captured = capsys.readouterr()
    assert "Deep Research in progress" in captured.err
