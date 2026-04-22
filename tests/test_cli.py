"""Tests for CLI dispatch via main()."""

import sys
import pytest

import gemini_search


def test_main_dispatch_search(mocker):
    """main() routes 'search' command to search()."""
    mock_search = mocker.patch("gemini_search.search")
    mocker.patch("sys.argv", ["gemini_search.py", "search", "my query"])

    gemini_search.main()

    mock_search.assert_called_once_with(
        "my query",
        model=gemini_search._DEFAULT_SEARCH_MODEL,
        raw_urls=False,
        as_json=False,
        file_path=None,
    )


def test_main_dispatch_deep_research(mocker):
    """main() routes 'deep-research' command to deep_research()."""
    mock_dr = mocker.patch("gemini_search.deep_research")
    mocker.patch("sys.argv", ["gemini_search.py", "deep-research", "my topic"])

    gemini_search.main()

    mock_dr.assert_called_once_with(
        "my topic",
        agent=gemini_search._DEFAULT_DEEP_RESEARCH_AGENT,
        as_json=False,
        file_path=None,
        previous_interaction_id=None,
    )


def test_main_rejects_raw_urls_with_deep_research(mocker, capsys):
    """main() emits a warning when --raw-urls is used with deep-research."""
    mock_dr = mocker.patch("gemini_search.deep_research")
    mocker.patch(
        "sys.argv",
        ["gemini_search.py", "deep-research", "topic", "--raw-urls"],
    )

    gemini_search.main()

    captured = capsys.readouterr()
    assert "WARNING" in captured.err
    assert "--raw-urls" in captured.err
    # deep_research() is still called despite the warning
    mock_dr.assert_called_once()


def test_main_passes_previous_interaction_id_to_deep_research(mocker):
    """main() forwards --previous-interaction-id to deep_research()."""
    mock_dr = mocker.patch("gemini_search.deep_research")
    mocker.patch(
        "sys.argv",
        [
            "gemini_search.py",
            "deep-research",
            "follow-up question",
            "--previous-interaction-id",
            "ia-prior-abc123",
        ],
    )

    gemini_search.main()

    mock_dr.assert_called_once_with(
        "follow-up question",
        agent=gemini_search._DEFAULT_DEEP_RESEARCH_AGENT,
        as_json=False,
        file_path=None,
        previous_interaction_id="ia-prior-abc123",
    )


def test_main_previous_interaction_id_defaults_to_none(mocker):
    """main() passes previous_interaction_id=None when flag is absent."""
    mock_dr = mocker.patch("gemini_search.deep_research")
    mocker.patch("sys.argv", ["gemini_search.py", "deep-research", "fresh query"])

    gemini_search.main()

    call_kwargs = mock_dr.call_args.kwargs
    assert call_kwargs["previous_interaction_id"] is None


def test_main_passes_model_to_search(mocker):
    """main() forwards --model to search()."""
    mock_search = mocker.patch("gemini_search.search")
    mocker.patch(
        "sys.argv",
        ["gemini_search.py", "search", "q", "--model", "gemini-3.1-pro-preview"],
    )

    gemini_search.main()

    mock_search.assert_called_once_with(
        "q",
        model="gemini-3.1-pro-preview",
        raw_urls=False,
        as_json=False,
        file_path=None,
    )


def test_main_passes_agent_to_deep_research(mocker):
    """main() forwards --agent to deep_research()."""
    mock_dr = mocker.patch("gemini_search.deep_research")
    mocker.patch(
        "sys.argv",
        ["gemini_search.py", "deep-research", "q", "--agent", "custom-agent-id"],
    )

    gemini_search.main()

    mock_dr.assert_called_once_with(
        "q",
        agent="custom-agent-id",
        as_json=False,
        file_path=None,
        previous_interaction_id=None,
    )


def test_main_invalid_command(mocker):
    """main() rejects unknown commands via argparse."""
    mocker.patch("sys.argv", ["gemini_search.py", "unknown-cmd", "q"])

    with pytest.raises(SystemExit) as exc:
        gemini_search.main()

    assert exc.value.code != 0


def test_main_passes_file_to_search(mocker):
    """main() forwards --file to search() as file_path."""
    mock_search = mocker.patch("gemini_search.search")
    mocker.patch(
        "sys.argv",
        ["gemini_search.py", "search", "q", "--file", "/tmp/notes.md"],
    )

    gemini_search.main()

    mock_search.assert_called_once_with(
        "q",
        model=gemini_search._DEFAULT_SEARCH_MODEL,
        raw_urls=False,
        as_json=False,
        file_path="/tmp/notes.md",
    )


def test_main_passes_file_to_deep_research(mocker):
    """main() forwards --file to deep_research() as file_path."""
    mock_dr = mocker.patch("gemini_search.deep_research")
    mocker.patch(
        "sys.argv",
        ["gemini_search.py", "deep-research", "q", "--file", "/tmp/brief.txt"],
    )

    gemini_search.main()

    mock_dr.assert_called_once_with(
        "q",
        agent=gemini_search._DEFAULT_DEEP_RESEARCH_AGENT,
        as_json=False,
        file_path="/tmp/brief.txt",
        previous_interaction_id=None,
    )
