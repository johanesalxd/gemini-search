#!/usr/bin/env python3
"""
gemini_search.py — Google Search via Gemini Grounding API + Deep Research
Official Google web index, no scraping. Returns synthesized answer + raw source URLs.

Usage:
  uv run gemini_search.py search "query"
  uv run gemini_search.py search "query" --raw-urls   # sources only, no synthesis
  uv run gemini_search.py search "query" --json       # full JSON output
  uv run gemini_search.py search "query" --model gemini-3-flash-preview

  uv run gemini_search.py deep-research "query"
  uv run gemini_search.py deep-research "query" --json
  uv run gemini_search.py deep-research "query" --agent deep-research-pro-preview-12-2025

Env: GOOGLE_API_KEY (required)
"""

import json
import os
import sys
import time
import warnings
import argparse

_DEFAULT_SEARCH_MODEL = "gemini-3-flash-preview"
_DEFAULT_DEEP_RESEARCH_AGENT = "deep-research-preview-04-2026"
_DEEP_RESEARCH_POLL_INTERVAL_SECONDS = 5


def get_api_key() -> str:
    key = os.environ.get("GOOGLE_API_KEY")
    if not key:
        print("ERROR: GOOGLE_API_KEY environment variable not set", file=sys.stderr)
        sys.exit(1)
    return key


def _make_client(key: str):
    from google import genai

    return genai.Client(api_key=key)


def _run_search(query: str, model: str, client) -> dict:
    from google.genai import types

    grounding_tool = types.Tool(google_search=types.GoogleSearch())
    config = types.GenerateContentConfig(tools=[grounding_tool])

    try:
        response = client.models.generate_content(
            model=model,
            contents=query,
            config=config,
        )
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    candidate = response.candidates[0] if response.candidates else None
    grounding_meta = None
    if candidate and hasattr(candidate, "grounding_metadata"):
        grounding_meta = candidate.grounding_metadata

    sources = []
    if grounding_meta and hasattr(grounding_meta, "grounding_chunks"):
        for chunk in grounding_meta.grounding_chunks or []:
            if hasattr(chunk, "web") and chunk.web:
                sources.append({
                    "title": chunk.web.title or "",
                    "url": chunk.web.uri or "",
                })

    search_queries = []
    if grounding_meta and hasattr(grounding_meta, "web_search_queries"):
        search_queries = list(grounding_meta.web_search_queries or [])

    return {
        "query": query,
        "model": model,
        "search_queries_used": search_queries,
        "answer": response.text or "",
        "sources": sources,
    }


def search(query: str, model: str = _DEFAULT_SEARCH_MODEL, raw_urls: bool = False, as_json: bool = False) -> None:
    key = get_api_key()
    client = _make_client(key)
    result = _run_search(query, model, client)

    if as_json:
        print(json.dumps(result, indent=2))
        return

    if raw_urls:
        print(f"Query: {result['query']}")
        sq = result["search_queries_used"]
        print(f"Google queries fired: {', '.join(sq) if sq else 'n/a'}")
        print()
        print("=== SOURCES ===")
        for i, s in enumerate(result["sources"], 1):
            print(f"  [{i}] {s['title']}")
            print(f"       {s['url']}")
        return

    print(f"Query: {result['query']}")
    print(f"Model: {result['model']}")
    if result["search_queries_used"]:
        print(f"Google queries fired: {', '.join(result['search_queries_used'])}")
    print()
    print("=== ANSWER (grounded in Google Search) ===")
    print(result["answer"])
    print()
    if result["sources"]:
        print("=== SOURCES ===")
        for i, s in enumerate(result["sources"], 1):
            print(f"  [{i}] {s['title']}")
            print(f"       {s['url']}")
    else:
        print("=== SOURCES: none returned ===")


def _run_deep_research(query: str, agent: str, client) -> dict:
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=".*Interactions usage is experimental.*",
                category=UserWarning,
            )
            interaction = client.interactions.create(
                agent=agent,
                input=query,
                background=True,
            )

            while interaction.status == "in_progress":
                time.sleep(_DEEP_RESEARCH_POLL_INTERVAL_SECONDS)
                interaction = client.interactions.get(interaction.id)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    if interaction.status != "completed":
        print(
            f"ERROR: Deep Research interaction ended with status '{interaction.status}'",
            file=sys.stderr,
        )
        sys.exit(1)

    answer = ""
    sources = []
    seen_urls: set[str] = set()

    for content in interaction.outputs or []:
        # interaction.outputs contains discriminated-union content items, each
        # with a `type` field. The two types relevant here are:
        #   - "text"                 → TextContent: has .text (str) and
        #                              optional .annotations (citation sources)
        #   - "google_search_result" → GoogleSearchResultContent: has
        #                              .result (List[GoogleSearchResult]), each
        #                              with .title, .url, .rendered_content
        content_type = getattr(content, "type", None)

        if content_type == "text":
            # Accumulate the synthesized research report text.
            text = getattr(content, "text", None)
            if text:
                answer += text
            # Annotations carry inline citation URLs; add as sources.
            for ann in getattr(content, "annotations", None) or []:
                url = getattr(ann, "source", None) or ""
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    sources.append({"title": url, "url": url})

        elif content_type == "google_search_result":
            # Each result item has .title and .url.
            for result_item in getattr(content, "result", None) or []:
                url = getattr(result_item, "url", None) or ""
                title = getattr(result_item, "title", None) or url
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    sources.append({"title": title, "url": url})

    return {
        "query": query,
        "agent": agent,
        "interaction_id": interaction.id,
        "status": interaction.status,
        "answer": answer.strip(),
        "sources": sources,
    }


def deep_research(query: str, agent: str = _DEFAULT_DEEP_RESEARCH_AGENT, as_json: bool = False) -> None:
    print(
        "Deep Research in progress — this may take 1–3 minutes...",
        file=sys.stderr,
    )
    key = get_api_key()
    client = _make_client(key)
    result = _run_deep_research(query, agent, client)

    if as_json:
        print(json.dumps(result, indent=2))
        return

    print(f"Query: {result['query']}")
    print(f"Agent: {result['agent']}")
    print(f"Status: {result['status']}")
    print(f"Interaction ID: {result['interaction_id']}")
    print()
    print("=== DEEP RESEARCH REPORT ===")
    print(result["answer"])
    print()
    if result["sources"]:
        print("=== SOURCES ===")
        for i, s in enumerate(result["sources"], 1):
            title = s.get("title", "")
            url = s.get("url", "")
            print(f"  [{i}] {title}")
            if url:
                print(f"       {url}")
    else:
        print("=== SOURCES: none returned ===")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Google Search via Gemini Grounding API + Deep Research"
    )
    parser.add_argument(
        "command",
        choices=["search", "deep-research"],
        help="Command to run: 'search' (fast, grounded) or 'deep-research' (thorough, slow)",
    )
    parser.add_argument("query", help="Query string")
    parser.add_argument(
        "--model",
        default=_DEFAULT_SEARCH_MODEL,
        help=f"Gemini model for search (default: {_DEFAULT_SEARCH_MODEL}). Ignored for deep-research.",
    )
    parser.add_argument(
        "--agent",
        default=_DEFAULT_DEEP_RESEARCH_AGENT,
        help=f"Agent identifier for deep-research (default: {_DEFAULT_DEEP_RESEARCH_AGENT}). Ignored for search.",
    )
    parser.add_argument(
        "--raw-urls",
        action="store_true",
        help="Return source URLs only — search mode only, not supported for deep-research.",
    )
    parser.add_argument("--json", action="store_true", help="Full JSON output")
    args = parser.parse_args()

    if args.command == "search":
        search(args.query, model=args.model, raw_urls=args.raw_urls, as_json=args.json)
    elif args.command == "deep-research":
        if args.raw_urls:
            print(
                "WARNING: --raw-urls is not supported for deep-research; ignoring.",
                file=sys.stderr,
            )
        deep_research(args.query, agent=args.agent, as_json=args.json)


if __name__ == "__main__":
    main()
