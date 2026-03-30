#!/usr/bin/env python3
"""
gemini_search.py — Google Search via Gemini Grounding API
Official Google web index, no scraping. Returns synthesized answer + raw source URLs.

Usage:
  uv run gemini_search.py search "query"
  uv run gemini_search.py search "query" --raw-urls   # sources only, no synthesis
  uv run gemini_search.py search "query" --json       # full JSON output
  uv run gemini_search.py search "query" --model gemini-3-flash-preview

Env: GOOGLE_API_KEY (required)
"""

import os
import sys
import json
import argparse


def get_api_key():
    key = os.environ.get("GOOGLE_API_KEY")
    if not key:
        print("ERROR: GOOGLE_API_KEY environment variable not set", file=sys.stderr)
        sys.exit(1)
    return key


def search(query, model="gemini-3-flash-preview", raw_urls=False, as_json=False):
    from google import genai
    from google.genai import types

    key = get_api_key()
    client = genai.Client(api_key=key)

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

    # Extract grounding metadata
    candidate = response.candidates[0] if response.candidates else None
    grounding_meta = None
    if candidate and hasattr(candidate, "grounding_metadata"):
        grounding_meta = candidate.grounding_metadata

    # Extract sources from groundingChunks
    sources = []
    if grounding_meta and hasattr(grounding_meta, "grounding_chunks"):
        for chunk in grounding_meta.grounding_chunks or []:
            if hasattr(chunk, "web") and chunk.web:
                sources.append({
                    "title": chunk.web.title or "",
                    "url": chunk.web.uri or "",
                })

    # Extract search queries used
    search_queries = []
    if grounding_meta and hasattr(grounding_meta, "web_search_queries"):
        search_queries = list(grounding_meta.web_search_queries or [])

    answer_text = response.text or ""

    if as_json:
        output = {
            "query": query,
            "model": model,
            "search_queries_used": search_queries,
            "answer": answer_text,
            "sources": sources,
        }
        print(json.dumps(output, indent=2))
        return

    if raw_urls:
        print(f"Query: {query}")
        print(f"Google queries fired: {', '.join(search_queries) if search_queries else 'n/a'}")
        print()
        print("=== SOURCES ===")
        for i, s in enumerate(sources, 1):
            print(f"  [{i}] {s['title']}")
            print(f"       {s['url']}")
        return

    # Full output
    print(f"Query: {query}")
    print(f"Model: {model}")
    if search_queries:
        print(f"Google queries fired: {', '.join(search_queries)}")
    print()
    print("=== ANSWER (grounded in Google Search) ===")
    print(answer_text)
    print()
    if sources:
        print("=== SOURCES ===")
        for i, s in enumerate(sources, 1):
            print(f"  [{i}] {s['title']}")
            print(f"       {s['url']}")
    else:
        print("=== SOURCES: none returned ===")


def main():
    parser = argparse.ArgumentParser(description="Google Search via Gemini Grounding API")
    parser.add_argument("command", choices=["search"], help="Command (only 'search' supported)")
    parser.add_argument("query", help="Search query")
    parser.add_argument(
        "--model",
        default="gemini-3-flash-preview",
        help="Gemini model (default: gemini-3-flash-preview)",
    )
    parser.add_argument(
        "--raw-urls",
        action="store_true",
        help="Return source URLs only — no synthesized answer",
    )
    parser.add_argument("--json", action="store_true", help="Full JSON output")
    args = parser.parse_args()

    search(args.query, model=args.model, raw_urls=args.raw_urls, as_json=args.json)


if __name__ == "__main__":
    main()
