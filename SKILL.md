---
name: gemini-search
description: "Google Search via Gemini Grounding API — official Google web index, AI-synthesized answer + source URLs. Use for: factual queries needing current Google data with citations, breaking news synthesis, queries where source attribution matters. NOT for: raw Google rankings, non-synthesis research."
---

# gemini-search — Google Search via Gemini Grounding API

Official Google web index via Gemini's `google_search` tool. Fires real Google searches, synthesizes results, returns answer + source URLs.

## Auth

`GOOGLE_API_KEY` environment variable required.

## Usage

```bash
# Full output (synthesized answer + sources)
uv run gemini_search.py search "query"

# Sources only (no synthesis)
uv run gemini_search.py search "query" --raw-urls

# JSON output
uv run gemini_search.py search "query" --json

# Use Pro model (default: gemini-3-flash-preview)
uv run gemini_search.py search "query" --model gemini-3.1-pro-preview
```

## Output

- **Query** — what you asked
- **Google queries fired** — what Gemini actually searched (useful for debugging)
- **ANSWER** — synthesized response grounded in Google's current web index
- **SOURCES** — source sites (title + proxied redirect URL)

## Gotchas

- **Gemini may not search.** Use explicit phrasing like "today", "current", "latest" to force grounding — otherwise it answers from training data.
- **Source URLs are proxied redirects** via `vertexaisearch.cloud.google.com`. Titles are reliable; URLs need redirect-following for actual destinations.
- **Latency: 15-25s.** Account for this in any parallel execution setup.
