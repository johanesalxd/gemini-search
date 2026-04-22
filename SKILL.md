---
name: gemini-search
description: "Google Search via Gemini Grounding API + Deep Research. Two modes: 'search' for fast grounded synthesis (15–25s, current facts, citations); 'deep-research' for thorough multi-step research reports (1–3 min, complex topics). Use 'search' for: factual queries, breaking news, quick lookups. Use 'deep-research' for: in-depth analysis, complex research questions. NOT for: raw Google rankings (use Serper), non-synthesis research (use Brave/Exa)."
---

# gemini-search — Google Search via Gemini Grounding API + Deep Research

Two modes backed by separate Google APIs:
- **`search`** — Gemini `google_search` grounding tool. Fast, single-turn, synthesized answer + sources.
- **`deep-research`** — Gemini Interactions API (`client.interactions.create` + `background=True` polling via `client.interactions.get`). Multi-step research, thorough report, 1–3 min latency.

## Auth

`GOOGLE_API_KEY` environment variable required.

## Usage

### search — fast grounded synthesis (15–25 s)

```bash
# Full output (synthesized answer + sources)
uv run gemini_search.py search "query"

# Sources only (no synthesis)
uv run gemini_search.py search "query" --raw-urls

# JSON output
uv run gemini_search.py search "query" --json

# Use a different search model
uv run gemini_search.py search "query" --model gemini-3-flash-preview
```

### deep-research — thorough research report (1–3 min)

```bash
# Full output (report + sources)
uv run gemini_search.py deep-research "query"

# JSON output
uv run gemini_search.py deep-research "query" --json
```

> A progress notice is printed to stderr. Do not interpret silence as a hang — the call is blocking and may take up to 3 minutes.

## Output

### search output fields
- **Query** — what you asked
- **Google queries fired** — what Gemini actually searched (useful for debugging)
- **ANSWER** — synthesized response grounded in Google's current web index
- **SOURCES** — source sites (title + proxied redirect URL)

### deep-research output fields
- **Query** — what you asked
- **Agent** — agent identifier used
- **Status** — interaction status (`completed`, `failed`, etc.)
- **Interaction ID** — unique ID for this research session
- **DEEP RESEARCH REPORT** — multi-section research report
- **SOURCES** — source sites referenced during research

## JSON schemas

### search --json
```json
{"query", "model", "search_queries_used", "answer", "sources"}
```

### deep-research --json
```json
{"query", "agent", "interaction_id", "status", "answer", "sources"}
```

Note: `search_queries_used` is absent from deep-research output. Do not parse both with the same schema.

## Parallel Execution

| Mode | Latency | Recommended yieldMs |
|---|---|---|
| search | 15–25 s | 30000 |
| deep-research | 60–180 s | 180000 |

## Gotchas

- **search: Gemini may not search.** Use explicit phrasing like "today", "current", "latest" to force grounding — otherwise it answers from training data.
- **search: Source URLs are proxied redirects** via `vertexaisearch.cloud.google.com/grounding-api-redirect/...`. Titles are reliable; URLs need redirect-following for actual destinations.
- **deep-research: `--raw-urls` is not supported.** Passing it emits a warning and is ignored.
- **deep-research: `--model` is ignored.** The agent identifier (`--agent`) drives the backend, not a model name.
- **deep-research API is experimental.** `client.interactions` carries a `UserWarning` from the SDK; the tool suppresses the specific experimental warning internally. The API surface may change in future SDK versions.
