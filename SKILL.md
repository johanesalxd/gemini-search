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

# Attach a local file as context (text or PDF)
uv run gemini_search.py search "summarize the key points" --file ./notes.md
uv run gemini_search.py search "what risks does this report identify?" --file ./report.pdf
```

### deep-research — thorough research report (1–3 min)

```bash
# Full output (report + sources)
uv run gemini_search.py deep-research "query"

# JSON output
uv run gemini_search.py deep-research "query" --json

# Use the current default agent explicitly
uv run gemini_search.py deep-research "query" --agent deep-research-preview-04-2026

# Use the more exhaustive max agent explicitly
uv run gemini_search.py deep-research "query" --agent deep-research-max-preview-04-2026

# Attach a text file as a research brief (inline prepend)
uv run gemini_search.py deep-research "research based on this brief" --file ./brief.md

# Attach a PDF document (Files API upload)
uv run gemini_search.py deep-research "summarize and expand on this report" --file ./report.pdf

# Attach an image (Files API upload)
uv run gemini_search.py deep-research "research the topic shown in this diagram" --file ./chart.png

# Continue from a prior deep-research run (follow-up question)
# Use the interaction_id printed at the end of a previous run
uv run gemini_search.py deep-research "what are the regulatory implications?" --previous-interaction-id ia-abc123
```

> A progress notice is printed to stderr. Do not interpret silence as a hang — the call is blocking and may take up to 3 minutes.

## File input — `--file`

Both modes accept `--file <path>` to attach a local file as context.

| File type | `search` | `deep-research` |
|---|---|---|
| `.txt`, `.md`, any `text/*` | Supported — inline prepend | Supported — inline prepend |
| `.pdf` | Supported (≤ 20 MB, inline bytes) | Supported — Files API upload → typed `document` input |
| Images (`image/png`, `image/jpeg`, etc.) | Not a primary use case | Supported — Files API upload → typed `image` input |
| Audio, Video | Not a primary use case | Not implemented in CLI (underlying agent supports them) |
| Other binary | Not a primary use case | Warning emitted, query runs without file |

`query` remains required even when `--file` is given — it is the instruction to apply to the file.

This CLI exposes the core research flow plus `--agent`, `--file`, and `--previous-interaction-id`. Other API controls (tool selection, visualization settings, streaming, etc.) are not surfaced.

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
- **SOURCES** — source sites referenced during research (may be empty; citations are embedded in the report text as markdown links)

## JSON schemas

### search --json
```json
{"query", "model", "search_queries_used", "answer", "sources"}
```

### deep-research --json
```json
{"query", "agent", "interaction_id", "status", "answer", "sources"}
```

When `--previous-interaction-id` is provided, `"previous_interaction_id"` is also included in the output.

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
- **deep-research: `--file` supports text, PDF, and images.** Text files are prepended inline. PDF and image files are uploaded to the Gemini Files API, polled until `ACTIVE`, then passed as typed `document`/`image` input to the agent. Audio and video are supported by the underlying agent API but are not implemented in the CLI. Other binary types emit a WARNING and the query runs without the file.
- **deep-research: `--previous-interaction-id` enables follow-up turns.** Pass the `interaction_id` from a prior run to send a continuation request. The SDK field is `previous_interaction_id` (verified via `help(client.interactions.create)`). The result includes a new `interaction_id` for further chaining. Only valid for `deep-research`; ignored for `search`.
- **deep-research: citations may be embedded in `answer` text.** The `sources` field can still be `[]` on valid runs, even though the CLI also extracts supported citation/search-result structures when present.
- **deep-research requires background execution at the API layer.** Current Google docs require `background=True` for Deep Research interactions.
- **deep-research API is experimental.** `client.interactions` carries a `UserWarning` from the SDK; the tool suppresses the specific experimental warning internally. The API surface may change in future SDK versions.
