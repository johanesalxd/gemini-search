---
name: gemini-search
description: "Google Search via Gemini Grounding API + Deep Research. Two modes: 'search' for fast grounded synthesis (15–25s, current facts, citations); 'deep-research' for thorough multi-step research reports (1–3 min, complex topics). Use 'search' for: factual queries, breaking news, quick lookups. Use 'deep-research' for: in-depth analysis, complex research questions. NOT for: raw Google rankings (use Serper), non-synthesis research (use Brave/Exa)."
---

# gemini-search — Google Search via Gemini Grounding API + Deep Research

Two modes:
- **`search`** — Fast grounded synthesis (15–25 s). Single-turn answer + sources.
- **`deep-research`** — Thorough multi-step research report (1–3 min). Supports follow-up Q&A via `--previous-interaction-id`.

## Auth

`GOOGLE_API_KEY` environment variable required.

## Usage

### search — fast grounded synthesis (15–25 s)

```bash
uv run gemini_search.py search "query"
uv run gemini_search.py search "query" --raw-urls       # sources only
uv run gemini_search.py search "query" --json            # JSON output
uv run gemini_search.py search "query" --model gemini-3-flash-preview
uv run gemini_search.py search "summarize the key points" --file ./notes.md
uv run gemini_search.py search "what risks does this identify?" --file ./report.pdf
```

### deep-research — thorough research report (1–3 min)

```bash
uv run gemini_search.py deep-research "query"
uv run gemini_search.py deep-research "query" --json
uv run gemini_search.py deep-research "query" --agent deep-research-preview-04-2026
uv run gemini_search.py deep-research "research based on this brief" --file ./brief.md
uv run gemini_search.py deep-research "summarize this report" --file ./report.pdf
uv run gemini_search.py deep-research "research the topic in this diagram" --file ./chart.png
uv run gemini_search.py deep-research "what are the regulatory implications?" --previous-interaction-id ia-abc123
uv run gemini_search.py deep-research "summarize section 3" --previous-interaction-id ia-abc123 --followup-model gemini-3-flash-preview
uv run gemini_search.py deep-research "query" --no-visualization
```

> A progress notice is printed to stderr. Do not interpret silence as a hang — the call is blocking and may take up to 3 minutes.

## File input — `--file`

Both modes accept `--file <path>` to attach a local file as context.

| File type | `search` | `deep-research` |
|---|---|---|
| `.txt`, `.md`, any `text/*` | Supported | Supported |
| `.pdf` | Supported (up to 20 MB) | Supported |
| Images (`image/png`, `image/jpeg`, etc.) | Not a primary use case | Supported |
| Audio, Video | Not supported | Not implemented |
| Other binary | Not supported | Warning emitted, query runs without file |

`query` remains required even when `--file` is given — it is the instruction to apply to the file.

## JSON schemas

### search --json

```json
{
  "query": "...",
  "model": "gemini-3-flash-preview",
  "search_queries_used": ["..."],
  "answer": "...",
  "sources": [{"title": "...", "url": "..."}]
}
```

### deep-research --json

```json
{
  "query": "...",
  "agent": "deep-research-preview-04-2026",
  "interaction_id": "...",
  "status": "completed",
  "answer": "...",
  "sources": [{"title": "...", "url": "..."}],
  "images": [{"path": "/tmp/gemini-search-ia-123/image_001.png", "index": 1}]
}
```

When `--previous-interaction-id` is provided, the output also includes `"previous_interaction_id"` and `"followup_model"`.

Note: `search_queries_used` is absent from deep-research output. Do not parse both with the same schema.

## Parallel Execution

| Mode | Latency | Recommended timeout |
|---|---|---|
| search | 15–25 s | 30000 ms |
| deep-research | 60–180 s | 180000 ms |

## Gotchas

- **search: Gemini may not search.** Use explicit phrasing like "today", "current", "latest" to force grounding — otherwise it answers from training data.
- **search: Source URLs are proxied redirects.** Titles are reliable; URLs need redirect-following for actual destinations.
- **deep-research: `--raw-urls` is not supported.** Passing it emits a warning and is ignored.
- **deep-research: `--model` is ignored.** Use `--agent` to select the agent.
- **deep-research: `--previous-interaction-id` enables follow-up Q&A.** The follow-up runs as a fast model interaction, not another full research cycle. Use `--followup-model` to override the default model. The result gains `followup_model` and `previous_interaction_id` fields.
- **deep-research: visualization is on by default.** The agent may generate charts/graphs. Images are saved to `/tmp/gemini-search-<id>/` as PNG files. Use `--no-visualization` to disable.
- **deep-research: citations may be embedded in `answer` text.** The `sources` field can be `[]` on valid runs — this is expected, not a bug.
- **deep-research: the API is experimental.** The API surface may change in future SDK versions.
