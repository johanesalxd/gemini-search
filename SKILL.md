---
name: gemini-search
description: "Google Search via Gemini in two modes. Use `search` for fast grounded synthesis with sources, and `deep-research` for slower multi-step research reports and follow-up Q&A. Good for current facts, cited web synthesis, and deeper topic/article analysis. NOT for raw rankings (use Serper) or broad multi-source research orchestration (use Brave/Exa/deep-analysis when needed)."
---

# gemini-search

Google-backed research CLI with two distinct modes:
- **`search`** — fast grounded synthesis (typically 15-25 s)
- **`deep-research`** — multi-step research report (typically 1-3 min)

Use this skill when you specifically want the local `gemini_search.py` CLI behavior rather than a generic search tool.

## Auth

`GOOGLE_API_KEY` environment variable required.

## Use when

- you want a Gemini-grounded web answer with source links
- you want a deeper Gemini research report on a topic, URL, or attached file
- you want to continue a completed Deep Research run with a follow-up question

Do not use this skill when you need raw search rankings, broad multi-source research orchestration, or a non-Gemini search stack.

## Core usage

### `search` — fast grounded synthesis

```bash
uv run gemini_search.py search "query"
uv run gemini_search.py search "query" --raw-urls
uv run gemini_search.py search "query" --json
uv run gemini_search.py search "query" --model gemini-3-flash-preview
uv run gemini_search.py search "summarize the key points" --file ./notes.md
uv run gemini_search.py search "what risks does this identify?" --file ./report.pdf
```

### `deep-research` — multi-step research report

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

A progress notice is printed to stderr. Deep Research is blocking and may take up to a few minutes.

## File input — `--file`

Both modes accept `--file <path>` to attach local context.

| File type | `search` | `deep-research` |
|---|---|---|
| Text (`.txt`, `.md`, `text/*`) | Supported | Supported |
| PDF | Supported | Supported |
| Images | Not a primary use case | Supported |
| Audio / Video | Not supported | Not implemented |
| Other binary | Not supported | Warning emitted; query continues without file |

`query` remains required even when `--file` is present.

## Output expectations

- `search --json` returns: `query`, `model`, `search_queries_used`, `answer`, `sources`
- `deep-research --json` returns: `query`, `agent`, `interaction_id`, `status`, `answer`, `sources`, and optionally `images`
- when `--previous-interaction-id` is used, deep-research output also includes `previous_interaction_id` and `followup_model`

Do not assume `search` and `deep-research` share the same schema.

## Key gotchas

- **`search`: Gemini may not search unless the prompt is clearly current.** Use phrasing like `today`, `current`, or `latest` when grounding matters.
- **`search`: source URLs may be redirect/proxy URLs.** Titles are often the more stable surface.
- **`deep-research`: `--raw-urls` is not supported.** It warns and is ignored.
- **`deep-research`: `--model` is not the control knob for the research run.** Use `--agent` for the fresh run and `--followup-model` for post-report follow-up.
- **`deep-research`: fresh run and follow-up are different paths.** Fresh research uses the Deep Research agent; `--previous-interaction-id` triggers a model-based follow-up interaction against a completed report.
- **`deep-research`: visualization is on by default.** Generated charts/graphs are saved under `/tmp/gemini-search-<interaction_id>/` as PNG files. Use `--no-visualization` to disable.
- **`deep-research`: citations may be embedded in `answer`.** `sources` can be empty on valid runs.
- **Deep Research is preview/experimental.** Agent IDs and response details may drift with SDK/API changes.

## Operational guidance

Recommended timeout headroom:
- `search` -> about 30 s
- `deep-research` -> about 180 s

If this skill grows materially larger later, keep `SKILL.md` focused on triggering and operation, and move bulky reference detail out instead of duplicating the full README.
