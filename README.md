# gemini-search

Google-backed research CLI with two distinct modes:
- **`search`** — fast grounded synthesis (typically 15–25 s)
- **`deep-research`** — multi-step research report (typically 1–3 min)

Good for current facts, cited web synthesis, and deeper topic/article analysis.

## Prerequisites

- [uv](https://github.com/astral-sh/uv) (Python package manager)
- Python 3.13
- `GOOGLE_API_KEY` set in your environment

## Installation

```bash
git clone https://github.com/johanesalxd/gemini-search.git
cd gemini-search
uv sync
```

## When to use which mode

| | `search` | `deep-research` |
|---|---|---|
| Best for | Current facts, news, quick lookups | In-depth research, complex topics |
| Typical latency | 15–25 s | 1–3 min |
| Output | Synthesized answer + sources | Research report + sources |
| `--raw-urls` | Supported | Not supported |
| `--file` text | Supported | Supported |
| `--file` PDF | Supported | Supported |
| `--file` image | Not a primary use case | Supported |
| Visualization | N/A | Charts/graphs saved to `/tmp` unless disabled |

## Usage

### search — fast grounded synthesis

```bash
# Full output (synthesized answer + sources)
uv run gemini_search.py search "latest AI news"

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

### deep-research — thorough multi-step research report

```bash
# Full output (research report + sources)
uv run gemini_search.py deep-research "impact of CRISPR on hereditary disease treatment"

# JSON output
uv run gemini_search.py deep-research "query" --json

# Custom agent (default: deep-research-preview-04-2026)
uv run gemini_search.py deep-research "query" --agent deep-research-preview-04-2026

# Attach a text file as a research brief (inline prepend)
uv run gemini_search.py deep-research "conduct research based on this brief" --file ./brief.md

# Attach a PDF document (uploaded via Files API)
uv run gemini_search.py deep-research "summarize and expand on this report" --file ./report.pdf

# Attach an image (uploaded via Files API)
uv run gemini_search.py deep-research "research the topic shown in this diagram" --file ./chart.png

# Continue from a prior deep-research run (follow-up question)
uv run gemini_search.py deep-research "what are the regulatory implications?" --previous-interaction-id ia-abc123

# Use a different model for follow-up Q&A
uv run gemini_search.py deep-research "summarize section 3" --previous-interaction-id ia-abc123 --followup-model gemini-3-flash-preview

# Disable visualization (charts/graphs) in research output
uv run gemini_search.py deep-research "query" --no-visualization
```

> **Note:** Deep Research is a blocking call that takes **1–3 minutes** to complete. A progress notice is printed to stderr at the start.
>
> **Note:** `--previous-interaction-id` switches to a **model-based post-report follow-up** (not another Deep Research agent run). The follow-up uses `gemini-3.1-pro-preview` (or the model specified via `--followup-model`) with `previous_interaction_id` to load the completed report's history — this is the docs-backed continuation contract. Using the Deep Research agent again with a completed interaction ID causes HTTP 400.
>
> **Note:** When visualization is enabled (default), the agent may generate charts and graphs as part of the report. Images are saved to `/tmp/gemini-search-<interaction_id>/` and their paths are included in the output.

## File input — `--file`

Both modes accept `--file <path>` to attach local context.

| File type | `search` | `deep-research` |
|---|---|---|
| Text (`.txt`, `.md`, `text/*`) | Supported | Supported |
| PDF | Supported (up to 20 MB) | Supported |
| Images (`image/png`, `image/jpeg`, etc.) | Not a primary use case | Supported |
| Audio / Video | Not a primary use case | Not implemented in CLI |
| Other binary | Not a primary use case | Warning emitted; query runs without file |

Notes:
- `query` remains required even when `--file` is given; it is the instruction to apply to the file.
- For `deep-research`, PDFs and images are uploaded through the Gemini Files API before being attached to the interaction.
- Audio and video may exist in the underlying API surface, but are not exposed by this CLI.
- File content is context only; the JSON `query` field remains the original query string.

## Output schemas

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

### deep-research --json (fresh run)

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

### deep-research --json (with --previous-interaction-id, post-report follow-up)

```json
{
  "query": "...",
  "agent": "deep-research-preview-04-2026",
  "interaction_id": "...",
  "status": "completed",
  "answer": "...",
  "sources": [{"title": "...", "url": "..."}],
  "images": [],
  "previous_interaction_id": "ia-prior-abc123",
  "followup_model": "gemini-3.1-pro-preview"
}
```

Notes:
- `--previous-interaction-id` switches to a fast post-report follow-up path rather than another full Deep Research run.
- `followup_model` shows which model handled the follow-up interaction.
- `search_queries_used` is absent from deep-research output; `interaction_id` and `status` are absent from search output.
- Deep Research citations may appear inline in `answer`, so `sources` can be empty on valid runs.

## Agent Skill

This repo includes a `SKILL.md` for use with AI coding agents (Claude Code, Cursor, Windsurf, OpenCode, and others).

### Install via skills CLI

```bash
npx skills add johanesalxd/gemini-search
```

### Manual install

```bash
cp SKILL.md ~/.claude/skills/gemini-search/SKILL.md         # Claude Code
cp SKILL.md .cursor/skills/gemini-search/SKILL.md           # Cursor
cp SKILL.md .opencode/skills/gemini-search/SKILL.md         # OpenCode
```

Once installed, your agent will load this skill when you ask for web searches or current information.

## License

MIT
