# gemini-search

Google Search via Gemini Grounding API + Deep Research — official Google web index, AI-synthesized answer + source URLs.

Two modes:
- **`search`** — fast grounded synthesis (15–25 s), uses Gemini `google_search` tool
- **`deep-research`** — thorough multi-step research report (1–3 min), uses Gemini Interactions API

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

# More exhaustive Deep Research agent
uv run gemini_search.py deep-research "query" --agent deep-research-max-preview-04-2026

# Attach a text file as a research brief (inline prepend)
uv run gemini_search.py deep-research "conduct research based on this brief" --file ./brief.md

# Attach a PDF document (uploaded via Files API)
uv run gemini_search.py deep-research "summarize and expand on this report" --file ./report.pdf

# Attach an image (uploaded via Files API)
uv run gemini_search.py deep-research "research the topic shown in this diagram" --file ./chart.png

# Continue from a prior deep-research run (follow-up question)
uv run gemini_search.py deep-research "what are the regulatory implications?" --previous-interaction-id ia-abc123
```

> **Note:** Deep Research is a blocking call that takes **1–3 minutes** to complete. A progress notice is printed to stderr at the start.

## File input — `--file`

Both modes accept `--file <path>` to attach a local file as context for the query.

| File type | `search` | `deep-research` |
|---|---|---|
| `.txt`, `.md`, any `text/*` | Supported — inline string prepended to query | Supported — inline string prepended to query |
| `.pdf` | Supported — inline `Part.from_bytes` multipart | Supported — Files API upload → typed `document` input |
| Images (`image/png`, `image/jpeg`, etc.) | Not a primary use case | Supported — Files API upload → typed `image` input |
| Audio, Video | Not a primary use case | Not implemented in CLI (underlying agent supports them; deferred) |
| Other binary | Not a primary use case | Warning emitted, query runs without file |

**Notes:**
- `query` remains required even when `--file` is given. The query is the instruction to apply to the file (e.g., "summarize this", "what risks does this identify?").
- PDF and image inputs for `deep-research` are uploaded to the Gemini Files API (`client.files.upload`), polled until `ACTIVE`, then passed as a typed content list to `client.interactions.create`. No local size limit applies (Files API handles large files).
- Audio and video are supported by the underlying Deep Research agent API but are not implemented in the CLI (impractical for typical research workflows).
- To continue a prior Deep Research session, pass `--previous-interaction-id <id>` using the `interaction_id` from a previous run. This sends the follow-up as a stateful continuation turn via the `previous_interaction_id` field in the Interactions API. The result includes a new `interaction_id` for chaining further follow-ups.
- File content is context only. The `query` field in JSON output reflects the original query string, not the file content.

## When to use search vs deep-research

| | search | deep-research |
|---|---|---|
| Latency | 15–25 s | 1–3 min |
| Output | Synthesized paragraph + sources | Multi-section research report + sources |
| Use case | Current facts, news, quick lookups | In-depth research, complex topics |
| `--raw-urls` | Supported | Not supported |
| `--file` text | Supported | Supported |
| `--file` PDF | Supported (inline multipart) | Supported (Files API upload) |
| `--file` image | Not a primary use case | Supported (Files API upload) |
| Cost | Lower | Higher |

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

### deep-research --json

```json
{
  "query": "...",
  "agent": "deep-research-preview-04-2026",
  "interaction_id": "...",
  "status": "completed",
  "answer": "...",
  "sources": [{"title": "...", "url": "..."}]
}
```

When `--previous-interaction-id` is provided, the output also includes:

```json
{
  "previous_interaction_id": "ia-prior-abc123"
}
```

Note: `search_queries_used` is absent from deep-research output; `interaction_id` and `status` are absent from search output.

Deep Research runs asynchronously under the hood with `background=True` and polls until completion.

Deep Research citations may appear inline in the `answer` text (for example as markdown links), and `sources` can still be empty on some valid runs. This is expected API behavior, not a parsing bug.

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
