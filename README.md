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
>
> **Note:** `--previous-interaction-id` switches to a **model-based post-report follow-up** (not another Deep Research agent run). The follow-up uses `gemini-3.1-pro-preview` with `previous_interaction_id` to load the completed report's history — this is the docs-backed continuation contract. Using the Deep Research agent again with a completed interaction ID causes HTTP 400.

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
- To ask a follow-up question against a completed Deep Research report, pass `--previous-interaction-id <id>` using the `interaction_id` from a previous run. The follow-up is sent as a **model-based Interactions request** (not another Deep Research agent run) using `model="gemini-3.1-pro-preview"` + `previous_interaction_id`. This is the docs-backed post-report Q&A contract ([ai.google.dev/gemini-api/docs/deep-research](https://ai.google.dev/gemini-api/docs/deep-research#follow-up-questions-and-interactions)). The result includes a new `interaction_id` and a `followup_model` field.
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

### deep-research --json (fresh run)

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

### deep-research --json (with --previous-interaction-id, post-report follow-up)

```json
{
  "query": "...",
  "agent": "deep-research-preview-04-2026",
  "interaction_id": "...",
  "status": "completed",
  "answer": "...",
  "sources": [{"title": "...", "url": "..."}],
  "previous_interaction_id": "ia-prior-abc123",
  "followup_model": "gemini-3.1-pro-preview"
}
```

The `followup_model` field indicates that the follow-up used a model-based interaction (not the Deep Research agent). The `agent` field still reflects the original Deep Research agent whose report is being followed up on.

Note: `search_queries_used` is absent from deep-research output; `interaction_id` and `status` are absent from search output.

**Two distinct Interactions API paths:**

| Mode | API path | When |
|---|---|---|
| Deep Research run | `agent="deep-research-preview-04-2026"` + `background=True` + polling | No `--previous-interaction-id` |
| Post-report follow-up | `model="gemini-3.1-pro-preview"` + `previous_interaction_id` (synchronous) | With `--previous-interaction-id` |

Deep Research runs asynchronously under the hood with `background=True` and polls until completion. Post-report follow-ups are synchronous model interactions that load conversation history via `previous_interaction_id`.

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
