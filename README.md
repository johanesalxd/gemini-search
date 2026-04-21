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
```

### deep-research — thorough multi-step research report

```bash
# Full output (research report + sources)
uv run gemini_search.py deep-research "impact of CRISPR on hereditary disease treatment"

# JSON output
uv run gemini_search.py deep-research "query" --json

# Custom agent (default: deep-research-pro-preview-12-2025)
uv run gemini_search.py deep-research "query" --agent deep-research-pro-preview-12-2025
```

> **Note:** Deep Research is a blocking call that takes **1–3 minutes** to complete. A progress notice is printed to stderr at the start.

## When to use search vs deep-research

| | search | deep-research |
|---|---|---|
| Latency | 15–25 s | 1–3 min |
| Output | Synthesized paragraph + sources | Multi-section research report + sources |
| Use case | Current facts, news, quick lookups | In-depth research, complex topics |
| `--raw-urls` | Supported | Not supported |
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
  "agent": "deep-research-pro-preview-12-2025",
  "interaction_id": "...",
  "status": "completed",
  "answer": "...",
  "sources": [{"title": "...", "url": "..."}]
}
```

Note: `search_queries_used` is absent from deep-research output; `interaction_id` and `status` are absent from search output.

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
