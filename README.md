# gemini-search

Google Search via Gemini Grounding API — official Google web index, AI-synthesized answer + source URLs.

Instead of scraping or using a search API directly, this tool uses Gemini's `google_search` grounding tool to fire real Google searches and synthesize results into a natural-language answer with cited sources.

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

```bash
# Full output (synthesized answer + sources)
uv run gemini_search.py search "latest AI news"

# Sources only (no synthesis)
uv run gemini_search.py search "query" --raw-urls

# JSON output
uv run gemini_search.py search "query" --json

# Use Pro model (default: gemini-3-flash-preview)
uv run gemini_search.py search "query" --model gemini-3.1-pro-preview
```

## Output

- **ANSWER** — synthesized response grounded in Google's current web index
- **Google queries fired** — what Gemini actually searched
- **SOURCES** — source sites with titles and URLs

## Agent Skill

This repo includes a `SKILL.md` for use with AI coding agents (Claude Code, Cursor, Windsurf, OpenCode, and others).

### Install via skills CLI

```bash
npx skills add johanesalxd/gemini-search
```

### Manual install

```bash
# Copy to your agent's skills directory
cp SKILL.md ~/.claude/skills/gemini-search/SKILL.md         # Claude Code
cp SKILL.md .cursor/skills/gemini-search/SKILL.md           # Cursor
cp SKILL.md .opencode/skills/gemini-search/SKILL.md         # OpenCode
```

Once installed, your agent will load this skill when you ask for web searches or current information.

## License

MIT
