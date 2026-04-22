#!/usr/bin/env python3
"""
gemini_search.py — Google Search via Gemini Grounding API + Deep Research
Official Google web index, no scraping. Returns synthesized answer + raw source URLs.

Usage:
  uv run gemini_search.py search "query"
  uv run gemini_search.py search "query" --raw-urls   # sources only, no synthesis
  uv run gemini_search.py search "query" --json       # full JSON output
  uv run gemini_search.py search "query" --model gemini-3-flash-preview
  uv run gemini_search.py search "query" --file ./notes.md
  uv run gemini_search.py search "summarize this" --file ./report.pdf

  uv run gemini_search.py deep-research "query"
  uv run gemini_search.py deep-research "query" --json
  uv run gemini_search.py deep-research "query" --agent deep-research-preview-04-2026
  uv run gemini_search.py deep-research "what does this say about X?" --file ./brief.md
  uv run gemini_search.py deep-research "research based on this report" --file ./report.pdf
  uv run gemini_search.py deep-research "analyze this diagram" --file ./chart.png
  uv run gemini_search.py deep-research "follow-up question" --previous-interaction-id <id>

Env: GOOGLE_API_KEY (required)

File input support:
  search:        text files (.txt, .md, etc.) and PDF (<=20 MB) via Part.from_bytes
  deep-research: text files (inline string prepend); PDF and images via Files API
                 upload (client.files.upload) → typed document/image input list.
                 Audio and video are supported by the underlying agent API but are
                 deferred from the CLI (impractical for research workflows).

Continuation — two distinct modes:
  Deep Research agent (planning):
    agent="deep-research-preview-04-2026" + background=True + polling.
    Fresh one-shot research run (no --previous-interaction-id).

  Post-report follow-up Q&A (model-based):
    model="gemini-3.1-pro-preview" + previous_interaction_id. Synchronous, no polling.
    Docs: ai.google.dev/gemini-api/docs/deep-research#follow-up-questions-and-interactions
    Pass --previous-interaction-id <id> using the interaction_id from a prior deep-research run.
"""

import argparse
import base64
import json
import mimetypes
import os
import pathlib
import sys
import time
import warnings

_DEFAULT_SEARCH_MODEL = "gemini-3-flash-preview"
_DEFAULT_DEEP_RESEARCH_AGENT = "deep-research-preview-04-2026"
# Model used for post-report follow-up Q&A (model-based Interactions, not the Deep Research agent).
# Docs: ai.google.dev/gemini-api/docs/deep-research#follow-up-questions-and-interactions
_DEFAULT_FOLLOWUP_MODEL = "gemini-3.1-pro-preview"
_DEEP_RESEARCH_POLL_INTERVAL_SECONDS = 5
_IMAGE_OUTPUT_DIR_PREFIX = "/tmp/gemini-search-"


def _save_image(data_b64: str, interaction_id: str, index: int) -> str:
    """Decode a base64 image and save it to /tmp/gemini-search-<id>/.

    Args:
        data_b64: Base64-encoded image data.
        interaction_id: Interaction ID for directory naming.
        index: 1-based image index for file naming.

    Returns:
        Absolute path to the saved image file.
    """
    out_dir = pathlib.Path(f"{_IMAGE_OUTPUT_DIR_PREFIX}{interaction_id}")
    out_dir.mkdir(parents=True, exist_ok=True)
    image_bytes = base64.b64decode(data_b64)
    file_path = out_dir / f"image_{index:03d}.png"
    file_path.write_bytes(image_bytes)
    return str(file_path)


def _validate_file_path(file_path: str) -> pathlib.Path:
    """Resolve file_path and exit if it does not exist."""
    path = pathlib.Path(file_path)
    if not path.exists():
        print(f"ERROR: --file path does not exist: {file_path}", file=sys.stderr)
        sys.exit(1)
    return path


def get_api_key() -> str:
    """Read GOOGLE_API_KEY from environment or exit with an error."""
    key = os.environ.get("GOOGLE_API_KEY")
    if not key:
        print("ERROR: GOOGLE_API_KEY environment variable not set", file=sys.stderr)
        sys.exit(1)
    return key


def _make_client(key: str):
    from google import genai

    return genai.Client(api_key=key)


def _detect_mime(path: pathlib.Path) -> str:
    """Return best-guess MIME type for path. Unknown types become octet-stream."""
    mime, _ = mimetypes.guess_type(str(path))
    return mime or "application/octet-stream"


def _is_text_file(path: pathlib.Path) -> bool:
    """Return True if path is a known text-type file."""
    mime = _detect_mime(path)
    return mime.startswith("text/")


def _build_search_contents(query: str, file_path: str | None):
    """Build the contents argument for generate_content.

    Args:
        query: The user query string.
        file_path: Optional path to a local file to include as context.

    Returns:
        A string (no file) or list of Parts/strings (with file).
    """
    if not file_path:
        return query

    from google.genai import types

    path = _validate_file_path(file_path)
    mime = _detect_mime(path)
    if mime == "application/pdf":
        data = path.read_bytes()
        return [types.Part.from_bytes(data=data, mime_type="application/pdf"), query]
    else:
        # text/markdown/plain or any text file: read as string and prepend
        text = path.read_text(encoding="utf-8")
        return [text, query]


def _build_dr_input(query: str, file_path: str) -> str:
    """Prepend a text file's content to the query for Deep Research input.

    Args:
        query: The user query string.
        file_path: Path to a text file (caller must verify it exists and is text).

    Returns:
        A string with file content prepended, separated from the query.
    """
    path = pathlib.Path(file_path)
    content = path.read_text(encoding="utf-8")
    return f"[Document: {path.name}]\n\n{content}\n\n---\n\n{query}"


def _build_dr_multimodal_input(
    query: str,
    path: pathlib.Path,
    client,
    *,
    _upload_wait_seconds: int = 60,
) -> list:
    """Upload a local file to the Files API and return a typed input list.

    Supports PDF (document type) and images (image type). The file is uploaded
    to the Files API, polled until ACTIVE, then referenced by URI in the
    typed input list accepted by client.interactions.create.

    Args:
        query: The user query string.
        path: Local file path (must exist).
        client: Authenticated genai.Client.
        _upload_wait_seconds: Maximum seconds to wait for file to become ACTIVE.

    Returns:
        List of typed content dicts for the Interactions API.

    Raises:
        SystemExit: On upload failure or timeout.
    """
    mime = _detect_mime(path)
    if mime == "application/pdf":
        input_type = "document"
    elif mime.startswith("image/"):
        input_type = "image"
    else:
        print(
            f"ERROR: _build_dr_multimodal_input called with unsupported MIME {mime}",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        # Pass mime_type explicitly via config to avoid relying on SDK inference,
        # ensuring the upload MIME matches the typed input dict MIME.
        # SDK signature: files.upload(*, file, config: UploadFileConfig | None)
        file_obj = client.files.upload(file=str(path), config={"mime_type": mime})

        waited = 0
        while True:
            state = client.files.get(name=file_obj.name).state
            # state may be enum or string depending on SDK version
            state_str = state.name if hasattr(state, "name") else str(state)
            if state_str == "ACTIVE":
                break
            if waited >= _upload_wait_seconds:
                raise RuntimeError(
                    f"File did not become ACTIVE within {_upload_wait_seconds}s"
                )
            time.sleep(2)
            waited += 2
    except SystemExit:
        raise
    except Exception as e:
        print(f"ERROR: Failed to upload --file for deep-research: {e}", file=sys.stderr)
        sys.exit(1)

    return [
        {"type": "text", "text": query},
        {"type": input_type, "uri": file_obj.uri, "mime_type": mime},
    ]


def _run_search(query: str, model: str, client, file_path: str | None = None) -> dict:
    """Call generate_content with Google Search grounding and return structured result."""
    from google.genai import types

    grounding_tool = types.Tool(google_search=types.GoogleSearch())
    config = types.GenerateContentConfig(tools=[grounding_tool])
    contents = _build_search_contents(query, file_path)

    try:
        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=config,
        )
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    candidate = response.candidates[0] if response.candidates else None
    grounding_meta = None
    if candidate and hasattr(candidate, "grounding_metadata"):
        grounding_meta = candidate.grounding_metadata

    sources = []
    if grounding_meta and hasattr(grounding_meta, "grounding_chunks"):
        for chunk in grounding_meta.grounding_chunks or []:
            if hasattr(chunk, "web") and chunk.web:
                sources.append({
                    "title": chunk.web.title or "",
                    "url": chunk.web.uri or "",
                })

    search_queries = []
    if grounding_meta and hasattr(grounding_meta, "web_search_queries"):
        search_queries = list(grounding_meta.web_search_queries or [])

    return {
        "query": query,
        "model": model,
        "search_queries_used": search_queries,
        "answer": response.text or "",
        "sources": sources,
    }


def search(
    query: str,
    model: str = _DEFAULT_SEARCH_MODEL,
    raw_urls: bool = False,
    as_json: bool = False,
    file_path: str | None = None,
) -> None:
    """Run a grounded search and print results to stdout."""
    key = get_api_key()
    client = _make_client(key)
    result = _run_search(query, model, client, file_path=file_path)

    if as_json:
        print(json.dumps(result, indent=2))
        return

    if raw_urls:
        print(f"Query: {result['query']}")
        sq = result["search_queries_used"]
        print(f"Google queries fired: {', '.join(sq) if sq else 'n/a'}")
        print()
        print("=== SOURCES ===")
        for i, s in enumerate(result["sources"], 1):
            print(f"  [{i}] {s['title']}")
            print(f"       {s['url']}")
        return

    print(f"Query: {result['query']}")
    print(f"Model: {result['model']}")
    if result["search_queries_used"]:
        print(f"Google queries fired: {', '.join(result['search_queries_used'])}")
    print()
    print("=== ANSWER (grounded in Google Search) ===")
    print(result["answer"])
    print()
    if result["sources"]:
        print("=== SOURCES ===")
        for i, s in enumerate(result["sources"], 1):
            print(f"  [{i}] {s['title']}")
            print(f"       {s['url']}")
    else:
        print("=== SOURCES: none returned ===")


def _run_deep_research(
    query: str,
    agent: str,
    client,
    file_path: str | None = None,
    previous_interaction_id: str | None = None,
    followup_model: str = _DEFAULT_FOLLOWUP_MODEL,
    visualization: bool = True,
) -> dict:
    """Run a Deep Research interaction and return structured result.

    Fresh runs use agent-based background execution with polling.
    Follow-ups (previous_interaction_id set) use model-based synchronous Q&A.
    """
    # Dispatch input construction based on file type.
    if not file_path:
        dr_input: str | list = query
    else:
        path = _validate_file_path(file_path)
        mime = _detect_mime(path)
        if mime.startswith("text/"):
            dr_input = _build_dr_input(query, file_path)
        elif mime == "application/pdf" or mime.startswith("image/"):
            dr_input = _build_dr_multimodal_input(query, path, client)
        else:
            print(
                f"WARNING: --file with {mime} is not supported for deep-research; "
                "passing query only.",
                file=sys.stderr,
            )
            dr_input = query

    # Dispatch to the correct Interactions API path based on whether this is a fresh
    # Deep Research run or a post-report follow-up Q&A.
    #
    # Fresh research run: agent-based, background execution + polling.
    # Post-report follow-up: model-based, synchronous (no background, no polling).
    # Docs: ai.google.dev/gemini-api/docs/deep-research#follow-up-questions-and-interactions
    if previous_interaction_id:
        # Model-based follow-up: use a standard Gemini model, not the Deep Research agent.
        # The API uses previous_interaction_id to load the conversation history from the
        # completed Deep Research interaction. No background=True; call is synchronous.
        create_kwargs: dict = {
            "model": followup_model,
            "input": dr_input,
            "previous_interaction_id": previous_interaction_id,
        }
        is_followup = True
    else:
        # Fresh Deep Research run: agent-based with background execution + polling.
        # agent_config controls agent behavior (type, visualization, etc.).
        # Docs: ai.google.dev/gemini-api/docs/deep-research#agent-configuration
        agent_config = {
            "type": "deep-research",
            "visualization": "auto" if visualization else "off",
        }
        create_kwargs = {
            "agent": agent,
            "input": dr_input,
            "background": True,
            "agent_config": agent_config,
        }
        is_followup = False

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=".*Interactions usage is experimental.*",
                category=UserWarning,
            )
            interaction = client.interactions.create(**create_kwargs)

            if not is_followup:
                # Agent-based Deep Research runs asynchronously; poll until complete.
                while interaction.status == "in_progress":
                    time.sleep(_DEEP_RESEARCH_POLL_INTERVAL_SECONDS)
                    interaction = client.interactions.get(interaction.id)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    if interaction.status != "completed":
        print(
            f"ERROR: Deep Research interaction ended with status '{interaction.status}'",
            file=sys.stderr,
        )
        sys.exit(1)

    answer = ""
    sources = []
    images = []
    seen_urls: set[str] = set()
    image_index = 0

    for content in interaction.outputs or []:
        # interaction.outputs contains discriminated-union content items, each
        # with a `type` field. The two types relevant here are:
        #   - "text"                 → TextContent: has .text (str) and
        #                              optional .annotations (citation sources)
        #   - "google_search_result" → GoogleSearchResultContent: has
        #                              .result (List[GoogleSearchResult]), each
        #                              with .title, .url, .rendered_content
        content_type = getattr(content, "type", None)

        if content_type == "text":
            # Accumulate the synthesized research report text.
            text = getattr(content, "text", None)
            if text:
                answer += text
            # TextContent.annotations is documented in the Interactions API
            # reference (ai.google.dev/api/interactions-api, TextContent section):
            #   annotations: Annotation (optional) — citation info for model-generated
            #   content. Polymorphic on `type`. UrlCitation (type="url_citation") has
            #   .url, .title, .start_index, .end_index. This matches the canonical
            #   Python handling example in the interactions docs (URL context section):
            #     for annotation in output.annotations:
            #         if annotation.get("type") == "url_citation":
            #             print(annotation["url"])
            # Deep Research can surface url_citation annotations for cited URLs.
            # appear in practice. The SDK may deserialize them as dicts or objects
            # depending on version; handle both defensively.
            for ann in getattr(content, "annotations", None) or []:
                # SDK object path
                ann_type = getattr(ann, "type", None)
                if ann_type is None and isinstance(ann, dict):
                    ann_type = ann.get("type")
                if ann_type == "url_citation":
                    url = (getattr(ann, "url", None) or
                           (ann.get("url") if isinstance(ann, dict) else None) or "")
                    title = (getattr(ann, "title", None) or
                             (ann.get("title") if isinstance(ann, dict) else None) or url)
                    if url and url not in seen_urls:
                        seen_urls.add(url)
                        sources.append({"title": title, "url": url})

        elif content_type == "image":
            # Image content from visualization. The agent generates charts/graphs
            # as base64-encoded image data when visualization="auto".
            # Docs: ai.google.dev/gemini-api/docs/deep-research#visualization
            data_b64 = getattr(content, "data", None)
            if data_b64:
                image_index += 1
                try:
                    saved_path = _save_image(
                        data_b64, interaction.id, image_index
                    )
                    images.append({"path": saved_path, "index": image_index})
                except Exception as e:
                    print(
                        f"WARNING: Failed to save image {image_index}: {e}",
                        file=sys.stderr,
                    )

        elif content_type == "google_search_result":
            # Each result item has .title and .url.
            for result_item in getattr(content, "result", None) or []:
                url = getattr(result_item, "url", None) or ""
                title = getattr(result_item, "title", None) or url
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    sources.append({"title": title, "url": url})

    result: dict = {
        "query": query,
        "agent": agent,
        "interaction_id": interaction.id,
        "status": interaction.status,
        "answer": answer.strip(),
        "sources": sources,
        "images": images,
    }
    if previous_interaction_id:
        result["previous_interaction_id"] = previous_interaction_id
        result["followup_model"] = followup_model
    return result


def deep_research(
    query: str,
    agent: str = _DEFAULT_DEEP_RESEARCH_AGENT,
    as_json: bool = False,
    file_path: str | None = None,
    previous_interaction_id: str | None = None,
    followup_model: str = _DEFAULT_FOLLOWUP_MODEL,
    visualization: bool = True,
) -> None:
    """Run deep research or post-report follow-up and print results to stdout."""
    if previous_interaction_id:
        print(
            "Post-report follow-up in progress (model-based Q&A)...",
            file=sys.stderr,
        )
    else:
        print(
            "Deep Research in progress — this may take 1–3 minutes...",
            file=sys.stderr,
        )
    key = get_api_key()
    client = _make_client(key)
    result = _run_deep_research(
        query,
        agent,
        client,
        file_path=file_path,
        previous_interaction_id=previous_interaction_id,
        followup_model=followup_model,
        visualization=visualization,
    )

    if as_json:
        print(json.dumps(result, indent=2))
        return

    print(f"Query: {result['query']}")
    if "followup_model" in result:
        print(f"Follow-up Model: {result['followup_model']} (post-report Q&A)")
        print(f"Prior Interaction ID: {result['previous_interaction_id']}")
    else:
        print(f"Agent: {result['agent']}")
    print(f"Status: {result['status']}")
    print(f"Interaction ID: {result['interaction_id']}")
    print()
    print("=== DEEP RESEARCH REPORT ===")
    print(result["answer"])
    print()
    if result.get("images"):
        print("=== IMAGES ===")
        for img in result["images"]:
            print(f"  [{img['index']}] {img['path']}")
        print()
    if result["sources"]:
        print("=== SOURCES ===")
        for i, s in enumerate(result["sources"], 1):
            title = s.get("title", "")
            url = s.get("url", "")
            print(f"  [{i}] {title}")
            if url:
                print(f"       {url}")
    else:
        print("=== SOURCES: none returned ===")


def main() -> None:
    """CLI entrypoint — parse args and dispatch to search or deep_research."""
    parser = argparse.ArgumentParser(
        description="Google Search via Gemini Grounding API + Deep Research"
    )
    parser.add_argument(
        "command",
        choices=["search", "deep-research"],
        help="Command to run: 'search' (fast, grounded) or 'deep-research' (thorough, slow)",
    )
    parser.add_argument("query", help="Query string")
    parser.add_argument(
        "--model",
        default=_DEFAULT_SEARCH_MODEL,
        help=f"Gemini model for search (default: {_DEFAULT_SEARCH_MODEL}). Ignored for deep-research.",
    )
    parser.add_argument(
        "--agent",
        default=_DEFAULT_DEEP_RESEARCH_AGENT,
        help=(
            f"Agent identifier for deep-research (default: {_DEFAULT_DEEP_RESEARCH_AGENT}). "
            "Ignored for search."
        ),
    )
    parser.add_argument(
        "--raw-urls",
        action="store_true",
        help="Return source URLs only — search mode only, not supported for deep-research.",
    )
    parser.add_argument("--json", action="store_true", help="Full JSON output")
    parser.add_argument(
        "--file",
        metavar="PATH",
        default=None,
        help=(
            "Path to a local file to include as context for the query. "
            "search: supports text files and PDF (<=20 MB, inline bytes). "
            "deep-research: supports text files (inline), PDF and images (Files API upload). "
            "Unsupported types emit a warning and are skipped."
        ),
    )
    parser.add_argument(
        "--previous-interaction-id",
        metavar="ID",
        default=None,
        help=(
            "Interaction ID from a prior deep-research run to continue from. "
            "When provided, the new request is sent as a follow-up turn using the "
            "previous_interaction_id field in the Interactions API. "
            "Only valid for deep-research; ignored for search."
        ),
    )
    parser.add_argument(
        "--followup-model",
        default=_DEFAULT_FOLLOWUP_MODEL,
        help=(
            f"Gemini model for post-report follow-up Q&A (default: {_DEFAULT_FOLLOWUP_MODEL}). "
            "Only used with --previous-interaction-id; ignored otherwise."
        ),
    )
    parser.add_argument(
        "--no-visualization",
        action="store_true",
        help=(
            "Disable agent-generated charts and images in deep-research output. "
            "By default, visualization is enabled (visualization='auto'). "
            "Only applies to deep-research; ignored for search."
        ),
    )
    args = parser.parse_args()

    if args.command == "search":
        search(
            args.query,
            model=args.model,
            raw_urls=args.raw_urls,
            as_json=args.json,
            file_path=args.file,
        )
    elif args.command == "deep-research":
        if args.raw_urls:
            print(
                "WARNING: --raw-urls is not supported for deep-research; ignoring.",
                file=sys.stderr,
            )
        deep_research(
            args.query,
            agent=args.agent,
            as_json=args.json,
            file_path=args.file,
            previous_interaction_id=args.previous_interaction_id,
            followup_model=args.followup_model,
            visualization=not args.no_visualization,
        )


if __name__ == "__main__":
    main()
