"""Convert data export from claude.ai to Markdown."""

import json
import logging
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Final, cast

if TYPE_CHECKING:
    from pathlib import Path

from chatclerk.argparse import Args, build_argument_parser
from chatclerk.datetime import iso_timestamp_to_str

logger: Final = logging.getLogger(__name__)


class Language(Enum):
    """Supported language types for code block formatting.

    Attributes:
        JSON: JSON syntax highlighting.
        XML: XML syntax highlighting.
        TEXT: Plain text (no syntax highlighting).
    """

    JSON = "json"
    XML = "xml"
    TEXT = ""


@dataclass(frozen=True)
class ToolOutput:
    """Representation of formatted tool result content with language hint.

    Attributes:
        text: The formatted text content.
        language: Optional `Language` enum specifying syntax highlighting.
    """

    text: str
    language: Language | None = None

    @property
    def code_block_language(self) -> str:
        """Get the language identifier for Markdown code blocks.

        Returns:
            `Language` identifier string, or empty string if no language specified.
        """
        if self.language is None:
            return ""
        return self.language.value


def _format_json_if_valid(text: str) -> ToolOutput:
    """Attempt to parse and format text as JSON or XML with syntax highlighting.

    Args:
        text: The text to analyse and format.

    Returns:
        Formatted text with appropriate language hint for syntax highlighting.
    """
    text = text.strip()
    if not text:
        return ToolOutput(text=text, language=Language.TEXT)

    if text.startswith("<"):
        return ToolOutput(text=text, language=Language.XML)

    if not text.startswith(("{", "[")):
        return ToolOutput(text=text, language=Language.TEXT)

    try:
        formatted = json.dumps(json.loads(text), indent=2)
        return ToolOutput(text=formatted, language=Language.JSON)
    except json.JSONDecodeError, ValueError:
        return ToolOutput(text=text, language=Language.TEXT)


def _code_block(text: str, language: str = "") -> str:
    """Fence content without letting embedded backticks close the block."""
    fence = "```"
    while fence in text:
        fence += "`"
    return f"{fence}{language}\n{text.rstrip()}\n{fence}"


def _format_metadata(item: dict[str, Any], handled: set[str]) -> str:
    """Preserve populated fields not rendered elsewhere."""
    metadata = {
        k: v
        for k, v in item.items()
        if k not in handled and v not in (None, "", [], {})
    }
    if not metadata:
        return ""
    return "*Metadata:*\n" + _code_block(
        json.dumps(metadata, indent=2, ensure_ascii=False), "json"
    )


def _format_citations(citations: list[dict[str, Any]]) -> str | None:
    """Format citations as a numbered Markdown list.

    Args:
        citations: List of citation dictionaries.

    Returns:
        Formatted Markdown string of citations, or `None` if no valid citations.
    """
    if not citations:
        return None

    citation_lines = ["", "*Citations:*"]
    for i, citation in enumerate(citations, 1):
        details = citation.get("details") or {}
        if url := details.get("url", ""):
            citation_lines.append(f"{i}. [{url}]({url})")
        if metadata := _format_metadata(citation, set()):
            citation_lines.append(metadata)

    min_citation_lines = 2
    if len(citation_lines) > min_citation_lines:
        logger.debug("Formatted %d citations", len(citation_lines) - min_citation_lines)
        return "\n".join(citation_lines)
    return None


def _format_artifact(item: dict[str, Any]) -> str:
    """Format an artifact tool use as Markdown.

    Args:
        item: Dictionary containing artifact data from tool use.

    Returns:
        Formatted Markdown string representing the artifact.
    """
    artifact_input = item["input"]
    artifact_title = artifact_input.get("title", "Untitled Artifact")
    artifact_type = artifact_input.get("type", "")
    artifact_content = artifact_input.get("content", "")
    artifact_id = artifact_input.get("id", "")

    logger.debug("Artifact: %s", artifact_title)

    artifact_header = f"### Artifact: {artifact_title}"
    if artifact_id:
        artifact_header += f"\n*Type: {artifact_type} | ID: {artifact_id}*"

    parts = [artifact_header]
    if artifact_content:
        if artifact_type == "text/markdown":
            parts.append(f"---\n\n{artifact_content.rstrip()}\n\n---")
        else:
            lang = artifact_input.get("language") or ""
            if not lang and artifact_type.startswith("text/"):
                lang = artifact_type.removeprefix("text/")
            parts.append(_code_block(artifact_content, lang))
    if metadata := _format_metadata(
        artifact_input, {"title", "type", "content", "id", "language"}
    ):
        parts.append(metadata)
    return "\n\n".join(parts)


def _format_web_search_input(tool_input: dict[str, Any]) -> str | None:
    """Format `web_search` tool input for display.

    Args:
        tool_input: Dictionary containing tool input parameters.

    Returns:
        Formatted search query string, or `None` if no query present.
    """
    if query := tool_input.get("query", ""):
        logger.debug("Web search: %s", query)
        return f"*[Searching for: {query}]*"
    return None


def _format_web_fetch_input(tool_input: dict[str, Any]) -> str | None:
    """Format `web_fetch` tool input for display.

    Args:
        tool_input: Dictionary containing tool input parameters.

    Returns:
        Formatted URL string, or `None` if no URL present.
    """
    if url := tool_input.get("url", ""):
        logger.debug("Web fetch: %s", url)
        return f"*[Fetching: {url}]*"
    return None


def _format_repl_input(tool_input: dict[str, Any]) -> str | None:
    """Format REPL tool input for display.

    Args:
        tool_input: Dictionary containing tool input parameters.

    Returns:
        Formatted code block string, or `None` if no code present.
    """
    if code := tool_input.get("code", ""):
        logger.debug("REPL: %d chars", len(code))
        return "*[Code executed]*\n" + _code_block(code.strip(), "javascript")
    return None


def _format_tool_input(tool_name: str, tool_input: dict[str, Any]) -> str | None:
    """Format tool input parameters for display.

    Known tools use a short description; other inputs are preserved as JSON.

    Args:
        tool_name: Name of the tool being used.
        tool_input: Dictionary containing tool input parameters.

    Returns:
        Formatted input string, or `None` if empty.
    """
    if not tool_input:
        return None

    handlers = {
        "web_search": (_format_web_search_input, "query"),
        "web_fetch": (_format_web_fetch_input, "url"),
        "repl": (_format_repl_input, "code"),
    }

    parts = []
    handled = None
    if tool_name in handlers:
        handler, handled = handlers[tool_name]
        if formatted := handler(tool_input):
            parts.append(formatted)
    remaining = {k: v for k, v in tool_input.items() if k != handled}
    if remaining:
        parts.append(
            f"*[Tool Input: {tool_name}]*\n"
            + _code_block(json.dumps(remaining, indent=2, ensure_ascii=False), "json")
        )
    return "\n\n".join(parts) or None


def _format_display_content(
    display_content: dict[str, Any], tool_name: str
) -> str | None:
    """Format rich display content from tool results as Markdown.

    Args:
        display_content: Dictionary containing display content metadata.
        tool_name: Name of the tool that generated the content.

    Returns:
        Formatted Markdown string, or `None` if no displayable content.
    """
    if not display_content:
        return None

    display_type = display_content.get("type", "")
    logger.debug("Display_content: type=%s, tool=%s", display_type, tool_name)

    parts = [f"*[Tool Result: {tool_name}]*"]
    if display_type in {"rich_link", "rich_content"}:
        content_items = (
            [display_content.get("link") or {}]
            if display_type == "rich_link"
            else display_content.get("content", [])
        )
        for content_item in content_items:
            url = content_item.get("url")
            title = content_item.get("title") or url or "Untitled"
            parts.append(f"- [{title}]({url})" if url else f"- {title}")
            if metadata := _format_metadata(content_item, {"title", "url"}):
                parts.append(metadata)
    elif display_type == "text":
        parts.append(display_content.get("text", ""))
    else:
        parts.append(_code_block(json.dumps(display_content, indent=2), "json"))
    if metadata := _format_metadata(
        display_content, {"type", "link", "content", "text"}
    ):
        parts.append(metadata)
    return "\n\n".join(parts)


def _format_tool_result_text(result_text: str, tool_name: str) -> str | None:
    """Format text content from a tool result as a code block.

    Args:
        result_text: The text content returned by the tool.
        tool_name: Name of the tool that generated the result.

    Returns:
        Formatted Markdown code block, or `None` if empty or just `"OK"`.
    """
    if not result_text or result_text == "OK":
        return None

    formatted = _format_json_if_valid(result_text)
    return f"*[Tool Result: {tool_name}]*\n" + _code_block(
        formatted.text, formatted.code_block_language
    )


def _process_text_content(item: dict[str, Any], text_parts: list[str]) -> None:
    """Process a text content item and append formatted text.

    Args:
        item: Dictionary containing text content and citations.
        text_parts: List to append formatted text parts to.
    """
    if text := item["text"].strip():
        text_parts.append(text)

    if citations_text := _format_citations(item.get("citations") or []):
        text_parts.append(citations_text)


def _process_tool_use(item: dict[str, Any], text_parts: list[str]) -> str:
    """Process a `tool_use` content item and append formatted output.

    Args:
        item: Dictionary containing tool use data.
        text_parts: List to append formatted tool use parts to.

    Returns:
        The name of the tool that was used.
    """
    tool_name = cast("str", item["name"])

    if tool_name == "artifacts":
        text_parts.append(_format_artifact(item))

    elif tool_input_text := _format_tool_input(tool_name, item["input"]):
        text_parts.append(tool_input_text)

    if display_text := _format_display_content(
        item.get("display_content") or {}, tool_name
    ):
        text_parts.append(display_text)
    return tool_name


def _process_tool_result_item(
    item: dict[str, Any], tool_name: str, text_parts: list[str]
) -> None:
    """Render text and search results, preserving other result types as JSON."""
    result_type = item.get("type")
    if result_type == "text":
        if result_text := _format_tool_result_text(item["text"], tool_name):
            text_parts.append(result_text)
        handled = {"type", "text"}
    elif result_type == "knowledge":
        url = item.get("url")
        title = item.get("title") or url or "Search result"
        text_parts.append(f"### [{title}]({url})" if url else f"### {title}")
        if text := item.get("text"):
            text_parts.append(text)
        handled = {"type", "title", "url", "text"}
    else:
        handled = set()
    if metadata := _format_metadata(item, handled):
        text_parts.append(metadata)


def _process_tool_result(item: dict[str, Any], text_parts: list[str]) -> None:
    """Process a `tool_result` content item and append formatted output.

    Args:
        item: Dictionary containing tool result data.
        text_parts: List to append formatted result parts to.
    """
    tool_name = item["name"]

    if display_text := _format_display_content(
        item.get("display_content") or {}, tool_name
    ):
        text_parts.append(display_text)

    result_content = item["content"]
    if isinstance(result_content, str):
        if result_text := _format_tool_result_text(result_content, tool_name):
            text_parts.append(result_text)
    elif isinstance(result_content, list):
        for result_item in result_content:
            if not isinstance(result_item, dict):
                text_parts.append(_code_block(json.dumps(result_item), "json"))
                continue
            _process_tool_result_item(result_item, tool_name, text_parts)
    elif result_content:
        text_parts.append(_code_block(json.dumps(result_content, indent=2), "json"))

    if structured := item.get("structured_content"):
        text_parts.append(
            "*Structured result:*\n"
            + _code_block(json.dumps(structured, indent=2, ensure_ascii=False), "json")
        )


def _extract_text_from_content(content: list[dict[str, Any]]) -> str:
    """Extract and format text content from a message's content array.

    Processes text, tool use, and tool result items into a unified Markdown
    string.

    Args:
        content: List of content item dictionaries.

    Returns:
        Formatted Markdown string combining all content items.
    """
    text_parts: list[str] = []
    tool_uses: list[str] = []

    for item in content:
        content_type = item["type"]

        handled = {"type"}
        if content_type == "text":
            _process_text_content(item, text_parts)
            handled |= {"text", "citations"}
        elif content_type == "thinking":
            text_parts.append("### Thinking")
            if thinking := item.get("thinking"):
                text_parts.append(thinking)
            text_parts.extend(
                summary["summary"] if isinstance(summary, dict) else summary
                for summary in item.get("summaries") or []
            )
            handled |= {"thinking", "summaries"}
        elif content_type == "tool_use":
            tool_name = _process_tool_use(item, text_parts)
            tool_uses.append(tool_name)
            handled |= {"name", "input", "display_content"}
        elif content_type == "tool_result":
            _process_tool_result(item, text_parts)
            handled |= {"name", "content", "display_content", "structured_content"}
        else:
            handled = set()
        if metadata := _format_metadata(item, handled):
            text_parts.append(metadata)

    result = "\n\n".join(text_parts)

    if tool_uses:
        tool_summary = f"*[Used tools: {', '.join(tool_uses)}]*"
        result = f"{result}\n\n{tool_summary}" if result else tool_summary

    return result


def _format_file_item(file_item: dict[str, Any]) -> str:
    """Format a single file or attachment item as Markdown.

    Args:
        file_item: Dictionary containing file metadata.

    Returns:
        Formatted Markdown list item with file details.
    """
    file_name = file_item.get("file_name") or file_item.get("file_uuid") or "unknown"
    file_size = file_item.get("file_size", "")
    file_type = file_item.get("file_type", "")

    info = f"- `{file_name}`"
    if file_type:
        info += f" (type: {file_type})"
    if file_size:
        info += f" (size: {file_size} bytes)"

    if extracted := file_item.get("extracted_content"):
        info += "\n\n" + _code_block(extracted)
    if metadata := _format_metadata(
        file_item, {"file_name", "file_size", "file_type", "extracted_content"}
    ):
        info += "\n\n" + metadata
    return info


def _format_attachments(attachments: list[dict[str, Any]]) -> str | None:
    """Format attachments list as Markdown.

    Args:
        attachments: List of attachment metadata dictionaries.

    Returns:
        Formatted Markdown list of attachments, or `None` if empty.
    """
    if not attachments:
        return None

    attachment_info = [_format_file_item(attachment) for attachment in attachments]

    if attachment_info:
        return f"*[Attachments: {len(attachments)}]*\n" + "\n".join(attachment_info)
    return None


def _format_files(files: list[dict[str, Any]]) -> str | None:
    """Format files list as Markdown.

    Args:
        files: List of file metadata dictionaries.

    Returns:
        Formatted Markdown list of files, or `None` if empty.
    """
    if not files:
        return None

    file_info = [_format_file_item(file) for file in files]

    if file_info:
        return f"*[Files: {len(files)}]*\n" + "\n".join(file_info)

    return None


def _format_message(
    message: dict[str, Any], message_ids: set[str] | None = None
) -> str:
    """Format a single chat message as Markdown.

    Args:
        message: Dictionary containing message data.
        message_ids: IDs available for parent links in this conversation.

    Returns:
        The formatted message as a Markdown string.
    """
    sender = message["sender"]
    created_at = message["created_at"]

    logger.debug("Message: %s (%d items)", sender, len(message["content"]))

    header = f"## {sender.title()}"
    timestamp = iso_timestamp_to_str(created_at)

    content = _extract_text_from_content(message["content"])
    content_parts: list[str] = []
    if uuid := message.get("uuid"):
        header = f'<a id="message-{uuid}"></a>\n\n{header}'
        content_parts.append(f"*UUID: {uuid}*")
    if parent := message.get("parent_message_uuid"):
        parent_text = (
            f"[{parent}](#message-{parent})"
            if message_ids is not None and parent in message_ids
            else f"{parent} (not included in export)"
        )
        content_parts.append(f"*Parent: {parent_text}*")
    if (fallback_text := message["text"].strip()) and not any(
        item["type"] == "text" and item.get("text", "").strip()
        for item in message["content"]
    ):
        content_parts.append(fallback_text)
    if content:
        content_parts.append(content)

    if attachments_text := _format_attachments(message["attachments"]):
        content_parts.append(attachments_text)

    if files_text := _format_files(message["files"]):
        content_parts.append(files_text)

    if metadata := _format_metadata(
        message,
        {
            "uuid",
            "parent_message_uuid",
            "sender",
            "created_at",
            "content",
            "text",
            "attachments",
            "files",
        },
    ):
        content_parts.append(metadata)
    return f"{header}\n\n*{timestamp}*\n\n" + "\n\n".join(content_parts)


def _convert_to_markdown(conversation: dict[str, Any]) -> str:
    """Convert a Claude conversation to Markdown format.

    Args:
        conversation: Dictionary containing the conversation data.

    Returns:
        The formatted conversation as a Markdown string.
    """
    uuid = conversation["uuid"]
    name = conversation["name"]
    created_at = iso_timestamp_to_str(conversation["created_at"])
    updated_at = iso_timestamp_to_str(conversation["updated_at"])

    logger.info(
        "Converting: %s (%s, %d messages)",
        name,
        uuid,
        len(conversation["chat_messages"]),
    )

    header_parts = [
        f"# {name}\n",
        f"- **UUID:** {uuid}",
        f"- **Created:** {created_at}",
        f"- **Updated:** {updated_at}\n",
    ]

    if summary := conversation.get("summary"):
        header_parts.append(f"## Summary\n\n{summary}\n")
    if metadata := _format_metadata(
        conversation,
        {"uuid", "name", "created_at", "updated_at", "summary", "chat_messages"},
    ):
        header_parts.append(metadata + "\n")

    messages: list[dict[str, Any]] = conversation["chat_messages"]
    message_ids = {message["uuid"] for message in messages}
    message_blocks = [_format_message(message, message_ids) for message in messages]

    header = "\n".join(header_parts)
    messages_section = "\n\n".join(message_blocks)

    return f"{header}\n{messages_section}"


def _has_content(conversation: dict[str, Any]) -> bool:
    """Check whether a conversation contains meaningful content.

    Args:
        conversation: Dictionary containing the conversation data.

    Returns:
        `True` if the conversation has a name, messages, or attachments,
        otherwise `False`.
    """
    if conversation["name"].strip() or conversation.get("summary"):
        return True

    messages = conversation["chat_messages"]
    for message in messages:
        if message["text"].strip():
            return True

        for content_item in message["content"]:
            if any(
                value.strip() if isinstance(value, str) else value
                for key, value in content_item.items()
                if key not in {"type", "start_timestamp", "stop_timestamp", "flags"}
            ):
                return True

        if message.get("attachments") or message.get("files"):
            return True

    return False


def _convert_memories(data: dict[str, Any] | list[Any]) -> str:
    """Render conversation memory and memory files with their metadata."""
    parts = ["# Memories"]
    records = data if isinstance(data, list) else [data]
    for record in records:
        if not isinstance(record, dict):
            parts.append(_code_block(json.dumps(record, indent=2), "json"))
            continue
        if memory := record.get("conversations_memory"):
            parts.append(f"## Conversation memory\n\n{memory}")
        for memory_file in record.get("memory_files") or []:
            parts.append(f"## {memory_file.get('path') or 'Memory file'}")
            parts.append(memory_file.get("content") or "")
            if metadata := _format_metadata(memory_file, {"path", "content"}):
                parts.append(metadata)
        if metadata := _format_metadata(
            record, {"conversations_memory", "memory_files"}
        ):
            parts.append(metadata)
    return "\n\n".join(parts) + "\n"


def _export_memories(input_dir: Path, output_dir: Path) -> None:
    """Convert memories.json and memories/*.json into memories.md."""
    paths = sorted((input_dir / "memories").glob("*.json"))
    legacy_path = input_dir / "memories.json"
    if legacy_path.is_file():
        paths.insert(0, legacy_path)
    if not paths:
        return

    records: list[Any] = []
    for path in paths:
        with path.open() as file:
            data = json.load(file)
        records.extend(data if isinstance(data, list) else [data])
        logger.info("Loaded memories: %s", path.relative_to(input_dir))

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "memories.md").write_text(_convert_memories(records))


def main() -> None:
    """Convert Claude conversation exports to Markdown files.

    Reads `conversations.json` from the input directory, converts each
    conversation with content to Markdown, and writes the results to the output
    directory. Optional memories.json and memories/*.json files are exported too.
    """
    parser = build_argument_parser("claude.ai")
    args = parser.parse_args(namespace=Args())

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="[%(levelname)-8s] %(message)s")

    conversations_path = args.input_dir.joinpath("conversations.json")
    with conversations_path.open() as f:
        conversations = json.load(f)

    logger.info("Loaded %d conversations", len(conversations))

    args.output_dir.mkdir(parents=True, exist_ok=True)

    exported_count = 0
    skipped_count = 0

    for conversation in conversations:
        if not _has_content(conversation):
            logger.debug("Skipping empty conversation (%s)", conversation["uuid"])
            skipped_count += 1
            continue

        uuid = conversation["uuid"]
        md_content = _convert_to_markdown(conversation)
        output_file = args.output_dir.joinpath(f"{uuid}.md")
        output_file.write_text(md_content)

        exported_count += 1

    _export_memories(args.input_dir, args.output_dir)
    logger.info(
        "Export complete: %d exported, %d skipped", exported_count, skipped_count
    )
