#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.14"
# dependencies = []
# ///


"""Convert data export from claude.ai to Markdown."""

# pyright: reportAny=false, reportExplicitAny=false

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.DEBUG, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class Language(Enum):
    """Supported language types for code blocks."""

    JSON = "json"
    XML = "xml"
    TEXT = ""


@dataclass
class ToolOutput:
    """Represents formatted tool result content with optional language hint."""

    text: str
    language: Language | None = None

    @property
    def code_block_language(self) -> str:
        """Get the language identifier for markdown code blocks."""
        if self.language is None:
            return ""
        return self.language.value


def format_timestamp(timestamp_str: str) -> str:
    """Convert ISO timestamp to readable format."""
    timestamp = datetime.fromisoformat(timestamp_str)
    return timestamp.strftime("%Y-%m-%d %H:%M:%S UTC")


def format_json_if_valid(text: str) -> ToolOutput:
    """Try to parse and format text as JSON/XML with appropriate language hint."""
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
    except (json.JSONDecodeError, ValueError):
        return ToolOutput(text=text, language=Language.TEXT)


def format_citations(citations: list[dict[str, Any]]) -> str | None:
    """Format citations as a markdown list."""
    if not citations:
        return None

    citation_lines = ["", "*Citations:*"]
    for i, citation in enumerate(citations, 1):
        details = citation.get("details", {})  # details can be null
        url = details.get("url", "") if details else ""
        if url:
            citation_lines.append(f"{i}. [{url}]({url})")

    min_citation_lines = 2
    if len(citation_lines) > min_citation_lines:
        logger.debug(
            "  Formatted %d citations", len(citation_lines) - min_citation_lines
        )
        return "\n".join(citation_lines)
    return None


def format_artifact(item: dict[str, Any]) -> str | None:
    """Format an artifact tool use as markdown."""
    artifact_input = item["input"]
    artifact_title = artifact_input.get("title", "Untitled Artifact")
    artifact_type = artifact_input.get("type", "")
    artifact_content = artifact_input.get("content", "")
    artifact_id = artifact_input.get("id", "")

    if not artifact_content:
        return None

    logger.debug("  Artifact: %s", artifact_title)

    artifact_header = f"### Artifact: {artifact_title}"
    if artifact_id:
        artifact_header += f"\n*Type: {artifact_type} | ID: {artifact_id}*"

    if artifact_type == "text/markdown":
        return f"{artifact_header}\n\n---\n\n{artifact_content.rstrip()}\n\n---"

    lang = ""
    if artifact_type == "application/vnd.ant.code":
        lang = "python"
    elif artifact_type.startswith("text/"):
        lang = artifact_type.replace("text/", "")

    if lang:
        return f"{artifact_header}\n\n```{lang}\n{artifact_content.rstrip()}\n```"
    return f"{artifact_header}\n\n```\n{artifact_content.rstrip()}\n```"


def _format_web_search_input(tool_input: dict[str, Any]) -> str | None:
    """Format web_search tool input."""
    query = tool_input.get("query", "")
    if query:
        logger.debug("  Web search: %s", query)
        return f"*[Searching for: {query}]*"
    return None


def _format_web_fetch_input(tool_input: dict[str, Any]) -> str | None:
    """Format web_fetch tool input."""
    url = tool_input.get("url", "")
    if url:
        logger.debug("  Web fetch: %s", url)
        return f"*[Fetching: {url}]*"
    return None


def _format_repl_input(tool_input: dict[str, Any]) -> str | None:
    """Format repl tool input."""
    code = tool_input.get("code", "")
    if code:
        logger.debug("  REPL: %d chars", len(code))
        return f"*[Code executed]*\n```javascript\n{code.strip()}\n```"
    return None


def format_tool_input(tool_name: str, tool_input: dict[str, Any]) -> str | None:
    """Format tool input for display (for interesting tools only)."""
    if not tool_input:  # Can be empty dict
        return None

    handlers = {
        "web_search": _format_web_search_input,
        "web_fetch": _format_web_fetch_input,
        "repl": _format_repl_input,
    }

    handler = handlers.get(tool_name)
    if handler:
        return handler(tool_input)

    return None


def format_display_content(
    display_content: dict[str, Any], tool_name: str
) -> str | None:
    """Format rich display content from tool results."""
    if not display_content:  # Can be null or empty
        return None

    display_type = display_content.get("type", "")
    logger.debug("  Display_content: type=%s, tool=%s", display_type, tool_name)

    if display_type == "rich_link":
        link = display_content.get("link", {})
        url = link.get("url", "")
        title = link.get("title", url)
        if url:
            logger.debug("  Rich link: %s", title)
            return f"*[Tool Result: {tool_name}]*\n- [{title}]({url})"

    elif display_type == "rich_content":
        content_items = display_content.get("content", [])
        if content_items:
            links: list[str] = []
            for content_item in content_items:
                title = content_item.get("title", "")
                url = content_item.get("url", "")
                if title and url:
                    links.append(f"- [{title}]({url})")
            if links:
                logger.debug("  Rich content: %d items", len(links))
                return f"*[Tool Result: {tool_name}]*\n" + "\n".join(links)

    return None


def format_tool_result_text(result_text: str, tool_name: str) -> str | None:
    """Format text content from a tool result."""
    if not result_text or result_text == "OK":
        return None

    formatted = format_json_if_valid(result_text)
    clean_text = formatted.text.rstrip()
    lang = formatted.code_block_language

    if lang:
        return f"*[Tool Result: {tool_name}]*\n```{lang}\n{clean_text}\n```"
    return f"*[Tool Result: {tool_name}]*\n```\n{clean_text}\n```"


def _process_text_content(item: dict[str, Any], text_parts: list[str]) -> None:
    """Process a 'text' content item and append to text_parts."""
    text = item["text"].strip()
    if text:
        text_parts.append(text)

    citations_text = format_citations(item["citations"])
    if citations_text:
        text_parts.append(citations_text)


def _process_tool_use(item: dict[str, Any], text_parts: list[str]) -> str:
    """Process a 'tool_use' content item and return the tool name."""
    tool_name = item["name"]

    if tool_name == "artifacts":
        artifact_text = format_artifact(item)
        if artifact_text:
            text_parts.append(artifact_text)

    elif tool_name in ["web_search", "web_fetch", "repl"]:
        tool_input_text = format_tool_input(tool_name, item["input"])
        if tool_input_text:
            text_parts.append(tool_input_text)

    return tool_name


def _process_tool_result(item: dict[str, Any], text_parts: list[str]) -> None:
    """Process a 'tool_result' content item and append formatted results."""
    tool_name = item["name"]

    display_text = format_display_content(item["display_content"], tool_name)
    if display_text:
        text_parts.append(display_text)

    result_content = item["content"]
    if isinstance(result_content, list):
        for result_item in result_content:
            if isinstance(result_item, dict) and result_item.get("type") == "text":
                result_text = format_tool_result_text(result_item["text"], tool_name)
                if result_text:
                    text_parts.append(result_text)


def extract_text_from_content(content: list[dict[str, Any]]) -> str:
    """Extract text content from a message's content array."""
    if not content:
        return ""

    text_parts: list[str] = []
    tool_uses: list[str] = []

    for item in content:
        content_type = item["type"]

        if content_type == "text":
            _process_text_content(item, text_parts)
        elif content_type == "tool_use":
            tool_name = _process_tool_use(item, text_parts)
            tool_uses.append(tool_name)
        elif content_type == "tool_result":
            _process_tool_result(item, text_parts)

    result = "\n\n".join(text_parts)

    if tool_uses:
        tool_summary = f"*[Used tools: {', '.join(tool_uses)}]*"
        result = f"{result}\n\n{tool_summary}" if result else tool_summary

    return result


def format_file_item(file_item: dict[str, Any]) -> str:
    """Format a single file/attachment item."""
    file_name = file_item.get("file_name", "unknown")
    file_size = file_item.get("file_size", "")
    file_type = file_item.get("file_type", "")

    info = f"- `{file_name}`"
    if file_type:
        info += f" (type: {file_type})"
    if file_size:
        info += f" (size: {file_size} bytes)"

    return info


def format_attachments(attachments: list[dict[str, Any]]) -> str | None:
    """Format attachments list as markdown."""
    if not attachments:
        return None

    attachment_info = [format_file_item(attachment) for attachment in attachments]

    if attachment_info:
        return f"*[Attachments: {len(attachments)}]*\n" + "\n".join(attachment_info)
    return None


def format_files(files: list[dict[str, Any]]) -> str | None:
    """Format files list as markdown."""
    if not files:
        return None

    file_info = [format_file_item(file) for file in files]

    if file_info:
        return f"*[Files: {len(files)}]*\n" + "\n".join(file_info)

    return None


def format_message(message: dict[str, Any]) -> str:
    """Format a single chat message as markdown."""
    sender = message["sender"]
    created_at = message["created_at"]

    logger.debug("  Message: %s (%d items)", sender, len(message["content"]))

    header = f"## {sender.title()}"
    timestamp = format_timestamp(created_at)

    content = extract_text_from_content(message["content"])
    if not content:
        content = message["text"].strip()

    content_parts: list[str] = []
    if content:
        content_parts.append(content)

    attachments_text = format_attachments(message["attachments"])
    if attachments_text:
        content_parts.append(attachments_text)

    files_text = format_files(message["files"])
    if files_text:
        content_parts.append(files_text)

    return f"{header}\n\n*{timestamp}*\n\n" + "\n\n".join(content_parts)


def convert_to_markdown(conversation: dict[str, Any]) -> str:
    """Convert a single conversation object to markdown format."""
    uuid = conversation["uuid"]
    name = conversation["name"]
    created_at = format_timestamp(conversation["created_at"])
    updated_at = format_timestamp(conversation["updated_at"])

    logger.info(
        "Converting conversation: %s (%s, %d messages)",
        name,
        uuid,
        len(conversation["chat_messages"]),
    )

    markdown = f"# {name}\n\n"
    markdown += f"**UUID:** {uuid}\n"
    markdown += f"**Created:** {created_at}  \n"
    markdown += f"**Updated:** {updated_at}\n\n"

    messages: list[dict[str, Any]] = conversation["chat_messages"]
    message_blocks: list[str] = []

    for message in messages:
        message_block = format_message(message)
        if message_block:
            message_blocks.append(message_block)

    markdown += "\n\n".join(message_blocks)

    return markdown


def has_content(conversation: dict[str, Any]) -> bool:
    """Check if a conversation has any meaningful content."""
    if conversation["name"].strip():
        return True

    messages = conversation["chat_messages"]
    for message in messages:
        if message["text"].strip():
            return True

        for content_item in message["content"]:
            if content_item["type"] == "text" and content_item["text"].strip():
                return True

        if message.get("attachments") or message.get("files"):
            return True

    return False


def main() -> None:
    conversations_path = Path("raw-logs/claude/conversations.json")
    with conversations_path.open() as f:
        conversations = json.load(f)

    logger.info("Loaded %d conversations", len(conversations))

    output_dir = Path("processed-logs/claude")
    output_dir.mkdir(exist_ok=True)

    exported_count = 0
    skipped_count = 0

    for conversation in conversations:
        if has_content(conversation):
            uuid = conversation["uuid"]
            md_content = convert_to_markdown(conversation)
            output_file = output_dir.joinpath(f"{uuid}.md")
            _ = output_file.write_text(md_content)

            exported_count += 1
        else:
            logger.info("Skipping empty conversation (%s)", conversation["uuid"])
            skipped_count += 1

    logger.info(
        "Export complete: %d exported, %d skipped", exported_count, skipped_count
    )


if __name__ == "__main__":
    main()
