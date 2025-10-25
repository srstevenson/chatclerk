#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.14"
# dependencies = []
# ///

"""Convert data export from claude.ai to Markdown."""

# pyright: reportAny=false, reportExplicitAny=false

import argparse
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


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
            str: `Language` identifier string, or empty string if no language
                specified.
        """
        if self.language is None:
            return ""
        return self.language.value


def format_timestamp(timestamp_str: str) -> str:
    """Convert an ISO 8601 timestamp to a human-readable string.

    Args:
        timestamp_str: ISO 8601 formatted timestamp string.

    Returns:
        str: Formatted timestamp string in the format `YYYY-MM-DD HH:MM:SS UTC`.
    """
    timestamp = datetime.fromisoformat(timestamp_str)
    return timestamp.strftime("%Y-%m-%d %H:%M:%S UTC")


def format_json_if_valid(text: str) -> ToolOutput:
    """Attempt to parse and format text as JSON or XML with syntax highlighting.

    Args:
        text: The text to analyse and format.

    Returns:
        ToolOutput: Formatted text with appropriate language hint for syntax
            highlighting.
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
    except (json.JSONDecodeError, ValueError):
        return ToolOutput(text=text, language=Language.TEXT)


def format_citations(citations: list[dict[str, Any]]) -> str | None:
    """Format citations as a numbered Markdown list.

    Args:
        citations: List of citation dictionaries.

    Returns:
        str | None: Formatted Markdown string of citations, or `None` if no
            valid citations.
    """
    if not citations:
        return None

    citation_lines = ["", "*Citations:*"]
    for i, citation in enumerate(citations, 1):
        details = citation.get("details", {})  # details can be null
        if url := details.get("url", ""):
            citation_lines.append(f"{i}. [{url}]({url})")

    min_citation_lines = 2
    if len(citation_lines) > min_citation_lines:
        logger.debug("Formatted %d citations", len(citation_lines) - min_citation_lines)
        return "\n".join(citation_lines)
    return None


def format_artifact(item: dict[str, Any]) -> str | None:
    """Format an artifact tool use as Markdown.

    Args:
        item: Dictionary containing artifact data from tool use.

    Returns:
        str | None: Formatted Markdown string representing the artifact, or
            `None` if empty.
    """
    artifact_input = item["input"]
    artifact_title = artifact_input.get("title", "Untitled Artifact")
    artifact_type = artifact_input.get("type", "")
    artifact_content = artifact_input.get("content", "")
    artifact_id = artifact_input.get("id", "")

    if not artifact_content:
        return None

    logger.debug("Artifact: %s", artifact_title)

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
    """Format `web_search` tool input for display.

    Args:
        tool_input: Dictionary containing tool input parameters.

    Returns:
        str | None: Formatted search query string, or `None` if no query
            present.
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
        str | None: Formatted URL string, or `None` if no URL present.
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
        str | None: Formatted code block string, or `None` if no code present.
    """
    if code := tool_input.get("code", ""):
        logger.debug("REPL: %d chars", len(code))
        return f"*[Code executed]*\n```javascript\n{code.strip()}\n```"
    return None


def format_tool_input(tool_name: str, tool_input: dict[str, Any]) -> str | None:
    """Format tool input parameters for display.

    Only formats inputs for selected tools that provide useful context.

    Args:
        tool_name: Name of the tool being used.
        tool_input: Dictionary containing tool input parameters.

    Returns:
        str | None: Formatted input string, or `None` if tool not supported
            or empty.
    """
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
    """Format rich display content from tool results as Markdown.

    Args:
        display_content: Dictionary containing display content metadata.
        tool_name: Name of the tool that generated the content.

    Returns:
        str | None: Formatted Markdown string, or `None` if no displayable
            content.
    """
    if not display_content:  # Can be null or empty
        return None

    display_type = display_content.get("type", "")
    logger.debug("Display_content: type=%s, tool=%s", display_type, tool_name)

    if display_type == "rich_link":
        link = display_content.get("link", {})
        url = link.get("url", "")
        title = link.get("title", url)
        if url:
            logger.debug("Rich link: %s", title)
            return f"*[Tool Result: {tool_name}]*\n- [{title}]({url})"

    elif display_type == "rich_content":
        if content_items := display_content.get("content", []):
            links: list[str] = []
            for content_item in content_items:
                title = content_item.get("title", "")
                url = content_item.get("url", "")
                if title and url:
                    links.append(f"- [{title}]({url})")
            if links:
                logger.debug("Rich content: %d items", len(links))
                return f"*[Tool Result: {tool_name}]*\n" + "\n".join(links)

    return None


def format_tool_result_text(result_text: str, tool_name: str) -> str | None:
    """Format text content from a tool result as a code block.

    Args:
        result_text: The text content returned by the tool.
        tool_name: Name of the tool that generated the result.

    Returns:
        str | None: Formatted Markdown code block, or `None` if empty or just
            `"OK"`.
    """
    if not result_text or result_text == "OK":
        return None

    formatted = format_json_if_valid(result_text)
    clean_text = formatted.text.rstrip()
    lang = formatted.code_block_language

    if lang:
        return f"*[Tool Result: {tool_name}]*\n```{lang}\n{clean_text}\n```"
    return f"*[Tool Result: {tool_name}]*\n```\n{clean_text}\n```"


def _process_text_content(item: dict[str, Any], text_parts: list[str]) -> None:
    """Process a text content item and append formatted text.

    Args:
        item: Dictionary containing text content and citations.
        text_parts: List to append formatted text parts to.
    """
    if text := item["text"].strip():
        text_parts.append(text)

    if citations_text := format_citations(item["citations"]):
        text_parts.append(citations_text)


def _process_tool_use(item: dict[str, Any], text_parts: list[str]) -> str:
    """Process a `tool_use` content item and append formatted output.

    Args:
        item: Dictionary containing tool use data.
        text_parts: List to append formatted tool use parts to.

    Returns:
        str: The name of the tool that was used.
    """
    tool_name = item["name"]

    if tool_name == "artifacts":
        if artifact_text := format_artifact(item):
            text_parts.append(artifact_text)

    elif tool_name in ["web_search", "web_fetch", "repl"] and (
        tool_input_text := format_tool_input(tool_name, item["input"])
    ):
        text_parts.append(tool_input_text)

    return tool_name


def _process_tool_result(item: dict[str, Any], text_parts: list[str]) -> None:
    """Process a `tool_result` content item and append formatted output.

    Args:
        item: Dictionary containing tool result data.
        text_parts: List to append formatted result parts to.
    """
    tool_name = item["name"]

    if display_text := format_display_content(item["display_content"], tool_name):
        text_parts.append(display_text)

    result_content = item["content"]
    if isinstance(result_content, list):
        text_parts.extend(
            result_text
            for result_item in result_content  # pyright: ignore[reportUnknownVariableType]
            if isinstance(result_item, dict)
            and result_item.get("type") == "text"  # pyright: ignore[reportUnknownMemberType]
            and (result_text := format_tool_result_text(result_item["text"], tool_name))  # pyright: ignore[reportUnknownArgumentType]
        )


def extract_text_from_content(content: list[dict[str, Any]]) -> str:
    """Extract and format text content from a message's content array.

    Processes text, tool use, and tool result items into a unified Markdown
    string.

    Args:
        content: List of content item dictionaries.

    Returns:
        str: Formatted Markdown string combining all content items.
    """
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
    """Format a single file or attachment item as Markdown.

    Args:
        file_item: Dictionary containing file metadata.

    Returns:
        str: Formatted Markdown list item with file details.
    """
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
    """Format attachments list as Markdown.

    Args:
        attachments: List of attachment metadata dictionaries.

    Returns:
        str | None: Formatted Markdown list of attachments, or `None` if empty.
    """
    if not attachments:
        return None

    attachment_info = [format_file_item(attachment) for attachment in attachments]

    if attachment_info:
        return f"*[Attachments: {len(attachments)}]*\n" + "\n".join(attachment_info)
    return None


def format_files(files: list[dict[str, Any]]) -> str | None:
    """Format files list as Markdown.

    Args:
        files: List of file metadata dictionaries.

    Returns:
        str | None: Formatted Markdown list of files, or `None` if empty.
    """
    if not files:
        return None

    file_info = [format_file_item(file) for file in files]

    if file_info:
        return f"*[Files: {len(files)}]*\n" + "\n".join(file_info)

    return None


def format_message(message: dict[str, Any]) -> str:
    """Format a single chat message as Markdown.

    Args:
        message: Dictionary containing message data.

    Returns:
        str: The formatted message as a Markdown string.
    """
    sender = message["sender"]
    created_at = message["created_at"]

    logger.debug("Message: %s (%d items)", sender, len(message["content"]))

    header = f"## {sender.title()}"
    timestamp = format_timestamp(created_at)

    content = extract_text_from_content(message["content"])
    if not content:
        content = message["text"].strip()

    content_parts: list[str] = []
    if content:
        content_parts.append(content)

    if attachments_text := format_attachments(message["attachments"]):
        content_parts.append(attachments_text)

    if files_text := format_files(message["files"]):
        content_parts.append(files_text)

    return f"{header}\n\n*{timestamp}*\n\n" + "\n\n".join(content_parts)


def convert_to_markdown(conversation: dict[str, Any]) -> str:
    """Convert a Claude conversation to Markdown format.

    Args:
        conversation: Dictionary containing the conversation data.

    Returns:
        str: The formatted conversation as a Markdown string.
    """
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

    # Build header metadata
    header_parts = [
        f"# {name}\n",
        f"**UUID:** {uuid}",
        f"**Created:** {created_at}  ",
        f"**Updated:** {updated_at}\n",
    ]

    messages: list[dict[str, Any]] = conversation["chat_messages"]
    message_blocks = [
        message_block
        for message in messages
        if (message_block := format_message(message))
    ]

    # Combine header and messages
    header = "\n".join(header_parts)
    messages_section = "\n\n".join(message_blocks)

    return f"{header}\n{messages_section}"


def has_content(conversation: dict[str, Any]) -> bool:
    """Check whether a conversation contains meaningful content.

    Args:
        conversation: Dictionary containing the conversation data.

    Returns:
        bool: `True` if the conversation has a name, messages, or attachments,
            otherwise `False`.
    """
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


@dataclass
class Args(argparse.Namespace):
    """Command-line arguments.

    Attributes:
        input_dir: Directory containing the Claude export data.
        output_dir: Directory where Markdown files will be written.
        verbose: Enable verbose logging.
    """

    input_dir: Path = field(init=False)
    output_dir: Path = field(init=False)
    verbose: bool = False


def parse_arguments() -> Args:
    """Parse command-line arguments.

    Returns:
        Args: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="convert data export from claude.ai to Markdown",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-i",
        "--input-dir",
        type=Path,
        default=Path("raw-logs/claude"),
        help="directory containing claude.ai export",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("processed-logs/claude"),
        help="directory to write Markdown files",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="enable verbose logging"
    )
    return parser.parse_args(namespace=Args())


def main() -> None:
    """Convert Claude conversation exports to Markdown files.

    Reads `conversations.json` from the input directory, converts each
    conversation with content to Markdown, and writes the results to the output
    directory.
    """
    args = parse_arguments()

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
        if not has_content(conversation):
            logger.debug("Skipping empty conversation (%s)", conversation["uuid"])
            skipped_count += 1
            continue

        uuid = conversation["uuid"]
        md_content = convert_to_markdown(conversation)
        output_file = args.output_dir.joinpath(f"{uuid}.md")
        output_file.write_text(md_content)

        exported_count += 1

    logger.info(
        "Export complete: %d exported, %d skipped", exported_count, skipped_count
    )


if __name__ == "__main__":
    main()
