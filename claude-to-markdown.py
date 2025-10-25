#!/usr/bin/env python3

"""
Convert Claude.ai conversations.json export to markdown files.

This script parses a conversations.json file from Claude.ai and creates
individual markdown files for each conversation in a directory.
"""

# pyright: reportAny=false, reportExplicitAny=false

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional

logging.basicConfig(
    level=logging.DEBUG,
    format="[%(asctime)s %(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
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
    language: Optional[Language] = None

    @property
    def code_block_language(self) -> str:
        """Get the language identifier for markdown code blocks."""
        if self.language is None:
            return ""
        return self.language.value


def format_timestamp(timestamp_str: str) -> str:
    """Convert ISO timestamp to readable format."""
    timestamp = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
    return timestamp.strftime("%Y-%m-%d %H:%M:%S UTC")


def format_json_if_valid(text: str) -> ToolOutput:
    """Try to parse and format text as JSON/XML with appropriate language hint."""
    text = text.strip()
    if not text:
        return ToolOutput(text=text, language=Language.TEXT)

    # Check if it looks like XML (starts with <).
    if text.startswith("<"):
        return ToolOutput(text=text, language=Language.XML)

    # Check if it looks like JSON (starts with { or [).
    if not (text.startswith("{") or text.startswith("[")):
        return ToolOutput(text=text, language=Language.TEXT)

    try:
        formatted = json.dumps(json.loads(text), indent=2)
        return ToolOutput(text=formatted, language=Language.JSON)
    except (json.JSONDecodeError, ValueError):
        return ToolOutput(text=text, language=Language.TEXT)


def format_citations(citations: list[dict[str, Any]]) -> Optional[str]:
    """Format citations as a markdown list."""
    if not citations:
        return None

    citation_lines = ["", "*Citations:*"]
    for i, citation in enumerate(citations, 1):
        details = citation.get("details", {})  # details can be null
        url = details.get("url", "") if details else ""
        if url:
            citation_lines.append(f"{i}. [{url}]({url})")

    # Only return if we have actual citations
    if len(citation_lines) > 2:
        logger.info(f"  Formatted {len(citation_lines) - 2} citations")
        return "\n".join(citation_lines)
    return None


def format_artifact(item: dict[str, Any]) -> Optional[str]:
    """Format an artifact tool use as markdown."""
    artifact_input = item["input"]
    artifact_title = artifact_input.get("title", "Untitled Artifact")
    artifact_type = artifact_input.get("type", "")
    artifact_content = artifact_input.get("content", "")
    artifact_id = artifact_input.get("id", "")

    if not artifact_content:
        return None

    logger.info(f"  Artifact: {artifact_title}")

    # Format the artifact header
    artifact_header = f"### Artifact: {artifact_title}"
    if artifact_id:
        artifact_header += f"\n*Type: {artifact_type} | ID: {artifact_id}*"

    # Handle markdown artifacts specially - render them natively
    if artifact_type == "text/markdown":
        return f"{artifact_header}\n\n---\n\n{artifact_content.rstrip()}\n\n---"

    # For non-markdown artifacts, use code blocks
    lang = ""
    if artifact_type == "application/vnd.ant.code":
        lang = "python"
    elif artifact_type.startswith("text/"):
        lang = artifact_type.replace("text/", "")

    if lang:
        return f"{artifact_header}\n\n```{lang}\n{artifact_content.rstrip()}\n```"
    else:
        return f"{artifact_header}\n\n```\n{artifact_content.rstrip()}\n```"


def format_tool_input(tool_name: str, tool_input: dict[str, Any]) -> Optional[str]:
    """Format tool input for display (for interesting tools only)."""
    if not tool_input:  # Can be empty dict
        return None

    if tool_name == "web_search":
        query = tool_input.get("query", "")
        if query:
            logger.info(f"  Web search: {query}")
            return f"*[Searching for: {query}]*"
        return None

    elif tool_name == "web_fetch":
        url = tool_input.get("url", "")
        if url:
            logger.info(f"  Web fetch: {url}")
            return f"*[Fetching: {url}]*"
        return None

    elif tool_name == "repl":
        code = tool_input.get("code", "")
        if code:
            logger.info(f"  REPL: {len(code)} chars")
            return f"*[Code executed]*\n```javascript\n{code.strip()}\n```"

    return None


def format_display_content(
    display_content: dict[str, Any], tool_name: str
) -> Optional[str]:
    """Format rich display content from tool results."""
    if not display_content:  # Can be null or empty
        return None

    display_type = display_content.get("type", "")
    logger.info(f"  Display_content: type={display_type}, tool={tool_name}")

    if display_type == "rich_link":
        link = display_content.get("link", {})
        url = link.get("url", "")
        title = link.get("title", url)
        if url:
            logger.info(f"  Rich link: {title}")
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
                logger.info(f"  Rich content: {len(links)} items")
                return f"*[Tool Result: {tool_name}]*\n" + "\n".join(links)

    return None


def format_tool_result_text(result_text: str, tool_name: str) -> Optional[str]:
    """Format text content from a tool result."""
    if not result_text or result_text == "OK":
        return None

    formatted = format_json_if_valid(result_text)
    clean_text = formatted.text.rstrip()
    lang = formatted.code_block_language

    if lang:
        return f"*[Tool Result: {tool_name}]*\n```{lang}\n{clean_text}\n```"
    else:
        return f"*[Tool Result: {tool_name}]*\n```\n{clean_text}\n```"


def extract_text_from_content(content: list[dict[str, Any]]) -> str:
    """Extract text content from a message's content array."""
    if not content:
        return ""

    text_parts: list[str] = []
    tool_uses: list[str] = []

    for item in content:
        content_type = item["type"]

        if content_type == "text":
            text = item["text"].strip()
            if text:
                text_parts.append(text)

            # Handle citations if present (always present but often empty)
            citations_text = format_citations(item["citations"])
            if citations_text:
                text_parts.append(citations_text)

        elif content_type == "tool_use":
            tool_name = item["name"]
            tool_uses.append(tool_name)

            # Handle artifacts
            if tool_name == "artifacts":
                artifact_text = format_artifact(item)
                if artifact_text:
                    text_parts.append(artifact_text)

            # Handle other interesting tools
            elif tool_name in ["web_search", "web_fetch", "repl"]:
                tool_input_text = format_tool_input(tool_name, item["input"])
                if tool_input_text:
                    text_parts.append(tool_input_text)

        elif content_type == "tool_result":
            tool_name = item["name"]

            # Check for display_content first (rich formatted results)
            display_text = format_display_content(item["display_content"], tool_name)
            if display_text:
                text_parts.append(display_text)

            # Extract text from tool results
            result_content = item["content"]
            if isinstance(result_content, list):
                for result_item in result_content:
                    if (
                        isinstance(result_item, dict)
                        and result_item.get("type")
                        == "text"  # type might not be present in all result items
                    ):
                        result_text = format_tool_result_text(
                            result_item["text"], tool_name
                        )
                        if result_text:
                            text_parts.append(result_text)

    result = "\n\n".join(text_parts)

    if tool_uses:
        tool_summary = f"*[Used tools: {', '.join(tool_uses)}]*"
        if result:
            result = f"{result}\n\n{tool_summary}"
        else:
            result = tool_summary

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


def format_attachments(attachments: list[dict[str, Any]]) -> Optional[str]:
    """Format attachments list as markdown."""
    if not attachments:
        return None

    attachment_info: list[str] = []
    for attachment in attachments:
        attachment_info.append(format_file_item(attachment))

    if attachment_info:
        return f"*[Attachments: {len(attachments)}]*\n" + "\n".join(attachment_info)


def format_files(files: list[dict[str, Any]]) -> Optional[str]:
    """Format files list as markdown."""
    if not files:
        return None

    file_info: list[str] = []
    for file in files:
        file_info.append(format_file_item(file))

    if file_info:
        return f"*[Files: {len(files)}]*\n" + "\n".join(file_info)

    return None


def format_message(message: dict[str, Any]) -> str:
    """Format a single chat message as markdown."""
    sender = message["sender"]
    created_at = message["created_at"]

    logger.info(f"  Message: {sender} ({len(message['content'])} items)")

    header = f"## {sender.title()}"
    timestamp = format_timestamp(created_at)

    # Extract content from the structured content array
    content = extract_text_from_content(message["content"])

    # Fallback to the text field if no content was extracted
    if not content:
        content = message["text"].strip()

    # Build the message parts
    content_parts: list[str] = []
    if content:
        content_parts.append(content)

    # Add attachments and files
    attachments_text = format_attachments(message["attachments"])
    if attachments_text:
        content_parts.append(attachments_text)

    files_text = format_files(message["files"])
    if files_text:
        content_parts.append(files_text)

    # Assemble the final message block
    message_block = f"{header}\n\n*{timestamp}*\n\n" + "\n\n".join(content_parts)

    return message_block


def convert_to_markdown(conversation: dict[str, Any]) -> str:
    """Convert a single conversation object to markdown format."""

    uuid = conversation["uuid"]
    name = conversation["name"]
    created_at = format_timestamp(conversation["created_at"])
    updated_at = format_timestamp(conversation["updated_at"])

    logger.info(
        f"Converting conversation: {name} ({uuid}, {len(conversation['chat_messages'])} messages)"
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
    with open("raw-logs/claude/conversations.json") as f:
        conversations = json.load(f)

    logger.info(f"Loaded {len(conversations)} conversations")

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
            logger.info(f"Skipping empty conversation ({conversation['uuid']})")
            skipped_count += 1

    logger.info(f"Export complete: {exported_count} exported, {skipped_count} skipped")


if __name__ == "__main__":
    main()
