"""Convert data export from kagi.com to Markdown."""

import json
import logging
from html import unescape
from typing import Any, Final, cast

from chatclerk.argparse import Args, build_argument_parser
from chatclerk.datetime import iso_timestamp_to_str

logger: Final = logging.getLogger(__name__)


def _timestamp_or_none(timestamp: object) -> str | None:
    """Convert an ISO timestamp to a display string when possible."""
    if not isinstance(timestamp, str) or not timestamp:
        return None

    try:
        return iso_timestamp_to_str(timestamp)
    except ValueError:
        logger.debug("Failed to parse ISO timestamp: %s", timestamp)
        return None


def _extract_conversations(data: object) -> list[dict[str, Any]]:
    """Extract conversation exports from the Kagi Assistant JSON structure.

    Args:
        data: Loaded JSON data from the export file.

    Returns:
        List of conversation export dictionaries.
    """
    if not isinstance(data, dict):
        msg = "Unrecognised Kagi export format."
        raise TypeError(msg)

    data_dict = cast("dict[str, object]", data)
    conversations = data_dict.get("conversations")
    if not isinstance(conversations, list):
        msg = "Unrecognised Kagi export format."
        raise TypeError(msg)

    valid_conversations: list[dict[str, Any]] = []
    for item in conversations:
        if not isinstance(item, dict):
            continue

        item_dict = cast("dict[str, object]", item)
        conversation = item_dict.get("conversation")
        init = item_dict.get("init")
        if isinstance(conversation, dict) and isinstance(init, dict):
            valid_conversations.append(cast("dict[str, Any]", item))

    if not valid_conversations:
        msg = "Unrecognised Kagi export format."
        raise ValueError(msg)

    return valid_conversations


def _format_float(value: float) -> str:
    """Format a float without noisy trailing zeroes."""
    return f"{value:.6f}".rstrip("0").rstrip(".")


def _format_message_stats(message: dict[str, Any]) -> str | None:
    """Format assistant message stats when present."""
    stats: list[str] = []

    if input_tokens := message.get("input_tokens"):
        stats.append(f"Input tokens: {input_tokens}")
    if output_tokens := message.get("output_tokens"):
        stats.append(f"Output tokens: {output_tokens}")

    cost = message.get("cost_usd")
    if isinstance(cost, int | float):
        stats.append(f"Cost: ${_format_float(float(cost))}")

    duration_ms = message.get("duration_ms")
    if isinstance(duration_ms, int | float):
        stats.append(f"Duration: {_format_float(float(duration_ms) / 1000)}s")

    if tokens_per_second := message.get("tokens_per_second"):
        stats.append(f"Speed: {tokens_per_second} tokens/s")

    if not stats:
        return None

    return "*" + " | ".join(stats) + "*"


def _format_references(references: list[dict[str, Any]]) -> str | None:
    """Format message references as a Markdown numbered list."""
    if not references:
        return None

    lines = ["*References:*"]
    for index, reference in enumerate(references, 1):
        url = reference.get("url") or reference.get("source")
        if not isinstance(url, str) or not url:
            continue

        title = reference.get("title")
        if not isinstance(title, str) or not title:
            title = url

        lines.append(f"{index}. [{unescape(title)}]({url})")

        snippet = reference.get("snippet")
        if isinstance(snippet, str) and snippet.strip():
            clean_snippet = unescape(snippet.strip()).replace("\n", " ")
            lines.append(f"   > {clean_snippet}")

    if len(lines) == 1:
        return None

    return "\n".join(lines)


def _format_attachment(attachment: dict[str, Any]) -> str:
    """Format one Kagi attachment metadata item."""
    original_filename = attachment.get("original_filename")
    filename = attachment.get("filename")
    display_filename = original_filename or filename or "unknown"

    details: list[str] = []
    if isinstance(filename, str) and filename and filename != display_filename:
        details.append(f"filename: `{filename}`")

    mime_type = attachment.get("mime_type")
    if isinstance(mime_type, str) and mime_type:
        details.append(f"type: {mime_type}")

    size = attachment.get("size")
    if isinstance(size, int):
        details.append(f"size: {size} bytes")

    url = attachment.get("url")
    if isinstance(url, str) and url:
        details.append(f"URL: `{url}`")

    line = f"- `{display_filename}`"
    if details:
        line += f" ({', '.join(details)})"
    return line


def _format_attachments(attachments: list[dict[str, Any]]) -> str | None:
    """Format message attachment metadata as Markdown."""
    if not attachments:
        return None

    lines = [f"*Attachments: {len(attachments)}*"]
    lines.extend(_format_attachment(attachment) for attachment in attachments)
    return "\n".join(lines)


def _format_model_line(message: dict[str, Any]) -> str | None:
    """Format model metadata for assistant messages."""
    model = message.get("model_display_name") or message.get("model_name")
    if not isinstance(model, str) or not model:
        return None

    details: list[str] = [f"Model: {model}"]

    provider = message.get("model_provider")
    if isinstance(provider, str) and provider:
        details.append(f"Provider: {provider}")

    version = message.get("model_version")
    if isinstance(version, str) and version and version != model:
        details.append(f"Version: {version}")

    return "*" + " | ".join(details) + "*"


def _format_message(message: dict[str, Any]) -> str:
    """Format a single Kagi message as Markdown."""
    role = message.get("role", "unknown")
    role_name = role.title() if isinstance(role, str) else "Unknown"
    header = f"## {role_name}"

    header_parts = [header]
    if timestamp := _timestamp_or_none(message.get("created_at")):
        header_parts.append(f"*{timestamp}*")

    if role == "assistant" and (model_line := _format_model_line(message)):
        header_parts.append(model_line)

    content_parts: list[str] = []
    content = message.get("content")
    if isinstance(content, str) and content.strip():
        content_parts.append(content.strip())

    references = message.get("references")
    if isinstance(references, list):
        typed_references = [
            cast("dict[str, Any]", reference)
            for reference in references
            if isinstance(reference, dict)
        ]
        if references_text := _format_references(typed_references):
            content_parts.append(references_text)

    attachments = message.get("attachments")
    if isinstance(attachments, list):
        typed_attachments = [
            cast("dict[str, Any]", attachment)
            for attachment in attachments
            if isinstance(attachment, dict)
        ]
        if attachments_text := _format_attachments(typed_attachments):
            content_parts.append(attachments_text)

    if role == "assistant" and (stats := _format_message_stats(message)):
        content_parts.append(stats)

    return "\n\n".join(header_parts + content_parts)


def _format_conversation_header(conversation: dict[str, Any]) -> str:
    """Format Kagi conversation metadata as a Markdown document header."""
    title = conversation.get("title") or "Untitled"
    uuid = conversation.get("uuid") or "unknown"

    header_parts = [f"# {title}\n", f"- **Conversation ID:** {uuid}"]

    if created := _timestamp_or_none(conversation.get("created_at")):
        header_parts.append(f"- **Created:** {created}")
    if updated := _timestamp_or_none(conversation.get("updated_at")):
        header_parts.append(f"- **Updated:** {updated}")

    model_name = conversation.get("model_name")
    if isinstance(model_name, str) and model_name:
        header_parts.append(f"- **Model:** {model_name}")

    total_tokens = conversation.get("total_tokens")
    if isinstance(total_tokens, int):
        header_parts.append(f"- **Total Tokens:** {total_tokens}")

    total_cost = conversation.get("total_cost")
    if isinstance(total_cost, int | float):
        header_parts.append(f"- **Total Cost:** ${_format_float(float(total_cost))}")

    if conversation.get("has_branches"):
        branch_count = conversation.get("branch_count")
        if isinstance(branch_count, int):
            header_parts.append(f"- **Branches:** {branch_count}")

    header_parts.append("")
    return "\n".join(header_parts)


def _conversation_messages(conversation_export: dict[str, Any]) -> list[dict[str, Any]]:
    """Return renderable messages from a Kagi conversation export."""
    init = conversation_export["init"]
    messages = init.get("messages", {})
    if not isinstance(messages, dict):
        return []

    items = messages.get("items", [])
    if not isinstance(items, list):
        return []

    return [cast("dict[str, Any]", item) for item in items if isinstance(item, dict)]


def _convert_to_markdown(conversation_export: dict[str, Any]) -> str:
    """Convert a Kagi conversation export to Markdown."""
    conversation = conversation_export["conversation"]
    uuid = conversation.get("uuid", "unknown")
    title = conversation.get("title") or "Untitled"
    messages = _conversation_messages(conversation_export)

    logger.info("Converting: %s (%s, %d messages)", title, uuid, len(messages))

    header = _format_conversation_header(conversation)
    message_blocks = [_format_message(message) for message in messages]
    messages_section = "\n\n".join(message_blocks)

    return f"{header}\n{messages_section}" if messages_section else header.rstrip()


def _has_content(conversation_export: dict[str, Any]) -> bool:
    """Check whether a Kagi conversation export contains meaningful content."""
    conversation = conversation_export["conversation"]
    title = conversation.get("title")
    if isinstance(title, str) and title.strip():
        return True

    for message in _conversation_messages(conversation_export):
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return True
        if message.get("attachments") or message.get("references"):
            return True

    return False


def main() -> None:
    """Convert Kagi Assistant exports to Markdown files.

    Reads a Kagi Assistant export JSON file, converts each conversation with
    content to Markdown, and writes the results to the output directory.
    """
    parser = build_argument_parser(
        "kagi.com",
        input_arg="--input-file",
        input_help="Kagi Assistant export JSON file",
    )
    args = parser.parse_args(namespace=Args())

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="[%(levelname)-8s] %(message)s")

    if not args.input_dir.is_file():
        msg = "Input path must be a Kagi Assistant export JSON file."
        raise FileNotFoundError(msg)

    export_path = args.input_dir
    logger.info("Using export file: %s", export_path)

    with export_path.open() as f:
        data = json.load(f)

    conversations = _extract_conversations(data)
    logger.info("Loaded %d conversations", len(conversations))

    args.output_dir.mkdir(parents=True, exist_ok=True)

    exported_count = 0
    skipped_count = 0

    for conversation_export in conversations:
        conversation = conversation_export["conversation"]
        conversation_id = conversation.get("uuid", "unknown")

        if not _has_content(conversation_export):
            logger.debug("Skipping empty conversation (%s)", conversation_id)
            skipped_count += 1
            continue

        md_content = _convert_to_markdown(conversation_export)
        output_file = args.output_dir / f"{conversation_id}.md"
        output_file.write_text(md_content, encoding="utf-8")

        exported_count += 1

    logger.info(
        "Export complete: %d exported, %d skipped", exported_count, skipped_count
    )
