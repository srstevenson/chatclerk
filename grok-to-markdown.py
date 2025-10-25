#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.14"
# dependencies = []
# ///

"""Convert data export from grok.com to Markdown."""

# pyright: reportAny=false, reportExplicitAny=false

import argparse
import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def format_mongodb_timestamp(timestamp_obj: dict[str, Any]) -> str:
    """Convert a MongoDB timestamp to a human-readable string.

    Args:
        timestamp_obj: MongoDB timestamp dictionary with $date and $numberLong.

    Returns:
        str: Formatted timestamp string in the format `YYYY-MM-DD HH:MM:SS UTC`.
    """
    timestamp_ms = int(timestamp_obj["$date"]["$numberLong"])
    timestamp_s = timestamp_ms / 1000.0
    dt = datetime.fromtimestamp(timestamp_s, tz=UTC)
    return dt.strftime("%Y-%m-%d %H:%M:%S UTC")


def format_iso_timestamp(timestamp_str: str) -> str:
    """Convert an ISO 8601 timestamp to a human-readable string.

    Args:
        timestamp_str: ISO 8601 formatted timestamp string.

    Returns:
        str: Formatted timestamp string in the format `YYYY-MM-DD HH:MM:SS UTC`.
    """
    dt = datetime.fromisoformat(timestamp_str)
    return dt.strftime("%Y-%m-%d %H:%M:%S UTC")


def format_message(response: dict[str, Any]) -> str:
    """Format a single message as Markdown.

    Args:
        response: Dictionary containing response data.

    Returns:
        str: The formatted message as a Markdown string.
    """
    response_data = response["response"]
    sender = response_data["sender"]
    sender_lower = sender.lower()
    message = response_data.get("message", "").strip()
    create_time = response_data.get("create_time")
    model = response_data.get("model", "unknown")
    metadata = response_data.get("metadata", {})

    if sender_lower == "human":
        header = "## User"
    elif sender_lower == "assistant":
        header = "## Assistant"
    else:
        header = f"## {sender.title()}"

    header_parts: list[str] = [header]

    if create_time and (timestamp := format_mongodb_timestamp(create_time)):
        timestamp_line = f"*{timestamp}*"
        if sender_lower == "assistant" and model and model != "unknown":
            timestamp_line += f" | Model: {model}"
        header_parts.append(timestamp_line)

    content_parts: list[str] = []

    if message:
        content_parts.append(message)

    if (
        metadata
        and (width := metadata.get("generated_image_width"))
        and (height := metadata.get("generated_image_height"))
    ):
        content_parts.append(f"*[Generated Image: {width}x{height}]*")

    if metadata and metadata.get("usedCustomInstructions"):
        content_parts.append("*[Used custom instructions]*")

    header_line = "\n\n".join(header_parts)
    content = "\n\n".join(content_parts)
    return f"{header_line}\n\n{content}"


def convert_to_markdown(conversation: dict[str, Any]) -> str:
    """Convert a Grok conversation to Markdown format.

    Args:
        conversation: Dictionary containing the conversation data.

    Returns:
        str: The formatted conversation as a Markdown string.
    """
    conv_metadata = conversation["conversation"]
    conv_id = conv_metadata.get("id", "unknown")
    title = conv_metadata.get("title") or "Untitled"
    create_time = conv_metadata.get("create_time")
    modify_time = conv_metadata.get("modify_time")
    system_prompt = conv_metadata.get("system_prompt_name")

    logger.info(
        "Converting conversation: %s (%s, %d messages)",
        title,
        conv_id,
        len(conversation["responses"]),
    )

    header_parts = [f"# {title}\n", f"**Conversation ID:** {conv_id}"]

    if create_time and (created := format_iso_timestamp(create_time)):
        header_parts.append(f"**Created:** {created}  ")
    if modify_time and (modified := format_iso_timestamp(modify_time)):
        header_parts.append(f"**Updated:** {modified}  ")
    if system_prompt:
        header_parts.append(f"**System Prompt:** {system_prompt}  ")

    header_parts.append("")

    responses: list[dict[str, Any]] = conversation["responses"]
    message_blocks = [format_message(response) for response in responses]

    header = "\n".join(header_parts)
    messages_section = "\n\n".join(message_blocks)

    return f"{header}\n{messages_section}"


def has_content(conversation: dict[str, Any]) -> bool:
    """Check whether a conversation contains meaningful content.

    Args:
        conversation: Dictionary containing the conversation data.

    Returns:
        bool: `True` if the conversation has a title or messages, otherwise
            `False`.
    """
    conv_metadata = conversation["conversation"]
    if (title := conv_metadata.get("title")) and title.strip():
        return True

    for response in conversation.get("responses", []):
        response_data = response.get("response", {})
        if (message := response_data.get("message", "")) and message.strip():
            return True

    return False


@dataclass
class Args(argparse.Namespace):
    """Command-line arguments.

    Attributes:
        input_dir: Directory containing the Grok export data.
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
        description="convert data export from grok.com to Markdown",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-i",
        "--input-dir",
        type=Path,
        default=Path("raw-logs/grok"),
        help="directory containing grok.com export",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("processed-logs/grok"),
        help="directory to write Markdown files",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="enable verbose logging"
    )
    return parser.parse_args(namespace=Args())


def main() -> None:
    """Convert Grok conversation exports to Markdown files.

    Reads `prod-grok-backend.json` from the input directory, converts each
    conversation with content to Markdown, and writes the results to the output
    directory.
    """
    args = parse_arguments()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="[%(levelname)-8s] %(message)s")

    conversations_path = args.input_dir.joinpath("prod-grok-backend.json")
    with conversations_path.open() as f:
        data = json.load(f)

    conversations = data["conversations"]
    logger.info("Loaded %d conversations", len(conversations))

    args.output_dir.mkdir(parents=True, exist_ok=True)

    exported_count = 0
    skipped_count = 0

    for conversation in conversations:
        conv_id = conversation["conversation"].get("id", "unknown")

        if not has_content(conversation):
            logger.debug("Skipping empty conversation (%s)", conv_id)
            skipped_count += 1
            continue

        md_content = convert_to_markdown(conversation)
        output_file = args.output_dir.joinpath(f"{conv_id}.md")
        output_file.write_text(md_content)

        exported_count += 1

    logger.info(
        "Export complete: %d exported, %d skipped", exported_count, skipped_count
    )


if __name__ == "__main__":
    main()
