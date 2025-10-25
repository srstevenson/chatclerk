#!/usr/bin/env python3

"""Convert ChatGPT conversations.json export to markdown files.

This script parses a conversations.json file from ChatGPT and creates
individual markdown files for each conversation in a directory.
"""

# pyright: reportAny=false, reportExplicitAny=false

import json
import logging
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.DEBUG,
    format="[%(asctime)s %(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def find_chatgpt_user_dir() -> Path | None:
    """Find the ChatGPT user directory containing exported images."""
    base_dir = Path("raw-logs/chatgpt")
    if not base_dir.exists():
        return None

    # Look for directories matching user-* pattern
    user_dirs = list(base_dir.glob("user-*"))

    if not user_dirs:
        logger.warning("No user directory found in raw-logs/chatgpt/")
        return None

    if len(user_dirs) > 1:
        logger.warning(f"Multiple user directories found, using first: {user_dirs[0]}")

    return user_dirs[0]


def format_timestamp(timestamp: float) -> str:
    """Convert Unix timestamp to readable format."""
    dt = datetime.fromtimestamp(timestamp)
    return dt.strftime("%Y-%m-%d %H:%M:%S UTC")


@dataclass
class Message:
    """Represents a message in the conversation."""

    role: str
    content: str
    timestamp: float | None
    metadata: dict[str, Any]
    images: list[dict[str, Any]] = field(default_factory=list)


def clean_text(text: str) -> str:
    """Remove ChatGPT's internal formatting characters from text."""
    # Remove Private Use Area characters (U+E000 to U+F8FF)
    # These are invisible markers ChatGPT uses internally but filters on their website
    cleaned = ""
    for char in text:
        code = ord(char)
        # Keep normal characters and emojis, skip private use area
        if not (0xE000 <= code <= 0xF8FF):
            cleaned += char
    return cleaned


def traverse_message_tree(
    mapping: dict[str, Any],
    start_id: str = "client-created-root",
    user_dir: Path | None = None,
) -> list[Message]:
    """Traverse the message tree in depth-first order to extract messages.

    Returns a list of Message objects in conversation order.
    """
    messages: list[Message] = []

    def visit(node_id: str) -> None:
        if node_id not in mapping:
            return

        node = mapping[node_id]
        message = node.get("message")

        # Process this message if it exists and should be visible
        if message:
            author = message.get("author", {})
            role = author.get("role", "")

            # Skip system messages that are visually hidden
            metadata = message.get("metadata", {})
            if metadata.get("is_visually_hidden_from_conversation"):
                logger.debug(f"  Skipping hidden message: {role}")
            elif role in ["user", "assistant", "tool"]:
                # Extract content from parts
                content_obj = message.get("content", {})
                content_type = content_obj.get("content_type", "")

                # Skip user_editable_context messages (user profile/instructions)
                if content_type == "user_editable_context":
                    logger.debug("  Skipping user context message")

                # Handle multimodal content (images, etc.)
                elif content_type == "multimodal_text":
                    parts = content_obj.get("parts", [])
                    content_items = []
                    image_list = []

                    for part in parts:
                        if isinstance(part, dict):
                            part_type = part.get("content_type", "")
                            if part_type == "image_asset_pointer":
                                width = part.get("width", "unknown")
                                height = part.get("height", "unknown")
                                asset = part.get("asset_pointer", "")

                                # Extract metadata
                                part_metadata = part.get("metadata", {})
                                dalle_meta = part_metadata.get("dalle", {})
                                gen_id = dalle_meta.get("gen_id", "")

                                # Find the original filename
                                asset_id = asset.replace("sediment://", "")
                                filename = f"{asset_id}.png"  # Default
                                if user_dir:
                                    matching_files = list(
                                        user_dir.glob(f"{asset_id}-*.png")
                                    )
                                    if matching_files:
                                        filename = matching_files[0].name

                                # Store image info for later processing
                                image_info = {
                                    "asset": asset,
                                    "width": width,
                                    "height": height,
                                    "gen_id": gen_id,
                                    "filename": filename,
                                }
                                image_list.append(image_info)

                                # Add placeholder text (will be replaced with image link)
                                content_items.append(
                                    f"*[Generated Image: {width}x{height}]*"
                                )
                                logger.info(
                                    f"  Found generated image: {width}x{height}"
                                )
                        elif isinstance(part, str) and part.strip():
                            content_items.append(part.strip())

                    if content_items or image_list:
                        content = "\n\n".join(content_items) if content_items else ""
                        content = clean_text(content)
                        timestamp = message.get("create_time")

                        messages.append(
                            Message(
                                role=role,
                                content=content,
                                timestamp=timestamp,
                                metadata=metadata,
                                images=image_list,
                            )
                        )

                # Handle regular text content
                else:
                    parts = content_obj.get("parts", [])
                    content = "\n".join(str(part) for part in parts if part)

                    # Clean ChatGPT's internal formatting characters
                    content = clean_text(content)

                    # Only include messages with non-empty content
                    content_stripped = content.strip()

                    # Skip unhelpful tool status messages
                    if role == "tool":
                        skip_phrases = [
                            "GPT-4o returned",
                            "Model set context updated",
                            "From now on, do not say or show ANYTHING",
                        ]
                        if any(phrase in content_stripped for phrase in skip_phrases):
                            logger.debug("  Skipping tool status message")
                            content_stripped = ""

                    if content_stripped:
                        timestamp = message.get("create_time")

                        messages.append(
                            Message(
                                role=role,
                                content=content_stripped,
                                timestamp=timestamp,
                                metadata=metadata,
                            )
                        )

        # Visit all children in order
        children = node.get("children", [])
        for child_id in children:
            visit(child_id)

    visit(start_id)
    return messages


def format_search_results(metadata: dict[str, Any]) -> str | None:
    """Format web search results from message metadata."""
    search_result_groups = metadata.get("search_result_groups", [])
    if not search_result_groups:
        return None

    result_lines = ["\n*Web search results:*"]

    for group in search_result_groups:
        if group.get("type") != "search_result_group":
            continue

        entries = group.get("entries", [])
        for entry in entries:
            title = entry.get("title", "")
            url = entry.get("url", "")
            snippet = entry.get("snippet", "")

            if url:
                result_lines.append(f"- [{title}]({url})")
                if snippet:
                    # Clean up snippet (remove excessive whitespace)
                    clean_snippet = " ".join(snippet.split())[:200]
                    if len(snippet) > 200:
                        clean_snippet += "..."
                    result_lines.append(f"  > {clean_snippet}")

    if len(result_lines) > 1:
        logger.info(f"  Formatted {len(result_lines) - 1} search results")
        return "\n".join(result_lines)

    return None


def format_citations(metadata: dict[str, Any]) -> str | None:
    """Format citations from message metadata."""
    citations = metadata.get("citations", [])
    if not citations:
        return None

    citation_lines = ["\n*Citations:*"]
    for i, citation in enumerate(citations, 1):
        metadata_obj = citation.get("metadata", {})
        url = metadata_obj.get("url", "")
        title = metadata_obj.get("title", "")

        if url:
            if title:
                citation_lines.append(f"{i}. [{title}]({url})")
            else:
                citation_lines.append(f"{i}. {url}")

    if len(citation_lines) > 1:
        logger.info(f"  Formatted {len(citation_lines) - 1} citations")
        return "\n".join(citation_lines)

    return None


def format_message_content(content: str) -> str:
    """Format message content, detecting and formatting JSON."""
    content = content.strip()

    # Check if content looks like JSON
    if content.startswith("{") and content.endswith("}"):
        try:
            # Try to parse and pretty-print as JSON
            parsed = json.loads(content)
            formatted_json = json.dumps(parsed, indent=2)
            return f"```json\n{formatted_json}\n```"
        except (json.JSONDecodeError, ValueError):
            # Not valid JSON, return as-is
            pass

    return content


def format_message(
    message: Message, conversation_id: str = "", image_index_offset: int = 0
) -> str:
    """Format a single message as markdown.

    Args:
        message: The message to format
        conversation_id: The conversation ID for image paths
        image_index_offset: Starting index for images in this message

    """
    # Format role header
    if message.role == "tool":
        # Tool outputs are labeled differently
        header = "## Tool Output"
    else:
        header = f"## {message.role.title()}"

    # Format timestamp if available
    timestamp_str = ""
    if message.timestamp:
        timestamp_str = f"\n\n*{format_timestamp(message.timestamp)}*"

    # Build content parts
    content_parts = []

    # Add main content (with JSON formatting if applicable)
    if message.content:
        formatted_content = format_message_content(message.content)
        content_parts.append(formatted_content)

    # Add images if present
    if message.images:
        for i, image_info in enumerate(message.images):
            image_num = image_index_offset + i + 1
            width = image_info.get("width", "unknown")
            height = image_info.get("height", "unknown")
            filename = image_info.get("filename", f"image_{image_num}.png")

            # Create image reference
            if conversation_id:
                image_path = f"{conversation_id}/{filename}"
            else:
                image_path = filename

            image_md = (
                f"![Generated Image {image_num} ({width}x{height})]({image_path})"
            )
            content_parts.append(image_md)

    # Add search results if present
    search_results = format_search_results(message.metadata)
    if search_results:
        content_parts.append(search_results)

    # Add citations if present
    citations = format_citations(message.metadata)
    if citations:
        content_parts.append(citations)

    # Assemble the final message block
    content = "\n\n".join(content_parts)
    message_block = f"{header}{timestamp_str}\n\n{content}"

    return message_block


def convert_to_markdown(
    conversation: dict[str, Any], user_dir: Path | None = None
) -> tuple[str, list[dict[str, Any]]]:
    """Convert a single conversation object to markdown format.

    Returns a tuple of (markdown_text, images_list).
    """
    title = conversation.get("title", "Untitled")
    conversation_id = conversation.get("conversation_id") or conversation.get("id", "")
    create_time = conversation.get("create_time")
    update_time = conversation.get("update_time")
    is_archived = conversation.get("is_archived", False)
    model_slug = conversation.get("default_model_slug", "unknown")

    logger.info(f"Converting conversation: {title} ({conversation_id})")

    # Build header
    markdown = f"# {title}\n\n"
    markdown += f"**Conversation ID:** {conversation_id}\n"

    if create_time:
        markdown += f"**Created:** {format_timestamp(create_time)}  \n"
    if update_time:
        markdown += f"**Updated:** {format_timestamp(update_time)}  \n"

    markdown += f"**Archived:** {'Yes' if is_archived else 'No'}  \n"
    markdown += f"**Model:** {model_slug}\n\n"

    # Extract messages from the tree
    mapping = conversation.get("mapping", {})
    messages = traverse_message_tree(mapping, user_dir=user_dir)

    # Sort messages chronologically by timestamp
    messages = sorted(messages, key=lambda m: m.timestamp if m.timestamp else 0)

    logger.info(f"  Extracted {len(messages)} visible messages")

    # Collect all images from messages
    all_images = []
    for message in messages:
        if message.images:
            all_images.extend(message.images)

    # Format each message with proper image indexing
    message_blocks = []
    image_counter = 0
    for message in messages:
        message_block = format_message(message, conversation_id, image_counter)
        if message_block:
            message_blocks.append(message_block)
        # Update image counter
        if message.images:
            image_counter += len(message.images)

    markdown += "\n\n".join(message_blocks)

    return markdown, all_images


def copy_conversation_images(
    conversation_id: str,
    images: list[dict[str, Any]],
    output_dir: Path,
    user_dir: Path | None = None,
) -> None:
    """Copy images for a conversation to a subdirectory."""
    if not images:
        return

    if not user_dir:
        logger.warning("No user directory provided, cannot copy images")
        return

    # Create subdirectory for images
    image_dir = output_dir / conversation_id
    image_dir.mkdir(exist_ok=True)

    for image_info in images:
        asset = image_info.get("asset", "")
        filename = image_info.get("filename", "")

        if not filename:
            logger.warning(f"  No filename for asset: {asset}")
            continue

        # Extract asset ID from sediment:// URL
        asset_id = asset.replace("sediment://", "")

        # Find the matching PNG file (has format: file_<asset_id>-<uuid>.png)
        matching_files = list(user_dir.glob(f"{asset_id}-*.png"))

        if matching_files:
            source_file = matching_files[0]
            dest_file = image_dir / filename

            _ = shutil.copy2(source_file, dest_file)
            logger.info(f"  Copied image: {source_file.name} -> {dest_file.name}")
        else:
            logger.warning(f"  Image not found for asset: {asset_id}")


def has_content(conversation: dict[str, Any]) -> bool:
    """Check if a conversation has any meaningful content."""
    # Check if title exists
    if conversation.get("title", "").strip():
        # Also check if there are any visible messages
        mapping = conversation.get("mapping", {})
        messages = traverse_message_tree(mapping)
        return len(messages) > 0

    return False


def main() -> None:
    with open("raw-logs/chatgpt/conversations.json") as f:
        conversations = json.load(f)

    logger.info(f"Loaded {len(conversations)} conversations")

    # Find the user directory containing images
    user_dir = find_chatgpt_user_dir()
    if user_dir:
        logger.info(f"Found user directory: {user_dir}")
    else:
        logger.warning("No user directory found - images will not be embedded")

    output_dir = Path("processed-logs/chatgpt")
    output_dir.mkdir(parents=True, exist_ok=True)

    exported_count = 0
    skipped_count = 0

    for conversation in conversations:
        if has_content(conversation):
            conversation_id = conversation.get("conversation_id") or conversation.get(
                "id", "unknown"
            )
            md_content, images = convert_to_markdown(conversation, user_dir)
            output_file = output_dir.joinpath(f"{conversation_id}.md")
            _ = output_file.write_text(md_content)

            # Copy images if any
            if images:
                copy_conversation_images(conversation_id, images, output_dir, user_dir)

            exported_count += 1
        else:
            conv_id = conversation.get("conversation_id") or conversation.get(
                "id", "unknown"
            )
            logger.info(f"Skipping empty conversation ({conv_id})")
            skipped_count += 1

    logger.info(f"Export complete: {exported_count} exported, {skipped_count} skipped")


if __name__ == "__main__":
    main()
