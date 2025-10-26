"""Convert data export from chatgpt.com to Markdown."""

import json
import logging
import shutil
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Final

from chatclerk.argparse import Args, build_argument_parser
from chatclerk.datetime import unix_timestamp_to_str

if TYPE_CHECKING:
    from pathlib import Path

logger: Final = logging.getLogger(__name__)


def _get_user_dir(input_dir: Path) -> Path:
    """Determine the ChatGPT user directory path from `user.json`.

    Args:
        input_dir: Path to the directory containing the ChatGPT export data.

    Returns:
        Path: Absolute path to the user directory containing exported images.

    """
    user_json_path = input_dir.joinpath("user.json")
    with user_json_path.open() as f:
        user_data = json.load(f)
    return input_dir.joinpath(user_data["id"])


@dataclass(frozen=True)
class ImageInfo:
    """Representation of a generated image in a ChatGPT conversation.

    Attributes:
        asset: The asset pointer URI (e.g., `sediment://...`).
        width: Image width in pixels, or `None` if unavailable.
        height: Image height in pixels, or `None` if unavailable.
        gen_id: DALL-E generation ID, or `None` if unavailable.
        filename: Filename for the exported image file.

    """

    asset: str
    width: int | None
    height: int | None
    gen_id: str | None
    filename: str


@dataclass(frozen=True)
class Message:
    """Representation of a message in a ChatGPT conversation.

    Attributes:
        role: The sender's role (user, assistant, or tool).
        content: The text content of the message.
        timestamp: Unix timestamp when the message was created, or `None`.
        metadata: Additional metadata associated with the message.
        images: List of generated image metadata.

    """

    role: str
    content: str
    timestamp: float | None
    metadata: dict[str, Any]
    images: list[ImageInfo] = field(default_factory=list)


def clean_text(text: str) -> str:
    """Remove Unicode private use area characters from text.

    ChatGPT uses private use area characters for internal formatting.
    This function strips them to produce clean output text.

    Args:
        text: The text to clean.

    Returns:
        str: Text with private use area characters removed.

    """
    private_use_start = 0xE000
    private_use_end = 0xF8FF
    cleaned: list[str] = []
    for char in text:
        code = ord(char)
        if not (private_use_start <= code <= private_use_end):
            cleaned.append(char)
    return "".join(cleaned)


def _process_image_asset(part: dict[str, Any], user_dir: Path) -> tuple[str, ImageInfo]:
    """Process an image asset pointer and extract image metadata.

    Args:
        part: Dictionary containing image asset pointer data.
        user_dir: Path to the user directory containing exported images.

    Returns:
        tuple[str, ImageInfo]: A tuple containing a placeholder text string
            and an `ImageInfo` object with image metadata including asset ID,
            dimensions, generation ID, and filename.

    """
    width = part.get("width")
    height = part.get("height")
    asset = part.get("asset_pointer", "")

    part_metadata = part.get("metadata", {})
    dalle_meta = part_metadata.get("dalle", {})
    gen_id = dalle_meta.get("gen_id")

    asset_id = asset.removeprefix("sediment://")
    matching_files = list(user_dir.glob(f"{asset_id}-*.png"))
    filename = matching_files[0].name if matching_files else f"{asset_id}.png"

    image_info = ImageInfo(asset, width, height, gen_id, filename)

    width_str = str(width) if width else "unknown"
    height_str = str(height) if height else "unknown"
    placeholder = f"*[Generated Image: {width_str}x{height_str}]*"
    logger.debug("Found generated image: %sx%s", width_str, height_str)

    return placeholder, image_info


def _process_multimodal_content(
    content_obj: dict[str, Any],
    role: str,
    metadata: dict[str, Any],
    timestamp: float | None,
    user_dir: Path,
) -> Message | None:
    """Process multimodal message content containing text and images.

    Args:
        content_obj: Dictionary containing the message content.
        role: The sender's role (user, assistant, or tool).
        metadata: Additional metadata associated with the message.
        timestamp: Unix timestamp when the message was created, or `None`.
        user_dir: Path to the user directory containing exported images.

    Returns:
        Message | None: A `Message` object if valid content exists, otherwise
            `None`.

    """
    parts = content_obj.get("parts", [])
    content_items: list[str] = []
    image_list: list[ImageInfo] = []

    for part in parts:
        if isinstance(part, dict):
            part_type = part.get("content_type", "")  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
            if part_type == "image_asset_pointer":
                placeholder, image_info = _process_image_asset(part, user_dir)  # pyright: ignore[reportUnknownArgumentType]
                content_items.append(placeholder)
                image_list.append(image_info)
        elif isinstance(part, str) and (stripped := part.strip()):
            content_items.append(stripped)

    if content_items or image_list:
        content = "\n\n".join(content_items) if content_items else ""
        content = clean_text(content)

        return Message(role, content, timestamp, metadata, image_list)

    return None


def _process_regular_content(
    content_obj: dict[str, Any],
    role: str,
    metadata: dict[str, Any],
    timestamp: float | None,
) -> Message | None:
    """Process regular text message content.

    Args:
        content_obj: Dictionary containing the message content.
        role: The sender's role (user, assistant, or tool).
        metadata: Additional metadata associated with the message.
        timestamp: Unix timestamp when the message was created, or `None`.

    Returns:
        Message | None: A `Message` object if valid content exists, otherwise
            `None`.

    """
    parts = content_obj.get("parts", [])
    content = "\n".join(str(part) for part in parts if part)

    content = clean_text(content)
    content_stripped = content.strip()

    if content_stripped:
        return Message(role, content_stripped, timestamp, metadata)

    return None


def _process_message_content(
    message: dict[str, Any], role: str, metadata: dict[str, Any], user_dir: Path
) -> Message | None:
    """Process message content based on its type.

    Dispatches to appropriate handler based on content type (multimodal or regular).

    Args:
        message: Dictionary containing the raw message data.
        role: The sender's role (user, assistant, or tool).
        metadata: Additional metadata associated with the message.
        user_dir: Path to the user directory containing exported images.

    Returns:
        Message | None: A `Message` object if valid content exists, otherwise
            `None`.

    """
    content_obj = message.get("content", {})
    content_type = content_obj.get("content_type", "")
    timestamp = message.get("create_time")

    if content_type == "user_editable_context":
        logger.debug("Skipping user context message")
        return None

    if content_type == "multimodal_text":
        return _process_multimodal_content(
            content_obj, role, metadata, timestamp, user_dir
        )

    return _process_regular_content(content_obj, role, metadata, timestamp)


def _traverse_message_tree(
    mapping: dict[str, Any], user_dir: Path, start_id: str = "client-created-root"
) -> list[Message]:
    """Traverse the conversation message tree to extract all messages.

    ChatGPT exports conversations as a tree structure to support branching.
    This function performs a depth-first traversal to extract messages in order.

    Args:
        mapping: Dictionary mapping node IDs to message nodes.
        user_dir: Path to the user directory containing exported images.
        start_id: ID of the root node to start traversal from.

    Returns:
        list[Message]: List of `Message` objects in conversation order.

    """
    messages: list[Message] = []

    def _visit(node_id: str) -> None:
        if node_id not in mapping:
            return

        node = mapping[node_id]

        if message := node.get("message"):
            author = message.get("author", {})
            role = author.get("role", "")
            metadata = message.get("metadata", {})

            if metadata.get("is_visually_hidden_from_conversation"):
                logger.debug("Skipping hidden message: %s", role)
            elif role in ["user", "assistant", "tool"] and (
                msg := _process_message_content(message, role, metadata, user_dir)
            ):
                messages.append(msg)

        children = node.get("children", [])
        for child_id in children:
            _visit(child_id)

    _visit(start_id)
    return messages


def _format_search_results(metadata: dict[str, Any]) -> str | None:
    """Format web search results as a Markdown list.

    Args:
        metadata: Message metadata dictionary potentially containing search
            results.

    Returns:
        str | None: Formatted Markdown string of search results, or `None` if
            no results.

    """
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
                    result_lines.append(f"  > {snippet.strip()}")

    if len(result_lines) > 1:
        logger.debug("Formatted %d search results", len(result_lines) - 1)
        return "\n".join(result_lines)

    return None


def _format_citations(metadata: dict[str, Any]) -> str | None:
    """Format citations as a numbered Markdown list.

    Args:
        metadata: Message metadata dictionary potentially containing citations.

    Returns:
        str | None: Formatted Markdown string of citations, or `None` if no
            citations.

    """
    citations = metadata.get("citations", [])
    if not citations:
        return None

    citation_lines = ["\n*Citations:*"]
    for i, citation in enumerate(citations, 1):
        metadata_obj = citation.get("metadata", {})
        url = metadata_obj.get("url", "")
        title = metadata_obj.get("title", "")

        if not url:
            continue
        if title:
            citation_lines.append(f"{i}. [{title}]({url})")
        else:
            citation_lines.append(f"{i}. {url}")

    if len(citation_lines) > 1:
        logger.debug("Formatted %d citations", len(citation_lines) - 1)
        return "\n".join(citation_lines)

    return None


def _format_message_content(content: str) -> str:
    """Format message content, applying syntax highlighting to JSON.

    Args:
        content: The message content to format.

    Returns:
        str: Formatted content with JSON in code blocks if applicable.

    """
    content = content.strip()

    if content.startswith("{") and content.endswith("}"):
        try:
            formatted_json = json.dumps(json.loads(content), indent=2)
        except (json.JSONDecodeError, ValueError):
            return content
        else:
            return f"```json\n{formatted_json}\n```"

    return content


def _format_message(
    message: Message, conversation_id: str = "", image_index_offset: int = 0
) -> str:
    """Format a single message as Markdown.

    Args:
        message: The message to format.
        conversation_id: The conversation ID for constructing image paths.
        image_index_offset: Starting index for numbering images in this message.

    Returns:
        str: The formatted message as a Markdown string.

    """
    if message.role == "tool":
        header = "## Tool Output"
    else:
        header = f"## {message.role.title()}"

    timestamp_str = ""
    if message.timestamp:
        timestamp_str = f"\n\n*{unix_timestamp_to_str(message.timestamp)}*"

    content_parts: list[str] = []

    if message.content:
        formatted_content = _format_message_content(message.content)
        content_parts.append(formatted_content)

    if message.images:
        for i, image_info in enumerate(message.images):
            image_num = image_index_offset + i + 1

            if conversation_id:
                image_path = f"{conversation_id}/{image_info.filename}"
            else:
                image_path = image_info.filename

            width_str = str(image_info.width) if image_info.width else "unknown"
            height_str = str(image_info.height) if image_info.height else "unknown"
            image_md = (
                f"![Generated Image {image_num} "
                f"({width_str}x{height_str})]({image_path})"
            )
            content_parts.append(image_md)

    if search_results := _format_search_results(message.metadata):
        content_parts.append(search_results)

    if citations := _format_citations(message.metadata):
        content_parts.append(citations)

    content = "\n\n".join(content_parts)
    return f"{header}{timestamp_str}\n\n{content}"


def _convert_to_markdown(
    conversation: dict[str, Any], user_dir: Path
) -> tuple[str, list[ImageInfo]]:
    """Convert a ChatGPT conversation to Markdown format.

    Args:
        conversation: Dictionary containing the conversation data.
        user_dir: Path to the user directory containing exported images.

    Returns:
        tuple[str, list[ImageInfo]]: A tuple containing the formatted
            Markdown string and a list of `ImageInfo` objects.

    """
    title = conversation.get("title", "Untitled")
    conversation_id = conversation.get("conversation_id") or conversation.get("id", "")
    create_time = conversation.get("create_time")
    update_time = conversation.get("update_time")
    is_archived = conversation.get("is_archived", False)
    model_slug = conversation.get("default_model_slug", "unknown")

    logger.info("Converting conversation: %s (%s)", title, conversation_id)

    header_parts = [f"# {title}\n", f"- **Conversation ID:** {conversation_id}"]

    if create_time:
        header_parts.append(f"- **Created:** {unix_timestamp_to_str(create_time)}")
    if update_time:
        header_parts.append(f"- **Updated:** {unix_timestamp_to_str(update_time)}")

    header_parts.append(f"- **Archived:** {'Yes' if is_archived else 'No'}")
    header_parts.append(f"- **Model:** {model_slug}\n")

    mapping = conversation.get("mapping", {})
    messages = _traverse_message_tree(mapping, user_dir)

    messages = sorted(messages, key=lambda m: m.timestamp if m.timestamp else 0)

    logger.debug("Extracted %d visible messages", len(messages))

    all_images: list[ImageInfo] = []
    for message in messages:
        if message.images:
            all_images.extend(message.images)

    message_blocks: list[str] = []
    image_counter = 0
    for message in messages:
        message_block = _format_message(message, conversation_id, image_counter)
        if message_block:
            message_blocks.append(message_block)
        if message.images:
            image_counter += len(message.images)

    header = "\n".join(header_parts)
    messages_section = "\n\n".join(message_blocks)

    return f"{header}\n{messages_section}", all_images


def _copy_conversation_images(
    conversation_id: str, images: list[ImageInfo], output_dir: Path, user_dir: Path
) -> None:
    """Copy conversation images to a subdirectory in the output location.

    Args:
        conversation_id: The conversation identifier for naming the
            subdirectory.
        images: List of `ImageInfo` objects.
        output_dir: Base output directory for processed conversations.
        user_dir: Path to the user directory containing source images.

    """
    if not images:
        return

    image_dir = output_dir.joinpath(conversation_id)
    image_dir.mkdir(exist_ok=True)

    for image_info in images:
        if not image_info.filename:
            logger.warning("No filename for asset: %s", image_info.asset)
            continue

        asset_id = image_info.asset.removeprefix("sediment://")

        matching_files = list(user_dir.glob(f"{asset_id}-*.png"))

        if not matching_files:
            logger.warning("Image not found for asset: %s", asset_id)
            continue

        source_file = matching_files[0]
        dest_file = image_dir.joinpath(image_info.filename)

        shutil.copy2(source_file, dest_file)
        logger.debug("Copied image: %s -> %s", source_file.name, dest_file.name)


def _has_content(conversation: dict[str, Any], user_dir: Path) -> bool:
    """Check whether a conversation contains meaningful content.

    Args:
        conversation: Dictionary containing the conversation data.
        user_dir: Path to the user directory containing exported images.

    Returns:
        bool: `True` if the conversation has a title and messages, otherwise
            `False`.

    """
    if conversation.get("title", "").strip():
        mapping = conversation.get("mapping", {})
        messages = _traverse_message_tree(mapping, user_dir)
        return len(messages) > 0

    return False


def main() -> None:
    """Convert ChatGPT conversation exports to Markdown files.

    Reads `conversations.json` from the input directory, converts each
    conversation with content to Markdown, and writes the results to the output
    directory along with any associated images.
    """
    parser = build_argument_parser("chatgpt.com")
    args = parser.parse_args(namespace=Args())

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="[%(levelname)-8s] %(message)s")

    conversations_path = args.input_dir.joinpath("conversations.json")
    with conversations_path.open() as f:
        conversations = json.load(f)

    logger.info("Loaded %d conversations", len(conversations))

    user_dir = _get_user_dir(args.input_dir)
    logger.info("Found user directory: %s", user_dir)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    exported_count = 0
    skipped_count = 0

    for conversation in conversations:
        if not _has_content(conversation, user_dir):
            conv_id = conversation.get("conversation_id") or conversation.get(
                "id", "unknown"
            )
            logger.debug("Skipping empty conversation (%s)", conv_id)
            skipped_count += 1
            continue

        conversation_id = conversation.get("conversation_id") or conversation.get(
            "id", "unknown"
        )
        md_content, images = _convert_to_markdown(conversation, user_dir)
        output_file = args.output_dir.joinpath(f"{conversation_id}.md")
        output_file.write_text(md_content)

        if images:
            _copy_conversation_images(
                conversation_id, images, args.output_dir, user_dir
            )

        exported_count += 1

    logger.info(
        "Export complete: %d exported, %d skipped", exported_count, skipped_count
    )
