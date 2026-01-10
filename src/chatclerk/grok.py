"""Convert data export from grok.com to Markdown."""

import json
import logging
import re
import shutil
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Final

from chatclerk.argparse import Args, build_argument_parser
from chatclerk.datetime import iso_timestamp_to_str, mongodb_timestamp_to_str

if TYPE_CHECKING:
    from pathlib import Path

logger: Final = logging.getLogger(__name__)

# ASCII printable character range constants
ASCII_SPACE: Final = 32
ASCII_TILDE: Final = 127


@dataclass(frozen=True)
class ArtifactInfo:
    """Representation of an xaiArtifact in a message.

    Attributes:
        artifact_id: The artifact identifier.
        artifact_version_id: The version identifier (matches asset directory).
        title: Title of the artifact.
        content_type: MIME type of the artifact content.
        content: The actual artifact content.
        filename: Filename for the exported artifact file.

    """

    artifact_id: str
    artifact_version_id: str
    title: str
    content_type: str
    content: str
    filename: str


@dataclass(frozen=True)
class FileAttachmentInfo:
    """Representation of a file attachment in a message.

    Attributes:
        attachment_id: The attachment identifier (matches asset directory).
        filename: Filename for the exported attachment.
        exists: Whether the file exists in the asset server directory.

    """

    attachment_id: str
    filename: str
    exists: bool


@dataclass(frozen=True)
class MessageResult:
    """Result of formatting a message.

    Attributes:
        formatted_text: The formatted Markdown text.
        artifacts: List of artifacts extracted from the message.
        attachments: List of file attachments in the message.

    """

    formatted_text: str
    artifacts: list[ArtifactInfo] = field(default_factory=list)
    attachments: list[FileAttachmentInfo] = field(default_factory=list)


def _extract_artifacts_from_message(message: str) -> tuple[str, list[ArtifactInfo]]:
    """Extract xaiArtifact tags from message text and return cleaned message.

    Args:
        message: The message text potentially containing xaiArtifact tags.

    Returns:
        tuple[str, list[ArtifactInfo]]: Cleaned message text and list of
            extracted artifacts.

    """
    artifacts: list[ArtifactInfo] = []

    artifact_pattern = re.compile(
        r'<xaiArtifact\s+artifact_id="([^"]+)"\s+'
        r'artifact_version_id="([^"]+)"\s+'
        r'title="([^"]+)"\s+'
        r'contentType="([^"]+)">\s*'
        r"(.*?)\s*"
        r"</xaiArtifact>",
        re.DOTALL,
    )

    def _replace_artifact(match: re.Match[str]) -> str:
        artifact_id = match.group(1)
        artifact_version_id = match.group(2)
        title = match.group(3)
        content_type = match.group(4)
        content = match.group(5)

        ext_map = {
            "text/python": ".py",
            "text/javascript": ".js",
            "text/markdown": ".md",
            "text/html": ".html",
            "text/css": ".css",
        }
        ext = ext_map.get(content_type, ".txt")

        title_without_ext = title.rsplit(".", 1)[0] if "." in title else title
        safe_title = re.sub(r"[^\w\s-]", "", title_without_ext).strip()
        safe_title = safe_title.replace(" ", "_")
        filename = f"{safe_title}{ext}"

        artifacts.append(
            ArtifactInfo(
                artifact_id, artifact_version_id, title, content_type, content, filename
            )
        )

        logger.debug("Found artifact: %s (%s)", title, content_type)

        return f"*[Artifact: {title}]*"

    cleaned_message = artifact_pattern.sub(_replace_artifact, message)
    return cleaned_message, artifacts


def _detect_file_type(file_path: Path) -> str:  # noqa: PLR0911, C901
    """Detect file type from magic bytes.

    Args:
        file_path: Path to the file to detect.

    Returns:
        str: File extension based on detected type.

    """
    if not file_path.exists():
        return ".bin"

    try:
        with file_path.open("rb") as f:
            magic = f.read(16)

        # Check magic bytes for common types
        if magic[:8] == b"\x89PNG\r\n\x1a\n":
            return ".png"
        if magic[:3] == b"\xff\xd8\xff":
            return ".jpg"
        if magic[:4] == b"GIF8":
            return ".gif"
        if magic[:4] == b"RIFF" and magic[8:12] == b"WEBP":
            return ".webp"
        if magic[:2] == b"BM":
            return ".bmp"
        if magic[:4] == b"%PDF":
            return ".pdf"
        if magic[:2] == b"PK":
            return ".zip"

        # Fallback to text if printable
        is_printable = all(
            ASCII_SPACE <= b < ASCII_TILDE or b in (9, 10, 13)
            for b in magic[:100]
            if b != 0
        )
        if is_printable:
            return ".txt"
        return ".bin"  # noqa: TRY300
    except OSError:
        return ".bin"


def _extract_file_attachments(
    response_data: dict[str, Any], user_dir: Path
) -> list[FileAttachmentInfo]:
    """Extract file attachment information from response data.

    Args:
        response_data: Dictionary containing response data.
        user_dir: Path to the asset server directory.

    Returns:
        list[FileAttachmentInfo]: List of file attachment information.

    """
    attachments: list[FileAttachmentInfo] = []
    file_attachments = response_data.get("file_attachments", [])

    for attachment_id in file_attachments:
        asset_dir = user_dir / attachment_id
        asset_file = asset_dir / "content"
        exists = asset_file.exists()

        if exists:
            ext = _detect_file_type(asset_file)
            filename = f"{attachment_id}{ext}"
            logger.debug("Found attachment: %s (type: %s)", attachment_id, ext)
        else:
            filename = f"{attachment_id}.bin"
            logger.debug("Attachment missing: %s", attachment_id)

        attachments.append(FileAttachmentInfo(attachment_id, filename, exists))

    return attachments


def _format_message(  # noqa: PLR0912, C901
    response: dict[str, Any], user_dir: Path, conversation_id: str = ""
) -> MessageResult:
    """Format a single message as Markdown.

    Args:
        response: Dictionary containing response data.
        user_dir: Path to the asset server directory.
        conversation_id: The conversation ID for constructing asset paths.

    Returns:
        MessageResult: The formatted message and associated assets.

    """
    response_data = response["response"]
    sender = response_data["sender"]
    sender_lower = sender.lower()
    message = response_data.get("message", "").strip()
    create_time = response_data.get("create_time")
    model = response_data.get("model", "unknown")
    metadata = response_data.get("metadata", {})

    artifacts: list[ArtifactInfo] = []
    attachments: list[FileAttachmentInfo] = []

    if message:
        message, artifacts = _extract_artifacts_from_message(message)

    attachments = _extract_file_attachments(response_data, user_dir)

    if sender_lower == "human":
        header = "## User"
    elif sender_lower == "assistant":
        header = "## Assistant"
    else:
        header = f"## {sender.title()}"

    header_parts: list[str] = [header]

    if create_time and (timestamp := mongodb_timestamp_to_str(create_time)):
        timestamp_line = f"*{timestamp}*"
        if sender_lower == "assistant" and model != "unknown":
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

    for artifact in artifacts:
        if conversation_id:
            artifact_path = f"{conversation_id}/{artifact.filename}"
        else:
            artifact_path = artifact.filename
        content_parts.append(f"*[Artifact: {artifact.title}]({artifact_path})*")

    for attachment in attachments:
        if conversation_id:
            attachment_path = f"{conversation_id}/{attachment.filename}"
        else:
            attachment_path = attachment.filename

        if attachment.exists:
            link = f"*[Attachment: {attachment.filename}]({attachment_path})*"
            content_parts.append(link)
        else:
            content_parts.append(
                f"*[Attachment: {attachment.attachment_id} (file not in export)]*"
            )

    header_line = "\n\n".join(header_parts)
    content = "\n\n".join(content_parts)
    formatted_text = f"{header_line}\n\n{content}"

    return MessageResult(formatted_text, artifacts, attachments)


def _convert_to_markdown(
    conversation: dict[str, Any], user_dir: Path
) -> tuple[str, list[ArtifactInfo], list[FileAttachmentInfo]]:
    """Convert a Grok conversation to Markdown format.

    Args:
        conversation: Dictionary containing the conversation data.
        user_dir: Path to the asset server directory.

    Returns:
        tuple[str, list[ArtifactInfo], list[FileAttachmentInfo]]: The formatted
            conversation as Markdown, list of artifacts, and list of attachments.

    """
    conv_metadata = conversation["conversation"]
    conv_id = conv_metadata.get("id", "unknown")
    title = conv_metadata.get("title") or "Untitled"
    create_time = conv_metadata.get("create_time")
    modify_time = conv_metadata.get("modify_time")
    system_prompt = conv_metadata.get("system_prompt_name")

    logger.info(
        "Converting: %s (%s, %d messages)",
        title,
        conv_id,
        len(conversation["responses"]),
    )

    header_parts = [f"# {title}\n", f"- **Conversation ID:** {conv_id}"]

    if create_time and (created := iso_timestamp_to_str(create_time)):
        header_parts.append(f"- **Created:** {created}")
    if modify_time and (modified := iso_timestamp_to_str(modify_time)):
        header_parts.append(f"- **Updated:** {modified}")
    if system_prompt:
        header_parts.append(f"- **System Prompt:** {system_prompt}")

    header_parts.append("")

    responses: list[dict[str, Any]] = conversation["responses"]
    message_blocks: list[str] = []
    all_artifacts: list[ArtifactInfo] = []
    all_attachments: list[FileAttachmentInfo] = []

    for response in responses:
        result = _format_message(response, user_dir, conv_id)
        message_blocks.append(result.formatted_text)
        all_artifacts.extend(result.artifacts)
        all_attachments.extend(result.attachments)

    header = "\n".join(header_parts)
    messages_section = "\n\n".join(message_blocks)

    return f"{header}\n{messages_section}", all_artifacts, all_attachments


def _copy_conversation_assets(
    conversation_id: str,
    artifacts: list[ArtifactInfo],
    attachments: list[FileAttachmentInfo],
    output_dir: Path,
    user_dir: Path,
) -> None:
    """Copy conversation artifacts and attachments to output directory.

    Args:
        conversation_id: The conversation identifier for naming the subdirectory.
        artifacts: List of artifacts to write.
        attachments: List of attachments to copy.
        output_dir: Base output directory for processed conversations.
        user_dir: Path to the asset server directory containing source files.

    """
    if not artifacts and not attachments:
        return

    asset_dir = output_dir / conversation_id
    asset_dir.mkdir(exist_ok=True)

    for artifact in artifacts:
        output_file = asset_dir / artifact.filename
        output_file.write_text(artifact.content)
        logger.debug("Wrote artifact: %s", artifact.filename)

    for attachment in attachments:
        if not attachment.exists:
            logger.debug("Skipping missing attachment: %s", attachment.attachment_id)
            continue

        source_file = user_dir / attachment.attachment_id / "content"
        dest_file = asset_dir / attachment.filename

        shutil.copy2(source_file, dest_file)
        logger.debug("Copied attachment: %s -> %s", source_file.name, dest_file.name)


def _has_content(conversation: dict[str, Any]) -> bool:
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


def main() -> None:
    """Convert Grok conversation exports to Markdown files.

    Reads `prod-grok-backend.json` from the input directory, converts each
    conversation with content to Markdown, and writes the results to the output
    directory along with any associated artifacts and attachments.
    """
    parser = build_argument_parser("grok.com")
    args = parser.parse_args(namespace=Args())

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="[%(levelname)-8s] %(message)s")

    conversations_path = args.input_dir.joinpath("prod-grok-backend.json")
    with conversations_path.open() as f:
        data = json.load(f)

    conversations = data["conversations"]
    logger.info("Loaded %d conversations", len(conversations))

    user_dir = args.input_dir / "prod-mc-asset-server"
    if user_dir.exists():
        logger.info("Found asset server directory: %s", user_dir)
    else:
        logger.warning("Asset server directory not found: %s", user_dir)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    exported_count = 0
    skipped_count = 0

    for conversation in conversations:
        conv_id = conversation["conversation"].get("id", "unknown")

        if not _has_content(conversation):
            logger.debug("Skipping empty conversation (%s)", conv_id)
            skipped_count += 1
            continue

        md_content, artifacts, attachments = _convert_to_markdown(
            conversation, user_dir
        )
        output_file = args.output_dir.joinpath(f"{conv_id}.md")
        output_file.write_text(md_content)

        if artifacts or attachments:
            _copy_conversation_assets(
                conv_id, artifacts, attachments, args.output_dir, user_dir
            )

        exported_count += 1

    logger.info(
        "Export complete: %d exported, %d skipped", exported_count, skipped_count
    )
