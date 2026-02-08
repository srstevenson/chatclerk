"""Convert data export from kagi.com to Markdown."""

import json
import logging
from typing import Final, TypedDict, TypeGuard, cast

from chatclerk.argparse import Args, build_argument_parser

logger: Final = logging.getLogger(__name__)


class ThreadDetail(TypedDict, total=False):
    """Thread metadata from the Kagi export.

    Attributes:
        id: Unique thread identifier.
        title: Thread title.
    """

    id: str
    title: str


class Thread(TypedDict, total=False):
    """Thread export data from the Kagi export.

    Attributes:
        threadDetail: Thread metadata.
        markdownExport: Markdown export content.
    """

    threadDetail: ThreadDetail
    markdownExport: str


def _is_thread(item: object) -> TypeGuard[Thread]:
    """Validate whether a raw JSON item matches the Thread structure.

    Args:
        item: Raw JSON item to validate.

    Returns:
        `True` if the item matches the expected thread structure.
    """
    if not isinstance(item, dict):
        return False

    typed_item = cast("dict[str, object]", item)
    thread_detail = typed_item.get("threadDetail")
    if thread_detail is not None and not isinstance(thread_detail, dict):
        return False

    markdown = typed_item.get("markdownExport")
    if markdown is None:
        return True
    return isinstance(markdown, str)


def _as_thread(item: object) -> Thread | None:
    """Validate and cast a raw JSON item to a Thread.

    Args:
        item: Raw JSON item to validate and cast.

    Returns:
        The thread data if valid, otherwise `None`.
    """
    return item if _is_thread(item) else None


def _extract_threads(data: object) -> list[Thread]:
    """Extract thread data from the export JSON structure.

    Args:
        data: Loaded JSON data from the export file.

    Returns:
        List of thread dictionaries.
    """
    if isinstance(data, list):
        items = cast("list[object]", data)
        return [thread for item in items if (thread := _as_thread(item)) is not None]

    if isinstance(data, dict):
        data_dict = cast("dict[str, object]", data)
        for key in ["threads", "conversations", "data"]:
            if isinstance(data_dict.get(key), list):
                items = cast("list[object]", data_dict[key])
                return [
                    thread for item in items if (thread := _as_thread(item)) is not None
                ]

    msg = "Unrecognized Kagi export format."
    raise ValueError(msg)


def _has_content(thread: Thread) -> bool:
    """Check whether a thread contains Markdown content.

    Args:
        thread: Dictionary containing thread data.

    Returns:
        `True` if the thread has Markdown content, otherwise `False`.
    """
    markdown = thread.get("markdownExport") or ""
    return bool(markdown.strip())


def main() -> None:
    """Convert Kagi Assistant exports to Markdown files.

    Reads a Kagi Assistant export JSON file, writes each thread's `markdownExport`
    directly to a Markdown file named after the thread ID.
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

    threads = _extract_threads(data)
    logger.info("Loaded %d threads", len(threads))

    args.output_dir.mkdir(parents=True, exist_ok=True)

    exported_count = 0
    skipped_count = 0

    for thread in threads:
        thread_detail = thread.get("threadDetail")
        thread_id = thread_detail.get("id", "unknown") if thread_detail else "unknown"
        title = thread_detail.get("title", "Untitled") if thread_detail else "Untitled"

        if not _has_content(thread):
            logger.debug("Skipping empty thread (%s)", thread_id)
            skipped_count += 1
            continue

        logger.info("Converting: %s (%s)", title, thread_id)
        markdown = thread.get("markdownExport") or ""
        output_file = args.output_dir / f"{thread_id}.md"
        output_file.write_text(markdown, encoding="utf-8")

        exported_count += 1

    logger.info(
        "Export complete: %d exported, %d skipped", exported_count, skipped_count
    )
