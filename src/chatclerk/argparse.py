"""Command-line argument parsing functionality."""

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


@dataclass
class Args(argparse.Namespace):
    """Command-line arguments.

    Attributes:
        input_dir: Directory containing the export data.
        output_dir: Directory where Markdown files will be written.
        verbose: Enable verbose logging.
    """

    input_dir: Path = field(init=False)
    output_dir: Path = field(init=False)
    verbose: bool = False


def build_argument_parser(
    service: Literal["chatgpt.com", "claude.ai", "grok.com", "kagi.com"],
    *,
    input_arg: str = "--input-dir",
    input_help: str | None = None,
) -> argparse.ArgumentParser:
    """Build command-line argument parser.

    Returns:
        Command-line argument parser.
    """
    if input_help is None:
        input_help = f"directory containing {service} export"

    parser = argparse.ArgumentParser(
        description=f"convert data export from {service} to Markdown"
    )
    parser.add_argument(
        "-i", input_arg, type=Path, dest="input_dir", required=True, help=input_help
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        required=True,
        help="directory to write Markdown files",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="enable verbose logging"
    )
    return parser
