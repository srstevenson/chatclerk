"""Datetime helpers."""

from datetime import UTC, datetime
from typing import Any, Final

STR_FORMAT: Final = "%Y-%m-%d %H:%M:%S UTC"


def unix_timestamp_to_str(timestamp: float) -> str:
    """Convert a Unix timestamp to a human-readable string.

    Args:
        timestamp: Unix timestamp as a float.

    Returns:
        Formatted timestamp string in the format `YYYY-MM-DD HH:MM:SS UTC`.
    """
    dt = datetime.fromtimestamp(timestamp, tz=UTC)
    return dt.strftime(STR_FORMAT)


def iso_timestamp_to_str(timestamp_str: str) -> str:
    """Convert an ISO 8601 timestamp to a human-readable string.

    Args:
        timestamp_str: ISO 8601 formatted timestamp string.

    Returns:
        Formatted timestamp string in the format `YYYY-MM-DD HH:MM:SS UTC`.
    """
    timestamp = datetime.fromisoformat(timestamp_str)
    return timestamp.strftime(STR_FORMAT)


def mongodb_timestamp_to_str(timestamp_obj: dict[str, Any]) -> str:
    """Convert a MongoDB timestamp to a human-readable string.

    Args:
        timestamp_obj: MongoDB timestamp dictionary with $date and $numberLong.

    Returns:
        Formatted timestamp string in the format `YYYY-MM-DD HH:MM:SS UTC`.
    """
    timestamp_ms = int(timestamp_obj["$date"]["$numberLong"])
    timestamp_s = timestamp_ms / 1000.0
    dt = datetime.fromtimestamp(timestamp_s, tz=UTC)
    return dt.strftime(STR_FORMAT)
