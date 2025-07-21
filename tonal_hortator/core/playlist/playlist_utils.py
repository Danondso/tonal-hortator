"""
Playlist utility functions for Tonal Hortator.

Shared utility functions used across playlist generation components.
This module avoids circular imports by providing common functionality
that can be imported by multiple playlist modules.
"""

import re
import urllib.parse
from typing import Any, Dict


def create_playlist_name_static(query: str) -> str:
    """
    Create a playlist name from a query string (static version)

    Args:
        query: The user query string

    Returns:
        A cleaned playlist name
    """
    # Remove common prefixes
    prefixes_to_remove = [
        "generate",
        "create",
        "make",
        "build",
        "find",
        "get",
        "show",
        "give",
        "me",
        "some",
        "a",
        "an",
        "the",
        "playlist",
        "of",
        "for",
    ]

    # Clean the query
    words = query.lower().split()
    filtered_words = []

    for word in words:
        # Remove common prefixes
        if word not in prefixes_to_remove:
            filtered_words.append(word)

    # If we filtered out everything, use original query
    if not filtered_words:
        filtered_words = query.split()

    # Join and capitalize
    playlist_name = " ".join(filtered_words).title()

    # If the playlist name is very short (single word), add "Mix"
    if len(filtered_words) == 1 and len(playlist_name) < 10:
        playlist_name += " Mix"

    # Handle empty case
    if not playlist_name.strip():
        playlist_name = "Mix"

    return playlist_name


def normalize_file_location_static(location: str) -> str:
    """
    Normalize file location by removing username and decoding URL encoding (static version)

    Args:
        location: File path or URL

    Returns:
        Normalized file location
    """
    if not location:
        return ""

    # Decode URL encoding
    try:
        location = urllib.parse.unquote(location)
    except Exception:
        pass  # If decoding fails, use original

    # Remove file:// protocol if present
    if location.startswith("file://"):
        location = location[7:]

    # Normalize path separators
    location = location.replace("\\", "/")

    # Remove username from common paths
    # Windows: C:\Users\username\... -> C:\Users\[user]\...
    # macOS: /Users/username/... -> /Users/[user]/...
    # Linux: /home/username/... -> /home/[user]/...

    patterns = [
        (r"/Users/[^/]+/", "/Users/[user]/"),
        (r"/home/[^/]+/", "/home/[user]/"),
        (r"C:/Users/[^/]+/", "C:/Users/[user]/"),
        (r"C:\\Users\\[^\\]+\\", "C:\\Users/[user]/"),
    ]

    for pattern, replacement in patterns:
        location = re.sub(pattern, replacement, location)

    return location


def extract_base_name_static(name: str) -> str:
    """
    Extract base name by removing common suffixes (static version)

    Args:
        name: Track name

    Returns:
        Base name without suffixes
    """
    if not name:
        return name

    # Remove common suffixes in parentheses
    suffixes_to_remove = [
        r"\s*\(Remix\)",
        r"\s*\(Live\)",
        r"\s*\(Acoustic\)",
        r"\s*\(Radio Edit\)",
        r"\s*\(Extended Mix\)",
        r"\s*\(Club Mix\)",
        r"\s*\(Original Mix\)",
    ]

    base_name = name
    for suffix_pattern in suffixes_to_remove:
        base_name = re.sub(suffix_pattern, "", base_name, flags=re.IGNORECASE)

    return base_name.strip()
