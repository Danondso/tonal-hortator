"""
Apple Music integration utilities
"""

import os
import subprocess
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def open_in_apple_music(playlist_path: str) -> bool:
    """
    Open M3U playlist in Apple Music

    Args:
        playlist_path: Path to M3U playlist file

    Returns:
        True if successful, False otherwise
    """
    if not os.path.exists(playlist_path):
        logger.error(f"Playlist not found: {playlist_path}")
        return False

    try:
        abs_path = os.path.abspath(playlist_path)
        logger.info(f"Opening in Apple Music: {os.path.basename(playlist_path)}")

        # Open with Apple Music app
        subprocess.run(["open", "-a", "Music", abs_path], check=True)
        logger.info("Successfully opened in Apple Music")
        return True

    except subprocess.CalledProcessError as e:
        logger.error(f"Could not open in Apple Music: {e}")
        return False
    except Exception as e:
        logger.error(f"Error opening playlist: {e}")
        return False


def find_latest_playlist(playlist_dir: str = "playlists") -> Optional[str]:
    """
    Find the most recent M3U playlist file

    Args:
        playlist_dir: Directory to search for playlists

    Returns:
        Path to latest playlist or None if not found
    """
    playlist_path = Path(playlist_dir)

    if not playlist_path.exists():
        logger.warning(f"No playlist directory found: {playlist_dir}")
        return None

    m3u_files = list(playlist_path.glob("*.m3u"))

    if not m3u_files:
        logger.warning(f"No M3U files found in {playlist_dir}")
        return None

    # Get most recent playlist
    latest_playlist = max(m3u_files, key=lambda x: x.stat().st_mtime)
    logger.info(f"Found latest playlist: {latest_playlist.name}")

    return str(latest_playlist)


def list_available_playlists(playlist_dir: str = "playlists") -> list:
    """
    List all available M3U playlists

    Args:
        playlist_dir: Directory to search for playlists

    Returns:
        List of playlist file paths
    """
    playlist_path = Path(playlist_dir)

    if not playlist_path.exists():
        logger.warning(f"No playlist directory found: {playlist_dir}")
        return []

    m3u_files = list(playlist_path.glob("*.m3u"))

    if not m3u_files:
        logger.warning(f"No M3U files found in {playlist_dir}")
        return []

    # Sort by modification time (newest first)
    sorted_files = sorted(m3u_files, key=lambda x: x.stat().st_mtime, reverse=True)

    return [str(f) for f in sorted_files]
