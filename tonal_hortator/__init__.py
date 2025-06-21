#!/usr/bin/env python3
"""
Tonal Hortator - Local Music Playlist Generator

A Python package that generates music playlists using semantic search
with local AI embeddings, featuring seamless Apple Music integration.
"""

__version__ = "2.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from .core.embeddings import OllamaEmbeddingService
from .core.playlist_generator import LocalPlaylistGenerator
from .core.track_embedder import LocalTrackEmbedder

__all__ = [
    "OllamaEmbeddingService",
    "LocalPlaylistGenerator",
    "LocalTrackEmbedder",
]
