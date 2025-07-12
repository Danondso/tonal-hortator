#!/usr/bin/env python3
"""
Tonal Hortator - Local Music Playlist Generator

A Python package that generates music playlists using semantic search
with local AI embeddings, featuring seamless Apple Music integration.
"""

__version__ = "2.0.2"
__author__ = "Danondso"
__email__ = "7014871+Danondso@users.noreply.github.com"

from .core.embeddings.embeddings import OllamaEmbeddingService
from .core.embeddings.track_embedder import LocalTrackEmbedder
from .core.playlist.playlist_generator import LocalPlaylistGenerator

__all__ = [
    "OllamaEmbeddingService",
    "LocalPlaylistGenerator",
    "LocalTrackEmbedder",
]
