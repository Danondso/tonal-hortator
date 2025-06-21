"""
Tonal Hortator - AI-powered local music playlist generator

A Python package for generating music playlists using local Ollama embeddings
and semantic search, with seamless Apple Music integration.
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