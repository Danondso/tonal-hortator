#!/usr/bin/env python3
"""
Data models for Tonal Hortator.

This module contains dataclasses that define the structure of data used
throughout the application, replacing Any type annotations for better
type safety and code clarity.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Protocol, Union


@dataclass
class Track:
    """Represents a music track with all its metadata."""

    # Basic identification
    id: Optional[int] = None
    name: Optional[str] = None
    artist: Optional[str] = None
    album: Optional[str] = None

    # Extended metadata
    album_artist: Optional[str] = None
    composer: Optional[str] = None
    genre: Optional[str] = None
    year: Optional[int] = None

    # Track details
    total_time: Optional[int] = None  # in milliseconds
    track_number: Optional[int] = None
    disc_number: Optional[int] = None
    play_count: Optional[int] = None
    date_added: Optional[str] = None
    location: Optional[str] = None

    # Musical analysis
    bpm: Optional[float] = None
    musical_key: Optional[str] = None
    key_scale: Optional[str] = None
    mood: Optional[str] = None
    chord_changes_rate: Optional[float] = None
    chord_key: Optional[str] = None
    chord_scale: Optional[str] = None

    # Production info
    label: Optional[str] = None
    producer: Optional[str] = None
    arranger: Optional[str] = None
    lyricist: Optional[str] = None
    original_year: Optional[int] = None
    original_date: Optional[str] = None
    script: Optional[str] = None
    replay_gain: Optional[float] = None
    release_country: Optional[str] = None
    catalog_number: Optional[str] = None

    # Technical metadata
    isrc: Optional[str] = None
    barcode: Optional[str] = None
    acoustid_id: Optional[str] = None
    musicbrainz_track_id: Optional[str] = None
    musicbrainz_artist_id: Optional[str] = None
    musicbrainz_album_id: Optional[str] = None
    media_type: Optional[str] = None
    analyzed_genre: Optional[str] = None

    # User data
    avg_rating: Optional[float] = None
    rating_count: Optional[int] = None
    similarity_score: Optional[float] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Track":
        """Create a Track from a dictionary, safely handling missing keys."""
        # Filter out keys that don't exist in the dataclass
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered_data)

    def to_dict(self) -> Dict[str, Any]:
        """Convert Track to dictionary, excluding None values."""
        return {k: v for k, v in self.__dict__.items() if v is not None}


@dataclass
class TrackEmbedding:
    """Represents a track's embedding data."""

    track_id: int
    embedding: bytes
    embedding_text: str
    created_at: Optional[datetime] = None


@dataclass
class UserFeedback:
    """Represents user feedback on a playlist or query."""

    id: Optional[int] = None
    query: str = ""
    query_type: str = ""  # artist_specific, similarity, general
    parsed_artist: Optional[str] = None
    parsed_reference_artist: Optional[str] = None
    parsed_genres: Optional[List[str]] = None
    parsed_mood: Optional[str] = None
    generated_tracks: Optional[List[int]] = None
    user_rating: Optional[int] = None  # 1-5 stars
    user_comments: Optional[str] = None
    user_actions: Optional[List[str]] = None
    playlist_length: Optional[int] = None
    requested_length: Optional[int] = None
    similarity_threshold: Optional[float] = None
    search_breadth: Optional[int] = None
    created_at: Optional[datetime] = None


@dataclass
class UserPreference:
    """Represents a user preference setting."""

    id: Optional[int] = None
    preference_key: str = ""
    preference_value: str = ""
    preference_type: str = ""  # string, integer, float, boolean, json
    description: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


@dataclass
class TrackRating:
    """Represents a user's rating of a track."""

    id: Optional[int] = None
    track_id: int = 0
    rating: int = 0  # 1-5 stars
    context: Optional[str] = None
    created_at: Optional[datetime] = None


@dataclass
class QueryLearning:
    """Represents learning data for improving LLM query parsing."""

    id: Optional[int] = None
    original_query: str = ""
    llm_parsed_result: str = ""  # JSON string
    user_correction: Optional[str] = None  # JSON string
    feedback_score: Optional[float] = None  # -1 to 1
    learning_applied: bool = False
    created_at: Optional[datetime] = None


@dataclass
class LLMQueryResult:
    """Represents the parsed result from an LLM query."""

    query_type: str = "general"
    genres: List[str] = field(default_factory=list)
    mood: Optional[str] = None
    artist: Optional[str] = None
    reference_artist: Optional[str] = None
    track_count: Optional[int] = None
    similarity_threshold: Optional[float] = None
    vague: Optional[bool] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LLMQueryResult":
        """Create LLMQueryResult from dictionary."""
        return cls(
            query_type=data.get("query_type", "general"),
            genres=data.get("genres", []),
            mood=data.get("mood"),
            artist=data.get("artist"),
            reference_artist=data.get("reference_artist"),
            track_count=data.get("track_count"),
            similarity_threshold=data.get("similarity_threshold"),
            vague=data.get("vague"),
        )


@dataclass
class MetadataMapping:
    """Represents a metadata field mapping."""

    id: Optional[int] = None
    source_format: str = ""
    source_tag: str = ""
    normalized_tag: str = ""
    data_type: str = ""
    description: Optional[str] = None
    created_at: Optional[datetime] = None


@dataclass
class FileMetadata:
    """Represents metadata extracted from audio files."""

    # Basic info
    title: Optional[str] = None
    artist: Optional[str] = None
    album: Optional[str] = None
    album_artist: Optional[str] = None
    composer: Optional[str] = None
    genre: Optional[str] = None
    year: Optional[int] = None

    # Track info
    track_number: Optional[int] = None
    disc_number: Optional[int] = None
    total_time: Optional[int] = None

    # Musical metadata
    bpm: Optional[float] = None
    musical_key: Optional[str] = None
    key_scale: Optional[str] = None
    mood: Optional[str] = None

    # Production metadata
    label: Optional[str] = None
    producer: Optional[str] = None
    arranger: Optional[str] = None
    lyricist: Optional[str] = None

    # Technical metadata
    format_type: Optional[str] = None
    bitrate: Optional[int] = None
    sample_rate: Optional[int] = None
    channels: Optional[int] = None


@dataclass
class PlaylistGenerationRequest:
    """Represents a request to generate a playlist."""

    query: str
    max_tracks: int = 20
    min_similarity: float = 0.6
    search_breadth: int = 100
    genre_keywords: List[str] = field(default_factory=list)
    artist_name: Optional[str] = None
    reference_artist: Optional[str] = None
    parsed_data: Optional[LLMQueryResult] = None


@dataclass
class PlaylistGenerationResult:
    """Represents the result of playlist generation."""

    tracks: List[Track]
    query: str
    total_found: int
    generation_time: float
    playlist_name: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EmbeddingConfig:
    """Configuration for embedding operations."""

    model_name: str = "nomic-embed-text"
    host: str = "http://localhost:11434"
    batch_size: int = 500
    max_workers: int = 4
    embedding_dimension: Optional[int] = None


@dataclass
class HybridEmbeddingConfig:
    """Configuration for hybrid embedding strategies."""

    age_weight: float = 0.3
    metadata_weight: float = 0.3
    confidence_weight: float = 0.4
    decay_factor: float = 0.1
    min_confidence: float = 0.1
    max_age_days: int = 365


@dataclass
class DatabaseStats:
    """Database statistics."""

    total_tracks: int = 0
    total_embeddings: int = 0
    tracks_with_bpm: int = 0
    tracks_with_key: int = 0
    tracks_with_mood: int = 0
    avg_play_count: float = 0.0
    latest_date_added: Optional[str] = None


# Protocol for database connections
class DatabaseProtocol(Protocol):
    """Protocol for database connection objects."""

    def execute(self, query: str, params: tuple = ()) -> Any:
        """Execute a query with parameters."""
        ...

    def execute_fetchall(self, query: str, params: tuple = ()) -> List[Any]:
        """Execute a query and fetch all results."""
        ...

    def execute_fetchone(self, query: str, params: tuple = ()) -> Any:
        """Execute a query and fetch one result."""
        ...

    def commit(self) -> None:
        """Commit the current transaction."""
        ...

    def rollback(self) -> None:
        """Rollback the current transaction."""
        ...


@dataclass
class LoaderTask:
    """Represents a background loading task."""

    name: str
    status: str = "pending"  # pending, running, completed, failed
    progress: float = 0.0
    message: str = ""
    task_id: Optional[Any] = None
    error: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None


@dataclass
class ValidationRange:
    """Represents validation ranges for configuration values."""

    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    allowed_values: Optional[List[Any]] = None
    data_type: str = "string"  # string, int, float, bool


@dataclass
class ConfigSection:
    """Represents a configuration section with typed values."""

    name: str
    values: Dict[str, Any] = field(default_factory=dict)
    validation_ranges: Dict[str, ValidationRange] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value with default."""
        return self.values.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Set a configuration value."""
        self.values[key] = value
