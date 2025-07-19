"""
Helper functions for building dynamic SQL queries.

This module provides utility functions for constructing SQL queries
that require dynamic field lists or placeholder generation.
"""

from typing import List


def build_update_track_query(fields: List[str]) -> str:
    """
    Build a dynamic UPDATE query for tracks.

    Args:
        fields: List of field names to update

    Returns:
        Formatted SQL query string
    """
    if not fields:
        raise ValueError("No fields provided for update")

    set_clause = ", ".join(f"{field} = ?" for field in fields)
    return f"UPDATE tracks SET {set_clause} WHERE id = ?"


def build_insert_track_query(fields: List[str]) -> str:
    """
    Build a dynamic INSERT query for tracks.

    Args:
        fields: List of field names to insert

    Returns:
        Formatted SQL query string
    """
    if not fields:
        raise ValueError("No fields provided for insert")

    placeholders = ", ".join(["?" for _ in fields])
    fields_str = ", ".join(fields)
    return f"INSERT INTO tracks ({fields_str}) VALUES ({placeholders})"


def build_get_tracks_by_ids_query(track_ids: List[int]) -> str:
    """
    Build a query to get tracks by their IDs.

    Args:
        track_ids: List of track IDs

    Returns:
        Formatted SQL query string
    """
    if not track_ids:
        raise ValueError("No track IDs provided")

    placeholders = ", ".join(["?" for _ in track_ids])
    return f"SELECT * FROM tracks WHERE id IN ({placeholders}) ORDER BY id"


def build_delete_embeddings_by_ids_query(track_ids: List[int]) -> str:
    """
    Build a query to delete embeddings by track IDs.

    Args:
        track_ids: List of track IDs

    Returns:
        Formatted SQL query string
    """
    if not track_ids:
        raise ValueError("No track IDs provided")

    placeholders = ", ".join(["?" for _ in track_ids])
    return f"DELETE FROM track_embeddings WHERE track_id IN ({placeholders})"
