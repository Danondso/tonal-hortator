"""
Metadata mappings for Tonal Hortator.

This module contains the complete metadata mappings data structure
that maps various audio format tags to normalized database columns.
"""

from typing import List, Tuple

# Complete metadata mappings data structure
# Format: (source_format, source_tag, normalized_tag, data_type, description)
METADATA_MAPPINGS = [
    # MP3 EasyID3 mappings
    ("mp3", "easyid3_artist", "artist", "string", "Artist name"),
    ("mp3", "easyid3_album", "album", "string", "Album name"),
    ("mp3", "easyid3_genre", "genre", "string", "Genre"),
    ("mp3", "easyid3_title", "title", "string", "Track title"),
    ("mp3", "easyid3_date", "year", "integer", "Release year"),
    ("mp3", "easyid3_tracknumber", "track_number", "integer", "Track number"),
    ("mp3", "easyid3_discnumber", "disc_number", "integer", "Disc number"),
    ("mp3", "easyid3_albumartist", "album_artist", "string", "Album artist"),
    ("mp3", "easyid3_composer", "composer", "string", "Composer"),
    ("mp3", "easyid3_organization", "label", "string", "Record label"),
    ("mp3", "easyid3_media", "media_type", "string", "Media type"),
    ("mp3", "easyid3_isrc", "isrc", "string", "ISRC code"),
    ("mp3", "easyid3_barcode", "barcode", "string", "Album barcode"),
    ("mp3", "easyid3_catalognumber", "catalog_number", "string", "Catalog number"),
    ("mp3", "easyid3_releasecountry", "release_country", "string", "Release country"),
    ("mp3", "easyid3_originaldate", "original_date", "string", "Original release date"),
    ("mp3", "easyid3_acoustid_id", "acoustid_id", "string", "AcoustID"),
    (
        "mp3",
        "easyid3_musicbrainz_trackid",
        "musicbrainz_track_id",
        "string",
        "MusicBrainz track ID",
    ),
    (
        "mp3",
        "easyid3_musicbrainz_artistid",
        "musicbrainz_artist_id",
        "string",
        "MusicBrainz artist ID",
    ),
    (
        "mp3",
        "easyid3_musicbrainz_albumid",
        "musicbrainz_album_id",
        "string",
        "MusicBrainz album ID",
    ),
    # MP3 ID3 mappings
    ("mp3", "id3_TPE1", "artist", "string", "Artist name"),
    ("mp3", "id3_TALB", "album", "string", "Album name"),
    ("mp3", "id3_TCON", "genre", "string", "Genre"),
    ("mp3", "id3_TIT2", "title", "string", "Track title"),
    ("mp3", "id3_TDRC", "year", "integer", "Release year"),
    ("mp3", "id3_TRCK", "track_number", "integer", "Track number"),
    ("mp3", "id3_TPOS", "disc_number", "integer", "Disc number"),
    ("mp3", "id3_TPE2", "album_artist", "string", "Album artist"),
    ("mp3", "id3_TCOM", "composer", "string", "Composer"),
    ("mp3", "id3_TPUB", "label", "string", "Record label"),
    ("mp3", "id3_TMED", "media_type", "string", "Media type"),
    ("mp3", "id3_TSRC", "isrc", "string", "ISRC code"),
    ("mp3", "id3_TDOR", "original_date", "string", "Original release date"),
    # MP3 TXXX mappings for musical analysis
    ("mp3", "id3_TXXX:ab:lo:rhythm:bpm", "bpm", "float", "Tempo in BPM"),
    ("mp3", "id3_TXXX:ab:lo:tonal:key_key", "musical_key", "string", "Musical key"),
    ("mp3", "id3_TXXX:ab:lo:tonal:key_scale", "key_scale", "string", "Key scale"),
    ("mp3", "id3_TXXX:ab:lo:tonal:chords_key", "chord_key", "string", "Chord key"),
    (
        "mp3",
        "id3_TXXX:ab:lo:tonal:chords_scale",
        "chord_scale",
        "string",
        "Chord scale",
    ),
    (
        "mp3",
        "id3_TXXX:ab:lo:tonal:chords_changes_rate",
        "chord_changes_rate",
        "float",
        "Chord changes rate",
    ),
    ("mp3", "id3_TXXX:ab:mood", "mood", "string", "Mood classification"),
    ("mp3", "id3_TXXX:ab:genre", "analyzed_genre", "string", "Analyzed genre"),
    ("mp3", "id3_TXXX:Acoustid Id", "acoustid_id", "string", "AcoustID"),
    (
        "mp3",
        "id3_TXXX:MusicBrainz Track Id",
        "musicbrainz_track_id",
        "string",
        "MusicBrainz track ID",
    ),
    (
        "mp3",
        "id3_TXXX:MusicBrainz Artist Id",
        "musicbrainz_artist_id",
        "string",
        "MusicBrainz artist ID",
    ),
    (
        "mp3",
        "id3_TXXX:MusicBrainz Album Id",
        "musicbrainz_album_id",
        "string",
        "MusicBrainz album ID",
    ),
    ("mp3", "id3_TXXX:BARCODE", "barcode", "string", "Album barcode"),
    ("mp3", "id3_TXXX:CATALOGNUMBER", "catalog_number", "string", "Catalog number"),
    (
        "mp3",
        "id3_TXXX:MusicBrainz Album Release Country",
        "release_country",
        "string",
        "Release country",
    ),
    ("mp3", "id3_TXXX:originalyear", "original_year", "integer", "Original year"),
    # M4A mappings
    ("m4a", "m4a_©ART", "artist", "string", "Artist name"),
    ("m4a", "m4a_©alb", "album", "string", "Album name"),
    ("m4a", "m4a_©gen", "genre", "string", "Genre"),
    ("m4a", "m4a_©nam", "title", "string", "Track title"),
    ("m4a", "m4a_©day", "year", "integer", "Release year"),
    ("m4a", "m4a_trkn", "track_number", "integer", "Track number"),
    ("m4a", "m4a_disk", "disc_number", "integer", "Disc number"),
    ("m4a", "m4a_aART", "album_artist", "string", "Album artist"),
    ("m4a", "m4a_©wrt", "composer", "string", "Composer"),
    ("m4a", "m4a_©too", "producer", "string", "Producer"),
    ("m4a", "m4a_©lyr", "lyricist", "string", "Lyricist"),
    # M4A iTunes mappings
    ("m4a", "m4a_----:com.apple.iTunes:ARTISTS", "artist", "string", "Artist name"),
    (
        "m4a",
        "m4a_----:com.apple.iTunes:ALBUM_ARTISTS",
        "album_artist",
        "string",
        "Album artist",
    ),
    ("m4a", "m4a_----:com.apple.iTunes:LABEL", "label", "string", "Record label"),
    ("m4a", "m4a_----:com.apple.iTunes:MEDIA", "media_type", "string", "Media type"),
    ("m4a", "m4a_----:com.apple.iTunes:ISRC", "isrc", "string", "ISRC code"),
    ("m4a", "m4a_----:com.apple.iTunes:BARCODE", "barcode", "string", "Album barcode"),
    (
        "m4a",
        "m4a_----:com.apple.iTunes:CATALOGNUMBER",
        "catalog_number",
        "string",
        "Catalog number",
    ),
    (
        "m4a",
        "m4a_----:com.apple.iTunes:MusicBrainz Album Release Country",
        "release_country",
        "string",
        "Release country",
    ),
    (
        "m4a",
        "m4a_----:com.apple.iTunes:ORIGINALDATE",
        "original_date",
        "string",
        "Original release date",
    ),
    (
        "m4a",
        "m4a_----:com.apple.iTunes:ORIGINAL YEAR",
        "original_year",
        "integer",
        "Original year",
    ),
    (
        "m4a",
        "m4a_----:com.apple.iTunes:Acoustid Id",
        "acoustid_id",
        "string",
        "AcoustID",
    ),
    (
        "m4a",
        "m4a_----:com.apple.iTunes:MusicBrainz Track Id",
        "musicbrainz_track_id",
        "string",
        "MusicBrainz track ID",
    ),
    (
        "m4a",
        "m4a_----:com.apple.iTunes:MusicBrainz Artist Id",
        "musicbrainz_artist_id",
        "string",
        "MusicBrainz artist ID",
    ),
    (
        "m4a",
        "m4a_----:com.apple.iTunes:MusicBrainz Album Id",
        "musicbrainz_album_id",
        "string",
        "MusicBrainz album ID",
    ),
    ("m4a", "m4a_----:com.apple.iTunes:PRODUCER", "producer", "string", "Producer"),
    ("m4a", "m4a_----:com.apple.iTunes:LYRICIST", "lyricist", "string", "Lyricist"),
    ("m4a", "m4a_----:com.apple.iTunes:ARRANGER", "arranger", "string", "Arranger"),
    ("m4a", "m4a_----:com.apple.iTunes:publisher", "label", "string", "Record label"),
    # M4A musical analysis mappings
    (
        "m4a",
        "m4a_----:com.apple.iTunes:ab:lo:rhythm:bpm",
        "bpm",
        "float",
        "Tempo in BPM",
    ),
    (
        "m4a",
        "m4a_----:com.apple.iTunes:ab:lo:tonal:key_key",
        "musical_key",
        "string",
        "Musical key",
    ),
    (
        "m4a",
        "m4a_----:com.apple.iTunes:ab:lo:tonal:key_scale",
        "key_scale",
        "string",
        "Key scale",
    ),
    (
        "m4a",
        "m4a_----:com.apple.iTunes:ab:lo:tonal:chords_key",
        "chord_key",
        "string",
        "Chord key",
    ),
    (
        "m4a",
        "m4a_----:com.apple.iTunes:ab:lo:tonal:chords_scale",
        "chord_scale",
        "string",
        "Chord scale",
    ),
    (
        "m4a",
        "m4a_----:com.apple.iTunes:ab:lo:tonal:chords_changes_rate",
        "chord_changes_rate",
        "float",
        "Chord changes rate",
    ),
    (
        "m4a",
        "m4a_----:com.apple.iTunes:ab:mood",
        "mood",
        "string",
        "Mood classification",
    ),
    (
        "m4a",
        "m4a_----:com.apple.iTunes:ab:genre",
        "analyzed_genre",
        "string",
        "Analyzed genre",
    ),
]

# Organized mappings by format for easier access
MAPPING_CATEGORIES = {
    "mp3": {
        "easyid3": [
            m
            for m in METADATA_MAPPINGS
            if m[0] == "mp3" and m[1].startswith("easyid3_")
        ],
        "id3": [
            m
            for m in METADATA_MAPPINGS
            if m[0] == "mp3"
            and m[1].startswith("id3_")
            and not m[1].startswith("id3_TXXX")
        ],
        "txxx": [
            m
            for m in METADATA_MAPPINGS
            if m[0] == "mp3" and m[1].startswith("id3_TXXX")
        ],
    },
    "m4a": {
        "standard": [
            m
            for m in METADATA_MAPPINGS
            if m[0] == "m4a" and not m[1].startswith("m4a_----")
        ],
        "itunes": [
            m
            for m in METADATA_MAPPINGS
            if m[0] == "m4a" and m[1].startswith("m4a_----")
        ],
    },
}


# Helper functions
def get_mappings_by_format(format_name: str) -> List[Tuple[str, str, str, str, str]]:
    """
    Get all mappings for a specific format.

    Args:
        format_name: The format name (e.g., 'mp3', 'm4a')

    Returns:
        List of mappings for the specified format
    """
    return [m for m in METADATA_MAPPINGS if m[0] == format_name]


def get_mappings_by_normalized_tag(
    normalized_tag: str,
) -> List[Tuple[str, str, str, str, str]]:
    """
    Get all mappings for a specific normalized tag.

    Args:
        normalized_tag: The normalized tag name

    Returns:
        List of mappings for the specified normalized tag
    """
    return [m for m in METADATA_MAPPINGS if m[2] == normalized_tag]


def get_supported_formats() -> List[str]:
    """
    Get list of supported audio formats.

    Returns:
        List of supported format names
    """
    return list(set(m[0] for m in METADATA_MAPPINGS))


def get_mapping_count() -> dict:
    """
    Get count of mappings by format.

    Returns:
        Dictionary with format names as keys and mapping counts as values
    """
    counts: dict[str, int] = {}
    for mapping in METADATA_MAPPINGS:
        format_name = mapping[0]
        counts[format_name] = counts.get(format_name, 0) + 1
    return counts
