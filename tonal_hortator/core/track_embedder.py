#!/usr/bin/env python3
"""
Embed music tracks using local Ollama service
This script embeds track metadata and stores embeddings in SQLite database
"""

import sqlite3
import logging
import time
import os
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from tonal_hortator.core.embeddings import OllamaEmbeddingService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LocalTrackEmbedder:
    """Embed tracks using local Ollama service"""
    
    def __init__(self, db_path: str = "music_library.db", 
                 embedding_service: Optional[OllamaEmbeddingService] = None,
                 conn: Optional[sqlite3.Connection] = None):
        """
        Initialize the local track embedder
        
        Args:
            db_path: Path to SQLite database
            embedding_service: OllamaEmbeddingService instance
            conn: Optional existing database connection
        """
        self.db_path = db_path
        if conn:
            self.conn = conn
        else:
            self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        
        if embedding_service:
            self.embedding_service = embedding_service
        else:
            self.embedding_service = OllamaEmbeddingService()
        
        # Ensure database has embeddings table
        self._ensure_embeddings_table()
        
    def _ensure_embeddings_table(self):
        """Ensure the embeddings table exists in the database"""
        try:
            cursor = self.conn.cursor()
            
            # Check if embeddings table exists
            cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='track_embeddings'
            """)
            
            if not cursor.fetchone():
                logger.info("Creating track_embeddings table...")
                cursor.execute("""
                    CREATE TABLE track_embeddings (
                        track_id INTEGER PRIMARY KEY,
                        embedding BLOB,
                        embedding_text TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (track_id) REFERENCES tracks (id)
                    )
                """)
                self.conn.commit()
                logger.info("✅ Created track_embeddings table")
            else:
                logger.info("✅ track_embeddings table already exists")
                
        except Exception as e:
            logger.error(f"❌ Error ensuring embeddings table: {e}")
            raise
    
    def get_tracks_without_embeddings(self) -> List[Dict[str, Any]]:
        """Get all tracks from the database that don't have embeddings"""
        try:
            cursor = self.conn.cursor()
            
            # Get tracks that are not in the track_embeddings table
            cursor.execute("""
                SELECT
                    t.id, t.title as name, t.artist, t.album, t.genre, t.year,
                    t.play_count, t.album_artist, t.composer, t.bpm
                FROM tracks t
                LEFT JOIN track_embeddings te ON t.id = te.track_id
                WHERE te.track_id IS NULL
            """)
            
            tracks = [dict(row) for row in cursor.fetchall()]
            logger.info(f"🔍 Found {len(tracks)} tracks without embeddings")
            return tracks
            
        except sqlite3.Error as e:
            logger.error(f"❌ Error getting tracks without embeddings: {e}")
            raise
    
    def create_track_embedding_text(self, track: Dict[str, Any]) -> str:
        """Create text representation of track for embedding"""
        return self.embedding_service.create_track_embedding_text(track)
    
    def embed_tracks_batch(self, tracks: List[Dict[str, Any]], batch_size: int = 100) -> int:
        """
        Embed a batch of tracks
        
        Args:
            tracks: List of track dictionaries
            batch_size: Number of tracks to process in each batch (increased for speed)
            
        Returns:
            Number of tracks successfully embedded
        """
        if not tracks:
            logger.info("No tracks to embed")
            return 0
        
        total_tracks = len(tracks)
        embedded_count = 0
        
        logger.info(f"🔄 Starting to embed {total_tracks} tracks in batches of {batch_size}")
        
        for i in range(0, total_tracks, batch_size):
            batch_tracks = tracks[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (total_tracks + batch_size - 1) // batch_size
            
            logger.info(f"📦 Processing batch {batch_num}/{total_batches} ({len(batch_tracks)} tracks)")
            
            batch_start = time.time()
            
            # Create embedding texts for this batch
            embedding_texts = []
            for track in batch_tracks:
                text = self.create_track_embedding_text(track)
                embedding_texts.append(text)
            
            # Get embeddings from Ollama (using optimized batch API)
            try:
                embeddings = self.embedding_service.get_embeddings_batch(embedding_texts, batch_size=50)
                
                # Store embeddings in database
                batch_embedded = self._store_embeddings_batch(batch_tracks, embeddings, embedding_texts)
                embedded_count += batch_embedded
                
                batch_time = time.time() - batch_start
                logger.info(f"⏱️  Batch {batch_num} completed in {batch_time:.2f}s ({batch_embedded}/{len(batch_tracks)} embedded)")
                
            except Exception as e:
                logger.error(f"❌ Error processing batch {batch_num}: {e}")
                continue
        
        logger.info(f"✅ Completed embedding process. {embedded_count}/{total_tracks} tracks embedded")
        return embedded_count
    
    def _store_embeddings_batch(self, tracks: List[Dict[str, Any]], 
                               embeddings: List[np.ndarray], 
                               embedding_texts: List[str]) -> int:
        """Store a batch of embeddings in the database using bulk insert"""
        try:
            cursor = self.conn.cursor()
            
            # Prepare bulk insert data
            insert_data = []
            for track, embedding, text in zip(tracks, embeddings, embedding_texts):
                try:
                    # Convert embedding to bytes for storage
                    embedding_bytes = embedding.tobytes()
                    insert_data.append((track['id'], embedding_bytes, text))
                except Exception as e:
                    logger.error(f"❌ Error preparing embedding for track {track['id']}: {e}")
                    continue
            
            if insert_data:
                # Use bulk insert for better performance
                cursor.executemany("""
                    INSERT OR REPLACE INTO track_embeddings 
                    (track_id, embedding, embedding_text) 
                    VALUES (?, ?, ?)
                """, insert_data)
                
                self.conn.commit()
                return len(insert_data)
            else:
                return 0
            
        except Exception as e:
            logger.error(f"❌ Error storing embeddings batch: {e}")
            return 0
    
    def get_all_embeddings(self) -> Tuple[List[np.ndarray], List[Dict[str, Any]]]:
        """Get all embeddings and corresponding track data from the database"""
        try:
            cursor = self.conn.cursor()
            
            # Join tracks and track_embeddings tables
            cursor.execute("""
                SELECT
                    te.embedding,
                    t.id, t.title as name, t.artist, t.album, t.genre, t.year,
                    t.play_count, t.album_artist, t.composer, t.bpm, t.location
                FROM track_embeddings te
                JOIN tracks t ON te.track_id = t.id
            """)
            
            rows = cursor.fetchall()
            embeddings = [np.frombuffer(row['embedding'], dtype=np.float32) for row in rows]
            track_data = [dict(row) for row in rows]
            
            logger.info(f"📊 Retrieved {len(embeddings)} embeddings from database")
            return embeddings, track_data
            
        except sqlite3.Error as e:
            logger.error(f"❌ Error getting embeddings: {e}")
            raise
    
    def embed_all_tracks(self) -> int:
        """Embed all tracks that don't have embeddings yet"""
        tracks = self.get_tracks_without_embeddings()
        return self.embed_tracks_batch(tracks)
    
    def get_embedding_stats(self) -> Dict[str, Any]:
        """Get statistics about embeddings in the database"""
        try:
            cursor = self.conn.cursor()
            
            # Total tracks
            cursor.execute("SELECT COUNT(*) FROM tracks")
            total_tracks = cursor.fetchone()[0]
            
            # Tracks with embeddings
            cursor.execute("SELECT COUNT(*) FROM track_embeddings")
            tracks_with_embeddings = cursor.fetchone()[0]
            
            # Tracks without embeddings
            tracks_without_embeddings = total_tracks - tracks_with_embeddings
            
            # Embedding coverage percentage
            coverage = (tracks_with_embeddings / total_tracks * 100) if total_tracks > 0 else 0
            
            stats = {
                'total_tracks': total_tracks,
                'tracks_with_embeddings': tracks_with_embeddings,
                'tracks_without_embeddings': tracks_without_embeddings,
                'coverage_percentage': round(coverage, 2)
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"❌ Error getting embedding stats: {e}")
            raise


def main():
    """Main function to embed all tracks"""
    try:
        logger.info("🚀 Starting local track embedding process")
        
        # Initialize embedder
        embedder = LocalTrackEmbedder()
        
        # Get current stats
        stats = embedder.get_embedding_stats()
        logger.info(f"📊 Current embedding stats: {stats}")
        
        # Embed all tracks
        if stats['tracks_without_embeddings'] > 0:
            logger.info(f"🔄 Embedding {stats['tracks_without_embeddings']} tracks...")
            embedded_count = embedder.embed_all_tracks()
            logger.info(f"✅ Successfully embedded {embedded_count} tracks")
        else:
            logger.info("✅ All tracks already have embeddings")
        
        # Get final stats
        final_stats = embedder.get_embedding_stats()
        logger.info(f"📊 Final embedding stats: {final_stats}")
        
    except Exception as e:
        logger.error(f"❌ Error in main embedding process: {e}")
        raise


if __name__ == "__main__":
    main() 