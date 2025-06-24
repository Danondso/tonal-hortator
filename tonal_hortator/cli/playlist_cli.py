#!/usr/bin/env python3
"""
Command-line interface for playlist generation
Handles interactive CLI logic and user interaction
"""

import logging
import subprocess

from tonal_hortator.core.playlist_generator import LocalPlaylistGenerator
from tonal_hortator.core.playlist_output import PlaylistOutput

logger = logging.getLogger(__name__)


class PlaylistCLI:
    """Handle command-line interface for playlist generation"""

    def __init__(self, generator: LocalPlaylistGenerator):
        self.generator = generator
        self.output = PlaylistOutput()

    def check_embeddings_available(self) -> bool:
        """Check if embeddings are available in the database"""
        stats = self.generator.track_embedder.get_embedding_stats()
        logger.info(f"📊 Database stats: {stats}")

        if stats["tracks_with_embeddings"] == 0:
            logger.warning(
                "No embeddings found. Please run embed_tracks_local.py first."
            )
            return False
        return True

    def process_playlist_request(self, query: str) -> bool:
        """Process a single playlist request"""
        try:
            # Generate playlist
            tracks = self.generator.generate_playlist(query, max_tracks=20)

            if not tracks:
                print("❌ No tracks found for your query. Try a different search term.")
                return False

            # Print summary
            self.output.print_playlist_summary(tracks, query)

            # Ask if user wants to save
            save = input("\nSave playlist to file? (y/n): ").strip().lower()
            if save in ["y", "yes"]:
                filepath = self.output.save_playlist_m3u(tracks, query)
                print(f"✅ Playlist saved to: {filepath}")

                # Ask if user wants to open in Apple Music
                open_music = input("Open in Apple Music? (y/n): ").strip().lower()
                if open_music in ["y", "yes"]:
                    try:
                        subprocess.run(["open", "-a", "Music", filepath], check=True)
                        print("🎵 Opened in Apple Music!")
                    except Exception as e:
                        print(f"❌ Could not open in Apple Music: {e}")
                        print(
                            "💡 You can open it manually with: python open_in_apple_music.py"
                        )

            return True

        except Exception as e:
            logger.error(f"❌ Error generating playlist: {e}")
            print(f"❌ Error: {e}")
            return False

    def run_interactive_loop(self) -> None:
        """Run the interactive playlist generation loop"""
        print("\n🎵 Local Playlist Generator")
        print("=" * 40)

        while True:
            query = input("\nEnter your playlist query (or 'quit' to exit): ").strip()

            if query.lower() in ["quit", "exit", "q"]:
                break

            if not query:
                print("Please enter a query.")
                continue

            self.process_playlist_request(query)

        print("\n👋 Thanks for using the Local Playlist Generator!")


def main() -> None:
    """Main function for interactive playlist generation"""
    try:
        logger.info("🚀 Starting local playlist generator")

        # Initialize generator
        generator = LocalPlaylistGenerator()
        cli = PlaylistCLI(generator)

        # Check if embeddings are available
        if not cli.check_embeddings_available():
            return

        # Run interactive loop
        cli.run_interactive_loop()

    except Exception as e:
        logger.error(f"❌ Error in main function: {e}")
        raise


if __name__ == "__main__":
    main()
