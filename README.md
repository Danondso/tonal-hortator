# Tonal Hortator

AI-powered local music playlist generator using Ollama embeddings.

## Features

- **Local AI**: Uses Ollama with `nomic-embed-text` for offline operation
- **Semantic Search**: Generate playlists from natural language queries
- **Apple Music Integration**: One-click playlist opening
- **Smart Deduplication**: Multi-strategy duplicate removal
- **Feedback System**: Learn from user ratings and preferences

## Quick Start

### Prerequisites

- Python 3.11+
- Ollama with `nomic-embed-text:latest`
- iTunes XML library export
- macOS (for Apple Music integration)

### Installation

```bash
# Clone and setup
git clone https://github.com/yourusername/tonal-hortator.git
cd tonal-hortator
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install Ollama and model
curl -fsSL https://ollama.ai/install.sh | sh
ollama pull nomic-embed-text

# Install package (enables `th` command)
pip install -e .
```

### Usage

```bash
# Parse iTunes library
python parse_library.py

# Generate embeddings
th embed

# Generate playlists
th generate "upbeat rock songs" --tracks 20 --open
th interactive
```

## CLI Commands

```bash
# Generate playlists
th generate "moody electronic" --open
th generate "jazz for studying" --tracks 15 --similarity 0.4

# Interactive mode
th interactive

# Embed tracks
th embed --batch 100

# Feedback system
th feedback --query "rock songs" --rating 5
th set-preference max_playlist_length 25
th feedback-stats
```

## Python API

```python
from tonal_hortator import LocalPlaylistGenerator

generator = LocalPlaylistGenerator()
tracks = generator.generate_playlist("workout songs", max_tracks=20)
filepath = generator.save_playlist_m3u(tracks, "workout")
```

## Feedback System

Record ratings and preferences to improve future playlists:

```python
from tonal_hortator.core.feedback.feedback_manager import FeedbackManager

fm = FeedbackManager(db_path="music_library.db")

# Record feedback
fm.record_playlist_feedback(
    query="rock songs",
    user_rating=5,
    user_comments="Great playlist!"
)

# Set preferences
fm.set_preference("max_playlist_length", 25, "int", "Maximum tracks")

# Get recommendations
settings = fm.get_recommended_settings("similarity")
```

## Development

```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Run tests
python -m tonal_hortator.tests.run_tests

# Quality checks
black .
isort .
flake8 .
mypy tonal_hortator/
```

## Project Structure

```
tonal_hortator/
├── core/                    # Core functionality
│   ├── embeddings.py       # Ollama embedding service
│   ├── playlist_generator.py # Playlist generation
│   └── feedback/           # Feedback system
├── cli/                    # Command-line interface
├── utils/                  # Utilities (Apple Music, etc.)
└── tests/                  # Unit tests
```

## License

MIT License 