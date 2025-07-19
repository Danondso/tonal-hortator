# Tonal Hortator

AI-powered local music playlist generator using Ollama embeddings.

## Features

- **Local AI**: Uses Ollama with `nomic-embed-text` for offline operation
- **Semantic Search**: Generate playlists from natural language queries
- **Apple Music Integration**: One-click playlist opening
- **Smart Deduplication**: Multi-strategy duplicate removal
- **Feedback System**: Learn from user ratings and preferences
- **Adaptive Performance**: Intelligent batch sizing based on system resources

## Adaptive Threading & Batch Sizing

Tonal Hortator automatically optimizes performance based on your system's capabilities:

### 🧠 **Smart Batch Size Detection**
- **Auto-detection**: Automatically determines optimal batch sizes (50-1000 tracks) based on:
  - Available system memory
  - CPU core count
  - Current system load
- **Memory-aware**: Adjusts batch size to prevent memory exhaustion
- **CPU-optimized**: Scales with your processor's capabilities
- **Fallback protection**: Gracefully handles resource detection failures

### ⚡ **Performance Optimization**
- **Default behavior**: Uses 500 tracks per batch for most systems
- **Resource scaling**: Larger batches (500-1000) for high-end systems
- **Conservative mode**: Smaller batches (50-200) for limited resources
- **Manual override**: Specify custom batch sizes when needed

### 🔧 **CLI Integration**
All embedding and processing commands now use adaptive batch sizing:
```bash
# Auto-detected optimal batch size
th embed

# Manual batch size override
th embed --batch 200

# Workflow with adaptive embedding
th workflow music.csv --embed-batch 300
```

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

# Embed tracks with adaptive batch sizing
th embed                    # Auto-detected optimal batch size
th embed --batch 200        # Manual batch size override
th embed --workers 8        # Adjust worker processes

# Update specific track embeddings
th update-embeddings "1,2,3" --batch 100
th update-embeddings --file track_ids.txt

# Complete workflow with adaptive performance
th workflow music.csv --embed-batch 300 --workers 6

# Feedback system
th feedback --query "rock songs" --rating 5
th set-preference max_playlist_length 25
th feedback-stats
```

### Performance Tuning

**Batch Size Guidelines:**
- **Auto-detection (recommended)**: `th embed` - Let the system optimize automatically
- **High-end systems**: `th embed --batch 800` - For 16GB+ RAM, 8+ cores
- **Mid-range systems**: `th embed --batch 500` - For 8GB RAM, 4-6 cores  
- **Limited resources**: `th embed --batch 100` - For 4GB RAM, 2 cores
- **Debugging**: `th embed --batch 50` - For testing and troubleshooting

**Worker Process Guidelines:**
- **Default**: 4 workers (good for most systems)
- **High-end**: 8-12 workers (for powerful multi-core systems)
- **Limited**: 2 workers (for resource-constrained systems)

## Python API

```python
from tonal_hortator import LocalPlaylistGenerator

generator = LocalPlaylistGenerator()
tracks = generator.generate_playlist("workout songs", max_tracks=20)
filepath = generator.save_playlist_m3u(tracks, "workout")
```

## Technical Implementation

### Adaptive Batch Sizing Algorithm

The adaptive batch sizing system uses a multi-factor approach:

```python
from tonal_hortator.utils.loader import get_optimal_batch_size

# Auto-detect optimal batch size
optimal_size = get_optimal_batch_size(
    base_size=500,      # Default batch size
    memory_factor=0.1,  # Memory utilization factor
    cpu_factor=0.5,     # CPU scaling factor
    min_size=50,        # Minimum batch size
    max_size=1000       # Maximum batch size
)
```

**Resource Detection:**
- **Memory**: Analyzes available RAM and adjusts batch size to prevent memory exhaustion
- **CPU**: Scales batch size based on logical CPU cores for parallel processing
- **Fallback**: Gracefully handles resource detection failures with sensible defaults

**Performance Characteristics:**
- **Memory usage**: ~1KB per track embedding
- **Processing time**: Scales linearly with batch size
- **Parallel efficiency**: Optimal with 4-8 worker processes

### System Requirements

**Minimum:**
- 4GB RAM, 2 CPU cores
- Batch size: 50-100 tracks
- Workers: 2 processes

**Recommended:**
- 8GB RAM, 4-6 CPU cores  
- Batch size: 500 tracks (auto-detected)
- Workers: 4 processes

**High Performance:**
- 16GB+ RAM, 8+ CPU cores
- Batch size: 800-1000 tracks
- Workers: 8-12 processes

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

## Troubleshooting

### Performance Issues

**Slow embedding generation:**
```bash
# Check your system's optimal batch size
python -c "from tonal_hortator.utils.loader import get_optimal_batch_size; print(f'Optimal batch size: {get_optimal_batch_size()}')"

# Try smaller batch size for limited resources
th embed --batch 100 --workers 2

# Monitor system resources during processing
htop  # or Activity Monitor on macOS
```

**Memory errors during embedding:**
```bash
# Reduce batch size significantly
th embed --batch 50 --workers 1

# Check available memory
python -c "import psutil; print(f'Available RAM: {psutil.virtual_memory().available / 1024**3:.1f}GB')"
```

**High CPU usage:**
```bash
# Reduce worker processes
th embed --workers 2

# Use smaller batches with more workers
th embed --batch 200 --workers 4
```

### Batch Size Optimization

**For large libraries (10,000+ tracks):**
- Use auto-detection: `th embed`
- Monitor progress and adjust if needed
- Consider running during off-peak hours

**For small libraries (< 1,000 tracks):**
- Manual batch size: `th embed --batch 200`
- Faster completion with smaller batches

**For debugging:**
- Minimal batch size: `th embed --batch 50 --workers 1`
- Easier to track progress and identify issues

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