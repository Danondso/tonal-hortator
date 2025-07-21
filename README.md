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

## Configuration System

Tonal Hortator uses a centralized configuration system with `config.yml` for easy customization without code changes.

### Configuration File

The main configuration file `config.yml` in the project root contains all customizable parameters:

```yaml
# === PLAYLIST GENERATION PARAMETERS ===
playlist:
  default_max_tracks: 20
  default_min_similarity: 0.2
  default_max_artist_ratio: 0.5
  default_search_breadth_factor: 15

# === SIMILARITY AND SCORING ===
similarity:
  genre_boost_score: 0.1
  perfect_match_score: 1.0

# === FEEDBACK SYSTEM ===
feedback:
  adjustments:
    like: 0.2
    dislike: -0.2
    block: -1.0
    note: 0.0
  
  # iTunes rating thresholds
  itunes_rating:
    thresholds:
      excellent: 4.5
      very_good: 4.0
      good: 3.0
    adjustments:
      excellent: 0.15
      very_good: 0.1
      good: 0.05

# === LLM CONFIGURATION ===
llm:
  models:
    embedding: "nomic-embed-text:latest"
    query_parser: "llama3:8b"
  max_tokens: 512
```

### Environment Variable Overrides

Override any configuration value using environment variables with the `TH_` prefix:

```bash
# Override playlist defaults
export TH_PLAYLIST_DEFAULT_MAX_TRACKS=30
export TH_PLAYLIST_DEFAULT_MIN_SIMILARITY=0.3

# Override similarity scoring
export TH_SIMILARITY_GENRE_BOOST_SCORE=0.15

# Override feedback adjustments
export TH_FEEDBACK_LIKE_ADJUSTMENT=0.25
export TH_FEEDBACK_DISLIKE_ADJUSTMENT=-0.25

# Run with overrides
th generate "rock songs"
```

### A/B Testing Variants

Enable different parameter sets for experimentation:

```bash
# Enable A/B testing in config.yml
ab_testing:
  enabled: true
  variants:
    conservative:
      min_similarity: 0.3
      max_artist_ratio: 0.3
      genre_boost_score: 0.05
    
    aggressive:
      min_similarity: 0.1
      max_artist_ratio: 0.7
      genre_boost_score: 0.2
```

Use variants programmatically:

```python
from tonal_hortator.core.config import get_config

# Load specific variant
config = get_config(variant='conservative')
generator = LocalPlaylistGenerator()

# Or switch variants dynamically
config.set_variant('aggressive')
```

### Configuration Categories

**Playlist Generation:**
- `default_max_tracks`: Default playlist size (20)
- `default_min_similarity`: Minimum similarity threshold (0.2)
- `default_max_artist_ratio`: Maximum tracks per artist (0.5)
- `default_search_breadth_factor`: Search breadth multiplier (15)

**Similarity Scoring:**
- `genre_boost_score`: Bonus for genre matches (0.1)
- `perfect_match_score`: Maximum similarity score (1.0)

**Feedback System:**
- `adjustments`: User feedback impact values
- `itunes_rating`: Rating-based adjustments
- `time_decay`: Feedback aging parameters

**LLM Models:**
- `embedding`: Embedding model name
- `query_parser`: Query parsing model name
- `max_tokens`: Maximum LLM response tokens

**Validation:**
- `user_rating`: Rating range (0-5)
- `similarity_threshold`: Valid similarity range (0.0-1.0)
- `search_breadth`: Minimum search breadth (1)

### Using Configuration in Code

```python
from tonal_hortator.core.config import get_config

# Get configuration instance
config = get_config()

# Access configuration values
max_tracks = config.get("playlist.default_max_tracks", 20)
min_similarity = config.get("playlist.default_min_similarity", 0.2)

# Use convenience properties
defaults = config.playlist_defaults
feedback_adjustments = config.feedback_adjustments
llm_config = config.llm_config

# Get entire sections
playlist_config = config.get_section("playlist")
```

### Custom Configuration

Create environment-specific configurations:

```bash
# Development configuration
cp config.yml config-dev.yml
# Edit config-dev.yml with dev-specific values

# Use custom config file
python -c "
from tonal_hortator.core.config import ConfigurationManager
config = ConfigurationManager('config-dev.yml')
"
```

### Configuration Tips

**Performance Tuning:**
```yaml
batch_processing:
  embedding_batch_size: 1000  # Larger batches for powerful systems
  max_workers: 8              # More workers for multi-core systems

algorithm:
  diversity:
    distribution_attempts: 2000  # More attempts for better diversity
```

**Quality Tuning:**
```yaml
playlist:
  default_min_similarity: 0.3   # Higher similarity for cohesive playlists
  default_max_artist_ratio: 0.2  # Lower ratio for more artist diversity

similarity:
  genre_boost_score: 0.2        # Stronger genre preference
```

**Experimentation:**
```yaml
ab_testing:
  enabled: true
  variants:
    discovery:
      min_similarity: 0.1       # Lower threshold for discovery
      genre_boost_score: 0.05   # Less genre bias
    
    focused:
      min_similarity: 0.4       # Higher threshold for focus
      genre_boost_score: 0.3    # Strong genre preference
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
from tonal_hortator.core.config import get_config

# Basic usage
generator = LocalPlaylistGenerator()
tracks = generator.generate_playlist("workout songs", max_tracks=20)
filepath = generator.save_playlist_m3u(tracks, "workout")

# Using configuration system
config = get_config()
defaults = config.playlist_defaults

# Generate playlist with config defaults
tracks = generator.generate_playlist(
    "chill electronic",
    max_tracks=defaults['max_tracks'],
    min_similarity=defaults['min_similarity']
)

# Use custom configuration
from tonal_hortator.core.config import ConfigurationManager
custom_config = ConfigurationManager('config-production.yml')
generator = LocalPlaylistGenerator()
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

### Configuration Issues

**Config file not found:**
```bash
# Verify config.yml exists in project root
ls -la config.yml

# Create default config if missing
python -c "
from tonal_hortator.core.config import get_config
config = get_config()  # Will create default config
"
```

**Environment variables not working:**
```bash
# Check environment variable format
echo $TH_PLAYLIST_DEFAULT_MAX_TRACKS

# Verify mapping in config.yml
grep -A 5 "environment_overrides:" config.yml

# Test override
TH_PLAYLIST_DEFAULT_MAX_TRACKS=25 th generate "test query"
```

**A/B testing variants not applying:**
```python
# Check variant configuration
from tonal_hortator.core.config import get_config
config = get_config()
variants = config.get_section('ab_testing').get('variants', {})
print("Available variants:", list(variants.keys()))

# Manually set variant
config.set_variant('conservative')
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
├── config.yml              # Main configuration file
├── core/                    # Core functionality
│   ├── config.py           # Configuration management
│   ├── embeddings.py       # Ollama embedding service
│   ├── playlist_generator.py # Playlist generation
│   └── feedback/           # Feedback system
├── cli/                    # Command-line interface
├── utils/                  # Utilities (Apple Music, etc.)
└── tests/                  # Unit tests
```

## License

MIT License 