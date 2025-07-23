# Tonal Hortator

AI-powered local music playlist generator using Ollama embeddings with intelligent learning from user feedback.

## Features

- **Local AI**: Uses Ollama with `nomic-embed-text` for offline operation
- **Semantic Search**: Generate playlists from natural language queries
- **Smart Learning**: Self-improving query parsing from user feedback
- **Apple Music Integration**: Listen-first workflow with one-click playlist opening
- **Smart Deduplication**: Multi-strategy duplicate removal
- **Feedback System**: Learn from user ratings and preferences
- **Adaptive Performance**: Intelligent batch sizing based on system resources
- **Configuration System**: Centralized YAML configuration with environment overrides

## 🧠 Intelligent Learning System

Tonal Hortator continuously improves its understanding of your music preferences through a sophisticated feedback learning system:

### **Self-Improving Query Parsing**
- **Training Data Generation**: Automatically aggregates positive feedback (4+ stars) into training examples
- **Few-Shot Learning**: Uses your best-rated playlists to teach the LLM better query parsing
- **Real-Time Updates**: Run `th tune-prompt` to instantly improve parsing accuracy
- **Quality Tracking**: Monitors and displays training example quality metrics

### **Listen-First Workflow**
1. **Generate playlist** from your natural language query
2. **Auto-open in Apple Music** to listen before rating
3. **Provide informed feedback** after actually hearing the results
4. **System learns** from your ratings to improve future playlists

### **Example Learning Process**
```bash
# Generate a playlist
th generate "dusty desert rock" --open

# After listening and rating 4+ stars, the system learns:
# Input: "dusty desert rock"
# Output: {"query_type": "general", "genres": ["rock", "desert rock"], "mood": "melancholy"}

# Future similar queries become more accurate!
th tune-prompt  # Apply learning to improve parsing
```

## Quick Start

### Prerequisites

- Python 3.11+
- Ollama with `nomic-embed-text:latest` and `llama3:8b`
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

# Install Ollama and models
curl -fsSL https://ollama.ai/install.sh | sh
ollama pull nomic-embed-text
ollama pull llama3:8b

# Install package (enables `th` command)
pip install -e .
```

### Basic Workflow

```bash
# 1. Parse iTunes library
python parse_library.py

# 2. Generate embeddings
th embed

# 3. Generate playlists with learning workflow
th generate "moody indie folk" --tracks 15 --open
# - Playlist opens in Apple Music automatically
# - Listen and enjoy!
# - Return to terminal to rate playlist
# - System learns from your feedback

# 4. Improve the system with your feedback
th tune-prompt
# - Aggregates your positive ratings
# - Generates training examples
# - Improves future query parsing

# 5. Interactive mode for multiple playlists
th interactive
```

## Learning & Feedback System

### **Prompt Tuning Workflow**

The system learns from your feedback to continuously improve:

```bash
# Check current training data quality
th tune-prompt
# Output shows:
# ✅ Aggregated feedback to playlist_training_data.jsonl
# ✅ Wrote prompt to llm_prompt.txt  
# 🎉 LLM prompt tuning complete!

# View training examples generated from your feedback
cat llm_prompt.txt
```

### **Training Example Generation**

When you rate playlists 4+ stars, the system creates training examples:

```
User: smelling her hair after a long time apart
LLM: {"query_type": "general", "genres": ["acoustic", "folk"], "mood": "sentimental"}

User: aggressively grinding out weights because you're angry  
LLM: {"query_type": "general", "genres": ["rock", "hard rock"], "mood": "angry"}
```

### **Feedback Collection**

Enhanced feedback collection with informed ratings:

```bash
# Generate playlist with auto-open
th generate "chill electronic beats" --open

# Workflow:
# 1. Playlist generated and saved
# 2. Opens automatically in Apple Music  
# 3. Listen to evaluate quality
# 4. Return to terminal for rating
# 5. Provide overall rating (1-5 stars)
# 6. Optionally rate individual tracks
# 7. Comments help fine-tune recommendations
```

### **Quality Metrics**

Track the quality of your training data:

```python
# Check training example quality
from tonal_hortator.core.feedback.training_data_aggregator import TrainingDataAggregator
from tonal_hortator.core.database import DatabaseManager

db = DatabaseManager("music_library.db")
aggregator = TrainingDataAggregator(db)
aggregator.aggregate("training_data.jsonl")

# View positive examples with rich parsing data
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
  max_tokens: 1000
  
  # Query parsing improvements
  query_parsing:
    min_artist_name_length: 2
    training_examples_path: "llm_prompt.txt"
    
# === FEEDBACK LEARNING ===
feedback:
  # Minimum rating for positive training examples
  positive_rating_threshold: 4
  
  # Training data quality requirements
  training:
    require_genres: true
    require_mood: false
    min_examples: 3
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
# Playlist generation with learning
th generate "upbeat indie rock" --open              # Auto-open in Apple Music
th generate "jazz for studying" --tracks 15         # Custom track count
th generate "workout music" --similarity 0.4        # Adjust similarity threshold

# Learning and improvement
th tune-prompt                                       # Update training from feedback
th tune-prompt --auto-reload                        # Auto-reload LLM prompt

# Interactive workflows
th interactive                                       # Interactive playlist generation
th feedback                                         # Manage feedback and preferences

# System management
th embed                                            # Generate embeddings (auto-batch)
th embed --batch 200                               # Manual batch size
th yeet ~/Music/iTunes/iTunes\ Music\ Library.xml  # Complete rebuild
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

### **Basic Usage with Learning**

```python
from tonal_hortator import LocalPlaylistGenerator
from tonal_hortator.core.feedback import FeedbackManager

# Generate playlist
generator = LocalPlaylistGenerator()
tracks = generator.generate_playlist("melancholic indie", max_tracks=20)

# Save and get feedback
filepath = generator.save_playlist_m3u(tracks, "melancholic indie")

# Record feedback for learning
feedback_manager = FeedbackManager()
feedback_manager.record_playlist_feedback(
    query="melancholic indie",
    query_type="general", 
    parsed_data={"genres": ["indie"], "mood": "melancholic"},
    generated_tracks=tracks,
    user_rating=5,
    user_comments="Perfect vibe for rainy day coding"
)
```

### **Training System Integration**

```python
from tonal_hortator.core.feedback.training_data_aggregator import TrainingDataAggregator
from tonal_hortator.core.database import DatabaseManager

# Aggregate feedback into training data
db = DatabaseManager("music_library.db")
aggregator = TrainingDataAggregator(db)
aggregator.aggregate("custom_training.jsonl")

# Generate improved prompt
from tonal_hortator.core.llm.update_llm_prompt import generate_prompt_from_jsonl
generate_prompt_from_jsonl("custom_training.jsonl", "improved_prompt.txt")
```

## Technical Implementation

### **Learning System Architecture**

The learning system consists of several interconnected components:

```
User Feedback → Training Data → Prompt Tuning → Improved Parsing
     ↑                                              ↓
User Rating ← Playlist Generation ← Better Results ←┘
```

**Components:**
- **TrainingDataAggregator**: Converts feedback into training examples
- **LLMQueryParser**: Uses training examples for few-shot learning  
- **FeedbackManager**: Collects and validates user feedback
- **ConfigurationManager**: Manages learning parameters

### **Training Data Quality**

The system tracks training example quality:

```python
# Quality metrics for training examples
quality_factors = {
    'has_genres': 2 points,      # Examples with genres are valuable
    'has_mood': 1 point,         # Mood detection improves specificity  
    'high_rating': 1 point,      # 5-star ratings indicate excellent results
    'detailed_comments': 1 point # User comments provide context
}

# High-quality example (5 points):
{
    "input": "dusty desert rock",
    "system_parsed": {
        "query_type": "general",
        "genres": ["rock", "desert rock"],  # +2 points
        "mood": "melancholy"                # +1 point
    },
    "user_feedback": {
        "rating": 5,                        # +1 point
        "comments": "Perfect stoner rock vibe!"  # +1 point
    },
    "label": 1
}
```

### **Adaptive Performance**

Tonal Hortator automatically optimizes performance:

**Batch Size Detection:**
- Auto-detects optimal batch sizes (50-1000 tracks)
- Adapts to available system memory and CPU cores
- Graceful fallback for resource detection failures

**System Requirements:**
- **Minimum**: 4GB RAM, 2 cores, batch size 50-100
- **Recommended**: 8GB RAM, 4-6 cores, batch size 500 (auto)
- **High Performance**: 16GB+ RAM, 8+ cores, batch size 800-1000

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

### **Learning System Issues**

**Empty training prompts:**
```bash
# Check feedback data
python -c "
import sqlite3
conn = sqlite3.connect('music_library.db')
cursor = conn.cursor()
cursor.execute('SELECT COUNT(*) FROM user_feedback WHERE user_rating >= 4')
print(f'Positive feedback examples: {cursor.fetchone()[0]}')
conn.close()
"

# Generate more feedback by rating playlists
th generate "your favorite genre" --open
# Rate 4+ stars to create training data
```

**Poor parsing quality:**
```bash
# Check training example quality
th tune-prompt
cat llm_prompt.txt  # Review generated examples

# Add more diverse, high-quality ratings
th generate "jazz for focus" --open
th generate "upbeat workout songs" --open
th generate "melancholic indie folk" --open
# Rate each 4+ stars with detailed comments
```

**Apple Music not opening:**
```bash
# Check macOS requirements
which open  # Should return /usr/bin/open

# Test manual opening
open -a Music "playlists/Your_Playlist.m3u"

# Check file permissions
ls -la playlists/
```

### **Performance Issues**

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

### **Configuration Issues**

**Training examples not loading:**
```bash
# Verify prompt file exists
ls -la llm_prompt.txt

# Check file content
head -10 llm_prompt.txt

# Manually regenerate if needed
th tune-prompt
```

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

### **Advanced Learning Commands**

```bash
# Training data management
th tune-prompt --out custom_training.jsonl         # Custom training data file
th tune-prompt --prompt custom_prompt.txt          # Custom prompt file

# Feedback analysis
th feedback                                         # Interactive feedback management
python -m tonal_hortator.core.feedback.feedback_report  # Detailed feedback analysis
```

## Examples

### **Learning Workflow Example**

```bash
# Start with a vague query
$ th generate "music for coding" --open

# System generates playlist, opens in Apple Music
# After listening, you rate it 2 stars with comment:
# "Too upbeat, I prefer ambient/focus music"

# Try again with more specific query
$ th generate "ambient focus music for programming" --open  

# This time it's perfect! Rate 5 stars:
# "Exactly what I wanted - calm, atmospheric, helps concentration"

# Update the system with your feedback
$ th tune-prompt
✅ Aggregated feedback to playlist_training_data.jsonl
📚 Loaded 1 new training example for query parsing
🎉 LLM prompt tuning complete!

# Now similar queries work much better:
$ th generate "music for deep work"
🧠 Parsed intent: {'query_type': 'general', 'genres': ['ambient'], 'mood': 'focused'}
```

### **Advanced Usage**

```python
# Custom learning pipeline
from tonal_hortator.core.config import get_config
from tonal_hortator.core.playlist.playlist_generator import LocalPlaylistGenerator
from tonal_hortator.core.feedback import FeedbackManager

# Configure for learning
config = get_config()
config.set("feedback.positive_rating_threshold", 4)

# Generate with A/B testing
generator = LocalPlaylistGenerator()
config.set_variant("discovery")  # Lower similarity, more exploration

# Generate experimental playlist
tracks = generator.generate_playlist("experimental electronic", max_tracks=10)

# Collect detailed feedback
fm = FeedbackManager()
fm.record_playlist_feedback(
    query="experimental electronic",
    query_type="general",
    parsed_data={"genres": ["electronic", "experimental"], "mood": "adventurous"},
    generated_tracks=tracks,
    user_rating=4,
    user_comments="Great discoveries, but some tracks too harsh",
    similarity_threshold=0.2,
    search_breadth=20
)

# Apply learning
fm.record_query_learning(
    original_query="experimental electronic",
    llm_parsed_result={"genres": ["electronic", "experimental"], "mood": "adventurous"},
    user_correction={"genres": ["electronic", "ambient"], "mood": "exploratory"},
    feedback_score=0.8
)
```

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

# Run tests including learning system
python -m tonal_hortator.tests.run_tests
python -m tonal_hortator.tests.test_feedback_manager
python -m tonal_hortator.tests.test_llm_query_parser

# Test training data generation
python -c "
from tonal_hortator.core.feedback.training_data_aggregator import TrainingDataAggregator
from tonal_hortator.core.database import DatabaseManager
db = DatabaseManager('music_library.db')
agg = TrainingDataAggregator(db)
agg.aggregate('test_training.jsonl')
print('Training data generated successfully')
"

# Quality checks
black .
isort .
flake8 .
mypy tonal_hortator/
```

## Project Structure

```
tonal_hortator/
├── config.yml                    # Main configuration file
├── llm_prompt.txt                # Auto-generated training examples
├── playlist_training_data.jsonl  # Feedback aggregation output
├── core/
│   ├── config.py                # Configuration management
│   ├── embeddings/              # Ollama embedding service
│   ├── playlist/                # Playlist generation & filtering
│   ├── feedback/                # Learning & feedback system
│   │   ├── feedback_manager.py  # Feedback collection
│   │   ├── training_data_aggregator.py  # Learning data generation
│   │   └── feedback_validator.py       # Input validation
│   └── llm/                     # LLM integration
│       ├── llm_client.py        # Ollama client
│       └── llm_query_parser.py  # Intelligent query parsing
├── cli/                         # Command-line interface
├── utils/                       # Utilities (Apple Music, etc.)
└── tests/                       # Unit tests
```

## License

MIT License 