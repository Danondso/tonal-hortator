# Tonal Hortator v2

**AI-powered local music playlist generator using Ollama embeddings**

A Python package that generates music playlists using semantic search with local AI embeddings, featuring seamless Apple Music integration and robust deduplication.

## 🚀 Features

- **Local AI Embeddings**: Uses Ollama with `nomic-embed-text` model for offline operation
- **Semantic Playlist Generation**: Create playlists based on natural language queries
- **Apple Music Integration**: One-click playlist opening in Apple Music
- **Robust Deduplication**: Multi-strategy deduplication to ensure clean playlists
- **iTunes Library Support**: Parse iTunes XML library into SQLite database
- **Batch Processing**: Efficient embedding generation with configurable batch sizes

## 📋 Requirements

- Python 3.11+
- Ollama with `nomic-embed-text:latest` model
- iTunes XML library export
- macOS (for Apple Music integration)

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/tonal-hortator-v2.git
   cd tonal-hortator-v2
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv tonal-hortator-v2-env
   source tonal-hortator-v2-env/bin/activate  # On Windows: tonal-hortator-v2-env\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Install Ollama and download the embedding model**:
   ```bash
   # Install Ollama (if not already installed)
   curl -fsSL https://ollama.ai/install.sh | sh
   
   # Download the embedding model
   ollama pull nomic-embed-text
   ```

5. **Install the package** (optional):
   ```bash
   pip install -e .
   ```

## 🎵 Quick Start

### 1. Parse iTunes Library

Export your iTunes library as XML and parse it into the database:

```bash
python parse_library.py
```

### 2. Generate Embeddings

Embed all tracks using the local Ollama model:

```bash
# Using the CLI
tonal-hortator embed

# Or using the module directly
python -m tonal_hortator.cli.main embed
```

### 3. Generate Playlists

Create playlists using natural language queries:

```bash
# Interactive mode
tonal-hortator interactive

# Generate specific playlist
tonal-hortator generate "upbeat rock songs" --max-tracks 20 --auto-open

# Generate with custom parameters
tonal-hortator generate "jazz for studying" --max-tracks 15 --min-similarity 0.4
```

## 📖 Usage Examples

### Command Line Interface

```bash
# Generate a playlist and open in Apple Music
tonal-hortator generate "moody electronic music" --auto-open

# Generate playlist with custom parameters
tonal-hortator generate "classic rock from the 70s" --max-tracks 30 --min-similarity 0.5

# Start interactive mode
tonal-hortator interactive

# Embed tracks with custom batch size
tonal-hortator embed --batch-size 100
```

### Python API

```python
from tonal_hortator import LocalPlaylistGenerator, OllamaEmbeddingService

# Initialize playlist generator
generator = LocalPlaylistGenerator()

# Generate playlist
tracks = generator.generate_playlist("upbeat workout songs", max_tracks=20)

# Save to M3U file
filepath = generator.save_playlist_m3u(tracks, "workout playlist")

# Print summary
generator.print_playlist_summary(tracks, "workout playlist")
```

## 🏗️ Project Structure

```
tonal-hortator-v2/
├── tonal_hortator/           # Main package
│   ├── core/                # Core functionality
│   │   ├── embeddings.py    # Ollama embedding service
│   │   ├── playlist_generator.py  # Playlist generation
│   │   └── track_embedder.py      # Track embedding
│   ├── cli/                 # Command-line interface
│   │   └── main.py          # CLI entry point
│   ├── utils/               # Utility functions
│   │   └── apple_music.py   # Apple Music integration
│   └── tests/               # Unit tests
│       ├── test_embeddings.py
│       ├── test_playlist_generator.py
│       └── run_tests.py
├── playlists/               # Generated playlists
├── music_library.db         # SQLite database
├── requirements.txt         # Python dependencies
├── setup.py                 # Package setup
└── README.md               # This file
```

## 🧪 Testing

Run the comprehensive test suite:

```bash
# Run all tests
python -m tonal_hortator.tests.run_tests

# Run specific test module
python -m unittest tonal_hortator.tests.test_embeddings

# Run with coverage (if coverage is installed)
coverage run -m tonal_hortator.tests.run_tests
coverage report
```

## 🔄 CI/CD & Quality Assurance

This project uses comprehensive GitHub Actions workflows for continuous integration and quality assurance.

### Automated Workflows

- **CI Pipeline** (`.github/workflows/ci.yml`): Runs on every push and PR
  - **Testing**: Multi-Python version testing (3.11, 3.12)
  - **Linting**: flake8, black, isort code formatting checks
  - **Type Checking**: mypy static type analysis
  - **Security**: bandit and safety vulnerability scanning
  - **Build**: Package building and validation
  - **Coverage**: Code coverage reporting

- **CodeQL Security** (`.github/workflows/codeql.yml`): Advanced security analysis
  - Automated security scanning
  - Weekly scheduled scans
  - Vulnerability detection

- **Dependency Review** (`.github/workflows/dependency-review.yml`): Dependency security
  - Checks for known vulnerabilities in dependencies
  - Runs on all pull requests

- **Release Pipeline** (`.github/workflows/release.yml`): Automated PyPI publishing
  - Triggers on GitHub releases
  - Builds and publishes to PyPI

### Development Tools

Install development dependencies for local quality checks:

```bash
pip install -r requirements-dev.txt
```

### Pre-commit Hooks

Set up pre-commit hooks for automatic code quality checks:

```bash
# Install pre-commit
pip install pre-commit

# Install the git hook scripts
pre-commit install

# Run against all files
pre-commit run --all-files
```

### Local Quality Checks

Run quality checks locally before committing:

```bash
# Code formatting
black .
isort .

# Linting
flake8 .

# Type checking
mypy tonal_hortator/

# Security checks
bandit -r tonal_hortator/
safety check

# Tests with coverage
pytest tests/ -v --cov=tonal_hortator --cov-report=html
```

### Automated Dependency Updates

Dependabot is configured to automatically:
- Check for dependency updates weekly
- Create pull requests for security updates
- Update both Python packages and GitHub Actions

### Quality Gates

All pull requests must pass:
- ✅ All tests passing
- ✅ Code coverage maintained
- ✅ No linting errors
- ✅ Type checking passes
- ✅ Security scans clean
- ✅ Build successful

## 🔧 Configuration

### Environment Variables

- `ITUNES_XML_PATH`: Path to iTunes XML library file (default: `~/Music/iTunes/iTunes Music Library.xml`)

### Database Schema

The SQLite database contains two main tables:

- `tracks`: Track metadata (title, artist, album, genre, etc.)
- `track_embeddings`: AI embeddings for semantic search

## 🎯 Deduplication Strategies

The playlist generator uses multiple deduplication strategies:

1. **File Location Deduplication**: Removes tracks pointing to the same physical file
2. **Title/Artist Deduplication**: Removes exact duplicate title/artist combinations
3. **Track ID Deduplication**: Removes duplicate track IDs (safety check)
4. **Smart Title Deduplication**: Removes variations like "Song (Remix)" vs "Song"

## 🚀 Performance

- **Embedding Generation**: ~50 tracks per batch (configurable)
- **Playlist Generation**: ~1-2 seconds for 20 tracks
- **Database**: Supports libraries with 100,000+ tracks
- **Memory Usage**: Efficient batch processing to minimize memory usage

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and add tests
4. Run the test suite: `python -m tonal_hortator.tests.run_tests`
5. Commit your changes: `git commit -am 'Add feature'`
6. Push to the branch: `git push origin feature-name`
7. Submit a pull request

### Development Setup

For contributors, set up the full development environment:

```bash
# Clone and setup
git clone https://github.com/yourusername/tonal-hortator-v2.git
cd tonal-hortator-v2

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Install package in development mode
pip install -e .
```

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [Ollama](https://ollama.ai/) for local AI model serving
- [Nomic AI](https://nomic.ai/) for the embedding model
- Apple Music for playlist integration

## 🐛 Troubleshooting

### Common Issues

1. **Ollama not running**: Make sure Ollama is running with `ollama serve`
2. **Model not found**: Download the model with `ollama pull nomic-embed-text`
3. **Database errors**: Check that the iTunes XML was parsed correctly
4. **Apple Music not opening**: Ensure Apple Music is installed and running

### Debug Mode

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📊 Statistics

- **Test Coverage**: Comprehensive unit tests for all core functionality
- **Code Quality**: Follows Python best practices and PEP 8
- **Documentation**: Full API documentation and usage examples
- **Performance**: Optimized for large music libraries 