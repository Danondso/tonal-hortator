#!/usr/bin/env python3
"""
Setup script for Tonal Hortator - Local Music Playlist Generator
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="tonal-hortator",
    version="2.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="AI-powered local music playlist generator using Ollama embeddings",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/tonal-hortator-v2",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: End Users/Desktop",
        "Topic :: Multimedia :: Sound/Audio",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.11",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "tonal-hortator=tonal_hortator.cli:main",
            "generate-playlist=tonal_hortator.cli:generate_playlist",
            "embed-tracks=tonal_hortator.cli:embed_tracks",
        ],
    },
    include_package_data=True,
    package_data={
        "tonal_hortator": ["*.txt", "*.md"],
    },
) 