#!/usr/bin/env python3
"""
Version bumping script for tonal-hortator

Usage:
    python bump_version.py [major|minor|patch]

Examples:
    python bump_version.py patch    # 2.0.2 -> 2.0.3
    python bump_version.py minor    # 2.0.2 -> 2.1.0
    python bump_version.py major    # 2.0.2 -> 3.0.0
"""

import re
import sys
from pathlib import Path
from typing import Tuple


def parse_version(version_str: str) -> Tuple[int, int, int]:
    """Parse version string into major, minor, patch components"""
    match = re.match(r"^(\d+)\.(\d+)\.(\d+)$", version_str)
    if not match:
        raise ValueError(f"Invalid version format: {version_str}")
    return tuple(int(x) for x in match.groups())


def format_version(major: int, minor: int, patch: int) -> str:
    """Format version components into version string"""
    return f"{major}.{minor}.{patch}"


def bump_version(version_str: str, bump_type: str) -> str:
    """Bump version according to semantic versioning"""
    major, minor, patch = parse_version(version_str)

    if bump_type == "major":
        major += 1
        minor = 0
        patch = 0
    elif bump_type == "minor":
        minor += 1
        patch = 0
    elif bump_type == "patch":
        patch += 1
    else:
        raise ValueError(
            f"Invalid bump type: {bump_type}. Use 'major', 'minor', or 'patch'"
        )

    return format_version(major, minor, patch)


def update_pyproject_toml(project_root: Path, new_version: str) -> None:
    """Update version in pyproject.toml"""
    pyproject_path = project_root / "pyproject.toml"

    if not pyproject_path.exists():
        raise FileNotFoundError(f"pyproject.toml not found at {pyproject_path}")

    # Read current content
    with open(pyproject_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Update version line
    pattern = r'^version = "([^"]+)"'
    replacement = f'version = "{new_version}"'

    if not re.search(pattern, content, re.MULTILINE):
        raise ValueError("Could not find version line in pyproject.toml")

    updated_content = re.sub(pattern, replacement, content, flags=re.MULTILINE)

    # Write back
    with open(pyproject_path, "w", encoding="utf-8") as f:
        f.write(updated_content)

    print(f"✅ Updated version to {new_version} in pyproject.toml")


def get_current_version(project_root: Path) -> str:
    """Get current version from pyproject.toml"""
    pyproject_path = project_root / "pyproject.toml"

    with open(pyproject_path, "r", encoding="utf-8") as f:
        content = f.read()

    match = re.search(r'^version = "([^"]+)"', content, re.MULTILINE)
    if not match:
        raise ValueError("Could not find version in pyproject.toml")

    return match.group(1)


def main():
    """Main function"""
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)

    bump_type = sys.argv[1].lower()
    if bump_type not in ["major", "minor", "patch"]:
        print("Error: Invalid bump type. Use 'major', 'minor', or 'patch'")
        print(__doc__)
        sys.exit(1)

    try:
        # Get project root (assuming script is in project root)
        project_root = Path(__file__).parent

        # Get current version
        current_version = get_current_version(project_root)
        print(f"📦 Current version: {current_version}")

        # Calculate new version
        new_version = bump_version(current_version, bump_type)
        print(f"🚀 Bumping {bump_type} version: {current_version} -> {new_version}")

        # Update pyproject.toml
        update_pyproject_toml(project_root, new_version)

        print(f"🎉 Successfully bumped version to {new_version}")

    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
