#!/usr/bin/env python3
"""
Version management script for Python packages.
This script helps manage version tags in a git repository.
"""

import argparse
import subprocess
import sys
from pathlib import Path
import re

try:
    import semver
except ImportError:
    print("Error: semver package is required")
    print("Please install it with: pip install semver")
    sys.exit(1)

try:
    from setuptools_scm import get_version
except ImportError:
    print("Error: setuptools_scm package is required")
    print("Please install it with: pip install setuptools_scm")
    sys.exit(1)


def create_git_tag(package: str, tag: bool = False) -> None:
    """Create git tag for package version."""
    version = "0.0.0"
    tag_name = f"v{version}"
    message = f"Version {version}"
    
    try:
        if not tag:
            print(f"Would delete local tag: {tag_name}")
            print(f"Would delete remote tag: {tag_name}")
            print(f"Would create tag: {tag_name}")
            print(f"Would push tag to origin: {tag_name}")
        else:
            # Delete both local and remote tags if they exist
            subprocess.run(['git', 'tag', '-d', tag_name], check=False)
            subprocess.run(['git', 'push', 'origin', '--delete', tag_name], check=False)
            
            # Create new tag
            subprocess.run(['git', 'tag', '-a', tag_name, '-m', message], check=True)
            
            # Push tag with force
            subprocess.run(['git', 'push', 'origin', tag_name, '--force'], check=True)
            print(f"Created tag: {tag_name}")
    except subprocess.CalledProcessError as e:
        print(f"Error creating tag: {e}")


def get_tags(package_dir: Path) -> list[tuple[str, str]] | None:
    """Get all valid version tags for the package from main branch.
    
    Returns:
        List of tuples (tag_name, version) or None if no tags found
    """
    try:
        # Get all tags from main branch
        result = subprocess.run(
            ['git', 'tag', '-l', 'v*', 'origin/main'],
            capture_output=True, text=True, check=True
        )
        # Split into lines and filter valid tags
        tags = []
        for tag in result.stdout.strip().split('\n'):
            if tag.startswith('v'):
                try:
                    version = tag.lstrip('v')
                    # Validate version format
                    semver.VersionInfo.parse(version)
                    tags.append((tag, version))
                except (ValueError, IndexError):
                    continue
        # Sort by version number in descending order
        tags.sort(key=lambda x: semver.VersionInfo.parse(x[1]), reverse=True)
        return tags if tags else None
    except subprocess.CalledProcessError:
        return None

def get_current_version(package_dir: Path) -> str:
    """Get current version from git tags on the default branch.
    Falls back to _version.py if no tags exist.
    """
    # First try git tags
    tags = get_tags(package_dir)
    print(f"Tags found: {tags}")
    if tags is not None and tags:
        return tags[0][1]

    # Try _version.py
    version_file = package_dir / "src" / package_dir.name / "_version.py"
    print(f"Looking for version in: {version_file}")
    if version_file.exists():
        try:
            with open(version_file) as f:
                content = f.read()
                # Look for version = "x.y.z" pattern
                version_match = re.search(r'version\s*=\s*["\']([^"\']+)["\']', content)
                if version_match:
                    version = version_match.group(1)
                    print(f"Found version in _version.py: {version}")
                    return version
        except Exception as e:
            print(f"Error reading _version.py: {e}")

    print("No version found in any source, using fallback")
    return "0.1.0"

def get_next_version(package_dir: Path, bump_type: str) -> str:
    """Get the next version based on the current version and bump type.
    
    Args:
        package_dir: Path to the package directory
        bump_type: One of "major", "minor", or "patch"
        
    Returns:
        The next version string
    """
    try:
        # Get current version using get_current_version
        current_version = get_current_version(package_dir)
        
        # Parse the version
        version = semver.VersionInfo.parse(current_version)
        
        # Bump the version based on type
        if bump_type == "major":
            next_version = version.bump_major()
        elif bump_type == "minor":
            next_version = version.bump_minor()
        elif bump_type == "patch":
            next_version = version.bump_patch()
        else:
            raise ValueError(f"Invalid bump type: {bump_type}. Must be one of: major, minor, patch")
            
        return str(next_version)
    except Exception as e:
        print(f"Error getting next version: {e}")
        return None

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Manual version tag management. For automatic versioning, use PR labels (major/minor) in GitHub."
    )
    parser.add_argument("package", nargs="?", help="Package name to update (optional if running from package directory)")
    parser.add_argument("-n", "--next-version", choices=["major", "minor", "patch"],
                       help="Print next version without creating tag")
    parser.add_argument("-g", "--get-version", action="store_true", help="Print current version and exit")
    parser.add_argument("-t", "--tag", action="store_true", help="Actually create the tag (default is dry-run)")
    args = parser.parse_args()
    
    # If script is in package directory (eventstudies/eventstudies/scripts/version.py),
    # use the parent directory as package_dir
    script_dir = Path(__file__).parent
    if script_dir.name == "scripts" and script_dir.parent.name == "eventstudies":
        # We're inside the package directory
        package_dir = script_dir.parent
        if args.package:
            print(f"Warning: Ignoring package argument '{args.package}' - using current package directory")
    else:
        # Legacy behavior: look for package in parent directory
        root_dir = Path(__file__).parent.parent
        if args.package:
            package_dir = root_dir / args.package
        else:
            print("Error: package name is required when not running from package directory")
            return
    
    if not package_dir.exists():
        print(f"Package directory not found: {package_dir}")
        return
        
    if args.get_version:
        version = get_current_version(package_dir)
        print(version)
        return
        
    if args.next_version:
        version = get_next_version(package_dir, args.next_version)
        if version:
            print(version)
        return
        
    # Require -t for tag creation - only create development v0.0.0 tags
    if not args.tag:
        print("Error: -t/--tag flag is required to create development v0.0.0 tags")
        print("This is a safety measure to prevent accidental tag creation")
        print("Use -t to actually create the tag")
        return
        
    # Create git tag
    create_git_tag(args.package or package_dir.name, args.tag)

if __name__ == "__main__":
    main()

