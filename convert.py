#!/usr/bin/env python3
"""
XAC to GLB Batch Converter

Converts all .xac files in the 'input' folder to .glb files in the 'output' folder,
preserving the directory structure. Automatically finds .xsm animation files and
textures from parallel IPF folder structures.

Usage:
    python convert.py [--texture-paths PATH1 PATH2 ...]

Options:
    --texture-paths    Additional paths to search for textures (space-separated)
    --no-animations    Skip animation files (.xsm)
"""

import argparse
import os
import sys
import time
from pathlib import Path

from gltf_exporter import export_to_gltf


def get_script_dir() -> Path:
    """Get the directory where this script is located."""
    return Path(__file__).parent.resolve()


def extract_model_name(xac_file: Path) -> str:
    """
    Extract the base model name from an XAC filename.
    e.g., 'monster_popolion_set.xac' -> 'popolion'
    """
    stem = xac_file.stem.lower()
    # Remove common prefixes/suffixes
    for prefix in ['monster_', 'pc_', 'npc_', 'item_']:
        if stem.startswith(prefix):
            stem = stem[len(prefix):]
    for suffix in ['_set', '_hi', '_low', '_model']:
        if stem.endswith(suffix):
            stem = stem[:-len(suffix)]
    return stem


def find_xsm_files(xac_file: Path, input_dir: Path) -> list:
    """
    Find all .xsm files for a model, searching in:
    1. Same directory as the .xac file
    2. animation.ipf folder structure (parallel to char_hi.ipf)

    Returns a list of paths to .xsm files.
    """
    xsm_files = set()
    model_name = extract_model_name(xac_file)

    # 1. Look for .xsm files in the same directory
    for f in xac_file.parent.glob("*.xsm"):
        xsm_files.add(str(f))

    # 2. Search in animation.ipf folder structure
    # If xac is in char_hi.ipf/monster/xxx.xac, look in animation.ipf/monster/modelname/*.xsm
    animation_dirs = list(input_dir.rglob("animation.ipf"))
    for anim_dir in animation_dirs:
        # Search for folders matching the model name
        for model_folder in anim_dir.rglob(f"*{model_name}*"):
            if model_folder.is_dir():
                for f in model_folder.glob("*.xsm"):
                    xsm_files.add(str(f))

        # Also search directly for xsm files matching the model name
        for f in anim_dir.rglob(f"*{model_name}*.xsm"):
            xsm_files.add(str(f))

    # 3. Search entire input directory for matching xsm files as fallback
    for f in input_dir.rglob(f"*{model_name}*.xsm"):
        xsm_files.add(str(f))

    return sorted(xsm_files)


def find_texture_dirs(xac_file: Path, input_dir: Path) -> list:
    """
    Find texture directories to search, including:
    1. Same directory as the .xac file
    2. char_texture.ipf folder structure (parallel to char_hi.ipf)
    3. All texture-related .ipf folders

    Returns a list of directory paths to search for textures.
    """
    texture_dirs = set()
    model_name = extract_model_name(xac_file)

    # 1. Add the xac file's directory
    texture_dirs.add(str(xac_file.parent))

    # 2. Search for texture folders in common IPF structures
    texture_ipf_patterns = [
        "char_texture.ipf",
        "char_texture_low.ipf",
        "item_texture.ipf",
        "item_texture_low.ipf",
    ]

    for pattern in texture_ipf_patterns:
        for tex_dir in input_dir.rglob(pattern):
            if tex_dir.is_dir():
                texture_dirs.add(str(tex_dir))
                # Also add subdirectories that might match the model
                for subdir in tex_dir.rglob(f"*{model_name}*"):
                    if subdir.is_dir():
                        texture_dirs.add(str(subdir))

    # 3. Add all directories containing texture files as fallback
    for ext in ['*.dds', '*.png', '*.tga']:
        for tex_file in input_dir.rglob(ext):
            texture_dirs.add(str(tex_file.parent))

    return sorted(texture_dirs)


def convert_xac_files(input_dir: Path, output_dir: Path, extra_texture_paths: list, include_animations: bool = True):
    """
    Recursively convert all .xac files from input_dir to output_dir.

    Args:
        input_dir: Source directory containing .xac files
        output_dir: Destination directory for .glb files
        extra_texture_paths: Additional directories to search for textures
        include_animations: Whether to include .xsm animation files
    """
    # Find all .xac files recursively
    xac_files = list(input_dir.rglob("*.xac"))

    if not xac_files:
        print(f"No .xac files found in '{input_dir}'")
        return

    print(f"Found {len(xac_files)} .xac file(s) to convert")
    print(f"Include animations: {include_animations}")
    print("-" * 60)

    successful = 0
    failed = 0
    total_animations = 0
    start_time = time.time()

    for xac_file in xac_files:
        # Calculate relative path to preserve directory structure
        relative_path = xac_file.relative_to(input_dir)
        output_file = output_dir / relative_path.with_suffix(".glb")

        # Create output directory if needed
        output_file.parent.mkdir(parents=True, exist_ok=True)

        print(f"Converting: {relative_path}")

        try:
            # Build texture search paths for this file
            texture_paths = find_texture_dirs(xac_file, input_dir) + extra_texture_paths

            # Find animation files
            xsm_files = []
            if include_animations:
                xsm_files = find_xsm_files(xac_file, input_dir)
                if xsm_files:
                    print(f"  Found {len(xsm_files)} animation(s)")
                    total_animations += len(xsm_files)

            export_to_gltf(
                str(xac_file),
                str(output_file),
                texture_paths,
                xsm_paths=xsm_files if xsm_files else None
            )
            print(f"  -> {output_file.relative_to(output_dir)}")
            successful += 1
        except Exception as e:
            print(f"  ERROR: {e}")
            failed += 1

    elapsed = time.time() - start_time
    print("-" * 60)
    print(f"Conversion complete in {elapsed:.1f}s")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    if include_animations:
        print(f"  Total animations embedded: {total_animations}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert .xac files to .glb format with optional animations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python convert.py
        Convert all .xac files from 'input' to 'output' folder
        (automatically finds animations and textures in IPF folder structure)

    python convert.py --no-animations
        Convert without animations

    python convert.py --texture-paths "C:/game/textures" "D:/assets"
        Convert with additional texture search paths
        """
    )
    parser.add_argument(
        "--texture-paths",
        nargs="*",
        default=[],
        help="Additional directories to search for textures"
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Input directory (default: 'input' in script directory)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory (default: 'output' in script directory)"
    )
    parser.add_argument(
        "--no-animations",
        action="store_true",
        help="Skip animation files (.xsm)"
    )

    args = parser.parse_args()

    script_dir = get_script_dir()

    # Determine input/output directories
    input_dir = Path(args.input) if args.input else script_dir / "input"
    output_dir = Path(args.output) if args.output else script_dir / "output"

    # Create directories if they don't exist
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("XAC to GLB Batch Converter")
    print("=" * 60)
    print(f"Input directory:  {input_dir}")
    print(f"Output directory: {output_dir}")
    print()

    # Check if input directory has any files
    if not any(input_dir.iterdir()):
        print(f"The input directory is empty.")
        print(f"Please place .xac files in: {input_dir}")
        print(f"Optionally place .xsm animation files in the same directories.")
        sys.exit(0)

    convert_xac_files(
        input_dir,
        output_dir,
        args.texture_paths,
        include_animations=not args.no_animations
    )


if __name__ == "__main__":
    main()
