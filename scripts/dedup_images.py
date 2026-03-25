"""
Remove duplicate images by MD5 hash.

Scans a directory for images, computes MD5 hash for each file,
and removes duplicates (keeps the first occurrence by sorted filename).

Usage:
    python scripts/dedup_images.py --input <dir> [--dry-run] [--move <trash_dir>]
"""

import argparse
import hashlib
import shutil
import sys
from pathlib import Path

from tqdm import tqdm

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}


def md5_hash(file_path: Path) -> str:
    h = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description="Remove duplicate images by MD5 hash")
    parser.add_argument("--input", "-i", required=True, help="Input directory")
    parser.add_argument("--dry-run", action="store_true", help="Only show duplicates, don't delete/move")
    parser.add_argument("--move", "-m", default=None, help="Move duplicates to this directory instead of deleting")
    args = parser.parse_args()

    input_dir = Path(args.input)
    if not input_dir.is_dir():
        print(f"Error: {input_dir} not found")
        sys.exit(1)

    image_paths = sorted([p for p in input_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS])
    if not image_paths:
        print(f"No images found in {input_dir}")
        sys.exit(1)

    print(f"Found {len(image_paths)} images in {input_dir}")

    if args.move:
        trash_dir = Path(args.move)
        trash_dir.mkdir(parents=True, exist_ok=True)

    seen = {}
    duplicates = []

    for path in tqdm(image_paths, desc="Hashing"):
        h = md5_hash(path)
        if h in seen:
            duplicates.append((path, seen[h]))
        else:
            seen[h] = path

    print(f"\nUnique: {len(seen)}")
    print(f"Duplicates: {len(duplicates)}")

    if not duplicates:
        print("No duplicates found.")
        return

    if args.dry_run:
        for dup, original in duplicates:
            print(f"  DUP: {dup.name}  ==  {original.name}")
        print(f"\n[DRY RUN] No files were deleted/moved.")
        return

    for dup, original in duplicates:
        if args.move:
            dst = Path(args.move) / dup.name
            counter = 1
            while dst.exists():
                dst = Path(args.move) / f"{dup.stem}_{counter}{dup.suffix}"
                counter += 1
            shutil.move(str(dup), str(dst))
        else:
            dup.unlink()

    action = "Moved" if args.move else "Deleted"
    print(f"{action} {len(duplicates)} duplicate files.")


if __name__ == "__main__":
    main()
