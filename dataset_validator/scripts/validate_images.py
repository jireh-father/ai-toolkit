"""Validate image integrity and output valid/corrupted entries as JSON.

Wraps image_loader.filter_corrupted() as a CLI tool for the subagent orchestrator.

Usage:
    python dataset_validator/scripts/validate_images.py \
        --scan-result ./workspace/scan_result.json \
        --out ./workspace/valid_entries.json
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Ensure the project root is on sys.path so that
# `dataset_validator` package is importable regardless of cwd.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from dataset_validator.core.image_loader import filter_corrupted

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Validate image integrity and output valid/corrupted entries as JSON.",
    )
    parser.add_argument("--scan-result", required=True, help="scan_dataset.py output JSON path")
    parser.add_argument("--out", required=True, help="Output JSON file path")
    args = parser.parse_args()

    # Validate input file
    scan_path = Path(args.scan_result)
    if not scan_path.is_file():
        logger.error(f"Scan result file not found: {scan_path}")
        sys.exit(1)

    with open(scan_path, "r", encoding="utf-8") as f:
        scan = json.load(f)

    # Restore Path objects from string (scan_dataset.py serializes as str)
    matched = scan["matched"]
    for m in matched:
        for key in ("input", "reference", "output"):
            m[key] = Path(m[key])

    valid, corrupted = filter_corrupted(matched)

    # Convert Path objects back to strings for JSON serialization
    for v in valid:
        for key in ("input", "reference", "output"):
            v[key] = str(v[key])

    result = {"valid": valid, "corrupted": corrupted}

    # Ensure output directory exists
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    logger.info(f"Validation result saved: {out_path}")
    print(f"Valid: {len(valid)}, Corrupted: {len(corrupted)}")


if __name__ == "__main__":
    main()
