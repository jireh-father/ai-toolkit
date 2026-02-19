"""Scan dataset directories and output matched/mismatched files as JSON.

Wraps image_loader.scan_dataset() as a CLI tool for the subagent orchestrator.

Usage:
    python dataset_validator/scripts/scan_dataset.py \
        --input-dir ./data/input \
        --reference-dir ./data/reference \
        --output-dir ./data/output \
        --out ./workspace/scan_result.json
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Ensure the project root is on sys.path so that
# `dataset_validator` package is importable regardless of cwd.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from dataset_validator.core.image_loader import scan_dataset

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Scan dataset directories and output matched/mismatched files as JSON.",
    )
    parser.add_argument("--input-dir", required=True, help="Path to input images folder")
    parser.add_argument("--reference-dir", required=True, help="Path to reference images folder")
    parser.add_argument("--output-dir", required=True, help="Path to output images folder")
    parser.add_argument("--out", required=True, help="Output JSON file path")
    args = parser.parse_args()

    # Validate directories
    input_dir = Path(args.input_dir)
    reference_dir = Path(args.reference_dir)
    output_dir = Path(args.output_dir)

    for name, d in [("input", input_dir), ("reference", reference_dir), ("output", output_dir)]:
        if not d.is_dir():
            logger.error(f"{name} directory not found: {d}")
            sys.exit(1)

    # Run scan
    matched, mismatched = scan_dataset(input_dir, reference_dir, output_dir)

    # Convert Path objects to strings for JSON serialization
    for m in matched:
        for key in ("input", "reference", "output"):
            m[key] = str(m[key])

    result = {"matched": matched, "mismatched": mismatched}

    # Ensure output directory exists
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    logger.info(f"Scan result saved: {out_path}")
    print(f"Matched: {len(matched)}, Mismatched: {len(mismatched)}")


if __name__ == "__main__":
    main()
