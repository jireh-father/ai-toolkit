"""Generate reports from evaluation results JSON.

Wraps generator.generate_all_reports() as a CLI tool for the subagent
orchestrator. Also generates mismatch_report.json when --valid-entries
and --scan-result are provided.

Usage:
    python dataset_validator/scripts/generate_reports.py \
        --results ./workspace/checkpoint.json \
        --scan-result ./workspace/scan_result.json \
        --valid-entries ./workspace/valid_entries.json \
        --report-dir ./reports \
        --threshold 7
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Ensure the project root is on sys.path so that
# `dataset_validator` package is importable regardless of cwd.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from dataset_validator.report.generator import generate_all_reports

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Generate reports (JSON, CSV, HTML) from evaluation results.",
    )
    parser.add_argument("--results", required=True, help="Checkpoint JSON file with results array")
    parser.add_argument("--scan-result", required=True, help="scan_dataset.py output JSON (for entries_map)")
    parser.add_argument(
        "--valid-entries", default=None,
        help="validate_images.py output JSON (optional, for mismatch_report.json)",
    )
    parser.add_argument("--report-dir", default="./reports", help="Output directory for reports")
    parser.add_argument("--threshold", type=int, default=7, help="Global pass threshold (0-10)")
    parser.add_argument("--threshold-hair", type=int, default=None, help="Threshold for hair criteria")
    parser.add_argument("--threshold-naturalness", type=int, default=None, help="Threshold for naturalness")
    args = parser.parse_args()

    # Validate input files
    results_path = Path(args.results)
    if not results_path.is_file():
        logger.error(f"Results file not found: {results_path}")
        sys.exit(1)

    scan_path = Path(args.scan_result)
    if not scan_path.is_file():
        logger.error(f"Scan result file not found: {scan_path}")
        sys.exit(1)

    # Load results from checkpoint JSON
    with open(results_path, "r", encoding="utf-8") as f:
        checkpoint = json.load(f)

    results = checkpoint["results"]

    # Load scan result to build entries_map (image paths for HTML thumbnails)
    with open(scan_path, "r", encoding="utf-8") as f:
        scan = json.load(f)

    entries_map = {}
    for entry in scan["matched"]:
        entries_map[entry["stem"]] = {
            "stem": entry["stem"],
            "input": Path(entry["input"]),
            "reference": Path(entry["reference"]),
            "output": Path(entry["output"]),
        }

    metadata = {
        "engine": checkpoint.get("engine", "claude-opus-4-6"),
        "timestamp": checkpoint.get("timestamp", ""),
        "elapsed_time_sec": checkpoint.get("elapsed_time_sec", 0),
    }

    report_dir = Path(args.report_dir)

    report_paths = generate_all_reports(
        results=results,
        metadata=metadata,
        entries_map=entries_map,
        report_dir=report_dir,
        threshold=args.threshold,
        threshold_hair=args.threshold_hair,
        threshold_naturalness=args.threshold_naturalness,
    )

    # Generate mismatch_report.json if data is available
    mismatched = scan.get("mismatched", [])
    corrupted = []

    if args.valid_entries:
        valid_path = Path(args.valid_entries)
        if valid_path.is_file():
            with open(valid_path, "r", encoding="utf-8") as f:
                valid_data = json.load(f)
            corrupted = valid_data.get("corrupted", [])
        else:
            logger.warning(f"Valid entries file not found: {valid_path}")

    if mismatched or corrupted:
        mismatch_path = report_dir / "mismatch_report.json"
        report_dir.mkdir(parents=True, exist_ok=True)
        with open(mismatch_path, "w", encoding="utf-8") as f:
            json.dump(
                {"mismatched_files": mismatched, "corrupted_files": corrupted},
                f, ensure_ascii=False, indent=2, default=str,
            )
        logger.info(f"Mismatch report saved: {mismatch_path}")

    logger.info(f"Reports generated in: {report_dir}")
    print(f"Reports generated: {report_dir}")
    for fmt, path in report_paths.items():
        if fmt != "metadata":
            print(f"  {fmt}: {path}")


if __name__ == "__main__":
    main()
