"""Separate passed and failed samples based on same-face hairstyle check results.

Works with results.json produced by validate_dataset_simple_check_same_face.py,
which uses a true/false match per sample (no reference directory).

Usage:
    python dataset_validator/separate_failed_same_face.py \
        --report ./reports_same_face/results.json \
        --input-dir ./data/input \
        --output-dir ./data/output \
        --failed-dir ./data/failed \
        --passed-dir ./data/passed
"""

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Separate passed/failed samples based on same-face hairstyle check results",
    )
    parser.add_argument(
        "--report", type=str, required=True,
        help="Path to results.json from validate_dataset_simple_check_same_face.py",
    )
    parser.add_argument(
        "--input-dir", type=str, required=True,
        help="Path to input (original) images folder",
    )
    parser.add_argument(
        "--output-dir", type=str, required=True,
        help="Path to output (edited) images folder",
    )
    parser.add_argument(
        "--failed-dir", type=str, default=None,
        help="Destination folder for failed samples",
    )
    parser.add_argument(
        "--passed-dir", type=str, default=None,
        help="Destination folder for passed samples",
    )
    parser.add_argument(
        "--mode", type=str, default="copy",
        choices=["copy", "move"],
        help="File operation mode: copy or move (default: copy)",
    )
    return parser.parse_args()


def _copy_files(filenames, input_dir, output_dir, dest_dir, op_func, op_name):
    """Copy or move files to destination directory.

    Returns (success_count, skip_count).
    """
    dest_input = dest_dir / "input"
    dest_output = dest_dir / "output"

    for d in [dest_input, dest_output]:
        d.mkdir(parents=True, exist_ok=True)

    success_count = 0
    skip_count = 0

    for filename in filenames:
        stem = Path(filename).stem

        moved_any = False
        for src_dir, dst_dir, label in [
            (input_dir, dest_input, "input"),
            (output_dir, dest_output, "output"),
        ]:
            src_file = src_dir / filename
            if not src_file.is_file():
                candidates = [
                    c for c in src_dir.glob(f"{stem}.*")
                    if c.suffix.lower() in SUPPORTED_EXTENSIONS
                ]
                if candidates:
                    src_file = candidates[0]
                else:
                    logger.warning(f"  {label}: {filename} not found in {src_dir}")
                    continue

            dst_file = dst_dir / src_file.name
            try:
                op_func(str(src_file), str(dst_file))
                moved_any = True
            except Exception as e:
                logger.error(f"  Failed to {op_name.lower()} {src_file}: {e}")

        if moved_any:
            success_count += 1
        else:
            skip_count += 1

    return success_count, skip_count


def main():
    args = parse_args()

    if args.passed_dir is None and args.failed_dir is None:
        logger.error("At least one of --passed-dir or --failed-dir must be specified")
        sys.exit(1)

    # Load report
    report_path = Path(args.report)
    if not report_path.is_file():
        logger.error(f"Report file not found: {report_path}")
        sys.exit(1)

    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    results = report.get("results", [])

    # Classify files based on match true/false
    failed_files = []
    passed_files = []
    for r in results:
        match_val = r.get("match")
        if r.get("error") or match_val is None:
            failed_files.append(r["filename"])
            continue
        if match_val is True:
            passed_files.append(r["filename"])
        else:
            failed_files.append(r["filename"])

    logger.info(f"Total results: {len(results)}")
    logger.info(f"Passed (same hairstyle): {len(passed_files)}, Failed (different hairstyle): {len(failed_files)}")

    # Validate source directories
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    for name, d in [("input", input_dir), ("output", output_dir)]:
        if not d.is_dir():
            logger.error(f"{name} directory not found: {d}")
            sys.exit(1)

    op_func = shutil.copy2 if args.mode == "copy" else shutil.move
    op_name = "Copying" if args.mode == "copy" else "Moving"

    # Process failed
    if args.failed_dir and failed_files:
        failed_dir = Path(args.failed_dir)
        logger.info(f"{op_name} {len(failed_files)} failed samples to {failed_dir}...")
        success, skip = _copy_files(
            failed_files, input_dir, output_dir, failed_dir, op_func, op_name,
        )
        logger.info(f"  Failed -- processed: {success}, skipped: {skip}")

    # Process passed
    if args.passed_dir and passed_files:
        passed_dir = Path(args.passed_dir)
        logger.info(f"{op_name} {len(passed_files)} passed samples to {passed_dir}...")
        success, skip = _copy_files(
            passed_files, input_dir, output_dir, passed_dir, op_func, op_name,
        )
        logger.info(f"  Passed -- processed: {success}, skipped: {skip}")

    logger.info("=" * 50)
    logger.info("Done!")
    if args.failed_dir:
        logger.info(f"  Failed dir: {Path(args.failed_dir).resolve()}")
    if args.passed_dir:
        logger.info(f"  Passed dir: {Path(args.passed_dir).resolve()}")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
