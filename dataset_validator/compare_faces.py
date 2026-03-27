"""Face Pixel Comparator — compare face regions between input/output image pairs.

Uses FaRL face parsing to segment face regions (skin, eyes, nose, eyebrows, lips)
and computes MAE/MSE metrics on the intersection of the two masks.

Usage:
    python dataset_validator/compare_faces.py \
        --input-dir E:/backup/aihub2025_gpu/output/input_image \
        --output-dir E:/backup/aihub2025_gpu/output \
        --report-dir ./face_reports \
        --threshold-mae 20.0
"""

import argparse
import logging
import multiprocessing as mp
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataset_validator.core.face_segmentor import FaceSegmentor
from dataset_validator.core.image_loader import SUPPORTED_EXTENSIONS, _get_image_stems
from dataset_validator.core.pixel_comparator import compare_faces
from dataset_validator.report.face_report_generator import generate_all_reports

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Face Pixel Comparator: compare face regions between input/output pairs"
    )
    parser.add_argument(
        "--input-dir", type=str, required=True,
        help="Input images directory",
    )
    parser.add_argument(
        "--output-dir", type=str, required=True,
        help="Output images directory",
    )
    parser.add_argument(
        "--report-dir", type=str, default="./face_reports",
        help="Report output directory (default: ./face_reports)",
    )
    parser.add_argument(
        "--threshold-mae", type=float, default=5.0,
        help="MAE threshold for pass/fail (default: 5.0)",
    )
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Max number of pairs to process (for debugging)",
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0",
        help="Device for face parsing (default: cuda:0)",
    )
    parser.add_argument(
        "--num-workers", type=int, default=1,
        help="Number of parallel worker processes (default: 1). "
             "Each worker gets its own FaceSegmentor. "
             "With multiple GPUs, workers are distributed across devices.",
    )
    parser.add_argument(
        "--log-level", type=str, default="info",
        choices=["debug", "info", "warning", "error"],
        help="Logging level (default: info)",
    )
    return parser.parse_args()


def match_image_pairs(
    input_dir: Path, output_dir: Path
) -> list[dict]:
    """Match input/output images by filename stem.

    Returns list of {"stem": str, "input": Path, "output": Path}
    """
    input_stems = _get_image_stems(input_dir)
    output_stems = _get_image_stems(output_dir)

    common_stems = sorted(set(input_stems) & set(output_stems))

    pairs = []
    for stem in common_stems:
        pairs.append({
            "stem": stem,
            "input": input_stems[stem],
            "output": output_stems[stem],
        })

    return pairs


def _process_single_pair(segmentor, pair, threshold_mae):
    """Process a single image pair with the given segmentor."""
    input_path = pair["input"]
    output_path = pair["output"]
    filename = input_path.name

    try:
        input_img = np.array(Image.open(input_path).convert("RGB"))
        output_img = np.array(Image.open(output_path).convert("RGB"))

        if output_img.shape[:2] != input_img.shape[:2]:
            h, w = input_img.shape[:2]
            output_img = np.array(
                Image.fromarray(output_img).resize((w, h), Image.LANCZOS)
            )

        input_mask = segmentor.segment(input_img)
        if input_mask is None:
            return {"filename": filename, "pass": False, "skipped": True,
                    "reason": "Face not detected in input"}

        output_mask = segmentor.segment(output_img)
        if output_mask is None:
            return {"filename": filename, "pass": False, "skipped": True,
                    "reason": "Face not detected in output"}

        metrics = compare_faces(input_img, output_img, input_mask, output_mask)
        passed = metrics["mae"] <= threshold_mae

        result = {"filename": filename, "pass": passed, "skipped": False, **metrics}
        if not passed:
            result["reason"] = f"MAE {metrics['mae']:.1f} exceeds threshold {threshold_mae}"
        return result

    except Exception as e:
        return {"filename": filename, "pass": False, "skipped": True,
                "reason": f"Error: {str(e)}"}


def _worker_process(worker_id, device, pairs, threshold_mae, result_queue, log_level):
    """Worker process: creates its own FaceSegmentor and processes assigned pairs."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=f"%(asctime)s [%(levelname)s] worker-{worker_id}: %(message)s",
    )
    wlogger = logging.getLogger(f"worker-{worker_id}")
    wlogger.info(f"Starting on {device}, {len(pairs)} pairs to process")

    segmentor = FaceSegmentor(device=device)

    for pair in pairs:
        result = _process_single_pair(segmentor, pair, threshold_mae)
        result_queue.put(result)

    wlogger.info("Done")


def _resolve_devices(base_device: str, num_workers: int) -> list[str]:
    """Assign a device to each worker.

    If base_device is a CUDA device, distribute workers across available GPUs
    in round-robin fashion. Otherwise all workers share the same device.
    """
    if not base_device.startswith("cuda"):
        return [base_device] * num_workers

    import torch
    num_gpus = torch.cuda.device_count()
    if num_gpus <= 1:
        return [base_device] * num_workers

    # Round-robin across available GPUs
    return [f"cuda:{i % num_gpus}" for i in range(num_workers)]


def main():
    args = parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    report_dir = Path(args.report_dir)

    if not input_dir.exists():
        logger.error(f"Input directory not found: {input_dir}")
        sys.exit(1)
    if not output_dir.exists():
        logger.error(f"Output directory not found: {output_dir}")
        sys.exit(1)

    # Step 1: Match image pairs
    logger.info(f"Scanning input: {input_dir}")
    logger.info(f"Scanning output: {output_dir}")
    pairs = match_image_pairs(input_dir, output_dir)
    logger.info(f"Matched pairs: {len(pairs)}")

    if not pairs:
        logger.error("No matching image pairs found")
        sys.exit(1)

    if args.max_samples:
        pairs = pairs[:args.max_samples]
        logger.info(f"Limited to {len(pairs)} pairs (--max-samples)")

    num_workers = args.num_workers
    start_time = time.time()

    if num_workers <= 1:
        # Single-process mode (original behavior)
        logger.info(f"Loading FaceSegmentor on {args.device}")
        segmentor = FaceSegmentor(device=args.device)

        results = []
        for pair in tqdm(pairs, desc="Comparing faces"):
            result = _process_single_pair(segmentor, pair, args.threshold_mae)
            results.append(result)
    else:
        # Multi-process mode
        devices = _resolve_devices(args.device, num_workers)
        device_summary = {}
        for d in devices:
            device_summary[d] = device_summary.get(d, 0) + 1
        logger.info(
            f"Launching {num_workers} workers: "
            + ", ".join(f"{d}x{c}" for d, c in device_summary.items())
        )

        # Split pairs into chunks for each worker
        chunks = [[] for _ in range(num_workers)]
        for i, pair in enumerate(pairs):
            chunks[i % num_workers].append(pair)

        ctx = mp.get_context("spawn")
        result_queue = ctx.Queue()

        processes = []
        for worker_id in range(num_workers):
            p = ctx.Process(
                target=_worker_process,
                args=(worker_id, devices[worker_id], chunks[worker_id],
                      args.threshold_mae, result_queue, args.log_level),
            )
            p.start()
            processes.append(p)

        # Collect results with progress bar
        total = len(pairs)
        results = []
        with tqdm(total=total, desc="Comparing faces") as pbar:
            while len(results) < total:
                result = result_queue.get()
                results.append(result)
                pbar.update(1)

        for p in processes:
            p.join()

    elapsed = time.time() - start_time

    # Step 4: Compute metadata
    valid_results = [r for r in results if not r.get("skipped")]
    passed_count = sum(1 for r in valid_results if r.get("pass"))
    failed_count = sum(1 for r in valid_results if not r.get("pass"))
    skipped_count = sum(1 for r in results if r.get("skipped"))
    total = len(results)

    metadata = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "threshold_mae": args.threshold_mae,
        "total_samples": total,
        "passed": passed_count,
        "failed": failed_count,
        "skipped": skipped_count,
        "pass_rate": round(passed_count / len(valid_results) * 100, 1) if valid_results else 0,
        "fail_rate": round(failed_count / len(valid_results) * 100, 1) if valid_results else 0,
        "timestamp": datetime.now().isoformat(),
        "elapsed_time_sec": round(elapsed, 1),
        "device": args.device,
    }

    logger.info(
        f"Results: {passed_count} passed, {failed_count} failed, "
        f"{skipped_count} skipped ({elapsed:.1f}s)"
    )

    # Step 5: Generate reports
    logger.info(f"Generating reports in {report_dir}")
    image_dirs = {
        "input": str(input_dir.resolve()),
        "output": str(output_dir.resolve()),
    }
    report_paths = generate_all_reports(
        results=results,
        metadata=metadata,
        image_dirs=image_dirs,
        report_dir=report_dir,
    )

    logger.info(f"JSON: {report_paths['json']}")
    logger.info(f"CSV:  {report_paths['csv']}")
    logger.info(f"HTML: {report_paths['html']}")
    logger.info("Done!")


if __name__ == "__main__":
    main()
