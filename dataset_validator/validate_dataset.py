"""Main entry point for dataset validation.

Usage:
    python dataset_validator/validate_dataset.py \
        --input-dir ./data/input \
        --reference-dir ./data/reference \
        --output-dir ./data/output \
        --model qwen2.5-vl-7b \
        --report-dir ./reports
"""

import argparse
import json
import logging
import random
import sys
import time
from datetime import datetime
from pathlib import Path

def setup_logging(level: str):
    """Configure logging with the specified level."""
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Hair Transfer Dataset Quality Validator using VLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Supported models:
  qwen2.5-vl-7b          ~5-6GB (INT4), ~14GB (FP16)
  qwen3-vl-8b            ~6-7GB (INT4), ~16GB (FP16)
  internvl2.5-8b         ~6-7GB (INT4), ~16GB (FP16)
  internvl3-8b           ~6-7GB (INT4), ~16GB (FP16)
  qwen3-vl-30b-a3b       ~7-8GB (INT4), ~18GB (FP16) [MoE]
  minicpm-v-2.6          ~6GB (INT4),   ~16GB (FP16)
  gemma-3-12b-vision     ~8-9GB (INT4), ~24GB (FP16)

Examples:
  # Basic usage
  python validate_dataset.py --input-dir ./input --reference-dir ./ref --output-dir ./output

  # Debug with 5 samples
  python validate_dataset.py --input-dir ./input --reference-dir ./ref --output-dir ./output \\
      --max-samples 5 --log-level debug

  # Low-VRAM mode (V100 32GB with large models)
  python validate_dataset.py --input-dir ./input --reference-dir ./ref --output-dir ./output \\
      --model qwen3-vl-30b-a3b --low-vram --resize-short-side 384

  # Multi-GPU with custom thresholds
  python validate_dataset.py --input-dir ./input --reference-dir ./ref --output-dir ./output \\
      --num-gpus 2 --threshold-hair 8 --threshold-naturalness 6
""",
    )

    # Required arguments
    parser.add_argument(
        "--input-dir", type=str, required=True,
        help="Path to input images folder",
    )
    parser.add_argument(
        "--reference-dir", type=str, required=True,
        help="Path to reference images folder",
    )
    parser.add_argument(
        "--output-dir", type=str, required=True,
        help="Path to output images folder",
    )

    # Model settings
    parser.add_argument(
        "--model", type=str, default="qwen3-vl-8b",
        help="VLM model name (default: qwen3-vl-8b)",
    )
    parser.add_argument(
        "--quantization", type=str, default="int4",
        choices=["int4", "int8", "fp16"],
        help="Quantization mode (default: int4)",
    )
    parser.add_argument(
        "--num-gpus", type=int, default=1,
        help="Number of GPUs to use (default: 1)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=4,
        help="Batch size per GPU (default: 4)",
    )
    parser.add_argument(
        "--low-vram", action="store_true",
        help="Enable low-VRAM mode: uses CPU offloading via device_map='auto'",
    )

    # Threshold settings
    parser.add_argument(
        "--threshold", type=int, default=7,
        help="Global pass threshold for all criteria (0-10, default: 7)",
    )
    parser.add_argument(
        "--threshold-hair", type=int, default=None,
        help="Threshold for hair criteria (overrides --threshold for hair fields)",
    )
    parser.add_argument(
        "--threshold-preservation", type=int, default=None,
        help="Threshold for non_hair_preservation (overrides --threshold)",
    )
    parser.add_argument(
        "--threshold-naturalness", type=int, default=None,
        help="Threshold for naturalness (overrides --threshold)",
    )
    parser.add_argument(
        "--threshold-face", type=int, default=None,
        help="Threshold for face criteria: face_shape_preservation, face_color_preservation (overrides --threshold)",
    )

    # Output settings
    parser.add_argument(
        "--report-dir", type=str, default="./reports",
        help="Directory for output reports (default: ./reports)",
    )

    # Checkpoint settings
    parser.add_argument(
        "--checkpoint-dir", type=str, default="./checkpoints",
        help="Directory for checkpoint files (default: ./checkpoints)",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from latest checkpoint",
    )
    parser.add_argument(
        "--checkpoint-interval", type=int, default=100,
        help="Save checkpoint every N samples (default: 100)",
    )

    # Image settings
    parser.add_argument(
        "--resize-short-side", type=int, default=512,
        help="Resize images so short side = this value (default: 512)",
    )

    # Misc
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Max samples to evaluate (for debugging)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--log-level", type=str, default="info",
        choices=["debug", "info", "warning", "error"],
        help="Logging level (default: info)",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    setup_logging(args.log_level)
    set_seed(args.seed)

    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("Hair Transfer Dataset Validator")
    logger.info("=" * 60)

    # Ensure dataset_validator package is importable
    pkg_dir = Path(__file__).resolve().parent
    if str(pkg_dir.parent) not in sys.path:
        sys.path.insert(0, str(pkg_dir.parent))

    # Validate directories
    input_dir = Path(args.input_dir)
    reference_dir = Path(args.reference_dir)
    output_dir = Path(args.output_dir)

    for name, d in [("input", input_dir), ("reference", reference_dir), ("output", output_dir)]:
        if not d.is_dir():
            logger.error(f"{name} directory not found: {d}")
            sys.exit(1)

    # Step 1: Scan dataset
    logger.info("Step 1: Scanning dataset...")
    from dataset_validator.core.image_loader import scan_dataset, filter_corrupted

    matched, mismatched = scan_dataset(input_dir, reference_dir, output_dir)

    if not matched:
        logger.error("No matched image triplets found. Check your directories.")
        sys.exit(1)

    # Step 2: Filter corrupted images
    logger.info("Step 2: Validating image integrity...")
    valid_entries, corrupted = filter_corrupted(matched)

    if not valid_entries:
        logger.error("No valid image triplets after corruption check.")
        sys.exit(1)

    logger.info(f"Valid entries: {len(valid_entries)}")

    # Limit samples if requested
    if args.max_samples and args.max_samples < len(valid_entries):
        valid_entries = valid_entries[:args.max_samples]
        logger.info(f"Limited to {args.max_samples} samples (--max-samples)")

    # Step 3: Initialize checkpoint manager
    logger.info("Step 3: Initializing checkpoint manager...")
    from dataset_validator.core.checkpoint import CheckpointManager

    ckpt = CheckpointManager(
        checkpoint_dir=Path(args.checkpoint_dir),
        interval=args.checkpoint_interval,
        model_name=args.model,
        quantization=args.quantization,
    )
    ckpt.total_count = len(valid_entries)

    if args.resume:
        ckpt.load()

    remaining = ckpt.get_remaining(valid_entries)

    start_time = time.time()

    if not remaining:
        logger.info("All samples already processed (checkpoint). Generating reports...")
        results = ckpt.results
    else:
        # Step 4: Run evaluation
        logger.info(f"Step 4: Evaluating {len(remaining)} samples...")
        logger.info(f"Model: {args.model}, Quantization: {args.quantization}, GPUs: {args.num_gpus}")

        from dataset_validator.core.parallel import run_parallel_evaluation

        new_results = run_parallel_evaluation(
            entries=remaining,
            model_name=args.model,
            quantization=args.quantization,
            num_gpus=args.num_gpus,
            short_side=args.resize_short_side,
            max_retries=3,
            checkpoint_manager=ckpt,
            checkpoint_interval=args.checkpoint_interval,
            low_vram=args.low_vram,
        )

        elapsed = time.time() - start_time
        logger.info(f"Evaluation completed in {elapsed:.1f}s")

        # Combine checkpoint results with new results
        results = ckpt.results

    elapsed_total = time.time() - start_time

    # Step 5: Generate reports
    logger.info("Step 5: Generating reports...")
    from dataset_validator.report.generator import generate_all_reports

    # Build entries map for HTML image embedding
    entries_map = {e["stem"]: e for e in valid_entries}

    metadata = {
        "model": args.model,
        "quantization": args.quantization,
        "timestamp": datetime.now().isoformat(),
        "elapsed_time_sec": elapsed_total,
        "seed": args.seed,
        "resize_short_side": args.resize_short_side,
        "num_gpus": args.num_gpus,
    }

    report_paths = generate_all_reports(
        results=results,
        metadata=metadata,
        entries_map=entries_map,
        report_dir=Path(args.report_dir),
        threshold=args.threshold,
        threshold_hair=args.threshold_hair,
        threshold_preservation=args.threshold_preservation,
        threshold_naturalness=args.threshold_naturalness,
        threshold_face=args.threshold_face,
    )

    # Final save checkpoint
    ckpt.save()
    ckpt.restore_signal_handlers()

    # Save mismatch report if any
    if mismatched or corrupted:
        mismatch_path = Path(args.report_dir) / "mismatch_report.json"
        with open(mismatch_path, "w", encoding="utf-8") as f:
            json.dump({
                "mismatched_files": mismatched,
                "corrupted_files": corrupted,
            }, f, ensure_ascii=False, indent=2, default=str)
        logger.info(f"Mismatch report saved: {mismatch_path}")

    # Summary
    meta = report_paths["metadata"]
    logger.info("=" * 60)
    logger.info("Validation Complete!")
    logger.info(f"  Total:  {meta['total_samples']}")
    logger.info(f"  Passed: {meta['passed']} ({meta['pass_rate']}%)")
    logger.info(f"  Failed: {meta['failed']}")
    if meta.get("errors", 0) > 0:
        logger.info(f"  Errors: {meta['errors']}")
    logger.info(f"  Reports: {Path(args.report_dir).resolve()}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
