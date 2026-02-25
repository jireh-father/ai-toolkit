"""Report generation for face pixel comparison: JSON, CSV, and HTML dashboard."""

import base64
import csv
import io
import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
from jinja2 import Environment, FileSystemLoader
from PIL import Image

logger = logging.getLogger(__name__)

TEMPLATE_DIR = Path(__file__).parent / "templates"

METRIC_FIELDS = ["mae", "mse", "mask_iou", "mask_area_ratio"]


def _compute_statistics(results: list[dict]) -> dict:
    """Compute per-metric statistics from results (excluding skipped)."""
    stats = {}
    valid = [r for r in results if not r.get("skipped")]
    for field in METRIC_FIELDS:
        values = [r[field] for r in valid if field in r]
        if values:
            arr = np.array(values, dtype=float)
            stats[field] = {
                "mean": round(float(np.mean(arr)), 2),
                "std": round(float(np.std(arr)), 2),
                "min": round(float(np.min(arr)), 2),
                "max": round(float(np.max(arr)), 2),
            }
        else:
            stats[field] = {"mean": 0, "std": 0, "min": 0, "max": 0}
    return stats


def _compute_mae_histogram(results: list[dict], bin_size: float = 5.0) -> dict:
    """Compute MAE distribution histogram with bins of given size.

    Returns dict with 'labels' (list of str) and 'counts' (list of int).
    """
    valid = [r for r in results if not r.get("skipped") and "mae" in r]
    if not valid:
        return {"labels": [], "counts": []}

    mae_values = [r["mae"] for r in valid]
    max_mae = max(mae_values)
    num_bins = max(int(max_mae / bin_size) + 1, 1)

    labels = []
    counts = [0] * num_bins
    for i in range(num_bins):
        lo = i * bin_size
        hi = (i + 1) * bin_size
        labels.append(f"{lo:.0f}-{hi:.0f}")

    for v in mae_values:
        idx = min(int(v / bin_size), num_bins - 1)
        counts[idx] += 1

    return {"labels": labels, "counts": counts}


def _image_to_base64(image: np.ndarray | Image.Image, max_size: int = 256) -> str:
    """Convert image to base64 thumbnail string."""
    try:
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        w, h = image.size
        if max(w, h) > max_size:
            ratio = max_size / max(w, h)
            image = image.resize(
                (int(w * ratio), int(h * ratio)), Image.LANCZOS
            )
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=75)
        return base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception as e:
        logger.warning(f"Failed to encode image: {e}")
        return ""


def _image_path_to_base64(path: Path, max_size: int = 256) -> str:
    """Load image from path and convert to base64 thumbnail."""
    try:
        img = Image.open(path).convert("RGB")
        return _image_to_base64(img, max_size)
    except Exception as e:
        logger.warning(f"Failed to load image {path}: {e}")
        return ""


def generate_json_report(
    results: list[dict],
    metadata: dict,
    output_path: Path,
) -> Path:
    """Generate detailed JSON report."""
    statistics = _compute_statistics(results)

    report = {
        "metadata": metadata,
        "statistics": statistics,
        "results": results,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)

    logger.info(f"JSON report saved: {output_path}")
    return output_path


def generate_csv_report(
    results: list[dict],
    output_path: Path,
) -> Path:
    """Generate CSV summary report."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "filename", "pass", "mae", "mse",
        "mask_iou", "mask_area_ratio", "intersection_pixels",
        "skipped", "reason",
    ]

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            row = {k: r.get(k, "") for k in fieldnames}
            writer.writerow(row)

    logger.info(f"CSV report saved: {output_path}")
    return output_path


def generate_html_report(
    results: list[dict],
    metadata: dict,
    image_dirs: dict,
    output_path: Path,
) -> Path:
    """Generate HTML dashboard report with infinite scroll and file-path images.

    Args:
        results: list of result dicts (non-skipped items need mae, mse, etc.)
        metadata: report metadata dict
        image_dirs: {"input": str, "output": str} absolute directory paths
        output_path: where to write HTML
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    statistics = _compute_statistics(results)
    histogram = _compute_mae_histogram(results)

    # Elapsed time formatting
    elapsed = metadata.get("elapsed_time_sec", 0)
    hours, remainder = divmod(int(elapsed), 3600)
    minutes, seconds = divmod(remainder, 60)
    metadata["elapsed_time_str"] = f"{hours}h {minutes}m {seconds}s"

    # Render template
    env = Environment(loader=FileSystemLoader(str(TEMPLATE_DIR)))
    template = env.get_template("face_summary.html")

    html = template.render(
        metadata=metadata,
        statistics=statistics,
        results_json=json.dumps(results, ensure_ascii=False, default=str),
        statistics_json=json.dumps(statistics),
        histogram_json=json.dumps(histogram),
        image_dirs_json=json.dumps(image_dirs, ensure_ascii=False),
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    logger.info(f"HTML report saved: {output_path}")
    return output_path


def generate_all_reports(
    results: list[dict],
    metadata: dict,
    image_dirs: dict,
    report_dir: Path,
    # Legacy parameters (ignored, kept for backwards compatibility)
    image_paths: dict | None = None,
    heatmap_images: dict | None = None,
) -> dict:
    """Generate all report formats (JSON, CSV, HTML).

    Returns dict with paths to generated files.
    """
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    json_path = generate_json_report(
        results, metadata, report_dir / "face_results.json"
    )
    csv_path = generate_csv_report(
        results, report_dir / "face_results.csv"
    )
    html_path = generate_html_report(
        results, metadata, image_dirs,
        report_dir / "face_summary.html",
    )

    return {
        "json": json_path,
        "csv": csv_path,
        "html": html_path,
    }
