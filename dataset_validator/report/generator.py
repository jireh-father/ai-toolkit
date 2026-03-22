"""Report generation: JSON, CSV, and HTML dashboard."""

import base64
import csv
import io
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
from jinja2 import Environment, FileSystemLoader
from PIL import Image

from ..core.evaluator import SCORE_FIELDS, is_pass

logger = logging.getLogger(__name__)

TEMPLATE_DIR = Path(__file__).parent / "templates"


def _compute_statistics(results: list[dict]) -> dict:
    """Compute per-field statistics from results."""
    stats = {}
    for field in SCORE_FIELDS:
        values = [
            r["scores"][field]
            for r in results
            if not r.get("error") and r.get("scores") and field in r["scores"]
        ]
        if values:
            arr = np.array(values, dtype=float)
            stats[field] = {
                "mean": round(float(np.mean(arr)), 2),
                "std": round(float(np.std(arr)), 2),
                "min": int(np.min(arr)),
                "max": int(np.max(arr)),
            }
        else:
            stats[field] = {"mean": 0, "std": 0, "min": 0, "max": 0}
    return stats


def _compute_histogram(results: list[dict]) -> dict:
    """Compute score distribution histograms (bins 0-10) per field."""
    histograms = {}
    for field in SCORE_FIELDS:
        counts = [0] * 11  # bins 0..10
        for r in results:
            if not r.get("error") and r.get("scores"):
                score = r["scores"].get(field, 0)
                score = max(0, min(10, int(score)))
                counts[score] += 1
        histograms[field] = counts
    return histograms


def _image_to_base64(path: Path, max_size: int = 256) -> str:
    """Load image, resize for thumbnail, and return base64 string."""
    try:
        img = Image.open(path).convert("RGB")
        # Resize to thumbnail
        w, h = img.size
        if max(w, h) > max_size:
            ratio = max_size / max(w, h)
            img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=75)
        return base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception as e:
        logger.warning(f"Failed to encode image {path}: {e}")
        return ""


def _avg_score(scores: Optional[dict]) -> float:
    """Compute average of all score fields."""
    if not scores:
        return 0.0
    vals = [scores.get(f, 0) for f in SCORE_FIELDS]
    return round(sum(vals) / len(vals), 2)


def _build_result_entry(
    result: dict,
    threshold: int,
    threshold_hair: Optional[int],
    threshold_naturalness: Optional[int],
) -> dict:
    """Build a single result entry for the JSON report."""
    scores = result.get("scores")
    passed = is_pass(
        scores, threshold, threshold_hair,
        threshold_naturalness,
    ) if scores else False

    # Determine filename with extension
    filename = result["filename"]
    # Try to find original extension from input path if available
    if "input_path" in result:
        filename = Path(result["input_path"]).name
    elif not Path(filename).suffix:
        filename = filename + ".jpg"  # fallback

    return {
        "filename": filename,
        "pass": passed,
        "scores": scores if scores else {},
        "reason": result.get("reason", ""),
        "error": result.get("error", False),
    }


def generate_json_report(
    results: list[dict],
    metadata: dict,
    output_path: Path,
) -> Path:
    """Generate detailed JSON report."""
    report = {
        "metadata": metadata,
        "statistics": _compute_statistics(results),
        "results": results,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    logger.info(f"JSON report saved: {output_path}")
    return output_path


def generate_csv_report(
    results: list[dict],
    output_path: Path,
) -> Path:
    """Generate CSV summary report."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["filename", "pass"] + SCORE_FIELDS + ["reason"]
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            row = {
                "filename": r["filename"],
                "pass": r.get("pass", False),
                "reason": r.get("reason", ""),
            }
            scores = r.get("scores", {})
            for field in SCORE_FIELDS:
                row[field] = scores.get(field, "")
            writer.writerow(row)

    logger.info(f"CSV report saved: {output_path}")
    return output_path


def generate_html_report(
    results: list[dict],
    metadata: dict,
    statistics: dict,
    entries_map: dict,
    output_path: Path,
    max_passed_display: int = 20,
) -> Path:
    """Generate HTML dashboard report.

    Args:
        results: list of result dicts with pass/fail status
        metadata: report metadata dict
        statistics: per-field statistics dict
        entries_map: {stem: entry_dict} mapping for image paths
        output_path: where to write HTML
        max_passed_display: max number of passed samples to show in gallery
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Separate passed/failed
    failed_items = []
    passed_items = []
    error_items = []

    for r in results:
        if r.get("error"):
            error_items.append(r)
            continue

        stem = Path(r["filename"]).stem
        entry = entries_map.get(stem)

        # Build gallery item
        item = {
            "filename": r["filename"],
            "scores": r.get("scores", {}),
            "reason": r.get("reason", ""),
            "avg_score": _avg_score(r.get("scores")),
            "images": None,
        }

        if entry:
            item["images"] = {
                "input": _image_to_base64(entry["input"]),
                "reference": _image_to_base64(entry["reference"]),
                "output": _image_to_base64(entry["output"]),
            }

        if r.get("pass"):
            passed_items.append(item)
        else:
            failed_items.append(item)

    # Sort failed by avg score ascending (worst first)
    failed_items.sort(key=lambda x: x["avg_score"])

    # Sort passed by avg score descending (best first), take top N
    passed_items.sort(key=lambda x: x["avg_score"], reverse=True)
    passed_display = passed_items[:max_passed_display]

    # Histogram data
    histogram = _compute_histogram(results)

    # Elapsed time formatting
    elapsed = metadata.get("elapsed_time_sec", 0)
    hours, remainder = divmod(int(elapsed), 3600)
    minutes, seconds = divmod(remainder, 60)
    metadata["elapsed_time_str"] = f"{hours}h {minutes}m {seconds}s"

    # Render template
    env = Environment(loader=FileSystemLoader(str(TEMPLATE_DIR)))
    template = env.get_template("summary.html")

    html = template.render(
        metadata=metadata,
        statistics_json=json.dumps(statistics),
        histogram_json=json.dumps(histogram),
        failed_samples=failed_items,
        passed_samples=passed_display,
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    logger.info(f"HTML report saved: {output_path}")
    return output_path


def generate_html_scroll_report(
    results: list[dict],
    metadata: dict,
    statistics: dict,
    image_dirs: dict,
    output_path: Path,
) -> Path:
    """Generate lightweight HTML report with infinite scroll (no base64 images).

    Args:
        results: list of result dicts with pass/fail status
        metadata: report metadata dict
        statistics: per-field statistics dict
        image_dirs: {"input": str, "reference": str, "output": str} absolute paths
        output_path: where to write HTML
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    histogram = _compute_histogram(results)

    elapsed = metadata.get("elapsed_time_sec", 0)
    hours, remainder = divmod(int(elapsed), 3600)
    minutes, seconds = divmod(remainder, 60)
    metadata["elapsed_time_str"] = f"{hours}h {minutes}m {seconds}s"

    env = Environment(loader=FileSystemLoader(str(TEMPLATE_DIR)))
    template = env.get_template("summary_scroll.html")

    html = template.render(
        metadata=metadata,
        results_json=json.dumps(results, ensure_ascii=False),
        statistics_json=json.dumps(statistics),
        histogram_json=json.dumps(histogram),
        image_dirs_json=json.dumps(image_dirs, ensure_ascii=False),
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    logger.info(f"HTML scroll report saved: {output_path}")
    return output_path


def generate_all_reports(
    results: list[dict],
    metadata: dict,
    entries_map: dict,
    report_dir: Path,
    threshold: int = 7,
    threshold_hair: Optional[int] = None,
    threshold_naturalness: Optional[int] = None,
    image_dirs: Optional[dict] = None,
) -> dict:
    """Generate all report formats (JSON, CSV, HTML).

    Args:
        image_dirs: {"input": str, "reference": str, "output": str} absolute paths.
                    If provided, generates scroll-based HTML report (file paths, infinite scroll).
                    If None, generates legacy base64-embedded HTML report.

    Returns dict with paths to generated files.
    """
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    # Build final results with pass/fail status
    final_results = []
    for r in results:
        entry = _build_result_entry(
            r, threshold, threshold_hair,
            threshold_naturalness,
        )
        final_results.append(entry)

    # Compute stats
    statistics = _compute_statistics(final_results)

    # Count pass/fail
    passed = sum(1 for r in final_results if r.get("pass"))
    failed = sum(1 for r in final_results if not r.get("pass") and not r.get("error"))
    errors = sum(1 for r in final_results if r.get("error"))
    total = len(final_results)

    metadata.update({
        "total_samples": total,
        "passed": passed,
        "failed": failed,
        "errors": errors,
        "pass_rate": round(passed / total * 100, 1) if total > 0 else 0,
        "threshold": threshold,
    })

    # Generate reports
    json_path = generate_json_report(
        final_results, metadata, report_dir / "results.json"
    )
    csv_path = generate_csv_report(
        final_results, report_dir / "results.csv"
    )

    # Use scroll-based HTML if image_dirs provided, otherwise legacy base64
    if image_dirs:
        html_path = generate_html_scroll_report(
            final_results, metadata, statistics, image_dirs,
            report_dir / "summary.html",
        )
    else:
        html_path = generate_html_report(
            final_results, metadata, statistics, entries_map,
            report_dir / "summary.html",
        )

    return {
        "json": json_path,
        "csv": csv_path,
        "html": html_path,
        "metadata": metadata,
    }
