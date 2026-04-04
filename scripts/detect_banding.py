"""
Detect banding/posterization artifacts in AI-generated images.

Uses multiple detection techniques optimized for AI-generated hair images:
  1. Gradient bimodality in smooth regions (flat zones + step jumps)
  2. False contour detection (connected edges in smooth gradient areas)
  3. Smooth-region residual analysis (blur-difference structure)
  4. Local effective bit-depth analysis
  5. Color channel banding (Lab space chrominance steps)

Usage:
    python scripts/detect_banding.py --input <dir> [--threshold 0.3] [--output-dir <dir>]

Examples:
    # Report only
    python scripts/detect_banding.py --input data/hair_output

    # Report + separate files
    python scripts/detect_banding.py --input data/hair_output --output-dir data/banding_result
"""

import argparse
import base64
import io
import json
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}


def _load_image(image_path: Path):
    """Load image handling non-ASCII paths (Korean filenames on Windows)."""
    buf = np.fromfile(str(image_path), dtype=np.uint8)
    color = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    gray = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    return color, gray


def _get_smooth_gradient_mask(gray: np.ndarray, kernel: int = 15,
                               low: float = 3.0, high: float = 25.0) -> np.ndarray:
    """Find smooth gradient regions: areas with gradual intensity change.

    These are the regions where banding would be visible - not flat backgrounds,
    not textures, not sharp edges. Just smooth gradients.
    """
    img_f = gray.astype(np.float32)
    blur = cv2.GaussianBlur(img_f, (kernel, kernel), 0)
    local_std = np.sqrt(cv2.GaussianBlur((img_f - blur) ** 2, (kernel, kernel), 0))
    return (local_std > low) & (local_std < high)


def detect_gradient_bimodality(gray: np.ndarray) -> dict:
    """Detect banding's signature: alternating flat zones and step jumps.

    In smooth gradient regions, natural images have mostly small (1-2 level)
    gradients. Banding creates a bimodal pattern: many zero-gradient pixels
    (flat zones within a band) interspersed with small jumps (band boundaries).

    Key metric: the product of near-zero ratio and step ratio captures this
    bimodal distribution.
    """
    img_f = gray.astype(np.float32)
    sg_mask = _get_smooth_gradient_mask(gray)

    # Vertical gradients in smooth regions
    grad_y = np.abs(np.diff(img_f, axis=0))
    mask_y = sg_mask[:-1, :]
    sg_grads = grad_y[mask_y]

    if len(sg_grads) < 500:
        return {"bimodality": 0.0, "near_zero_ratio": 0.0,
                "step_ratio": 0.0, "smooth_gradient_area": 0.0}

    # Near-zero: pixels within a flat band (gradient < 0.5)
    near_zero = float((sg_grads < 0.5).mean())
    # Step jumps: band boundaries (gradient 2-8, distinct from JPEG noise ~1)
    steps = float(((sg_grads >= 2.0) & (sg_grads <= 8.0)).mean())
    # Large steps (edge-like, not banding)
    large = float((sg_grads > 15.0).mean())

    # Bimodality: high when both flat zones AND steps are present
    bimodality = near_zero * steps

    # Staircase score: step jumps that are NOT near real edges
    # Penalize if most energy is in large gradients (= real texture/edge)
    if large > 0.1:
        bimodality *= 0.5  # discount if many real edges in "smooth" area

    return {
        "bimodality": round(bimodality, 6),
        "near_zero_ratio": round(near_zero, 4),
        "step_ratio": round(steps, 4),
        "smooth_gradient_area": round(float(sg_mask.mean()), 4),
    }


def detect_false_contours(gray: np.ndarray) -> dict:
    """Detect connected false contour lines in smooth gradient regions.

    Banding creates visible contour lines (like topographic maps) in areas
    that should be smooth gradients. We detect these by finding Canny edges
    in heavily blurred smooth regions — real edges disappear after blurring
    but banding contours persist because they span larger areas.
    """
    img_f = gray.astype(np.float32)
    sg_mask = _get_smooth_gradient_mask(gray)

    # Light blur to reduce JPEG noise but preserve banding structure
    blurred = cv2.GaussianBlur(img_f, (5, 5), 1.0)

    # Gradient magnitude in blurred image
    grad_x = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)

    # False contours: edges in smooth regions with moderate gradient
    # (not too weak = noise, not too strong = real edges)
    false_contour_mask = sg_mask & (grad_mag > 2.0) & (grad_mag < 20.0)

    # Count connected components — banding creates long connected contour lines
    fc_uint8 = false_contour_mask.astype(np.uint8) * 255
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(fc_uint8, connectivity=8)

    # Filter: only count components with significant length (area/width ratio)
    long_contours = 0
    long_contour_pixels = 0
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        w = max(1, stats[i, cv2.CC_STAT_WIDTH])
        h = max(1, stats[i, cv2.CC_STAT_HEIGHT])
        aspect = max(w, h) / min(w, h) if min(w, h) > 0 else 1
        # Long, thin components = contour lines
        if area > 20 and aspect > 3:
            long_contours += 1
            long_contour_pixels += area

    smooth_area = max(1, int(sg_mask.sum()))
    contour_density = long_contour_pixels / smooth_area

    return {
        "contour_density": round(contour_density, 6),
        "long_contours": long_contours,
    }


def detect_residual_structure(gray: np.ndarray) -> dict:
    """Detect banding via blur-difference residual analysis.

    Heavily blur the image and subtract from original. In smooth regions:
    - Clean image: residual is random noise (low structure)
    - Banding image: residual shows staircase pattern (high structure)

    We measure structure as the ratio of gradient energy in residual
    vs the residual magnitude itself.
    """
    img_f = gray.astype(np.float32)
    sg_mask = _get_smooth_gradient_mask(gray)

    # Heavy blur to get "ideal" smooth version
    heavy_blur = cv2.GaussianBlur(img_f, (21, 21), 5.0)
    residual = img_f - heavy_blur

    # Residual in smooth regions
    res_smooth = residual[sg_mask]
    if len(res_smooth) < 500:
        return {"residual_structure": 0.0, "residual_std": 0.0}

    res_std = float(np.std(res_smooth))

    # Structure: gradient of residual (banding residual has organized gradient)
    res_grad_y = np.abs(np.diff(residual, axis=0))
    res_grad_x = np.abs(np.diff(residual, axis=1))

    # Mean gradient of residual in smooth regions
    rg_y = res_grad_y[sg_mask[:-1, :]] if sg_mask[:-1, :].sum() > 100 else np.array([0])
    rg_x = res_grad_x[sg_mask[:, :-1]] if sg_mask[:, :-1].sum() > 100 else np.array([0])

    res_grad_mean = float(np.mean(rg_y) + np.mean(rg_x)) / 2

    # Structure ratio: if residual has high gradient relative to its std,
    # it's organized (banding) rather than random (noise)
    if res_std > 0.1:
        structure = res_grad_mean / res_std
    else:
        structure = 0.0

    return {
        "residual_structure": round(structure, 4),
        "residual_std": round(res_std, 4),
    }


def detect_effective_bitdepth(gray: np.ndarray) -> dict:
    """Detect posterization via local effective bit-depth analysis.

    In smooth gradient patches, count unique intensity levels relative to
    the intensity range. Banding reduces effective bit-depth because smooth
    gradients get quantized into fewer levels.
    """
    h, w = gray.shape
    patch_sz = 16  # smaller patches for finer detection
    stride = 8

    low_depth_count = 0
    analyzed_patches = 0
    depth_ratios = []

    for y in range(0, h - patch_sz, stride):
        for x in range(0, w - patch_sz, stride):
            patch = gray[y:y + patch_sz, x:x + patch_sz]
            prange = int(patch.max()) - int(patch.min())

            # Only analyze gradient patches (not flat, not textured)
            if prange < 8 or prange > 100:
                continue

            # Check if this is a smooth gradient (low local variance)
            p_std = float(np.std(patch.astype(np.float32)))
            if p_std < 2 or p_std > 30:
                continue

            analyzed_patches += 1
            unique = len(np.unique(patch))

            # Expected: for a range of N, smooth gradient should use ~N/2 levels
            expected = max(4, prange // 2)
            ratio = unique / expected

            depth_ratios.append(ratio)

            if unique < expected * 0.4:
                low_depth_count += 1

    if analyzed_patches < 10:
        return {"low_bitdepth_ratio": 0.0, "avg_depth_ratio": 1.0,
                "analyzed_patches": 0}

    return {
        "low_bitdepth_ratio": round(low_depth_count / analyzed_patches, 4),
        "avg_depth_ratio": round(float(np.mean(depth_ratios)), 4),
        "analyzed_patches": analyzed_patches,
    }


def detect_color_banding(color_bgr: np.ndarray) -> dict:
    """Detect color banding in Lab chrominance channels.

    Color banding in AI-generated images often manifests as quantized
    chrominance (a, b channels) in smooth gradient regions.
    Focuses on step jumps in the 2-6 range (above JPEG noise, below real edges)
    within smooth gradient regions only.
    """
    lab = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2LAB)
    gray = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2GRAY)
    sg_mask = _get_smooth_gradient_mask(gray)

    scores = {}
    for i, name in enumerate(["a", "b"]):
        ch = lab[:, :, i + 1].astype(np.float32)
        grad_y = np.abs(np.diff(ch, axis=0))

        mask_y = sg_mask[:-1, :]
        ch_grads = grad_y[mask_y]

        if len(ch_grads) < 500:
            scores[f"color_step_{name}"] = 0.0
            continue

        # Step jumps distinguishable from JPEG noise
        steps = float(((ch_grads >= 2.0) & (ch_grads <= 6.0)).mean())
        scores[f"color_step_{name}"] = round(steps, 6)

    # Combined color banding score
    scores["color_banding_score"] = round(
        (scores.get("color_step_a", 0) + scores.get("color_step_b", 0)) / 2, 6
    )
    return scores


def compute_banding_score(image_path: Path) -> dict:
    """Run all banding detection methods and compute combined score."""
    color, gray = _load_image(image_path)
    if gray is None:
        return {"error": f"Cannot read image: {image_path.name}"}

    # Run all detectors
    bimod = detect_gradient_bimodality(gray)
    fc = detect_false_contours(gray)
    resid = detect_residual_structure(gray)
    bitdepth = detect_effective_bitdepth(gray)
    color_band = detect_color_banding(color)

    # Normalize each detector to 0-1 range with calibrated multipliers
    # Calibration based on: banding images typically show
    #   bimodality: 0.02-0.09, clean images: <0.01
    #   contour_density: banding creates more long contours
    #   residual_structure: higher for organized banding patterns
    #   low_bitdepth_ratio: >0 when posterization is present
    #   color_banding_score: chrominance steps in smooth regions

    n_bimod = min(1.0, bimod["bimodality"] / 0.08)
    n_contour = min(1.0, fc["contour_density"] / 0.10)
    n_resid = min(1.0, max(0, resid["residual_structure"] - 0.3) / 0.7)
    n_bitdepth = min(1.0, bitdepth["low_bitdepth_ratio"] / 0.3)
    n_color = min(1.0, color_band["color_banding_score"] / 0.12)

    # Combined score with weights
    score = float(np.clip(
        0.30 * n_bimod
        + 0.25 * n_contour
        + 0.20 * n_resid
        + 0.10 * n_bitdepth
        + 0.15 * n_color,
        0, 1
    ))

    return {
        "banding_score": round(score, 4),
        # Normalized detector scores
        "gradient_bimodality": round(n_bimod, 4),
        "false_contour": round(n_contour, 4),
        "residual_structure": round(n_resid, 4),
        "effective_bitdepth": round(n_bitdepth, 4),
        "color_banding": round(n_color, 4),
        # Raw details
        "details": {**bimod, **fc, **resid, **bitdepth, **color_band},
    }


def image_to_base64_thumbnail(path: Path, max_size: int = 256) -> str:
    """Load image, resize for thumbnail, return base64 string."""
    try:
        img = Image.open(path)
        img.thumbnail((max_size, max_size))
        if img.mode == "RGBA":
            img = img.convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=80)
        return base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception:
        return ""


SCORE_FIELDS = [
    ("gradient_bimodality", "Gradient Bimodality"),
    ("false_contour", "False Contour"),
    ("residual_structure", "Residual Structure"),
    ("effective_bitdepth", "Effective Bitdepth"),
    ("color_banding", "Color Banding"),
]


def generate_html_report(results: list[dict], metadata: dict, output_path: Path):
    """Generate standalone HTML report with Chart.js dashboard."""

    passed = [r for r in results if not r.get("error") and not r["has_banding"]]
    failed = [r for r in results if not r.get("error") and r["has_banding"]]
    errors = [r for r in results if r.get("error")]

    bin_labels = [f"{i/10:.1f}-{(i+1)/10:.1f}" for i in range(10)]
    bin_counts = [0] * 10
    for r in results:
        if not r.get("error"):
            idx = min(9, int(r["banding_score"] * 10))
            bin_counts[idx] += 1

    failed.sort(key=lambda x: x["banding_score"], reverse=True)
    passed.sort(key=lambda x: x["banding_score"], reverse=True)

    scores = [r["banding_score"] for r in results if not r.get("error")]
    avg_score = round(sum(scores) / len(scores), 4) if scores else 0
    std_score = round(float(np.std(scores)), 4) if scores else 0

    detector_avgs = {}
    for key, label in SCORE_FIELDS:
        vals = [r[key] for r in results if not r.get("error") and key in r]
        detector_avgs[label] = round(sum(vals) / len(vals), 4) if vals else 0

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Banding Artifact Detection Report</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #f5f5f5; color: #333; }}
  .container {{ max-width: 1400px; margin: 0 auto; padding: 20px; }}
  h1 {{ text-align: center; margin-bottom: 10px; color: #1a1a2e; }}
  .meta {{ text-align: center; color: #666; margin-bottom: 30px; font-size: 14px; }}
  .dashboard {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-bottom: 30px; }}
  .card {{ background: #fff; border-radius: 12px; padding: 24px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }}
  .card h3 {{ margin-bottom: 12px; color: #1a1a2e; font-size: 16px; }}
  .stat-big {{ font-size: 48px; font-weight: 700; }}
  .stat-label {{ color: #666; font-size: 14px; }}
  .pass-rate {{ color: #2ecc71; }}
  .fail-rate {{ color: #e74c3c; }}
  .chart-row {{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 20px; margin-bottom: 30px; }}
  .chart-card {{ background: #fff; border-radius: 12px; padding: 24px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }}
  canvas {{ max-height: 350px; }}
  .controls {{ background: #fff; border-radius: 12px; padding: 16px 24px; margin-bottom: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); display: flex; gap: 16px; flex-wrap: wrap; align-items: center; }}
  .controls input, .controls select {{ padding: 8px 12px; border: 1px solid #ddd; border-radius: 6px; font-size: 14px; }}
  .controls input[type="text"] {{ width: 220px; }}
  .controls label {{ font-size: 14px; color: #666; }}
  .gallery-section {{ margin-bottom: 40px; }}
  .gallery-section h2 {{ margin-bottom: 16px; color: #1a1a2e; }}
  .sample-count {{ color: #666; font-size: 14px; margin-bottom: 16px; }}
  .sample-card {{ background: #fff; border-radius: 12px; padding: 20px; margin-bottom: 16px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }}
  .sample-header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; }}
  .sample-filename {{ font-weight: 600; font-size: 16px; word-break: break-all; }}
  .sample-badge {{ padding: 4px 12px; border-radius: 20px; font-size: 12px; font-weight: 600; }}
  .badge-pass {{ background: #d4edda; color: #155724; }}
  .badge-fail {{ background: #f8d7da; color: #721c24; }}
  .badge-error {{ background: #fff3cd; color: #856404; }}
  .sample-body {{ display: flex; gap: 20px; align-items: flex-start; }}
  .sample-thumb {{ flex-shrink: 0; }}
  .sample-thumb img {{ width: 200px; border-radius: 8px; border: 1px solid #eee; }}
  .sample-thumb img.lazy {{ opacity: 0; transition: opacity 0.3s; }}
  .sample-thumb img.loaded {{ opacity: 1; }}
  .scores-detail {{ flex: 1; }}
  .score-bar-container {{ margin-bottom: 8px; }}
  .score-bar-label {{ display: flex; justify-content: space-between; font-size: 13px; margin-bottom: 4px; }}
  .score-bar {{ height: 8px; background: #e9ecef; border-radius: 4px; overflow: hidden; }}
  .score-bar-fill {{ height: 100%; border-radius: 4px; transition: width 0.3s; }}
  @media (max-width: 768px) {{
    .chart-row {{ grid-template-columns: 1fr; }}
    .sample-body {{ flex-direction: column; }}
    .sample-thumb img {{ width: 100%; }}
  }}
</style>
</head>
<body>
<div class="container">
  <h1>Banding / Posterization Artifact Detection Report</h1>
  <div class="meta">
    Threshold: {metadata['threshold']} |
    Generated: {metadata['timestamp']} |
    Duration: {metadata['elapsed']} |
    Input: {metadata['input_dir']}
  </div>
  <div class="dashboard">
    <div class="card"><h3>Total</h3><div class="stat-big">{metadata['total']}</div></div>
    <div class="card"><h3>Clean</h3><div class="stat-big pass-rate">{len(passed)}</div>
      <div class="stat-label">{len(passed)/max(1,metadata['total'])*100:.1f}%</div></div>
    <div class="card"><h3>Banding</h3><div class="stat-big fail-rate">{len(failed)}</div>
      <div class="stat-label">{len(failed)/max(1,metadata['total'])*100:.1f}%</div></div>
    <div class="card"><h3>Avg Score</h3><div class="stat-big" style="font-size:36px">{avg_score}</div>
      <div class="stat-label">std: {std_score}</div></div>
    {"" if not errors else f'<div class="card"><h3>Errors</h3><div class="stat-big" style="color:#f39c12">{len(errors)}</div></div>'}
  </div>
  <div class="chart-row">
    <div class="chart-card"><h3>Clean / Banding Ratio</h3><canvas id="pieChart"></canvas></div>
    <div class="chart-card"><h3>Score Distribution</h3><canvas id="histChart"></canvas></div>
    <div class="chart-card"><h3>Detector Averages</h3><canvas id="radarChart"></canvas></div>
  </div>
  <div class="controls">
    <div><label>Search: </label><input type="text" id="searchInput" placeholder="Filename..." oninput="filterGallery()"></div>
    <div><label>Filter: </label><select id="filterSelect" onchange="filterGallery()">
      <option value="all">All</option><option value="banding">Banding Only</option><option value="clean">Clean Only</option></select></div>
    <div><label>Sort: </label><select id="sortSelect" onchange="sortGallery()">
      <option value="score-desc">Score (High to Low)</option><option value="score-asc">Score (Low to High)</option>
      <option value="name-asc">Filename (A to Z)</option><option value="name-desc">Filename (Z to A)</option></select></div>
  </div>
  <div class="gallery-section">
    <h2>All Samples</h2>
    <div class="sample-count" id="sampleCount"></div>
    <div id="gallery">
"""

    all_items = failed + passed
    colors_map = ["#e74c3c", "#e67e22", "#9b59b6", "#3498db", "#1abc9c"]
    for r in all_items:
        if r.get("error"):
            continue
        status = "banding" if r["has_banding"] else "clean"
        badge_class = "badge-fail" if r["has_banding"] else "badge-pass"
        badge_text = "BANDING" if r["has_banding"] else "CLEAN"
        score = r["banding_score"]
        bar_color = f"hsl({max(0, 120 - int(score * 120))}, 70%, 50%)"

        thumb = r.get("thumbnail", "")
        thumb_html = ""
        if thumb:
            thumb_html = f'<div class="sample-thumb"><img class="lazy" data-src="data:image/jpeg;base64,{thumb}" alt="{r["filename"]}"></div>'

        bars_html = f"""
          <div class="score-bar-container">
            <div class="score-bar-label"><span><b>Combined Score</b></span><span style="font-weight:700">{score}</span></div>
            <div class="score-bar" style="height:10px"><div class="score-bar-fill" style="width:{score*100}%;background:{bar_color}"></div></div>
          </div>"""

        for idx, (key, label) in enumerate(SCORE_FIELDS):
            val = r.get(key, 0)
            c = colors_map[idx % len(colors_map)]
            bars_html += f"""
          <div class="score-bar-container">
            <div class="score-bar-label"><span>{label}</span><span>{val}</span></div>
            <div class="score-bar"><div class="score-bar-fill" style="width:{max(0, float(val))*100}%;background:{c}"></div></div>
          </div>"""

        html += f"""
    <div class="sample-card gallery-item" data-status="{status}" data-filename="{r['filename']}" data-score="{score}">
      <div class="sample-header">
        <span class="sample-filename">{r['filename']}</span>
        <span class="sample-badge {badge_class}">{badge_text}</span>
      </div>
      <div class="sample-body">{thumb_html}
        <div class="scores-detail">{bars_html}
        </div>
      </div>
    </div>
"""

    for r in errors:
        html += f"""
    <div class="sample-card gallery-item" data-status="error" data-filename="{r['filename']}" data-score="0">
      <div class="sample-header">
        <span class="sample-filename">{r['filename']}</span>
        <span class="sample-badge badge-error">ERROR</span>
      </div>
      <div style="color:#856404;font-size:13px;padding:8px 12px;background:#fff3cd;border-radius:6px;">{r['error']}</div>
    </div>
"""

    radar_labels = json.dumps([label for _, label in SCORE_FIELDS])
    radar_data = json.dumps([detector_avgs[label] for _, label in SCORE_FIELDS])

    html += f"""
    </div>
  </div>
</div>
<script>
const passCount = {len(passed)};
const failCount = {len(failed)};
const binLabels = {json.dumps(bin_labels)};
const binCounts = {json.dumps(bin_counts)};

new Chart(document.getElementById('pieChart'), {{
  type: 'doughnut',
  data: {{ labels: ['Clean', 'Banding'], datasets: [{{ data: [passCount, failCount], backgroundColor: ['#2ecc71', '#e74c3c'] }}] }},
  options: {{ responsive: true, plugins: {{ legend: {{ position: 'bottom' }} }} }}
}});

new Chart(document.getElementById('histChart'), {{
  type: 'bar',
  data: {{
    labels: binLabels,
    datasets: [{{
      label: 'Images', data: binCounts,
      backgroundColor: binCounts.map((_, i) => i < 3 ? '#2ecc7188' : i < 7 ? '#f39c1288' : '#e74c3c88'),
      borderColor: binCounts.map((_, i) => i < 3 ? '#2ecc71' : i < 7 ? '#f39c12' : '#e74c3c'),
      borderWidth: 1, borderRadius: 6,
    }}]
  }},
  options: {{ responsive: true, scales: {{ y: {{ beginAtZero: true }} }}, plugins: {{ legend: {{ display: false }} }} }}
}});

new Chart(document.getElementById('radarChart'), {{
  type: 'radar',
  data: {{
    labels: {radar_labels},
    datasets: [{{ label: 'Avg Score', data: {radar_data},
      backgroundColor: 'rgba(231, 76, 60, 0.2)', borderColor: '#e74c3c', pointBackgroundColor: '#e74c3c' }}]
  }},
  options: {{ responsive: true, scales: {{ r: {{ min: 0, max: 1, ticks: {{ stepSize: 0.2 }} }} }}, plugins: {{ legend: {{ display: false }} }} }}
}});

function filterGallery() {{
  const search = document.getElementById('searchInput').value.toLowerCase();
  const filter = document.getElementById('filterSelect').value;
  document.querySelectorAll('.gallery-item').forEach(el => {{
    const status = el.dataset.status;
    const name = el.dataset.filename.toLowerCase();
    let show = true;
    if (filter !== 'all' && status !== filter) show = false;
    if (search && !name.includes(search)) show = false;
    el.style.display = show ? '' : 'none';
  }});
  updateCount();
}}

function sortGallery() {{
  const sortBy = document.getElementById('sortSelect').value;
  const container = document.getElementById('gallery');
  const items = Array.from(container.querySelectorAll('.gallery-item'));
  items.sort((a, b) => {{
    if (sortBy === 'score-asc') return parseFloat(a.dataset.score) - parseFloat(b.dataset.score);
    if (sortBy === 'score-desc') return parseFloat(b.dataset.score) - parseFloat(a.dataset.score);
    if (sortBy === 'name-asc') return a.dataset.filename.localeCompare(b.dataset.filename);
    if (sortBy === 'name-desc') return b.dataset.filename.localeCompare(a.dataset.filename);
    return 0;
  }});
  items.forEach(item => container.appendChild(item));
}}

function updateCount() {{
  const visible = document.querySelectorAll('#gallery .gallery-item:not([style*="display: none"])');
  document.getElementById('sampleCount').textContent = 'Showing ' + visible.length + ' of {metadata["total"]} samples';
}}
updateCount();

const lazyObserver = new IntersectionObserver((entries) => {{
  entries.forEach(entry => {{
    if (entry.isIntersecting) {{
      const img = entry.target;
      img.src = img.dataset.src;
      img.classList.remove('lazy');
      img.classList.add('loaded');
      lazyObserver.unobserve(img);
    }}
  }});
}}, {{ rootMargin: '200px 0px' }});
document.querySelectorAll('img.lazy').forEach(img => lazyObserver.observe(img));
</script>
</body>
</html>"""

    output_path.write_text(html, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description="Detect banding/posterization artifacts in AI-generated images"
    )
    parser.add_argument("--input", "-i", required=True, help="Input image directory")
    parser.add_argument(
        "--threshold", "-t", type=float, default=0.3,
        help="Banding score threshold (0-1, default: 0.3)"
    )
    parser.add_argument(
        "--output-dir", "-o", default=None,
        help="If set, copy images into banding/ and clean/ subdirs"
    )
    parser.add_argument(
        "--no-thumbnail", action="store_true",
        help="Skip embedding thumbnails in HTML (faster, smaller file)"
    )
    args = parser.parse_args()

    input_dir = Path(args.input)
    if not input_dir.is_dir():
        print(f"Error: {input_dir} not found")
        sys.exit(1)

    image_paths = sorted([
        p for p in input_dir.iterdir()
        if p.suffix.lower() in IMAGE_EXTENSIONS
    ])
    if not image_paths:
        print(f"No images found in {input_dir}")
        sys.exit(1)

    print(f"Found {len(image_paths)} images in {input_dir}")
    print(f"Threshold: {args.threshold}")

    start_time = time.time()
    results = []

    for img_path in tqdm(image_paths, desc="Analyzing"):
        scores = compute_banding_score(img_path)
        entry = {
            "filename": img_path.name,
            "filepath": str(img_path.resolve()),
        }

        if "error" in scores:
            entry["error"] = scores["error"]
        else:
            entry.update(scores)
            entry["has_banding"] = scores["banding_score"] >= args.threshold

        if not args.no_thumbnail and "error" not in entry:
            entry["thumbnail"] = image_to_base64_thumbnail(img_path)

        results.append(entry)

    elapsed = time.time() - start_time
    elapsed_str = f"{int(elapsed // 60)}m {int(elapsed % 60)}s"

    banding_count = sum(1 for r in results if not r.get("error") and r.get("has_banding"))
    clean_count = sum(1 for r in results if not r.get("error") and not r.get("has_banding"))
    error_count = sum(1 for r in results if r.get("error"))

    print(f"\nResults: {clean_count} clean, {banding_count} banding, {error_count} errors")
    print(f"Elapsed: {elapsed_str}")

    metadata = {
        "input_dir": str(input_dir.resolve()),
        "threshold": args.threshold,
        "total": len(results),
        "banding": banding_count,
        "clean": clean_count,
        "errors": error_count,
        "elapsed": elapsed_str,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    json_results = [{k: v for k, v in r.items() if k != "thumbnail"} for r in results]
    json_path = input_dir / "banding_report.json"
    json_path.write_text(
        json.dumps({"metadata": metadata, "results": json_results}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"JSON report: {json_path}")

    html_path = input_dir / "banding_report.html"
    generate_html_report(results, metadata, html_path)
    print(f"HTML report: {html_path}")

    if args.output_dir:
        out = Path(args.output_dir)
        banding_dir = out / "banding"
        clean_dir = out / "clean"
        banding_dir.mkdir(parents=True, exist_ok=True)
        clean_dir.mkdir(parents=True, exist_ok=True)

        for r in tqdm(results, desc="Copying"):
            if r.get("error"):
                continue
            src = Path(r["filepath"])
            dst = banding_dir if r["has_banding"] else clean_dir
            shutil.copy2(src, dst / src.name)

        print(f"Copied to: {banding_dir} ({banding_count}), {clean_dir} ({clean_count})")


if __name__ == "__main__":
    main()
