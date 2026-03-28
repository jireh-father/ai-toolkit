"""Face identity preservation validator (INPUT vs OUTPUT only).

Compares input and output images to determine if face identity is
consistent after hair transfer. Returns true/false per sample.

Usage:
    python dataset_validator/validate_dataset_preservation.py \
        --input-dir ./data/input \
        --output-dir ./data/output \
        --model qwen3-vl-8b \
        --report-dir ./reports_preservation
"""

import argparse
import csv
import json
import logging
import multiprocessing as mp
import random
import re
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from PIL import Image

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}

PRESERVATION_PROMPT = """You are a face identity consistency checker for hair transfer models.

Above you see two images: [INPUT] and [OUTPUT].
- INPUT = original person BEFORE hair editing.
- OUTPUT = result AFTER applying a new hairstyle onto the INPUT person.

The hair WILL be different — that is expected and intentional. Ignore all hair differences completely.

TASK: Determine whether the face identity is consistent between INPUT and OUTPUT.

Focus ONLY on face identity:
- Is it the same person? Same facial structure, same eyes, nose, mouth, jawline?
- Are the core facial features preserved (not distorted, not a different person)?

IMPORTANT — ignore ALL of the following:
- Any hair differences (color, length, style, volume, etc.)
- Parts of the face (forehead, ears, temples, sideburns area) that become newly visible or hidden due to the hairstyle change — this is completely normal and expected
- Minor lighting or color tone shifts
- Background changes
- Clothing or accessory differences
- Slight expression changes

Answer true if the face identity is clearly the same person.
Answer false ONLY if the face is severely distorted, disfigured, or looks like a completely different person.

Respond with ONLY a JSON object, no other text:
{"match": true or false, "reason": "<1-2 sentences explaining your decision>"}"""


def setup_logging(level: str):
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def set_seed(seed: int):
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
    parser = argparse.ArgumentParser(
        description="Face Identity Preservation Validator (INPUT vs OUTPUT)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python dataset_validator/validate_dataset_preservation.py \\
      --input-dir ./input --output-dir ./output

  # Ollama backend
  python dataset_validator/validate_dataset_preservation.py \\
      --input-dir ./input --output-dir ./output \\
      --model qwen3.5-9b --backend ollama

  # Multi-GPU Ollama
  python dataset_validator/validate_dataset_preservation.py \\
      --input-dir ./input --output-dir ./output \\
      --backend ollama --ollama-urls http://localhost:11434,http://localhost:11435

  # Debug with 5 samples
  python dataset_validator/validate_dataset_preservation.py \\
      --input-dir ./input --output-dir ./output \\
      --max-samples 5 --log-level debug
""",
    )

    parser.add_argument("--input-dir", type=str, required=True,
                        help="Path to input images folder")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Path to output images folder")

    # Model settings
    parser.add_argument("--model", type=str, default="qwen3-vl-8b",
                        help="VLM model name (default: qwen3-vl-8b)")
    parser.add_argument("--quantization", type=str, default="int4",
                        choices=["int4", "int8", "fp16"])
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--low-vram", action="store_true")

    # Backend settings
    parser.add_argument("--backend", type=str, default="local",
                        choices=["local", "vllm", "ollama"])
    parser.add_argument("--vllm-url", type=str, default="http://localhost:8000")
    parser.add_argument("--ollama-url", type=str, default="http://localhost:11434")
    parser.add_argument("--ollama-urls", type=str, default=None,
                        help="Comma-separated Ollama server URLs for parallel processing. "
                             "Each URL runs in a separate process. "
                             "e.g. http://localhost:11434,http://localhost:11435")
    parser.add_argument("--ollama-port", type=int, default=None,
                        help="Ollama server port (overrides port in --ollama-url, default: 11434)")

    # Output settings
    parser.add_argument("--report-dir", type=str, default="./reports_preservation")
    parser.add_argument("--copy-images", action="store_true",
                        help="Copy images to report-dir/images/")

    # Checkpoint settings
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints_preservation")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint-interval", type=int, default=100)

    # Image settings
    parser.add_argument("--resize-short-side", type=int, default=512)

    # Misc
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-level", type=str, default="info",
                        choices=["debug", "info", "warning", "error"])

    args = parser.parse_args()
    if args.ollama_port is not None:
        args.ollama_url = f"http://localhost:{args.ollama_port}"
    return args


# ---------------------------------------------------------------------------
# Dataset scanning (2-dir: input + output only)
# ---------------------------------------------------------------------------

def scan_dataset_pair(input_dir: Path, output_dir: Path):
    """Scan input and output dirs, match by filename stem."""
    inp_stems = {}
    out_stems = {}

    for p in input_dir.iterdir():
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS:
            inp_stems[p.stem] = p
    for p in output_dir.iterdir():
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS:
            out_stems[p.stem] = p

    all_stems = sorted(set(inp_stems) | set(out_stems))
    matched = []
    mismatched = []

    for stem in all_stems:
        if stem in inp_stems and stem in out_stems:
            matched.append({
                "stem": stem,
                "input": inp_stems[stem],
                "output": out_stems[stem],
            })
        else:
            missing_in = []
            if stem not in inp_stems:
                missing_in.append("input")
            if stem not in out_stems:
                missing_in.append("output")
            mismatched.append({"stem": stem, "missing_in": missing_in})

    logger = logging.getLogger(__name__)
    logger.info(f"Dataset scan: input={len(inp_stems)}, output={len(out_stems)}")
    logger.info(f"Matched pairs: {len(matched)}")
    if mismatched:
        logger.warning(f"Mismatched: {len(mismatched)}")

    return matched, mismatched


def validate_images(matched):
    """Filter out corrupted images."""
    valid = []
    corrupted = []
    for entry in matched:
        bad = []
        for key in ("input", "output"):
            try:
                with Image.open(entry[key]) as img:
                    img.verify()
            except Exception:
                bad.append(key)
        if bad:
            corrupted.append({"stem": entry["stem"], "corrupted_in": bad})
        else:
            valid.append(entry)
    return valid, corrupted


def load_image_pair(entry: dict, short_side: int = 512):
    """Load and resize input + output images."""
    from dataset_validator.core.image_loader import resize_image
    images = []
    for key in ("input", "output"):
        try:
            img = Image.open(entry[key]).convert("RGB")
            img = resize_image(img, short_side)
            images.append(img)
        except Exception:
            return None
    return tuple(images)


# ---------------------------------------------------------------------------
# Evaluation (2-image, true/false)
# ---------------------------------------------------------------------------

def _parse_json_response(text: str) -> Optional[dict]:
    """Extract JSON from VLM response."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass
    return None


def _validate_simple_response(data: dict) -> Optional[dict]:
    """Validate true/false response."""
    if not isinstance(data, dict):
        return None
    match_val = data.get("match")
    if match_val is None:
        return None
    if isinstance(match_val, bool):
        return {"match": match_val, "reason": str(data.get("reason", ""))}
    if isinstance(match_val, str):
        lower = match_val.lower().strip()
        if lower in ("true", "yes", "1"):
            return {"match": True, "reason": str(data.get("reason", ""))}
        if lower in ("false", "no", "0"):
            return {"match": False, "reason": str(data.get("reason", ""))}
    return None


def _build_messages_2img_qwen(images, prompt):
    return [{
        "role": "user",
        "content": [
            {"type": "text", "text": "[INPUT] Original person before hair editing:"},
            {"type": "image", "image": images[0]},
            {"type": "text", "text": "[OUTPUT] Result after hair transfer:"},
            {"type": "image", "image": images[1]},
            {"type": "text", "text": prompt},
        ],
    }]


def _build_messages_2img_generic(images, prompt):
    return [{
        "role": "user",
        "content": [
            {"type": "text", "text": "[INPUT] Original person before hair editing:"},
            {"type": "image"},
            {"type": "text", "text": "[OUTPUT] Result after hair transfer:"},
            {"type": "image"},
            {"type": "text", "text": prompt},
        ],
    }]


def evaluate_single_preservation(vlm: dict, images, max_retries: int = 3):
    """Evaluate a single input/output pair. Returns {"match": bool, "reason": str} or None."""
    import torch

    model = vlm["model"]
    processor = vlm["processor"]
    family = vlm["family"]
    logger = logging.getLogger(__name__)

    for attempt in range(max_retries):
        inputs = None
        generated_ids = None
        try:
            if family in ("qwen2_vl", "qwen3_5"):
                messages = _build_messages_2img_qwen(images, PRESERVATION_PROMPT)
                from qwen_vl_utils import process_vision_info
                text = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                image_inputs, video_inputs = process_vision_info(messages)
                inputs = processor(
                    text=[text], images=image_inputs, videos=video_inputs,
                    padding=True, return_tensors="pt",
                ).to(model.device)
            else:
                messages = _build_messages_2img_generic(images, PRESERVATION_PROMPT)
                text = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                inputs = processor(
                    text=[text], images=list(images),
                    return_tensors="pt",
                ).to(model.device)

            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs, max_new_tokens=256,
                    do_sample=False, temperature=None, top_p=None,
                )

            input_len = inputs["input_ids"].shape[1]
            output_ids = generated_ids[:, input_len:]
            response_text = processor.batch_decode(
                output_ids, skip_special_tokens=True
            )[0]

            logger.debug(f"VLM response (attempt {attempt + 1}): {response_text[:200]}")

            parsed = _parse_json_response(response_text)
            if parsed is None:
                logger.warning(f"JSON parse failed (attempt {attempt + 1}/{max_retries})")
                continue

            validated = _validate_simple_response(parsed)
            if validated is None:
                logger.warning(f"Response validation failed (attempt {attempt + 1}/{max_retries})")
                continue

            return validated

        except Exception as e:
            logger.error(f"Evaluation error (attempt {attempt + 1}/{max_retries}): {e}")
            continue
        finally:
            del inputs, generated_ids
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    logger.error("All evaluation attempts failed")
    return None


def _evaluate_single_vllm_preservation(vllm_url, model_id, images, max_retries=3):
    """Evaluate via vLLM API with 2 images."""
    import base64
    import io
    import requests

    logger = logging.getLogger(__name__)

    def _img_to_b64(img):
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode()}"

    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": "[INPUT] Original person before hair editing:"},
            {"type": "image_url", "image_url": {"url": _img_to_b64(images[0])}},
            {"type": "text", "text": "[OUTPUT] Result after hair transfer:"},
            {"type": "image_url", "image_url": {"url": _img_to_b64(images[1])}},
            {"type": "text", "text": PRESERVATION_PROMPT},
        ],
    }]

    api_url = f"{vllm_url.rstrip('/')}/v1/chat/completions"

    for attempt in range(max_retries):
        try:
            resp = requests.post(api_url, json={
                "model": model_id, "messages": messages,
                "max_completion_tokens": 256, "temperature": 0,
            }, timeout=120)
            resp.raise_for_status()
            response_text = resp.json()["choices"][0]["message"]["content"]
            logger.debug(f"vLLM response (attempt {attempt + 1}): {response_text[:200]}")
            parsed = _parse_json_response(response_text)
            if parsed:
                validated = _validate_simple_response(parsed)
                if validated:
                    return validated
            logger.warning(f"Parse/validation failed (attempt {attempt + 1}/{max_retries})")
        except Exception as e:
            logger.error(f"vLLM error (attempt {attempt + 1}/{max_retries}): {e}")
    return None


def _evaluate_single_ollama_preservation(ollama_url, model_name, images, max_retries=3):
    """Evaluate via Ollama API with 2 images."""
    import base64
    import io
    import requests
    import yaml

    logger = logging.getLogger(__name__)

    # Resolve ollama tag
    config_path = Path(__file__).parent / "config" / "models.yaml"
    ollama_tag = model_name
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        if model_name in config["models"]:
            ollama_tag = config["models"][model_name].get("ollama_tag", model_name)
    except Exception:
        pass

    def _img_to_b64(img):
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode()}"

    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": "[INPUT] Original person before hair editing:"},
            {"type": "image_url", "image_url": {"url": _img_to_b64(images[0])}},
            {"type": "text", "text": "[OUTPUT] Result after hair transfer:"},
            {"type": "image_url", "image_url": {"url": _img_to_b64(images[1])}},
            {"type": "text", "text": PRESERVATION_PROMPT},
        ],
    }]

    api_url = f"{ollama_url.rstrip('/')}/v1/chat/completions"

    for attempt in range(max_retries):
        try:
            resp = requests.post(api_url, json={
                "model": ollama_tag, "messages": messages,
                "max_completion_tokens": 256, "temperature": 0, "stream": False,
            }, timeout=1000)
            resp.raise_for_status()
            response_text = resp.json()["choices"][0]["message"]["content"]
            logger.debug(f"Ollama response (attempt {attempt + 1}): {response_text[:200]}")
            parsed = _parse_json_response(response_text)
            if parsed:
                validated = _validate_simple_response(parsed)
                if validated:
                    return validated
            logger.warning(f"Parse/validation failed (attempt {attempt + 1}/{max_retries})")
        except Exception as e:
            logger.error(f"Ollama error (attempt {attempt + 1}/{max_retries}): {e}")
    return None


# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------

def _make_result(entry, match, reason, error):
    """Build a result dict with file names."""
    return {
        "filename": entry["stem"],
        "input_file": Path(entry["input"]).name,
        "output_file": Path(entry["output"]).name,
        "match": match,
        "reason": reason,
        "error": error,
    }


def _ollama_worker(worker_id, ollama_url, model_name, entries_with_idx,
                   short_side, result_queue):
    """Worker process for parallel Ollama evaluation."""
    wlogger = logging.getLogger(f"ollama-worker-{worker_id}")
    wlogger.info(f"Starting on {ollama_url}, {len(entries_with_idx)} samples")

    for idx, entry in entries_with_idx:
        pair = load_image_pair(entry, short_side=short_side)
        if pair is None:
            result = _make_result(entry, None, "Failed to load images", True)
        else:
            resp = _evaluate_single_ollama_preservation(ollama_url, model_name, pair)
            if resp is None:
                result = _make_result(entry, None, "Ollama evaluation failed", True)
            else:
                result = _make_result(entry, resp["match"], resp["reason"], False)
        result_queue.put((idx, result))

    wlogger.info("Done")


def run_evaluation(entries, args):
    """Run evaluation on all entries."""
    from tqdm import tqdm

    logger = logging.getLogger(__name__)
    results = []

    if args.backend == "local":
        from dataset_validator.core.evaluator import load_vlm
        vlm = load_vlm(
            args.model, quantization=args.quantization,
            device="cuda:0", low_vram=args.low_vram,
        )

        pbar = tqdm(entries, desc="Evaluating", unit="sample")
        for entry in pbar:
            pair = load_image_pair(entry, short_side=args.resize_short_side)
            if pair is None:
                results.append(_make_result(entry, None, "Failed to load images", True))
            else:
                resp = evaluate_single_preservation(vlm, pair, max_retries=3)
                if resp is None:
                    results.append(_make_result(entry, None, "VLM evaluation failed", True))
                else:
                    results.append(_make_result(entry, resp["match"], resp["reason"], False))
            pbar.set_postfix(
                done=len(results),
                preserved=sum(1 for r in results if r.get("match") is True),
            )

    elif args.backend == "vllm":
        from dataset_validator.core.evaluator import load_model_config
        from dataset_validator.core.vllm_client import check_server_health
        model_config = load_model_config(args.model)
        hf_id = model_config["hf_id"]
        if not check_server_health(args.vllm_url):
            raise ConnectionError(f"vLLM server not reachable at {args.vllm_url}")
        logger.info(f"Connected to vLLM server at {args.vllm_url}")

        pbar = tqdm(entries, desc="Evaluating (vLLM)", unit="sample")
        for entry in pbar:
            pair = load_image_pair(entry, short_side=args.resize_short_side)
            if pair is None:
                results.append(_make_result(entry, None, "Failed to load images", True))
            else:
                resp = _evaluate_single_vllm_preservation(args.vllm_url, hf_id, pair)
                if resp is None:
                    results.append(_make_result(entry, None, "vLLM evaluation failed", True))
                else:
                    results.append(_make_result(entry, resp["match"], resp["reason"], False))
            pbar.set_postfix(
                done=len(results),
                preserved=sum(1 for r in results if r.get("match") is True),
            )

    elif args.backend == "ollama":
        from dataset_validator.core.ollama_client import check_server_health

        # Determine Ollama URLs
        ollama_urls = [args.ollama_url]
        if args.ollama_urls:
            ollama_urls = [u.strip() for u in args.ollama_urls.split(",") if u.strip()]

        # Health check all servers
        for url in ollama_urls:
            if not check_server_health(url):
                raise ConnectionError(f"Ollama server not reachable at {url}")
            logger.info(f"Connected to Ollama server at {url}")

        if len(ollama_urls) <= 1:
            # Single-process mode
            pbar = tqdm(entries, desc="Evaluating (Ollama)", unit="sample")
            for entry in pbar:
                pair = load_image_pair(entry, short_side=args.resize_short_side)
                if pair is None:
                    results.append(_make_result(entry, None, "Failed to load images", True))
                else:
                    resp = _evaluate_single_ollama_preservation(
                        ollama_urls[0], args.model, pair,
                    )
                    if resp is None:
                        results.append(_make_result(entry, None, "Ollama evaluation failed", True))
                    else:
                        results.append(_make_result(entry, resp["match"], resp["reason"], False))
                pbar.set_postfix(
                    done=len(results),
                    preserved=sum(1 for r in results if r.get("match") is True),
                )
        else:
            # Multi-process mode: one process per Ollama URL
            num_workers = len(ollama_urls)
            logger.info(f"Launching {num_workers} parallel workers")

            # Split entries into chunks with original indices
            chunks = [[] for _ in range(num_workers)]
            for i, entry in enumerate(entries):
                chunks[i % num_workers].append((i, entry))

            ctx = mp.get_context("fork")
            result_queue = ctx.Queue()

            processes = []
            for worker_id in range(num_workers):
                p = ctx.Process(
                    target=_ollama_worker,
                    args=(worker_id, ollama_urls[worker_id], args.model,
                          chunks[worker_id], args.resize_short_side, result_queue),
                )
                p.start()
                processes.append(p)

            # Collect results with progress bar
            total = len(entries)
            indexed_results = []
            with tqdm(total=total, desc=f"Evaluating (Ollama x{num_workers})", unit="sample") as pbar:
                while len(indexed_results) < total:
                    idx, result = result_queue.get()
                    indexed_results.append((idx, result))
                    pbar.update(1)
                    pbar.set_postfix(
                        done=len(indexed_results),
                        preserved=sum(1 for _, r in indexed_results if r.get("match") is True),
                    )

            for p in processes:
                p.join()

            # Sort by original index to maintain order
            indexed_results.sort(key=lambda x: x[0])
            results = [r for _, r in indexed_results]

    return results


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_preservation_reports(results, metadata, entries_map, report_dir, image_dirs=None):
    """Generate JSON, CSV, and HTML reports for preservation results."""
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    total = len(results)
    preserved = sum(1 for r in results if r.get("match") is True)
    not_preserved = sum(1 for r in results if r.get("match") is False)
    errors = sum(1 for r in results if r.get("error"))

    metadata.update({
        "total_samples": total,
        "preserved": preserved,
        "not_preserved": not_preserved,
        "errors": errors,
        "preservation_rate": round(preserved / total * 100, 1) if total > 0 else 0,
    })

    # JSON report
    json_report = {"metadata": metadata, "results": results}
    json_path = report_dir / "results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_report, f, ensure_ascii=False, indent=2)

    # CSV report
    csv_path = report_dir / "results.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "match", "reason"])
        writer.writeheader()
        for r in results:
            writer.writerow({
                "filename": r["filename"],
                "match": r.get("match", ""),
                "reason": r.get("reason", ""),
            })

    # HTML report
    html_path = report_dir / "summary.html"
    _generate_preservation_html(results, metadata, image_dirs, html_path)

    logging.getLogger(__name__).info(f"Reports saved to {report_dir}")
    return {"json": json_path, "csv": csv_path, "html": html_path, "metadata": metadata}


def _generate_preservation_html(results, metadata, image_dirs, output_path):
    """Generate a self-contained HTML report for face identity preservation results."""
    elapsed = metadata.get("elapsed_time_sec", 0)
    hours, remainder = divmod(int(elapsed), 3600)
    minutes, seconds = divmod(remainder, 60)
    elapsed_str = f"{hours}h {minutes}m {seconds}s"

    results_json = json.dumps(results, ensure_ascii=False)
    image_dirs_json = json.dumps(image_dirs or {}, ensure_ascii=False)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Face Identity Preservation Report</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #f5f5f5; color: #333; }}
  .container {{ max-width: 1200px; margin: 0 auto; padding: 20px; }}
  h1 {{ text-align: center; margin-bottom: 10px; color: #1a1a2e; }}
  .meta {{ text-align: center; color: #666; margin-bottom: 30px; font-size: 14px; }}

  .dashboard {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 20px; margin-bottom: 30px; }}
  .card {{ background: #fff; border-radius: 12px; padding: 24px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }}
  .card h3 {{ margin-bottom: 12px; color: #1a1a2e; font-size: 16px; }}
  .stat-big {{ font-size: 48px; font-weight: 700; }}
  .stat-label {{ color: #666; font-size: 14px; }}
  .color-match {{ color: #2ecc71; }}
  .color-unmatch {{ color: #e74c3c; }}

  .chart-row {{ display: grid; grid-template-columns: 1fr; gap: 20px; margin-bottom: 30px; max-width: 400px; margin-left: auto; margin-right: auto; }}
  .chart-card {{ background: #fff; border-radius: 12px; padding: 24px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }}

  .controls {{ background: #fff; border-radius: 12px; padding: 16px 24px; margin-bottom: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); display: flex; gap: 16px; flex-wrap: wrap; align-items: center; }}
  .controls input, .controls select {{ padding: 8px 12px; border: 1px solid #ddd; border-radius: 6px; font-size: 14px; }}
  .controls input[type="text"] {{ width: 220px; }}
  .controls label {{ font-size: 14px; color: #666; }}
  .radio-group {{ display: flex; gap: 4px; }}
  .radio-group input[type="radio"] {{ display: none; }}
  .radio-group label {{ padding: 6px 16px; border: 1px solid #ddd; border-radius: 6px; font-size: 14px; color: #666; cursor: pointer; }}
  .radio-group input[type="radio"]:checked + label {{ background: #1a1a2e; color: #fff; border-color: #1a1a2e; }}

  .sample-card {{ background: #fff; border-radius: 12px; padding: 20px; margin-bottom: 16px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }}
  .sample-header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; }}
  .sample-filename {{ font-weight: 600; font-size: 16px; }}
  .sample-badge {{ padding: 4px 12px; border-radius: 20px; font-size: 12px; font-weight: 600; }}
  .badge-match {{ background: #d4edda; color: #155724; }}
  .badge-unmatch {{ background: #f8d7da; color: #721c24; }}
  .badge-error {{ background: #fff3cd; color: #856404; }}
  .image-row {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 12px; margin-bottom: 12px; }}
  .image-col {{ text-align: center; }}
  .image-col img {{ width: 100%; border-radius: 8px; border: 1px solid #eee; min-height: 150px; background: #f0f0f0; }}
  .image-col .label {{ font-size: 12px; color: #666; margin-top: 4px; font-weight: 600; }}
  .reason-text {{ font-size: 13px; color: #555; font-style: italic; padding: 8px 12px; background: #f8f9fa; border-radius: 6px; }}
  .sample-count {{ color: #666; font-size: 14px; margin-bottom: 16px; }}
</style>
</head>
<body>
<div class="container">
  <h1>Face Identity Preservation Report</h1>
  <div class="meta">
    Model: {metadata.get('model', '')} |
    Generated: {metadata.get('timestamp', '')} |
    Duration: {elapsed_str}
  </div>

  <div class="dashboard">
    <div class="card">
      <h3>Total Samples</h3>
      <div class="stat-big">{metadata['total_samples']}</div>
    </div>
    <div class="card">
      <h3>Preserved</h3>
      <div class="stat-big color-match">{metadata['preserved']}</div>
      <div class="stat-label">{metadata['preservation_rate']}%</div>
    </div>
    <div class="card">
      <h3>Not Preserved</h3>
      <div class="stat-big color-unmatch">{metadata['not_preserved']}</div>
      <div class="stat-label">{round(100 - metadata['preservation_rate'], 1)}%</div>
    </div>
    {"<div class='card'><h3>Errors</h3><div class='stat-big' style='color:#f39c12'>" + str(metadata['errors']) + "</div></div>" if metadata.get('errors', 0) > 0 else ""}
  </div>

  <div class="chart-row">
    <div class="chart-card">
      <h3>Preserved / Not Preserved</h3>
      <canvas id="pieChart"></canvas>
    </div>
  </div>

  <div class="controls">
    <div class="radio-group">
      <input type="radio" id="filterAll" name="filter" value="all" checked onchange="applyFilter()">
      <label for="filterAll">All</label>
      <input type="radio" id="filterMatch" name="filter" value="match" onchange="applyFilter()">
      <label for="filterMatch">Preserved</label>
      <input type="radio" id="filterUnmatch" name="filter" value="unmatch" onchange="applyFilter()">
      <label for="filterUnmatch">Not Preserved</label>
    </div>
    <div>
      <label>Search: </label>
      <input type="text" id="searchInput" placeholder="Filename..." oninput="applyFilter()">
    </div>
    <div>
      <label>Sort: </label>
      <select id="sortSelect" onchange="applyFilter()">
        <option value="name-asc">Filename (A -> Z)</option>
        <option value="name-desc">Filename (Z -> A)</option>
        <option value="unmatch-first">Not Preserved first</option>
        <option value="match-first">Preserved first</option>
      </select>
    </div>
  </div>

  <div>
    <div class="sample-count" id="galleryCount"></div>
    <div id="gallery"></div>
    <div id="sentinel" style="height:1px;"></div>
  </div>
</div>

<script>
const RESULTS = {results_json};
const IMAGE_DIRS = {image_dirs_json};
const preservedCount = {metadata['preserved']};
const notPreservedCount = {metadata['not_preserved']};

new Chart(document.getElementById('pieChart'), {{
  type: 'doughnut',
  data: {{ labels: ['Preserved', 'Not Preserved'], datasets: [{{ data: [preservedCount, notPreservedCount], backgroundColor: ['#2ecc71', '#e74c3c'] }}] }},
  options: {{ responsive: true, plugins: {{ legend: {{ position: 'bottom' }} }} }}
}});

const PAGE_SIZE = 10;
let filteredResults = [];
let displayedCount = 0;

function buildImagePath(dir, filename) {{
  const base = dir.replace(/\\\\/g, '/');
  if (IMAGE_DIRS._relative) return base + '/' + filename;
  return 'file:///' + base + '/' + filename;
}}

function renderCard(item) {{
  const isMatch = item.match === true;
  const isError = item.error === true;
  let badge;
  if (isError) badge = '<span class="sample-badge badge-error">ERROR</span>';
  else if (isMatch) badge = '<span class="sample-badge badge-match">PRESERVED</span>';
  else badge = '<span class="sample-badge badge-unmatch">NOT PRESERVED</span>';

  let imagesHtml = '';
  if (IMAGE_DIRS.input && IMAGE_DIRS.output) {{
    imagesHtml = `<div class="image-row">
      <div class="image-col"><img src="${{buildImagePath(IMAGE_DIRS.input, item.input_file)}}" alt="Input"><div class="label">INPUT</div></div>
      <div class="image-col"><img src="${{buildImagePath(IMAGE_DIRS.output, item.output_file)}}" alt="Output"><div class="label">OUTPUT</div></div>
    </div>`;
  }}

  const reasonHtml = item.reason ? `<div class="reason-text">${{item.reason}}</div>` : '';

  return `<div class="sample-card">
    <div class="sample-header"><span class="sample-filename">${{item.filename}}</span>${{badge}}</div>
    ${{imagesHtml}}${{reasonHtml}}
  </div>`;
}}

function loadMore() {{
  const gallery = document.getElementById('gallery');
  const end = Math.min(displayedCount + PAGE_SIZE, filteredResults.length);
  const fragment = document.createDocumentFragment();
  for (let i = displayedCount; i < end; i++) {{
    const div = document.createElement('div');
    div.innerHTML = renderCard(filteredResults[i]);
    fragment.appendChild(div.firstElementChild);
  }}
  gallery.appendChild(fragment);
  displayedCount = end;
  document.getElementById('galleryCount').textContent =
    `Showing ${{displayedCount}} of ${{filteredResults.length}} samples`;
}}

function applyFilter() {{
  const search = document.getElementById('searchInput').value.toLowerCase();
  const filter = document.querySelector('input[name="filter"]:checked').value;
  const sortBy = document.getElementById('sortSelect').value;

  filteredResults = RESULTS.filter(r => {{
    if (filter === 'match' && r.match !== true) return false;
    if (filter === 'unmatch' && r.match !== false) return false;
    if (search && !r.filename.toLowerCase().includes(search)) return false;
    return true;
  }});

  filteredResults.sort((a, b) => {{
    if (sortBy === 'name-asc') return a.filename.localeCompare(b.filename);
    if (sortBy === 'name-desc') return b.filename.localeCompare(a.filename);
    if (sortBy === 'unmatch-first') return (a.match === true ? 1 : 0) - (b.match === true ? 1 : 0);
    if (sortBy === 'match-first') return (a.match === true ? 0 : 1) - (b.match === true ? 0 : 1);
    return 0;
  }});

  document.getElementById('gallery').innerHTML = '';
  displayedCount = 0;
  loadMore();
}}

const sentinel = document.getElementById('sentinel');
new IntersectionObserver(entries => {{
  if (entries[0].isIntersecting) loadMore();
}}, {{ rootMargin: '400px 0px' }}).observe(sentinel);

applyFilter();
</script>
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    setup_logging(args.log_level)
    set_seed(args.seed)

    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("Face Identity Preservation Validator (INPUT vs OUTPUT)")
    logger.info("=" * 60)

    # Ensure package is importable
    pkg_dir = Path(__file__).resolve().parent
    if str(pkg_dir.parent) not in sys.path:
        sys.path.insert(0, str(pkg_dir.parent))

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    for name, d in [("input", input_dir), ("output", output_dir)]:
        if not d.is_dir():
            logger.error(f"{name} directory not found: {d}")
            sys.exit(1)

    # Step 1: Scan dataset
    logger.info("Step 1: Scanning dataset...")
    matched, mismatched = scan_dataset_pair(input_dir, output_dir)

    if not matched:
        logger.error("No matched image pairs found.")
        sys.exit(1)

    # Step 2: Filter corrupted
    logger.info("Step 2: Validating image integrity...")
    valid_entries, corrupted = validate_images(matched)

    if not valid_entries:
        logger.error("No valid image pairs after corruption check.")
        sys.exit(1)

    logger.info(f"Valid entries: {len(valid_entries)}")

    if args.max_samples and args.max_samples < len(valid_entries):
        valid_entries = valid_entries[:args.max_samples]
        logger.info(f"Limited to {args.max_samples} samples")

    # Step 3: Evaluate
    start_time = time.time()

    if args.backend == "vllm":
        logger.info(f"Model: {args.model}, Backend: vLLM ({args.vllm_url})")
    elif args.backend == "ollama":
        if args.ollama_urls:
            urls = [u.strip() for u in args.ollama_urls.split(",") if u.strip()]
            logger.info(f"Model: {args.model}, Backend: Ollama x{len(urls)} ({', '.join(urls)})")
        else:
            logger.info(f"Model: {args.model}, Backend: Ollama ({args.ollama_url})")
    else:
        logger.info(f"Model: {args.model}, Quantization: {args.quantization}")

    logger.info(f"Step 3: Evaluating {len(valid_entries)} samples...")
    results = run_evaluation(valid_entries, args)

    elapsed = time.time() - start_time
    logger.info(f"Evaluation completed in {elapsed:.1f}s")

    # Step 4: Generate reports
    logger.info("Step 4: Generating reports...")

    entries_map = {e["stem"]: e for e in valid_entries}

    metadata = {
        "model": args.model,
        "quantization": args.quantization,
        "backend": args.backend,
        "timestamp": datetime.now().isoformat(),
        "elapsed_time_sec": elapsed,
        "seed": args.seed,
        "resize_short_side": args.resize_short_side,
    }

    image_dirs = {
        "input": str(input_dir.resolve()),
        "output": str(output_dir.resolve()),
    }

    if args.copy_images:
        logger.info("Copying images to report directory...")
        report_images_dir = Path(args.report_dir) / "images"
        for role in ["input", "output"]:
            dest_dir = report_images_dir / role
            dest_dir.mkdir(parents=True, exist_ok=True)
        copied = 0
        for entry in valid_entries:
            for role in ["input", "output"]:
                src = Path(entry[role])
                dest = report_images_dir / role / src.name
                if not dest.exists():
                    try:
                        shutil.copy2(src, dest)
                        copied += 1
                    except Exception as e:
                        logger.warning(f"Failed to copy {src}: {e}")
        logger.info(f"Copied {copied} images")
        image_dirs = {
            "input": "images/input",
            "output": "images/output",
            "_relative": True,
        }

    report_paths = generate_preservation_reports(
        results, metadata, entries_map,
        report_dir=Path(args.report_dir),
        image_dirs=image_dirs,
    )

    # Save mismatch report
    if mismatched or corrupted:
        mismatch_path = Path(args.report_dir) / "mismatch_report.json"
        with open(mismatch_path, "w", encoding="utf-8") as f:
            json.dump({
                "mismatched_files": mismatched,
                "corrupted_files": corrupted,
            }, f, ensure_ascii=False, indent=2, default=str)

    # Summary
    meta = report_paths["metadata"]
    logger.info("=" * 60)
    logger.info("Validation Complete!")
    logger.info(f"  Total:          {meta['total_samples']}")
    logger.info(f"  Preserved:      {meta['preserved']} ({meta['preservation_rate']}%)")
    logger.info(f"  Not Preserved:  {meta['not_preserved']}")
    if meta.get("errors", 0) > 0:
        logger.info(f"  Errors:         {meta['errors']}")
    logger.info(f"  Reports:        {Path(args.report_dir).resolve()}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
