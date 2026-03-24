"""
Naver hair style image downloader.

Scans JSON files under a source directory, collects style_img_url from style_items,
and downloads them as RGB JPEGs into the specified output directory.
Skips duplicates by pre-checking all URLs before downloading.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from urllib.parse import urlparse, unquote_to_bytes

import requests
from PIL import Image
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed


def collect_urls(source_dir: str) -> list[tuple[str, str]]:
    """Collect all (style_img_url, filename) pairs from JSON files, deduplicating by URL."""
    seen_urls = set()
    results = []

    for root, _, files in os.walk(source_dir):
        for fname in files:
            if not fname.endswith(".json"):
                continue
            json_path = os.path.join(root, fname)
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                print(f"[WARN] Failed to parse {json_path}: {e}")
                continue

            style_items = data.get("style_items", [])
            if not style_items:
                continue

            for item in style_items:
                url = item.get("style_img_url")
                if not url or url in seen_urls:
                    continue
                seen_urls.add(url)

                # Extract original filename from URL, decode percent-encoding
                # Naver uses EUC-KR for Korean filenames, try EUC-KR first then UTF-8
                parsed = urlparse(url)
                raw_name = os.path.basename(parsed.path)
                raw_bytes = unquote_to_bytes(raw_name)
                try:
                    original_name = raw_bytes.decode("euc-kr")
                except (UnicodeDecodeError, ValueError):
                    original_name = raw_bytes.decode("utf-8", errors="replace")
                # Force .jpg extension
                name_stem = os.path.splitext(original_name)[0]
                output_name = name_stem + ".jpg"
                results.append((url, output_name))

    return results


def download_and_save(url: str, output_path: str) -> bool:
    """Download image from URL and save as RGB JPEG."""
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        img = Image.open(BytesIO(resp.content))
        if img.mode != "RGB":
            img = img.convert("RGB")
        img.save(output_path, "JPEG", quality=95)
        return True
    except Exception as e:
        print(f"[ERROR] {url} -> {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Download Naver hair style images from crawled JSON data.")
    parser.add_argument("--source_dir", type=str, default=r"D:\dataset\hair_crawling_naver",
                        help="Root directory containing subfolder/JSON files")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save downloaded images")
    parser.add_argument("--workers", type=int, default=8,
                        help="Number of parallel download threads (default: 8)")
    args = parser.parse_args()

    source_dir = args.source_dir
    output_dir = args.output_dir

    if not os.path.isdir(source_dir):
        print(f"[ERROR] Source directory not found: {source_dir}")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    print(f"Scanning JSON files in {source_dir} ...")
    url_pairs = collect_urls(source_dir)
    print(f"Found {len(url_pairs)} unique image URLs.")

    if not url_pairs:
        print("Nothing to download.")
        return

    # Check which files already exist in output_dir to skip re-downloads
    existing = set(os.listdir(output_dir))
    to_download = [(url, name) for url, name in url_pairs if name not in existing]
    skipped = len(url_pairs) - len(to_download)
    if skipped > 0:
        print(f"Skipping {skipped} already downloaded images.")
    print(f"Downloading {len(to_download)} images with {args.workers} threads ...")

    success = 0
    fail = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {}
        for url, name in to_download:
            out_path = os.path.join(output_dir, name)
            futures[executor.submit(download_and_save, url, out_path)] = (url, name)

        for i, future in enumerate(as_completed(futures), 1):
            url, name = futures[future]
            if future.result():
                success += 1
            else:
                fail += 1
            if i % 100 == 0 or i == len(to_download):
                print(f"  Progress: {i}/{len(to_download)} (success={success}, fail={fail})")

    print(f"\nDone. Success: {success}, Failed: {fail}, Skipped: {skipped}")


if __name__ == "__main__":
    main()
