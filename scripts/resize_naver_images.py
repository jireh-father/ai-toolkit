"""
Resize images in a directory using ComfyUI's FluxKontextImageScale logic.

Port of `FluxKontextImageScale` + `comfy.utils.common_upscale(..., "lanczos", "center")`
from ComfyUI. Picks the preferred Flux Kontext resolution whose aspect ratio is
closest to the source, center-crops to that aspect ratio, then resizes with
PIL LANCZOS (identical to ComfyUI's `lanczos` implementation).
"""

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from PIL import Image


PREFERRED_KONTEXT_RESOLUTIONS = [
    (672, 1568),
    (688, 1504),
    (720, 1456),
    (752, 1392),
    (800, 1328),
    (832, 1248),
    (880, 1184),
    (944, 1104),
    (1024, 1024),
    (1104, 944),
    (1184, 880),
    (1248, 832),
    (1328, 800),
    (1392, 752),
    (1456, 720),
    (1504, 688),
    (1568, 672),
]

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def pick_target_resolution(width: int, height: int) -> tuple[int, int]:
    aspect_ratio = width / height
    _, w, h = min(
        (abs(aspect_ratio - tw / th), tw, th)
        for tw, th in PREFERRED_KONTEXT_RESOLUTIONS
    )
    return w, h


def center_crop_to_aspect(img: Image.Image, target_w: int, target_h: int) -> Image.Image:
    """Replicates ComfyUI common_upscale center-crop (rounding matches the torch version)."""
    old_w, old_h = img.size
    old_aspect = old_w / old_h
    new_aspect = target_w / target_h

    x = 0
    y = 0
    if old_aspect > new_aspect:
        x = round((old_w - old_w * (new_aspect / old_aspect)) / 2)
    elif old_aspect < new_aspect:
        y = round((old_h - old_h * (old_aspect / new_aspect)) / 2)

    return img.crop((x, y, old_w - x, old_h - y))


def flux_kontext_resize(img: Image.Image) -> Image.Image:
    if img.mode not in ("RGB", "RGBA", "L"):
        img = img.convert("RGB")

    target_w, target_h = pick_target_resolution(img.width, img.height)
    cropped = center_crop_to_aspect(img, target_w, target_h)
    return cropped.resize((target_w, target_h), resample=Image.Resampling.LANCZOS)


def iter_image_files(root: Path):
    for dirpath, _, files in os.walk(root):
        for name in files:
            if Path(name).suffix.lower() in IMAGE_EXTS:
                yield Path(dirpath) / name


def resolve_output_path(src: Path, input_dir: Path, output_dir: Path | None) -> Path:
    if output_dir is None:
        return src
    rel = src.relative_to(input_dir)
    return output_dir / rel


def save_same_format(img: Image.Image, src_path: Path, dst_path: Path) -> None:
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    ext = src_path.suffix.lower()
    save_kwargs = {}
    save_img = img

    if ext in (".jpg", ".jpeg"):
        if save_img.mode != "RGB":
            save_img = save_img.convert("RGB")
        save_kwargs["quality"] = 95
    elif ext == ".webp":
        save_kwargs["quality"] = 95

    save_img.save(dst_path, **save_kwargs)


def _process_one(src_str: str, input_dir_str: str, output_dir_str: str | None) -> tuple[bool, str, str]:
    src = Path(src_str)
    input_dir = Path(input_dir_str)
    output_dir = Path(output_dir_str) if output_dir_str else None
    try:
        with Image.open(src) as im:
            im.load()
            resized = flux_kontext_resize(im)
        dst = resolve_output_path(src, input_dir, output_dir)
        save_same_format(resized, src, dst)
        return True, str(src), f"{dst} ({resized.width}x{resized.height})"
    except Exception as e:
        return False, str(src), repr(e)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Resize images using ComfyUI FluxKontextImageScale logic."
    )
    parser.add_argument("input_dir", help="Directory containing images to resize.")
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Optional output directory. If omitted, overwrites originals in input_dir.",
    )
    parser.add_argument(
        "--num_test",
        type=int,
        default=None,
        help="If set, only process the first N images (for testing).",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of worker processes. 1 = single-process (default).",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    if not input_dir.is_dir():
        print(f"[ERROR] Not a directory: {input_dir}", file=sys.stderr)
        return 1

    output_dir = Path(args.output_dir).resolve() if args.output_dir else None

    sources: list[Path] = []
    for src in iter_image_files(input_dir):
        if args.num_test is not None and len(sources) >= args.num_test:
            break
        sources.append(src)

    total = len(sources)
    ok = 0
    input_dir_str = str(input_dir)
    output_dir_str = str(output_dir) if output_dir else None

    if args.num_workers <= 1:
        for src in sources:
            success, src_s, info = _process_one(str(src), input_dir_str, output_dir_str)
            if success:
                ok += 1
                print(f"[OK] {src_s} -> {info}")
            else:
                print(f"[FAIL] {src_s}: {info}", file=sys.stderr)
    else:
        with ProcessPoolExecutor(max_workers=args.num_workers) as pool:
            futures = [
                pool.submit(_process_one, str(src), input_dir_str, output_dir_str)
                for src in sources
            ]
            for fut in as_completed(futures):
                success, src_s, info = fut.result()
                if success:
                    ok += 1
                    print(f"[OK] {src_s} -> {info}")
                else:
                    print(f"[FAIL] {src_s}: {info}", file=sys.stderr)

    print(f"\nDone: {ok}/{total} images resized.")
    return 0 if ok == total else 2


if __name__ == "__main__":
    sys.exit(main())
