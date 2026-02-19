"""Image loading, dataset scanning, and preprocessing utilities."""

import logging
from pathlib import Path
from typing import Optional

from PIL import Image

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}


def _is_image_file(path: Path) -> bool:
    return path.suffix.lower() in SUPPORTED_EXTENSIONS


def _get_image_stems(directory: Path) -> dict[str, Path]:
    """Return {stem: path} for all image files in a directory."""
    stems = {}
    for p in directory.iterdir():
        if p.is_file() and _is_image_file(p):
            stems[p.stem] = p
    return stems


def scan_dataset(
    input_dir: Path,
    reference_dir: Path,
    output_dir: Path,
) -> tuple[list[dict], list[dict]]:
    """Scan three folders and match filenames.

    Returns:
        matched: list of {"stem": str, "input": Path, "reference": Path, "output": Path}
        mismatched: list of {"stem": str, "missing_in": list[str], "present_in": list[str]}
    """
    input_stems = _get_image_stems(input_dir)
    ref_stems = _get_image_stems(reference_dir)
    out_stems = _get_image_stems(output_dir)

    all_stems = sorted(set(input_stems) | set(ref_stems) | set(out_stems))

    matched = []
    mismatched = []

    for stem in all_stems:
        present_in = []
        missing_in = []
        if stem in input_stems:
            present_in.append("input")
        else:
            missing_in.append("input")
        if stem in ref_stems:
            present_in.append("reference")
        else:
            missing_in.append("reference")
        if stem in out_stems:
            present_in.append("output")
        else:
            missing_in.append("output")

        if not missing_in:
            matched.append({
                "stem": stem,
                "input": input_stems[stem],
                "reference": ref_stems[stem],
                "output": out_stems[stem],
            })
        else:
            mismatched.append({
                "stem": stem,
                "missing_in": missing_in,
                "present_in": present_in,
            })

    logger.info(
        f"Dataset scan complete: input={len(input_stems)}, "
        f"reference={len(ref_stems)}, output={len(out_stems)}"
    )
    logger.info(f"Matched (inspection targets): {len(matched)}")
    if mismatched:
        logger.warning(f"Mismatched: {len(mismatched)} — see mismatch report for details")

    return matched, mismatched


def resize_image(image: Image.Image, short_side: int = 512) -> Image.Image:
    """Resize image keeping aspect ratio, short side = short_side px."""
    w, h = image.size
    if min(w, h) <= short_side:
        return image

    if w < h:
        new_w = short_side
        new_h = int(h * short_side / w)
    else:
        new_h = short_side
        new_w = int(w * short_side / h)

    return image.resize((new_w, new_h), Image.LANCZOS)


def validate_image(path: Path) -> bool:
    """Check if an image file can be opened without errors."""
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False


def load_image_triplet(
    entry: dict,
    short_side: int = 512,
) -> Optional[tuple[Image.Image, Image.Image, Image.Image]]:
    """Load and resize input/reference/output images.

    Args:
        entry: dict with 'input', 'reference', 'output' Path values
        short_side: target short side for resizing

    Returns:
        (input_img, reference_img, output_img) or None if any image is corrupted
    """
    images = []
    for key in ("input", "reference", "output"):
        path = entry[key]
        try:
            img = Image.open(path).convert("RGB")
            img = resize_image(img, short_side)
            images.append(img)
        except Exception as e:
            logger.error(f"Failed to load {key} image {path}: {e}")
            return None

    return tuple(images)


def filter_corrupted(
    matched: list[dict],
) -> tuple[list[dict], list[dict]]:
    """Validate all matched entries and separate corrupted ones.

    Returns:
        valid: entries where all 3 images are readable
        corrupted: entries where at least one image is corrupted
    """
    valid = []
    corrupted = []

    for entry in matched:
        all_ok = True
        bad_keys = []
        for key in ("input", "reference", "output"):
            if not validate_image(entry[key]):
                all_ok = False
                bad_keys.append(key)

        if all_ok:
            valid.append(entry)
        else:
            corrupted.append({
                "stem": entry["stem"],
                "corrupted_in": bad_keys,
            })
            logger.error(
                f"Corrupted image(s) for '{entry['stem']}': {bad_keys}"
            )

    if corrupted:
        logger.warning(f"Corrupted entries: {len(corrupted)}")

    return valid, corrupted
