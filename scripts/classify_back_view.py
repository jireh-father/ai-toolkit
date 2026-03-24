"""
Classify images by person orientation (front/side/back) using insightface.

Uses RetinaFace detector from insightface to detect faces and landmarks.
- Face detected with high confidence → FRONT/SIDE (other/)
- No face detected → BACK (back_view/)

Usage:
    python scripts/classify_back_view.py --input <input_dir> --output <output_dir> [--move] [--num-test 10] [--threshold 0.5] [--batch-size 16]
"""

import argparse
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
from insightface.app import FaceAnalysis
from tqdm import tqdm

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}
FACE_DET_THRESHOLD = 0.5


def load_detector(det_size: int = 640):
    app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(det_size, det_size), det_thresh=0.1)
    return app


def classify_image(app: FaceAnalysis, img_path: Path, threshold: float) -> str:
    """Classify a single image. Returns 'BACK', 'FRONT', or 'MULTI'."""
    img = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return "error"

    faces = app.get(img)
    confident_faces = [f for f in faces if f.det_score >= threshold]

    if not confident_faces:
        return "BACK"

    if len(confident_faces) >= 2:
        return "MULTI"

    return "FRONT"


def main():
    parser = argparse.ArgumentParser(description="Classify images by face detection (back view filter)")
    parser.add_argument("--input", "-i", required=True, help="Input directory containing images")
    parser.add_argument("--output", "-o", required=True, help="Output directory (back_view/ and other/ will be created)")
    parser.add_argument("--move", action="store_true", default=False, help="Move files instead of copy (default: copy)")
    parser.add_argument("--num-test", "-n", type=int, default=None, help="Process only first N images for testing")
    parser.add_argument("--threshold", "-t", type=float, default=FACE_DET_THRESHOLD, help=f"Face detection confidence threshold (default: {FACE_DET_THRESHOLD})")
    parser.add_argument("--det-size", type=int, default=640, help="Detection input size (default: 640, smaller=faster)")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    if not input_dir.is_dir():
        print(f"Error: Input directory not found: {input_dir}")
        sys.exit(1)

    # collect image files
    image_paths = sorted([p for p in input_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS])
    if not image_paths:
        print(f"No images found in {input_dir}")
        sys.exit(1)

    if args.num_test:
        image_paths = image_paths[: args.num_test]
        print(f"[TEST MODE] Processing first {len(image_paths)} images")

    print(f"Found {len(image_paths)} images in {input_dir}")
    print(f"Face detection threshold: {args.threshold}")
    print(f"Detection size: {args.det_size}")
    print(f"Mode: {'move' if args.move else 'copy'}")

    # create output dirs
    back_dir = output_dir / "back_view"
    other_dir = output_dir / "other"
    back_dir.mkdir(parents=True, exist_ok=True)
    other_dir.mkdir(parents=True, exist_ok=True)

    # load detector
    print("Loading insightface detector...")
    app = load_detector(args.det_size)
    print("Detector loaded.")

    # classify and distribute
    action = shutil.move if args.move else shutil.copy2
    action_name = "Moving" if args.move else "Copying"

    multi_dir = output_dir / "multi_face"
    multi_dir.mkdir(parents=True, exist_ok=True)

    counts = {"BACK": 0, "FRONT": 0, "MULTI": 0, "error": 0}
    for path in tqdm(image_paths, desc="Classifying"):
        label = classify_image(app, path, args.threshold)
        counts[label] = counts.get(label, 0) + 1

        if label == "error":
            print(f"  [SKIP] {path.name}: failed to read")
            continue

        if label == "BACK":
            dst_dir = back_dir
        elif label == "MULTI":
            dst_dir = multi_dir
        else:
            dst_dir = other_dir
        dst = dst_dir / path.name

        if dst.exists():
            stem = path.stem
            suffix = path.suffix
            counter = 1
            while dst.exists():
                dst = dst_dir / f"{stem}_{counter}{suffix}"
                counter += 1

        action(str(path), str(dst))

    # summary
    print(f"\nDone! {action_name} complete.")
    print(f"  Back view (no face): {counts.get('BACK', 0)}")
    print(f"  Front/Side (face):   {counts.get('FRONT', 0)}")
    print(f"  Multi face (3+):     {counts.get('MULTI', 0)}")
    if counts.get("error", 0):
        print(f"  Errors:              {counts['error']}")
    print(f"\nOutput:")
    print(f"  {back_dir}  ({counts.get('BACK', 0)} files)")
    print(f"  {other_dir}  ({counts.get('FRONT', 0)} files)")
    print(f"  {multi_dir}  ({counts.get('MULTI', 0)} files)")


if __name__ == "__main__":
    main()
