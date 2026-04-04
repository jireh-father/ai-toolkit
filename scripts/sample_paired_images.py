"""
3개 폴더에서 동일 파일명 기준으로 n개 쌍을 샘플링하여 output_dir에 복사하는 스크립트.

Usage:
    python scripts/sample_paired_images.py \
        --dir1 /path/to/folder1 \
        --dir2 /path/to/folder2 \
        --dir3 /path/to/folder3 \
        --output_dir /path/to/output \
        --n 100 \
        --seed 42
"""

import argparse
import random
import shutil
from pathlib import Path

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff', '.tif'}


def get_image_filenames(folder: Path) -> set[str]:
    return {
        f.name for f in folder.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    }


def main():
    parser = argparse.ArgumentParser(description="3개 폴더에서 동일 파일명 기준으로 n개 쌍 샘플링 후 복사")
    parser.add_argument("--dir1", type=str, required=True, help="첫 번째 폴더 경로")
    parser.add_argument("--dir2", type=str, required=True, help="두 번째 폴더 경로")
    parser.add_argument("--dir3", type=str, required=True, help="세 번째 폴더 경로")
    parser.add_argument("--output_dir", type=str, required=True, help="출력 폴더 경로")
    parser.add_argument("--n", type=int, required=True, help="샘플링할 쌍의 수")
    parser.add_argument("--seed", type=int, default=42, help="랜덤 시드 (default: 42)")
    args = parser.parse_args()

    dirs = [Path(args.dir1), Path(args.dir2), Path(args.dir3)]
    for d in dirs:
        if not d.is_dir():
            raise FileNotFoundError(f"폴더를 찾을 수 없습니다: {d}")

    filenames = [get_image_filenames(d) for d in dirs]
    common = filenames[0] & filenames[1] & filenames[2]

    if not common:
        print("3개 폴더에 공통된 파일명이 없습니다.")
        return

    print(f"공통 파일 수: {len(common)}")

    if args.n > len(common):
        print(f"요청한 n={args.n}이 공통 파일 수({len(common)})보다 큽니다. 전체를 사용합니다.")
        sampled = sorted(common)
    else:
        random.seed(args.seed)
        sampled = sorted(random.sample(sorted(common), args.n))

    print(f"샘플링된 쌍 수: {len(sampled)}")

    output_base = Path(args.output_dir)
    output_dirs = []
    for d in dirs:
        out = output_base / d.name
        out.mkdir(parents=True, exist_ok=True)
        output_dirs.append(out)

    copied = 0
    for fname in sampled:
        for src_dir, dst_dir in zip(dirs, output_dirs):
            shutil.copy2(src_dir / fname, dst_dir / fname)
        copied += 1

    print(f"복사 완료: {copied}쌍 × 3폴더 = {copied * 3}개 파일 → {output_base}")


if __name__ == "__main__":
    main()
