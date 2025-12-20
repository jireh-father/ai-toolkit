import argparse
import os
import shutil
from pathlib import Path
from typing import List, Optional, Set

# 지원하는 이미지 확장자
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff', '.tif'}


def get_image_files(folder_path: Path) -> Set[str]:
    """
    폴더에서 이미지 파일들의 파일명(확장자 제외)을 반환합니다.
    
    Args:
        folder_path: 검사할 폴더 경로
        
    Returns:
        이미지 파일명 집합 (확장자 제외)
    """
    image_names = set()
    for file in folder_path.iterdir():
        if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS:
            image_names.add(file.stem)
    return image_names


def get_all_image_files_with_path(folder_path: Path) -> List[Path]:
    """
    폴더에서 모든 이미지 파일 경로를 반환합니다.
    
    Args:
        folder_path: 검사할 폴더 경로
        
    Returns:
        이미지 파일 경로 리스트
    """
    image_files = []
    for file in folder_path.iterdir():
        if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS:
            image_files.append(file)
    return image_files


def remove_unmatched_image_files(
    folders: List[str], 
    removed_dir: Optional[str] = None
) -> None:
    """
    여러 폴더에서 파일명이 모든 폴더에 공통적으로 존재하지 않는 이미지 파일들을 삭제합니다.
    
    Args:
        folders: 검사할 폴더 경로들
        removed_dir: 삭제된 파일들을 이동시킬 디렉토리 (None이면 영구 삭제)
    """
    # 폴더 경로 유효성 검사
    folder_paths = []
    for folder in folders:
        folder_path = Path(folder)
        if not folder_path.exists():
            print(f"오류: 폴더를 찾을 수 없습니다: {folder}")
            return
        if not folder_path.is_dir():
            print(f"오류: 유효한 폴더가 아닙니다: {folder}")
            return
        folder_paths.append(folder_path)
    
    if len(folder_paths) < 2:
        print("오류: 최소 2개 이상의 폴더를 지정해야 합니다.")
        return
    
    # 각 폴더별 이미지 파일명 집합 구하기
    print("=" * 60)
    print("폴더별 이미지 파일 분석")
    print("=" * 60)
    
    image_names_per_folder = {}
    for folder_path in folder_paths:
        image_names = get_image_files(folder_path)
        image_names_per_folder[folder_path] = image_names
        print(f"{folder_path.name}: {len(image_names)}개 이미지 파일")
    
    # 모든 폴더에 공통으로 존재하는 파일명 구하기
    common_names = None
    for image_names in image_names_per_folder.values():
        if common_names is None:
            common_names = image_names.copy()
        else:
            common_names = common_names.intersection(image_names)
    
    if common_names is None:
        common_names = set()
    
    print("-" * 60)
    print(f"모든 폴더에 공통으로 존재하는 이미지: {len(common_names)}개")
    print("=" * 60)
    
    # removed_dir 설정
    removed_base_path = None
    if removed_dir:
        removed_base_path = Path(removed_dir)
        removed_base_path.mkdir(parents=True, exist_ok=True)
        print(f"삭제 파일 이동 경로: {removed_base_path}")
    
    # 각 폴더에서 공통 파일명에 포함되지 않는 이미지 삭제/이동
    total_removed = 0
    total_kept = 0
    
    for folder_path in folder_paths:
        print(f"\n처리 중: {folder_path}")
        print("-" * 40)
        
        image_files = get_all_image_files_with_path(folder_path)
        removed_count = 0
        kept_count = 0
        
        for image_file in image_files:
            if image_file.stem in common_names:
                kept_count += 1
            else:
                # 삭제 또는 이동
                try:
                    if removed_base_path:
                        # 기존 폴더명으로 하위 디렉토리 생성
                        dest_folder = removed_base_path / folder_path.name
                        dest_folder.mkdir(parents=True, exist_ok=True)
                        dest_path = dest_folder / image_file.name
                        shutil.move(str(image_file), str(dest_path))
                        print(f"  이동: {image_file.name} -> {dest_path}")
                    else:
                        image_file.unlink()
                        print(f"  삭제: {image_file.name}")
                    removed_count += 1
                except Exception as e:
                    print(f"  실패: {image_file.name} - {str(e)}")
        
        print(f"  결과: {removed_count}개 {'이동' if removed_base_path else '삭제'}, {kept_count}개 유지")
        total_removed += removed_count
        total_kept += kept_count
    
    print("\n" + "=" * 60)
    print(f"전체 결과: {total_removed}개 {'이동' if removed_base_path else '삭제'}, {total_kept}개 유지")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description='여러 폴더에서 파일명이 모든 폴더에 공통적으로 존재하는 이미지만 남기고 나머지를 삭제합니다.'
    )
    parser.add_argument(
        'folders',
        nargs='+',
        help='검사할 폴더 경로들 (최소 2개 이상)'
    )
    parser.add_argument(
        '--removed_dir',
        type=str,
        default=None,
        help='삭제된 파일들을 이동시킬 디렉토리 경로 (지정하지 않으면 영구 삭제)'
    )
    
    args = parser.parse_args()
    
    remove_unmatched_image_files(args.folders, args.removed_dir)


if __name__ == "__main__":
    main()
