import argparse
import os
from pathlib import Path
import shutil
from typing import Dict


def get_image_files_dict(directory: str) -> Dict[str, str]:
    """
    디렉토리 내의 모든 이미지 파일을 재귀적으로 찾아서 딕셔너리로 반환
    
    Args:
        directory: 검색할 디렉토리 경로
        
    Returns:
        파일명을 키로, 전체 경로를 값으로 하는 딕셔너리
    """
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}
    image_dict = {}
    
    directory_path = Path(directory)
    
    if not directory_path.exists():
        print(f"경고: 디렉토리가 존재하지 않습니다: {directory}")
        return image_dict
    
    # 재귀적으로 모든 파일 검색
    for file_path in directory_path.rglob('*'):
        if file_path.is_file() and file_path.suffix.lower() in image_extensions:
            filename = file_path.name
            if filename in image_dict:
                print(f"경고: 중복된 파일명 발견: {filename}")
                print(f"  기존: {image_dict[filename]}")
                print(f"  새로운: {file_path}")
            image_dict[filename] = str(file_path)
    
    return image_dict


def copy_filtered_images(ori_dir: str, filtered_dir: str, output_dir: str):
    """
    filtered_dir의 이미지 파일명을 기준으로 ori_dir에서 찾아 output_dir로 복사
    
    Args:
        ori_dir: 원본 이미지들이 있는 디렉토리
        filtered_dir: 필터링된 이미지들이 있는 디렉토리
        output_dir: 복사할 대상 디렉토리
    """
    # output_dir 생성
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # ori_dir에서 이미지 딕셔너리 생성
    print(f"원본 디렉토리에서 이미지 파일 검색 중: {ori_dir}")
    ori_images = get_image_files_dict(ori_dir)
    print(f"총 {len(ori_images)}개의 원본 이미지 파일 발견")
    
    # filtered_dir에서 이미지 파일 목록 가져오기
    print(f"\n필터링된 디렉토리에서 이미지 파일 검색 중: {filtered_dir}")
    filtered_images = get_image_files_dict(filtered_dir)
    print(f"총 {len(filtered_images)}개의 필터링된 이미지 파일 발견")
    
    # 복사 작업 수행
    copied_count = 0
    not_found_count = 0
    
    print(f"\n이미지 복사 시작...")
    for filename in sorted(filtered_images.keys()):
        if filename in ori_images:
            src_path = ori_images[filename]
            dst_path = output_path / filename
            
            try:
                shutil.copy2(src_path, dst_path)
                copied_count += 1
                print(f"복사 완료: {filename}")
            except Exception as e:
                print(f"복사 실패: {filename} - {e}")
        else:
            not_found_count += 1
            print(f"원본 없음: {filename}")
    
    # 결과 출력
    print(f"\n=== 복사 완료 ===")
    print(f"성공: {copied_count}개")
    print(f"실패 (원본 없음): {not_found_count}개")
    print(f"출력 디렉토리: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='필터링된 이미지 파일명을 기준으로 원본 디렉토리에서 이미지를 찾아 복사합니다.'
    )
    
    parser.add_argument(
        '--ori_dir',
        type=str,
        required=True,
        help='원본 이미지들이 있는 디렉토리 경로'
    )
    
    parser.add_argument(
        '--filtered_dir',
        type=str,
        required=True,
        help='필터링된 이미지들이 있는 디렉토리 경로'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='이미지를 복사할 출력 디렉토리 경로'
    )
    
    args = parser.parse_args()
    
    # 경로 검증
    if not os.path.exists(args.ori_dir):
        print(f"오류: 원본 디렉토리가 존재하지 않습니다: {args.ori_dir}")
        return
    
    if not os.path.exists(args.filtered_dir):
        print(f"오류: 필터링된 디렉토리가 존재하지 않습니다: {args.filtered_dir}")
        return
    
    # 복사 실행
    copy_filtered_images(args.ori_dir, args.filtered_dir, args.output_dir)


if __name__ == '__main__':
    main()

