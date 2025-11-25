import argparse
import os
import random
import shutil
from pathlib import Path
from typing import List


def get_image_files(directory: str) -> List[Path]:
    """
    디렉토리에서 이미지 파일들을 가져옵니다.
    
    Args:
        directory: 이미지 파일들이 있는 디렉토리 경로
        
    Returns:
        이미지 파일 경로 리스트
    """
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp', '.tiff'}
    image_files = []
    
    directory_path = Path(directory)
    if not directory_path.exists():
        raise ValueError(f"디렉토리를 찾을 수 없습니다: {directory}")
    
    for file_path in directory_path.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in image_extensions:
            image_files.append(file_path)
    
    return image_files


def copy_images(
    reference_image_dir: str,
    control_image_dir: str,
    output_image_dir: str
) -> None:
    """
    참조 이미지 디렉토리에서 컨트롤 이미지 개수만큼 이미지를 출력 디렉토리로 복사합니다.
    컨트롤 이미지 파일명과 동일하게 이름을 변경하여 복사합니다.
    
    Args:
        reference_image_dir: 참조 이미지가 있는 디렉토리
        control_image_dir: 컨트롤 이미지가 있는 디렉토리 (개수와 파일명 참조용)
        output_image_dir: 이미지를 복사할 출력 디렉토리
    """
    # 참조 이미지 파일들 가져오기
    reference_images = get_image_files(reference_image_dir)
    
    if not reference_images:
        raise ValueError(f"참조 디렉토리에 이미지 파일이 없습니다: {reference_image_dir}")
    
    # 컨트롤 이미지 파일들 가져오기
    control_images = get_image_files(control_image_dir)
    
    if not control_images:
        raise ValueError(f"컨트롤 디렉토리에 이미지 파일이 없습니다: {control_image_dir}")
    
    # 컨트롤 이미지를 파일명 순으로 정렬
    control_images = sorted(control_images, key=lambda x: x.name)
    num_target_images = len(control_images)
    
    print(f"참조 이미지 개수: {len(reference_images)}")
    print(f"컨트롤 이미지 개수: {num_target_images}")
    print(f"목표 이미지 개수: {num_target_images}")
    
    # 출력 디렉토리 생성
    output_path = Path(output_image_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 복사할 이미지 선택
    if num_target_images <= len(reference_images):
        # 참조 이미지가 더 많거나 같으면 랜덤으로 선택
        selected_images = random.sample(reference_images, num_target_images)
    else:
        # 목표 개수가 더 많으면 중복을 허용하여 랜덤 선택
        selected_images = random.choices(reference_images, k=num_target_images)
    
    # 이미지 복사 (컨트롤 이미지 파일명으로 변경)
    copied_count = 0
    for control_image, source_image in zip(control_images, selected_images):
        # 컨트롤 이미지의 파일명 사용
        output_filename = control_image.name
        output_file_path = output_path / output_filename
        
        # 파일 복사
        shutil.copy2(source_image, output_file_path)
        copied_count += 1
        
        if copied_count % 100 == 0:
            print(f"진행 중: {copied_count}/{num_target_images} 이미지 복사 완료")
    
    print(f"\n완료: {copied_count}개의 이미지를 {output_image_dir}에 복사했습니다.")


def main():
    parser = argparse.ArgumentParser(
        description="참조 이미지 디렉토리에서 컨트롤 이미지 개수만큼 이미지를 출력 디렉토리로 복사합니다."
    )
    
    parser.add_argument(
        '--reference_image_dir',
        type=str,
        required=True,
        help='참조 이미지가 있는 디렉토리 경로'
    )
    
    parser.add_argument(
        '--control_image_dir',
        type=str,
        required=True,
        help='컨트롤 이미지가 있는 디렉토리 경로 (개수와 파일명 참조용)'
    )
    
    parser.add_argument(
        '--output_image_dir',
        type=str,
        required=True,
        help='이미지를 복사할 출력 디렉토리 경로'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='랜덤 시드 (재현성을 위해 사용, 선택사항)'
    )
    
    args = parser.parse_args()
    
    # 랜덤 시드 설정 (지정된 경우)
    if args.seed is not None:
        random.seed(args.seed)
        print(f"랜덤 시드 설정: {args.seed}")
    
    try:
        copy_images(
            reference_image_dir=args.reference_image_dir,
            control_image_dir=args.control_image_dir,
            output_image_dir=args.output_image_dir
        )
    except Exception as e:
        print(f"오류 발생: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

