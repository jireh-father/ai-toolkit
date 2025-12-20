import argparse
import os
from pathlib import Path
from PIL import Image


def resize_image_keep_aspect_ratio(image, target_height):
    """
    이미지의 종횡비를 유지하면서 목표 높이에 맞게 리사이징합니다.
    
    Args:
        image: PIL Image 객체
        target_height: 목표 높이
    
    Returns:
        리사이징된 PIL Image 객체
    """
    original_width, original_height = image.size
    aspect_ratio = original_width / original_height
    new_width = int(target_height * aspect_ratio)
    return image.resize((new_width, target_height), Image.LANCZOS)


def stitch_images_horizontally(control_image, reference_image):
    """
    두 이미지를 수평으로 결합합니다.
    reference_image는 control_image의 높이에 맞게 리사이징됩니다.
    
    Args:
        control_image: 왼쪽에 배치될 PIL Image 객체
        reference_image: 오른쪽에 배치될 PIL Image 객체
    
    Returns:
        결합된 PIL Image 객체
    """
    # control_image의 높이에 맞게 reference_image 리사이징
    control_height = control_image.size[1]
    resized_reference = resize_image_keep_aspect_ratio(reference_image, control_height)
    
    # 새로운 이미지 크기 계산
    total_width = control_image.size[0] + resized_reference.size[0]
    max_height = control_height
    
    # 새로운 이미지 생성
    stitched_image = Image.new('RGB', (total_width, max_height))
    
    # 이미지 붙이기
    stitched_image.paste(control_image, (0, 0))
    stitched_image.paste(resized_reference, (control_image.size[0], 0))
    
    return stitched_image


def save_prompt_files(result_image_dir, prompt_text):
    """
    result_image_dir에 있는 모든 이미지 파일에 대해 동일한 파일명의 txt 파일을 생성합니다.
    
    Args:
        result_image_dir: 이미지 파일이 있는 디렉토리
        prompt_text: txt 파일에 저장할 프롬프트 텍스트
    """
    result_dir = Path(result_image_dir)
    
    # 지원하는 이미지 확장자
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}
    
    # result_dir의 모든 이미지 파일 가져오기
    image_files = [f for f in result_dir.iterdir() 
                   if f.is_file() and f.suffix.lower() in image_extensions]
    
    saved_count = 0
    
    for image_file in image_files:
        # 이미지 파일명에서 확장자를 제거하고 .txt 확장자로 변경
        txt_filename = image_file.stem + '.txt'
        txt_path = result_dir / txt_filename
        
        try:
            # 프롬프트 텍스트를 txt 파일에 저장 (덮어쓰기)
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(prompt_text)
            
            print(f"📝 프롬프트 저장: {txt_filename}")
            saved_count += 1
            
        except Exception as e:
            print(f"❌ 프롬프트 저장 오류 ({txt_filename}): {str(e)}")
    
    print(f"\n프롬프트 파일 저장 완료: {saved_count}개")


def process_images(control_image_dir, reference_image_dir, output_image_dir):
    """
    control_image_dir과 reference_image_dir의 동일한 파일명을 가진 이미지들을
    수평으로 결합하여 output_image_dir에 저장합니다.
    
    Args:
        control_image_dir: control 이미지가 있는 디렉토리
        reference_image_dir: reference 이미지가 있는 디렉토리
        output_image_dir: 결과 이미지를 저장할 디렉토리
    """
    # 디렉토리 경로 객체 생성
    control_dir = Path(control_image_dir)
    reference_dir = Path(reference_image_dir)
    output_dir = Path(output_image_dir)
    
    # 출력 디렉토리가 없으면 생성
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 지원하는 이미지 확장자
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}
    
    # control_image_dir의 모든 이미지 파일 가져오기
    control_images = [f for f in control_dir.iterdir() 
                     if f.is_file() and f.suffix.lower() in image_extensions]
    
    processed_count = 0
    skipped_count = 0
    
    for control_image_path in control_images:
        filename = control_image_path.name
        reference_image_path = reference_dir / filename
        
        # reference 이미지가 존재하는지 확인
        if not reference_image_path.exists():
            print(f"⚠️  건너뜀: {filename} (reference 이미지가 없음)")
            skipped_count += 1
            continue
        
        try:
            # 이미지 로드
            control_image = Image.open(control_image_path).convert('RGB')
            reference_image = Image.open(reference_image_path).convert('RGB')
            
            # 이미지 결합
            stitched_image = stitch_images_horizontally(control_image, reference_image)
            
            # 결과 저장
            output_path = output_dir / filename
            stitched_image.save(output_path)
            
            print(f"✅ 처리 완료: {filename}")
            processed_count += 1
            
        except Exception as e:
            print(f"❌ 오류 발생 ({filename}): {str(e)}")
            skipped_count += 1
    
    print(f"\n{'='*50}")
    print(f"처리 완료: {processed_count}개")
    print(f"건너뜀: {skipped_count}개")
    print(f"출력 디렉토리: {output_dir.absolute()}")


def main():
    parser = argparse.ArgumentParser(
        description='Control 이미지와 Reference 이미지를 수평으로 결합합니다.'
    )
    
    parser.add_argument(
        '--control_image_dir',
        type=str,
        required=True,
        help='Control 이미지가 있는 디렉토리 경로'
    )
    
    parser.add_argument(
        '--reference_image_dir',
        type=str,
        required=True,
        help='Reference 이미지가 있는 디렉토리 경로'
    )
    
    parser.add_argument(
        '--output_image_dir',
        type=str,
        required=True,
        help='결과 이미지를 저장할 디렉토리 경로'
    )
    
    parser.add_argument(
        '--result_image_dir',
        type=str,
        default=None,
        help='프롬프트 txt 파일을 생성할 이미지 디렉토리 경로 (선택사항)'
    )
    
    parser.add_argument(
        '--prompt',
        type=str,
        default=None,
        help='각 이미지와 함께 저장할 프롬프트 텍스트 (선택사항)'
    )
    
    args = parser.parse_args()
    
    # 디렉토리 존재 확인
    if not os.path.exists(args.control_image_dir):
        print(f"❌ 오류: control_image_dir이 존재하지 않습니다: {args.control_image_dir}")
        return
    
    if not os.path.exists(args.reference_image_dir):
        print(f"❌ 오류: reference_image_dir이 존재하지 않습니다: {args.reference_image_dir}")
        return
    
    print(f"Control 이미지 디렉토리: {args.control_image_dir}")
    print(f"Reference 이미지 디렉토리: {args.reference_image_dir}")
    print(f"출력 디렉토리: {args.output_image_dir}")
    if args.result_image_dir and args.prompt:
        print(f"프롬프트 저장 디렉토리: {args.result_image_dir}")
        print(f"프롬프트: {args.prompt}")
    print(f"{'='*50}\n")
    
    process_images(args.control_image_dir, args.reference_image_dir, args.output_image_dir)
    
    # result_image_dir과 prompt가 모두 제공된 경우 txt 파일 생성
    if args.result_image_dir and args.prompt:
        if not os.path.exists(args.result_image_dir):
            print(f"\n❌ 오류: result_image_dir이 존재하지 않습니다: {args.result_image_dir}")
        else:
            print(f"\n{'='*50}")
            save_prompt_files(args.result_image_dir, args.prompt)


if __name__ == "__main__":
    main()

