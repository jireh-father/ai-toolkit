import argparse
import os
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='이미지 파일들에 대한 기본 캡션 텍스트 파일을 생성합니다.')
    parser.add_argument('folder', type=str, help='이미지 파일들이 있는 폴더 경로')
    parser.add_argument('--caption', type=str, required=True, help='저장할 기본 캡션 텍스트')
    
    args = parser.parse_args()
    
    folder_path = Path(args.folder)
    default_caption = args.caption
    
    # 폴더가 존재하는지 확인
    if not folder_path.exists():
        print(f"오류: 폴더 '{folder_path}'가 존재하지 않습니다.")
        return
    
    if not folder_path.is_dir():
        print(f"오류: '{folder_path}'는 폴더가 아닙니다.")
        return
    
    # 지원하는 이미지 확장자들
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}
    
    # 이미지 파일들 찾기
    image_files = []
    for file_path in folder_path.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in image_extensions:
            image_files.append(file_path)
    
    if not image_files:
        print(f"폴더 '{folder_path}'에서 이미지 파일을 찾을 수 없습니다.")
        return
    
    print(f"총 {len(image_files)}개의 이미지 파일을 찾았습니다.")
    
    # 각 이미지 파일에 대해 텍스트 파일 생성
    created_count = 0
    for image_file in image_files:
        # 확장자를 .txt로 변경
        txt_file = image_file.with_suffix('.txt')
        
        try:
            # 텍스트 파일에 기본 캡션 저장
            with open(txt_file, 'w', encoding='utf-8') as f:
                f.write(default_caption)
            
            print(f"생성됨: {txt_file.name}")
            created_count += 1
            
        except Exception as e:
            print(f"오류: {image_file.name}에 대한 텍스트 파일 생성 실패 - {e}")
    
    print(f"\n작업 완료: {created_count}개의 캡션 파일이 생성되었습니다.")

if __name__ == "__main__":
    main()
