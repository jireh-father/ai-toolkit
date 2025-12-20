import os
import sys
from pathlib import Path


def remove_unmatched_txt_files(folder_path: str):
    """
    특정 폴더에서 동일한 파일명의 jpg 파일이 없는 txt 파일들을 삭제합니다.
    
    Args:
        folder_path: 검사할 폴더 경로
    """
    folder = Path(folder_path)
    
    if not folder.exists():
        print(f"오류: 폴더를 찾을 수 없습니다: {folder_path}")
        return
    
    if not folder.is_dir():
        print(f"오류: 유효한 폴더가 아닙니다: {folder_path}")
        return
    
    # txt 파일 목록 가져오기
    txt_files = list(folder.glob("*.txt"))
    
    if not txt_files:
        print(f"폴더에 txt 파일이 없습니다: {folder_path}")
        return
    
    print(f"검사 중: {folder_path}")
    print(f"총 {len(txt_files)}개의 txt 파일 발견")
    print("-" * 50)
    
    removed_count = 0
    kept_count = 0
    
    for txt_file in txt_files:
        # 동일한 이름의 jpg 파일 경로
        jpg_file = txt_file.with_suffix('.jpg')
        
        if jpg_file.exists():
            # jpg 파일이 있으면 유지
            kept_count += 1
            print(f"유지: {txt_file.name} (매칭되는 jpg 파일 존재)")
        else:
            # jpg 파일이 없으면 삭제
            try:
                txt_file.unlink()
                removed_count += 1
                print(f"삭제: {txt_file.name} (매칭되는 jpg 파일 없음)")
            except Exception as e:
                print(f"삭제 실패: {txt_file.name} - {str(e)}")
    
    print("-" * 50)
    print(f"완료: {removed_count}개 삭제, {kept_count}개 유지")


def main():
    if len(sys.argv) < 2:
        print("사용법: python remove_nomatched_txt_files.py <폴더경로>")
        print("예시: python remove_nomatched_txt_files.py ./dataset")
        sys.exit(1)
    
    folder_path = sys.argv[1]
    remove_unmatched_txt_files(folder_path)


if __name__ == "__main__":
    main()

