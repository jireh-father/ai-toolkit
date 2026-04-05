"""
특정 폴더의 파일 수가 N개 이상이 되면 서버를 종료하는 스크립트.

지정된 폴더를 주기적으로 모니터링하다가 파일(폴더 제외) 수가
임계값 이상이 되면 5초 카운트다운 후 시스템을 종료합니다.

Usage:
    python scripts/shutdown_on_file_count.py --path <dir> --max-files 1000
    python scripts/shutdown_on_file_count.py --path <dir> --max-files 500 --interval 30
"""

import argparse
import os
import sys
import time


def count_files(path: str) -> int:
    """폴더 내 파일 수를 카운트합니다 (하위 폴더 제외, 1단계만)."""
    return sum(1 for f in os.scandir(path) if f.is_file())


def main():
    parser = argparse.ArgumentParser(
        description="폴더 내 파일 수가 N개 이상이면 서버를 종료합니다"
    )
    parser.add_argument(
        "--path", "-p", type=str, required=True,
        help="모니터링할 폴더 경로"
    )
    parser.add_argument(
        "--max-files", "-n", type=int, required=True,
        help="이 수 이상이면 종료 트리거"
    )
    parser.add_argument(
        "--interval", "-i", type=int, default=60,
        help="체크 주기 (초, 기본값: 60)"
    )
    args = parser.parse_args()

    if not os.path.isdir(args.path):
        print(f"Error: 폴더를 찾을 수 없습니다: {args.path}")
        sys.exit(1)

    print(f"모니터링 시작: {args.path}")
    print(f"임계값: {args.max_files}개 이상이면 종료")
    print(f"체크 주기: {args.interval}초")
    print("-" * 40)

    try:
        while True:
            file_count = count_files(args.path)
            print(f"[{time.strftime('%H:%M:%S')}] 파일 수: {file_count} / {args.max_files}")

            if file_count >= args.max_files:
                print(f"\n*** 파일 수 {file_count}개 >= {args.max_files}개 — 5초 후 시스템 종료 ***")
                for i in range(5, 0, -1):
                    print(f"  {i}...")
                    time.sleep(1)
                print("시스템을 종료합니다.")
                if sys.platform == "win32":
                    os.system("shutdown /s /t 0")
                else:
                    os.system("sudo shutdown -h now")
                sys.exit(0)

            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\n모니터링 중단됨.")


if __name__ == "__main__":
    main()
