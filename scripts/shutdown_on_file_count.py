"""
특정 폴더의 파일 수가 N개 이상이 되면 서버를 종료하는 스크립트.

지정된 폴더를 주기적으로 모니터링하다가 파일(폴더 제외) 수가
임계값 이상이 되면 5초 카운트다운 후 시스템을 종료합니다.

Usage:
    python scripts/shutdown_on_file_count.py --path <dir> --max-files 1000
    python scripts/shutdown_on_file_count.py --path <dir> --max-files 500 --interval 30
"""

import argparse
import json
import os
import socket
import sys
import time
from urllib import request, error


def load_discord_webhook_url() -> str | None:
    """프로젝트 루트의 .env에서 DISCORD_WEBHOOK_URL을 읽습니다."""
    env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    if not os.path.isfile(env_path):
        return None
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("DISCORD_WEBHOOK_URL="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


def send_discord_message(webhook_url: str, message: str):
    """디스코드 웹훅으로 메시지를 보냅니다."""
    payload = json.dumps({"content": message}).encode("utf-8")
    req = request.Request(
        webhook_url,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "User-Agent": "ShutdownMonitor/1.0",
        },
    )
    try:
        request.urlopen(req, timeout=10)
        print(f"디스코드 알림 전송 완료")
    except (error.URLError, socket.timeout) as e:
        print(f"디스코드 알림 전송 실패: {e}")


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

    webhook_url = load_discord_webhook_url()
    if webhook_url:
        print(f"디스코드 웹훅: 활성화")
    else:
        print(f"디스코드 웹훅: .env에 DISCORD_WEBHOOK_URL 없음 (알림 생략)")

    print(f"모니터링 시작: {args.path}")
    print(f"임계값: {args.max_files}개 이상이면 종료")
    print(f"체크 주기: {args.interval}초")
    print("-" * 40)

    try:
        while True:
            file_count = count_files(args.path)
            print(f"[{time.strftime('%H:%M:%S')}] 파일 수: {file_count} / {args.max_files}")

            if file_count >= args.max_files:
                hostname = socket.gethostname()
                msg = (
                    f"🔴 **서버 종료 알림** ({hostname})\n"
                    f"폴더 `{args.path}` 파일 수 **{file_count}개** >= 임계값 **{args.max_files}개**\n"
                    f"5초 후 시스템을 종료합니다."
                )
                print(f"\n*** 파일 수 {file_count}개 >= {args.max_files}개 — 5초 후 시스템 종료 ***")
                if webhook_url:
                    send_discord_message(webhook_url, msg)
                for i in range(5, 0, -1):
                    print(f"  {i}...")
                    time.sleep(1)
                print("시스템을 종료합니다.")
                if sys.platform == "win32":
                    os.system("shutdown /s /t 0")
                else:
                    os.system("shutdown -h now")
                sys.exit(0)

            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\n모니터링 중단됨.")


if __name__ == "__main__":
    main()
