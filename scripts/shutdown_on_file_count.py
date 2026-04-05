"""
특정 폴더의 파일 수가 N개 이상이 되면 서버를 종료하는 스크립트.

지정된 폴더를 주기적으로 모니터링하다가 파일(폴더 제외) 수가
임계값 이상이 되면 5초 카운트다운 후 시스템을 종료합니다.

지원 환경:
  - Windows: shutdown /s /t 0
  - SimplePod: DELETE /instances/{hashId} API 호출
  - 기타 Linux: shutdown -h now → kill 1 fallback

.env 변수:
  - DISCORD_WEBHOOK_URL: 디스코드 웹훅 URL
  - SIMPLEPOD_API_KEY: SimplePod API 키 (X-AUTH-TOKEN)

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


SIMPLEPOD_INSTANCE_JSON = "/etc/simplepod/instance.json"


def load_env_var(key: str) -> str | None:
    """프로젝트 루트의 .env에서 특정 변수를 읽습니다."""
    env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    if not os.path.isfile(env_path):
        return None
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith(f"{key}="):
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
        print("디스코드 알림 전송 완료")
    except (error.URLError, socket.timeout) as e:
        print(f"디스코드 알림 전송 실패: {e}")


def get_simplepod_instance_id() -> str | None:
    """SimplePod 인스턴스 hashId를 읽습니다."""
    if not os.path.isfile(SIMPLEPOD_INSTANCE_JSON):
        return None
    try:
        with open(SIMPLEPOD_INSTANCE_JSON, "r") as f:
            data = json.load(f)
        return data.get("hashId")
    except Exception:
        return None


def terminate_simplepod(instance_id: str, api_key: str) -> bool:
    """SimplePod REST API로 인스턴스를 삭제합니다."""
    url = f"https://api.simplemining.net/instances/{instance_id}"
    req = request.Request(url, method="DELETE", headers={
        "X-AUTH-TOKEN": api_key,
        "User-Agent": "ShutdownMonitor/1.0",
    })
    try:
        resp = request.urlopen(req, timeout=15)
        print(f"SimplePod 인스턴스 삭제 성공 (status: {resp.code})")
        return True
    except error.HTTPError as e:
        print(f"SimplePod API 오류: {e.code} {e.reason}")
        return False
    except Exception as e:
        print(f"SimplePod API 요청 실패: {e}")
        return False


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
    parser.add_argument(
        "--cloud", type=str, default=None,
        choices=["simplepod", "runpod"],
        help="클라우드 환경 (simplepod 또는 runpod, 미지정시 자동 감지)"
    )
    args = parser.parse_args()

    if not os.path.isdir(args.path):
        print(f"Error: 폴더를 찾을 수 없습니다: {args.path}")
        sys.exit(1)

    # .env에서 설정 로드
    webhook_url = load_env_var("DISCORD_WEBHOOK_URL")
    simplepod_api_key = load_env_var("SIMPLEPOD_API_KEY")

    # SimplePod 감지
    simplepod_id = get_simplepod_instance_id()

    # 환경 정보 출력
    if webhook_url:
        print("디스코드 웹훅: 활성화")
    else:
        print("디스코드 웹훅: .env에 DISCORD_WEBHOOK_URL 없음 (알림 생략)")

    if simplepod_id:
        print(f"SimplePod 감지: {simplepod_id}")
        if simplepod_api_key:
            print("SimplePod API 키: 활성화")
        else:
            print("WARNING: .env에 SIMPLEPOD_API_KEY 없음 — API 종료 불가")

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
                    f"5초 후 인스턴스를 종료합니다."
                )
                print(f"\n*** 파일 수 {file_count}개 >= {args.max_files}개 — 5초 후 종료 ***")
                if webhook_url:
                    send_discord_message(webhook_url, msg)
                for i in range(5, 0, -1):
                    print(f"  {i}...")
                    time.sleep(1)

                print("시스템을 종료합니다.")

                cloud = args.cloud
                # 자동 감지
                if cloud is None:
                    if simplepod_id:
                        cloud = "simplepod"
                    elif os.environ.get("RUNPOD_POD_ID"):
                        cloud = "runpod"

                if sys.platform == "win32":
                    os.system("shutdown /s /t 0")
                elif cloud == "simplepod":
                    if simplepod_id and simplepod_api_key:
                        print(f"SimplePod 인스턴스 삭제: {simplepod_id}")
                        if not terminate_simplepod(simplepod_id, simplepod_api_key):
                            print("API 실패, kill 1 시도...")
                            os.system("kill 1")
                    else:
                        print("SimplePod 인스턴스 ID 또는 API 키 없음, kill 1 시도...")
                        os.system("kill 1")
                elif cloud == "runpod":
                    pod_id = os.environ.get("RUNPOD_POD_ID", "")
                    if pod_id:
                        print(f"RunPod Pod 삭제: {pod_id}")
                        ret = os.system(f"runpodctl remove pod {pod_id}")
                        if ret != 0:
                            print("runpodctl 실패, kill 1 시도...")
                            os.system("kill 1")
                    else:
                        print("RUNPOD_POD_ID 없음, kill 1 시도...")
                        os.system("kill 1")
                else:
                    # 기타 Linux
                    ret = os.system("shutdown -h now 2>/dev/null")
                    if ret != 0:
                        print("shutdown 명령 없음, kill 1 시도...")
                        os.system("kill 1")

                sys.exit(0)

            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\n모니터링 중단됨.")


if __name__ == "__main__":
    main()
