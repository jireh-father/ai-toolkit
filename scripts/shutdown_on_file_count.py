"""
특정 폴더의 파일 수가 N개 이상이 되면:
  1. 해당 폴더 전체를 압축 (파일+하위폴더 recursive)
  2. S3에 public으로 업로드
  3. 업로드 완료 후 압축 파일 삭제
  4. 다운로드 URL을 디스코드로 전송
  5. 서버 종료

.env 변수:
  - DISCORD_WEBHOOK_URL: 디스코드 웹훅 URL
  - SIMPLEPOD_API_KEY: SimplePod API 키
  - AWS_ACCESS_KEY_ID: S3 Access Key
  - AWS_SECRET_ACCESS_KEY: S3 Secret Key
  - AWS_S3_BUCKET: S3 버킷명
  - AWS_S3_REGION: S3 리전 (기본값: ap-northeast-2)

Usage:
    python scripts/shutdown_on_file_count_compress_and_upload_s3.py \
        --path /output --max-files 1000 --work-name hair_v1 \
        --cloud simplepod

    python scripts/shutdown_on_file_count_compress_and_upload_s3.py \
        --path /output --max-files 500 --work-name hair_v2 \
        --s3-prefix my_datasets --interval 30
"""

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path
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
            if not line or line.startswith("#"):
                continue
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


def compress_folder(folder_path: str, archive_path: str) -> bool:
    """폴더 전체를 tar.gz로 압축합니다 (파일+하위폴더 recursive).

    Args:
        folder_path: 압축할 폴더 경로
        archive_path: 출력 압축 파일 경로 (.tar.gz)

    Returns:
        성공 여부
    """
    print(f"압축 시작: {folder_path} → {archive_path}")
    start = time.time()
    try:
        # tar.gz로 압축 (폴더 내용물만, 상위 경로 제외)
        ret = subprocess.run(
            ["tar", "-czf", archive_path, "-C", folder_path, "."],
            capture_output=True, text=True,
        )
        if ret.returncode != 0:
            print(f"tar 실패: {ret.stderr}")
            # fallback: Python shutil
            shutil.make_archive(
                archive_path.replace(".tar.gz", ""),
                "gztar",
                root_dir=folder_path,
            )
    except FileNotFoundError:
        # tar 명령이 없는 경우 (Windows 등)
        shutil.make_archive(
            archive_path.replace(".tar.gz", ""),
            "gztar",
            root_dir=folder_path,
        )

    elapsed = time.time() - start
    if os.path.isfile(archive_path):
        size_mb = os.path.getsize(archive_path) / (1024 * 1024)
        print(f"압축 완료: {size_mb:.1f}MB ({elapsed:.0f}초)")
        return True
    else:
        print("압축 실패: 파일이 생성되지 않았습니다.")
        return False


def upload_to_s3(
    file_path: str,
    bucket: str,
    s3_key: str,
    region: str,
    access_key: str,
    secret_key: str,
) -> str | None:
    """파일을 S3에 public-read로 업로드하고 URL을 반환합니다.

    boto3가 있으면 사용하고, 없으면 aws cli를 사용합니다.

    Returns:
        다운로드 URL 또는 None (실패 시)
    """
    print(f"S3 업로드 시작: {file_path} → s3://{bucket}/{s3_key}")
    start = time.time()

    # boto3 시도
    try:
        import boto3
        session = boto3.Session(
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=region,
        )
        s3 = session.client("s3")
        s3.upload_file(
            file_path, bucket, s3_key,
            ExtraArgs={"ACL": "public-read"},
        )
        elapsed = time.time() - start
        url = f"https://{bucket}.s3.{region}.amazonaws.com/{s3_key}"
        print(f"S3 업로드 완료: {elapsed:.0f}초")
        return url
    except ImportError:
        pass
    except Exception as e:
        print(f"boto3 업로드 실패: {e}, aws cli 시도...")

    # aws cli fallback
    env = os.environ.copy()
    env["AWS_ACCESS_KEY_ID"] = access_key
    env["AWS_SECRET_ACCESS_KEY"] = secret_key
    env["AWS_DEFAULT_REGION"] = region

    ret = subprocess.run(
        ["aws", "s3", "cp", file_path, f"s3://{bucket}/{s3_key}", "--acl", "public-read"],
        capture_output=True, text=True, env=env,
    )
    if ret.returncode == 0:
        elapsed = time.time() - start
        url = f"https://{bucket}.s3.{region}.amazonaws.com/{s3_key}"
        print(f"S3 업로드 완료 (aws cli): {elapsed:.0f}초")
        return url
    else:
        print(f"aws cli 업로드 실패: {ret.stderr}")
        return None


def do_shutdown(cloud: str | None, simplepod_id: str | None, simplepod_api_key: str | None):
    """서버를 종료합니다."""
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
        ret = os.system("shutdown -h now 2>/dev/null")
        if ret != 0:
            print("shutdown 명령 없음, kill 1 시도...")
            os.system("kill 1")


def main():
    parser = argparse.ArgumentParser(
        description="파일 수 모니터링 → 압축 → S3 업로드 → 디스코드 알림 → 서버 종료"
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
        "--work-name", "-w", type=str, required=True,
        help="작업명 (압축 파일명으로 사용)"
    )
    parser.add_argument(
        "--interval", "-i", type=int, default=60,
        help="체크 주기 (초, 기본값: 60)"
    )
    parser.add_argument(
        "--s3-prefix", type=str, default="dataset",
        help="S3 업로드 경로 (기본값: dataset)"
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
    aws_access_key = load_env_var("AWS_ACCESS_KEY_ID")
    aws_secret_key = load_env_var("AWS_SECRET_ACCESS_KEY")
    aws_bucket = load_env_var("AWS_S3_BUCKET")
    aws_region = load_env_var("AWS_S3_REGION") or "ap-northeast-2"

    # SimplePod/RunPod 감지
    simplepod_id = get_simplepod_instance_id()
    cloud = args.cloud
    if cloud is None:
        if simplepod_id:
            cloud = "simplepod"
        elif os.environ.get("RUNPOD_POD_ID"):
            cloud = "runpod"

    # 환경 정보 출력
    print("=" * 50)
    print(f"디스코드 웹훅: {'활성화' if webhook_url else '없음'}")
    if simplepod_id:
        print(f"SimplePod: {simplepod_id} (API 키: {'있음' if simplepod_api_key else '없음'})")
    if cloud:
        print(f"클라우드: {cloud}")
    print(f"S3 버킷: {aws_bucket or '미설정'} / 리전: {aws_region}")
    print(f"S3 업로드 경로: {args.s3_prefix}/{args.work_name}.tar.gz")
    print(f"모니터링: {args.path}")
    print(f"임계값: {args.max_files}개 / 체크 주기: {args.interval}초")
    print("=" * 50)

    if not aws_access_key or not aws_secret_key or not aws_bucket:
        print("WARNING: S3 설정 불완전 (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_S3_BUCKET)")
        print("  → 압축은 하지만 S3 업로드는 건너뜁니다.")

    try:
        while True:
            file_count = count_files(args.path)
            print(f"[{time.strftime('%H:%M:%S')}] 파일 수: {file_count} / {args.max_files}")

            if file_count >= args.max_files:
                hostname = socket.gethostname()
                print(f"\n*** 파일 수 {file_count}개 >= {args.max_files}개 — 압축 및 업로드 시작 ***")

                # 1. 압축
                archive_name = f"{args.work_name}.tar.gz"
                archive_path = os.path.join(os.path.dirname(args.path.rstrip("/")), archive_name)

                if not compress_folder(args.path, archive_path):
                    print("압축 실패, 종료합니다.")
                    if webhook_url:
                        send_discord_message(webhook_url,
                            f"🔴 **압축 실패** ({hostname})\n"
                            f"폴더 `{args.path}` 압축 중 오류 발생. 서버를 종료합니다.")
                    do_shutdown(cloud, simplepod_id, simplepod_api_key)
                    sys.exit(1)

                # 2. S3 업로드
                download_url = None
                if aws_access_key and aws_secret_key and aws_bucket:
                    s3_key = f"{args.s3_prefix}/{archive_name}"
                    download_url = upload_to_s3(
                        archive_path, aws_bucket, s3_key,
                        aws_region, aws_access_key, aws_secret_key,
                    )

                    # 업로드 성공하면 압축 파일 삭제
                    if download_url:
                        os.remove(archive_path)
                        print(f"압축 파일 삭제 완료: {archive_path}")
                    else:
                        print("S3 업로드 실패, 압축 파일을 유지합니다.")

                # 3. 디스코드 알림
                if webhook_url:
                    size_mb = os.path.getsize(archive_path) if os.path.isfile(archive_path) else 0
                    msg = (
                        f"🔴 **서버 종료 알림** ({hostname})\n"
                        f"폴더 `{args.path}` 파일 수 **{file_count}개** 도달\n"
                    )
                    if download_url:
                        msg += f"📦 압축 완료 + S3 업로드 완료\n"
                        msg += f"📥 다운로드: {download_url}\n"
                    elif os.path.isfile(archive_path):
                        msg += f"📦 압축 완료 (S3 업로드 실패, 로컬에 보관: `{archive_path}`)\n"
                    msg += f"5초 후 인스턴스를 종료합니다."
                    send_discord_message(webhook_url, msg)

                # 4. 카운트다운 후 종료
                for i in range(5, 0, -1):
                    print(f"  {i}...")
                    time.sleep(1)

                print("시스템을 종료합니다.")
                do_shutdown(cloud, simplepod_id, simplepod_api_key)
                sys.exit(0)

            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\n모니터링 중단됨.")


if __name__ == "__main__":
    main()
