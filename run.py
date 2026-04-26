import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
import json
import sys
from typing import Union, OrderedDict
from urllib import request, error
from dotenv import load_dotenv
# Load the .env file if it exists
load_dotenv()

sys.path.insert(0, os.getcwd())
# must come before ANY torch or fastai imports
# import toolkit.cuda_malloc

# turn off diffusers telemetry until I can figure out how to make it opt-in
os.environ['DISABLE_TELEMETRY'] = 'YES'

# check if we have DEBUG_TOOLKIT in env
if os.environ.get("DEBUG_TOOLKIT", "0") == "1":
    # set torch to trace mode
    import torch
    torch.autograd.set_detect_anomaly(True)
import argparse
from toolkit.job import get_job
from toolkit.accelerator import get_accelerator
from toolkit.print import print_acc, setup_log_to_file

accelerator = get_accelerator()


SIMPLEPOD_INSTANCE_JSON = "/etc/simplepod/instance.json"


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_simplepod_instance_id():
    """SimplePod 인스턴스 hashId를 읽습니다."""
    if not os.path.isfile(SIMPLEPOD_INSTANCE_JSON):
        return None
    try:
        with open(SIMPLEPOD_INSTANCE_JSON, "r") as f:
            data = json.load(f)
        return data.get("hashId")
    except Exception:
        return None


def terminate_simplepod(instance_id, api_key):
    """SimplePod REST API로 인스턴스를 삭제합니다."""
    url = f"https://api.simplemining.net/instances/{instance_id}"
    req = request.Request(url, method="DELETE", headers={
        "X-AUTH-TOKEN": api_key,
        "User-Agent": "AiToolkit/1.0",
    })
    try:
        resp = request.urlopen(req, timeout=15)
        print_acc(f"SimplePod 인스턴스 삭제 성공 (status: {resp.code})")
        return True
    except error.HTTPError as e:
        print_acc(f"SimplePod API 오류: {e.code} {e.reason}")
        return False
    except Exception as e:
        print_acc(f"SimplePod API 요청 실패: {e}")
        return False


def delete_cloud_pod():
    """학습 완료 후 클라우드 팟(SimplePod / RunPod)을 삭제합니다."""
    simplepod_id = get_simplepod_instance_id()
    simplepod_api_key = os.environ.get("SIMPLEPOD_API_KEY")
    runpod_pod_id = os.environ.get("RUNPOD_POD_ID", "")

    if simplepod_id:
        if simplepod_api_key:
            print_acc(f"SimplePod 인스턴스 삭제: {simplepod_id}")
            if not terminate_simplepod(simplepod_id, simplepod_api_key):
                print_acc("API 실패, kill 1 시도...")
                os.system("kill 1")
        else:
            print_acc("SimplePod 인스턴스 ID는 있으나 API 키 없음, kill 1 시도...")
            os.system("kill 1")
    elif runpod_pod_id:
        print_acc(f"RunPod Pod 삭제: {runpod_pod_id}")
        ret = os.system(f"runpodctl remove pod {runpod_pod_id}")
        if ret != 0:
            print_acc("runpodctl 실패, kill 1 시도...")
            os.system("kill 1")
    else:
        print_acc("클라우드 팟을 감지할 수 없습니다 (SimplePod instance.json 없음, RUNPOD_POD_ID 없음). 팟 삭제를 건너뜁니다.")


def print_end_message(jobs_completed, jobs_failed):
    if not accelerator.is_main_process:
        return
    failure_string = f"{jobs_failed} failure{'' if jobs_failed == 1 else 's'}" if jobs_failed > 0 else ""
    completed_string = f"{jobs_completed} completed job{'' if jobs_completed == 1 else 's'}"

    print_acc("")
    print_acc("========================================")
    print_acc("Result:")
    if len(completed_string) > 0:
        print_acc(f" - {completed_string}")
    if len(failure_string) > 0:
        print_acc(f" - {failure_string}")
    print_acc("========================================")


def main():
    parser = argparse.ArgumentParser()

    # require at lease one config file
    parser.add_argument(
        'config_file_list',
        nargs='+',
        type=str,
        help='Name of config file (eg: person_v1 for config/person_v1.json/yaml), or full path if it is not in config folder, you can pass multiple config files and run them all sequentially'
    )

    # flag to continue if failed job
    parser.add_argument(
        '-r', '--recover',
        action='store_true',
        help='Continue running additional jobs even if a job fails'
    )

    # flag to continue if failed job
    parser.add_argument(
        '-n', '--name',
        type=str,
        default=None,
        help='Name to replace [name] tag in config file, useful for shared config file'
    )
    
    parser.add_argument(
        '-l', '--log',
        type=str,
        default=None,
        help='Log file to write output to'
    )

    parser.add_argument(
        '--delete_pod_after_training',
        type=str2bool,
        default=False,
        help='If True, delete the cloud pod (SimplePod or RunPod) after training completes'
    )
    args = parser.parse_args()
    
    if args.log is not None:
        setup_log_to_file(args.log)

    config_file_list = args.config_file_list
    if len(config_file_list) == 0:
        raise Exception("You must provide at least one config file")

    jobs_completed = 0
    jobs_failed = 0

    if accelerator.is_main_process:
        print_acc(f"Running {len(config_file_list)} job{'' if len(config_file_list) == 1 else 's'}")

    for config_file in config_file_list:
        try:
            job = get_job(config_file, args.name)
            job.run()
            job.cleanup()
            jobs_completed += 1
        except Exception as e:
            print_acc(f"Error running job: {e}")
            jobs_failed += 1
            try:
                job.process[0].on_error(e)
            except Exception as e2:
                print_acc(f"Error running on_error: {e2}")
            if not args.recover:
                print_end_message(jobs_completed, jobs_failed)
                raise e
        except KeyboardInterrupt as e:
            try:
                job.process[0].on_error(e)
            except Exception as e2:
                print_acc(f"Error running on_error: {e2}")
            if not args.recover:
                print_end_message(jobs_completed, jobs_failed)
                raise e

    print_end_message(jobs_completed, jobs_failed)

    if args.delete_pod_after_training and accelerator.is_main_process:
        print_acc("")
        print_acc("학습 완료 → 클라우드 팟 삭제를 시작합니다.")
        delete_cloud_pod()


if __name__ == '__main__':
    main()
