import json
from urllib import request
import traceback


def queue_prompt(prompt_workflow, ip):
    p = {"prompt": prompt_workflow}
    data = json.dumps(p).encode('utf-8')

    req = request.Request(f"http://{ip}/prompt", data=data)
    res = request.urlopen(req)
    if res.code != 200:
        raise Exception(f"Error: {res.code} {res.reason}")
    return json.loads(res.read().decode('utf-8'))['prompt_id']




def get_remain_queue(ip):
    req = request.Request(f"http://{ip}/prompt", method='GET')
    res = request.urlopen(req)
    result = json.loads(res.read().decode('utf-8'))
    if res.code != 200:
        raise Exception(f"Error: {res.code} {res.reason}")

    return result['exec_info']['queue_remaining']


def get_most_idle_comfyui_server_simple(servers):
    most_idle_server = None
    min_queues = 99999999999999
    for comfyui_server in servers:
        try:
            num_queues = get_remain_queue(comfyui_server)
            print("num_queues", num_queues)
            if num_queues < min_queues:
                min_queues = num_queues
                most_idle_server = comfyui_server
        except Exception as e:

            traceback.print_exc()
            continue
    return most_idle_server, min_queues
