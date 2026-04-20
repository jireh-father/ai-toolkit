#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ComfyUI 대량 이미지 합성 요청 스크립트

여러 이미지를 ComfyUI 서버에 라운드로빈 방식으로 요청을 보냅니다.
"""

import argparse
import json
import os
import glob
import random
from itertools import cycle
import shutil
from PIL import Image
from urllib import request


# 랜덤 프롬프트 생성을 위한 속성 리스트
AGES = ["20yo", "30yo", "40yo", "50yo"]
COUNTRIES = [
    "korean", "japanese", "chinese", "thai", "vietnamese", "indian", "indonesian", "iranian",
    "philippines", "brazilian", "mexican", "american", "british", "german",
    "russian", "swedish", "nigerian", "egyptian"
]
EXPRESSIONS = [
    "laugh expression", "expressionless", "angry expression",
    "happy expression", "smile expression"
]
EYE_SIZES = ["big eyes", "small eyes"]
MOUTH_SIZES = ["big mouth", "small mouth"]

# 배경 변경을 위한 색상 리스트
BACKGROUND_COLORS = [
    "white", "black", "gray", "red", "blue", "green", "yellow", "orange",
    "pink", "purple", "beige", "brown", "navy", "sky blue", "mint",
    "ivory", "cream", "lavender", "coral", "teal"
]

# 배경 변경을 위한 실제 배경 리스트
REAL_BACKGROUNDS = [
    "ocean", "salon", "river", "cute room", "gorgeous room", "city", "night city",
    "beach", "sunset", "sunrise", "garden", "park",
    "library", "cafe", "restaurant", "office", "studio", "rooftop"
]

# 카메라 방향 리스트
CAMERA_DIRECTIONS = ["left", "right"]

# 옷 색상 리스트
CLOTH_COLORS = [
    "white", "black", "gray", "red", "blue", "green", "yellow", "orange",
    "pink", "purple", "beige", "brown", "navy", "sky blue", "mint",
    "ivory", "cream", "lavender", "coral", "teal"
]

# 상의 종류 리스트
CLOTH_TOP_TYPES = [
    "t-shirt", "shirt", "blouse", "sweater", "cardigan",
    "jacket", "coat", "vest", "tank top", "polo shirt", "turtleneck",
    "crop top", "sweatshirt", "blazer"
]

# 상의 긴팔, 반팔
CLOTH_TOP_LENGTHS = [
    "long sleeved", "short sleeved"
]

def queue_prompt(prompt_workflow, ip, cookie=None):
    p = {"prompt": prompt_workflow}
    data = json.dumps(p).encode('utf-8')

    req = request.Request(f"http://{ip}/prompt", data=data)
    if cookie:
        req.add_header("Cookie", cookie)
    res = request.urlopen(req)
    if res.code != 200:
        raise Exception(f"Error: {res.code} {res.reason}")
    return json.loads(res.read().decode('utf-8'))['prompt_id']

def find_node_by_class_type(workflow: dict, class_type: str) -> tuple[str, dict] | None:
    """
    워크플로우에서 특정 class_type을 가진 노드를 찾습니다.
    
    Args:
        workflow: 워크플로우 딕셔너리
        class_type: 찾을 노드의 class_type
        
    Returns:
        (노드 ID, 노드 딕셔너리) 튜플 또는 None
    """
    for node_id, node in workflow.items():
        if node.get("class_type") == class_type:
            return node_id, node
    return None

def find_node_by_class_type_and_title(workflow: dict, class_type: str) -> tuple[str, dict] | None:
    """
    워크플로우에서 특정 class_type을 가진 노드를 찾습니다.
    
    Args:
        workflow: 워크플로우 딕셔너리
        class_type: 찾을 노드의 class_type
        
    Returns:
        (노드 ID, 노드 딕셔너리) 튜플 또는 None
    """
    for node_id, node in workflow.items():
        if node.get("class_type") == class_type and node.get("_meta").get("title") == class_type:
            return node_id, node
    return None


def truncate_filename(name: str, ext: str = "", max_bytes: int = 200) -> str:
    """파일명을 리눅스 최대 파일명 길이에 맞게 자른다. ext는 확장자(.json 등)."""
    limit = max_bytes - len(ext.encode('utf-8'))
    while len(name.encode('utf-8')) > limit:
        name = name[:len(name) - 1]
    return name + ext


def calculate_resolution_for_width(width: int) -> int:
    """
    이미지 가로 길이보다 작으면서 64로 나눠지는 가장 큰 값을 계산합니다.
    
    예: 1024 -> 960, 880 -> 832
    
    Args:
        width: 이미지 가로 길이
        
    Returns:
        64로 나눠지는 resolution 값
    """
    return ((width - 1) // 64) * 64


def generate_random_prompt(gender: str) -> str:
    """
    랜덤 속성들을 조합하여 프롬프트를 생성합니다.
    
    Args:
        gender: 성별 (male 또는 female)
        
    Returns:
        조합된 프롬프트 문자열
    """
    age = random.choice(AGES)
    country = random.choice(COUNTRIES)
    expression = random.choice(EXPRESSIONS)
    eye_size = random.choice(EYE_SIZES)
    mouth_size = random.choice(MOUTH_SIZES)
    
    # 성별에 따라 man/woman 결정
    gender_word = "woman" if gender.lower() == "female" else "man"
    
    return f"{age} {country} {gender_word}, {expression}, {eye_size}, {mouth_size}"


def generate_background_prompt() -> str:
    """
    배경 변경을 위한 동적 프롬프트를 생성합니다.
    simple_color 또는 real 배경 중 랜덤으로 선택합니다.
    
    Returns:
        배경 변경 프롬프트 문자열
    """
    prompt_type = random.choice(["simple_color", "real"])
    
    if prompt_type == "simple_color":
        color = random.choice(BACKGROUND_COLORS)
        return f"change only background to simple {color} background"
    else:
        real_bg = random.choice(REAL_BACKGROUNDS)
        return f"change only background to {real_bg}"


def generate_camera_angle_prompt() -> str:
    """
    카메라 각도 변경을 위한 동적 프롬프트를 생성합니다.
    
    Returns:
        카메라 각도 변경 프롬프트 문자열
    """
    direction = random.choice(CAMERA_DIRECTIONS)
    return f"move the camera 5 degrees to the {direction}"


def generate_cloth_prompt() -> str:
    """
    옷 변경을 위한 동적 프롬프트를 생성합니다.
    
    Returns:
        옷 변경 프롬프트 문자열
    """
    color = random.choice(CLOTH_COLORS)
    cloth_type = random.choice(CLOTH_TOP_TYPES)
    cloth_length = random.choice(CLOTH_TOP_LENGTHS)
    return f"only change top to {cloth_length} {color} {cloth_type}"


def modify_workflow_random_face_change(workflow: dict, image_path: str, gender: str, output_dir: str = None) -> dict:
    """
    random_face_change 워크플로우를 수정합니다.
    
    수정 사항:
    1. LoadImage 노드: 이미지 파일 경로 설정
    2. SaveImageJpg 노드: filename_prefix를 입력 이미지 파일명(확장자 제외)으로 설정
    3. MediaPipe-FaceMeshPreprocessor 노드: resolution을 이미지 가로 길이보다 작은 64의 배수로 설정
    4. CLIPTextEncode 노드: 랜덤 생성된 프롬프트로 text 설정
    
    Args:
        workflow: 원본 워크플로우 딕셔너리
        image_path: 입력 이미지 경로
        gender: 성별 (male 또는 female)
        
    Returns:
        수정된 워크플로우 딕셔너리
    """
    # 워크플로우 복사본 생성
    modified_workflow = json.loads(json.dumps(workflow))
    
    # 이미지 파일명 (확장자 제외)
    image_filename = os.path.basename(image_path)
    image_name_without_ext = os.path.splitext(image_filename)[0]
    
    # 이미지 크기 가져오기
    with Image.open(image_path) as img:
        width, height = img.size
    
    # 1. LoadImage 노드 수정
    load_image_result = find_node_by_class_type(modified_workflow, "LoadImage")
    if load_image_result:
        node_id, node = load_image_result
        node["inputs"]["image"] = image_path
    
    # 2. SaveImageJpg 노드 수정
    save_image_result = find_node_by_class_type(modified_workflow, "SaveImageJpg")
    if save_image_result:
        node_id, node = save_image_result
        prefix = os.path.join(output_dir, image_name_without_ext) if output_dir else image_name_without_ext
        node["inputs"]["filename_prefix"] = prefix

    # 3. MediaPipe-FaceMeshPreprocessor 노드 수정
    face_mesh_result = find_node_by_class_type(modified_workflow, "MediaPipe-FaceMeshPreprocessor")
    if face_mesh_result:
        node_id, node = face_mesh_result
        resolution = calculate_resolution_for_width(width)
        node["inputs"]["resolution"] = resolution
    
    # 4. CLIPTextEncode 노드 수정 (첫 번째 발견되는 노드 - Positive 프롬프트)
    clip_text_result = find_node_by_class_type(modified_workflow, "CLIPTextEncode")
    if clip_text_result:
        node_id, node = clip_text_result
        random_prompt = generate_random_prompt(gender)
        node["inputs"]["text"] = random_prompt
    
    return modified_workflow


def modify_workflow_random_background_change(workflow: dict, image_path: str, output_dir: str = None) -> dict:
    """
    random_background_change 워크플로우를 수정합니다.
    
    수정 사항:
    1. LoadImage 노드: 이미지 파일 경로 설정
    2. SaveImageJpg 노드: filename_prefix를 입력 이미지 파일명(확장자 제외)으로 설정
    3. CLIPTextEncode 노드: 랜덤 생성된 배경 프롬프트로 text 설정
    
    Args:
        workflow: 원본 워크플로우 딕셔너리
        image_path: 입력 이미지 경로
        
    Returns:
        수정된 워크플로우 딕셔너리
    """
    # 워크플로우 복사본 생성
    modified_workflow = json.loads(json.dumps(workflow))
    
    # 이미지 파일명 (확장자 제외)
    image_filename = os.path.basename(image_path)
    image_name_without_ext = os.path.splitext(image_filename)[0]
    
    # 1. LoadImage 노드 수정
    load_image_result = find_node_by_class_type(modified_workflow, "LoadImage")
    if load_image_result:
        node_id, node = load_image_result
        node["inputs"]["image"] = image_path
    
    # 2. SaveImageJpg 노드 수정
    save_image_result = find_node_by_class_type(modified_workflow, "SaveImageJpg")
    if save_image_result:
        node_id, node = save_image_result
        prefix = os.path.join(output_dir, image_name_without_ext) if output_dir else image_name_without_ext
        node["inputs"]["filename_prefix"] = prefix

    # 3. CLIPTextEncode 노드 수정 (랜덤 배경 프롬프트)
    clip_text_result = find_node_by_class_type(modified_workflow, "CLIPTextEncode")
    if clip_text_result:
        node_id, node = clip_text_result
        background_prompt = generate_background_prompt()
        node["inputs"]["text"] = background_prompt
    
    return modified_workflow


def modify_workflow_with_prompt(workflow: dict, image_path: str, prompt: str, output_dir: str = None) -> dict:
    """
    범용 워크플로우 수정 함수입니다.
    이미지 경로와 프롬프트만 설정합니다.
    
    수정 사항:
    1. LoadImage 노드: 이미지 파일 경로 설정
    2. SaveImageJpg 노드: filename_prefix를 입력 이미지 파일명(확장자 제외)으로 설정
    3. CLIPTextEncode 노드: 주어진 프롬프트로 text 설정
    
    Args:
        workflow: 원본 워크플로우 딕셔너리
        image_path: 입력 이미지 경로
        prompt: 프롬프트 텍스트
        
    Returns:
        수정된 워크플로우 딕셔너리
    """
    # 워크플로우 복사본 생성
    modified_workflow = json.loads(json.dumps(workflow))
    
    # 이미지 파일명 (확장자 제외)
    image_filename = os.path.basename(image_path)
    image_name_without_ext = os.path.splitext(image_filename)[0]
    
    # 1. LoadImage 노드 수정
    load_image_result = find_node_by_class_type(modified_workflow, "LoadImage")
    if load_image_result:
        node_id, node = load_image_result
        node["inputs"]["image"] = image_path
    
    # 2. SaveImageJpg 노드 수정
    save_image_result = find_node_by_class_type(modified_workflow, "SaveImageJpg")
    if save_image_result:
        node_id, node = save_image_result
        prefix = os.path.join(output_dir, image_name_without_ext) if output_dir else image_name_without_ext
        node["inputs"]["filename_prefix"] = prefix

    # 3. CLIPTextEncode 노드 수정
    clip_text_result = find_node_by_class_type_and_title(modified_workflow, "TextEncodeQwenImageEditPlus")
    if clip_text_result:
        node_id, node = clip_text_result
        node["inputs"]["prompt"] = prompt
    
    return modified_workflow


def get_image_files(image_dir: str) -> list[str]:
    """
    이미지 디렉토리에서 이미지 파일 목록을 가져옵니다.
    콤마로 구분된 여러 디렉토리를 지원합니다.
    
    Args:
        image_dir: 이미지 파일들이 있는 디렉토리 경로 (콤마로 구분 가능)
        
    Returns:
        이미지 파일 절대 경로 리스트 (정렬됨)
    """
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.webp', '*.bmp']
    image_files = []
    
    # 콤마로 구분된 여러 디렉토리 처리
    dirs = [d.strip() for d in image_dir.split(',')]
    
    for dir_path in dirs:
        # 절대 경로로 변환
        abs_image_dir = os.path.abspath(dir_path)
        
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(abs_image_dir, ext)))
            image_files.extend(glob.glob(os.path.join(abs_image_dir, ext.upper())))
    
    return sorted(set(image_files))


def load_workflow(workflow_path: str) -> dict:
    """
    ComfyUI 워크플로우 JSON 파일을 로드합니다.
    
    Args:
        workflow_path: 워크플로우 JSON 파일 경로
        
    Returns:
        워크플로우 딕셔너리
    """
    with open(workflow_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def modify_workflow_for_image(
    workflow: dict,
    workflow_type: str,
    image_path: str,
    gender: str,
    output_dir: str = None,
) -> dict:
    """
    워크플로우 타입에 따라 이미지 경로, MP 이미지 사이즈, 프롬프트를 동적으로 변경합니다.
    
    Args:
        workflow: 원본 워크플로우 딕셔너리
        workflow_type: 워크플로우 타입 (예: random_face_change)
        image_path: 입력 이미지 경로
        gender: 성별 (male 또는 female)
        mp_image_size: MP 이미지 사이즈 (width, height)
        prompt: 프롬프트 텍스트
        
    Returns:
        수정된 워크플로우 딕셔너리
    """
    if workflow_type == "random_face_change":
        return modify_workflow_random_face_change(workflow, image_path, gender, output_dir=output_dir)
    elif workflow_type == "random_background_change":
        return modify_workflow_random_background_change(workflow, image_path, output_dir=output_dir)
    elif workflow_type == "random_camera_angle_move":
        return modify_workflow_with_prompt(workflow, image_path, generate_camera_angle_prompt(), output_dir=output_dir)
    elif workflow_type == "random_cloth_change":
        return modify_workflow_with_prompt(workflow, image_path, generate_cloth_prompt(), output_dir=output_dir)
    else:
        # 알 수 없는 워크플로우 타입은 원본 그대로 반환
        return workflow


def modify_workflow_qwen_hairstyle_edit(
    workflow: dict, image_path1: str, image_path2: str, gen_index: int,
    steps: int = None, cfg: float = None, scheduler: str = None,
    unet_name: str = None, lora_name: str = None,
    lightning_lora_name: str = None,
    prompt: str = None, output_dir: str = None,
) -> dict:
    """
    qwen_hairstyle_edit / qwen_lora_hairstyle_edit 워크플로우를 수정합니다.

    수정 사항:
    1. 노드 137(LoadImage): 이미지 1 파일 경로 설정
    2. 노드 78(LoadImage): 이미지 2 파일 경로 설정
    3. 노드 138(SaveImageJpg): filename_prefix 설정
    4. 노드 3(KSampler): steps, cfg 설정 (옵션)
    5. 노드 132(UNETLoader): unet_name 설정 (옵션)
    6. 노드 134(LoraLoaderModelOnly): lora_name 설정 (옵션)
    7. 노드 143(LoraLoaderModelOnly, Lightning LoRA): lightning_lora_name 설정 (옵션, 노드가 존재하는 워크플로우에만 적용)
    8. 노드 141(TextEncodeQwenImageEditPlus): prompt 설정 (옵션)

    Args:
        workflow: 원본 워크플로우 딕셔너리
        image_path1: 입력 이미지 1 경로 (노드 137)
        image_path2: 입력 이미지 2 경로 (노드 78)
        gen_index: 생성 인덱스 (파일명에 사용)
        steps: KSampler steps (None이면 워크플로우 원본값 사용)
        cfg: KSampler cfg (None이면 워크플로우 원본값 사용)
        unet_name: UNETLoader unet_name (None이면 워크플로우 원본값 사용)
        lora_name: LoraLoaderModelOnly lora_name (None이면 워크플로우 원본값 사용)
        lightning_lora_name: 노드 143 Lightning LoRA lora_name (None이면 워크플로우 원본값 사용)
        prompt: TextEncodeQwenImageEditPlus prompt (None이면 워크플로우 원본값 사용)

    Returns:
        수정된 워크플로우 딕셔너리
    """
    # 워크플로우 복사본 생성
    modified_workflow = json.loads(json.dumps(workflow))
    
    # 이미지 파일명 (확장자 제외)
    image1_filename = os.path.basename(image_path1)
    image1_name_without_ext = os.path.splitext(image1_filename)[0]
    image2_filename = os.path.basename(image_path2)
    image2_name_without_ext = os.path.splitext(image2_filename)[0]
    
    # 1. 노드 78(LoadImage) 수정 - 이미지 1
    if "78" in modified_workflow:
        modified_workflow["78"]["inputs"]["image"] = image_path1
    
    # 2. 노드 106(LoadImage) 수정 - 이미지 2
    if "106" in modified_workflow:
        modified_workflow["106"]["inputs"]["image"] = image_path2
    
    # 3. 노드 138(SaveImageJpg) 수정 - filename_prefix 설정
    output_prefix = truncate_filename(f"{image1_name_without_ext}_{image2_name_without_ext}_{gen_index:04d}")
    if "138" in modified_workflow:
        prefix = os.path.join(output_dir, output_prefix) if output_dir else output_prefix
        modified_workflow["138"]["inputs"]["filename_prefix"] = prefix

    # 4. 노드 139(SaveImageJpg) 수정 - input_image/ prefix 추가
    if "139" in modified_workflow:
        prefix = os.path.join(output_dir, "input_image", output_prefix) if output_dir else os.path.join("input_image", output_prefix)
        modified_workflow["139"]["inputs"]["filename_prefix"] = prefix

    # 5. 노드 140(SaveImageJpg) 수정 - reference_image/ prefix 추가
    if "140" in modified_workflow:
        prefix = os.path.join(output_dir, "reference_image", output_prefix) if output_dir else os.path.join("reference_image", output_prefix)
        modified_workflow["140"]["inputs"]["filename_prefix"] = prefix

    # 6. 노드 3(KSampler) 수정 - seed 랜덤 설정 + steps/cfg/scheduler 오버라이드
    if "3" in modified_workflow:
        modified_workflow["3"]["inputs"]["seed"] = random.randint(0, 2**64 - 1)
        if steps is not None:
            modified_workflow["3"]["inputs"]["steps"] = steps
        if cfg is not None:
            modified_workflow["3"]["inputs"]["cfg"] = cfg
        if scheduler is not None:
            modified_workflow["3"]["inputs"]["scheduler"] = scheduler

    # 7. 노드 132(UNETLoader) 수정 - unet_name 오버라이드
    if unet_name is not None and "132" in modified_workflow:
        modified_workflow["132"]["inputs"]["unet_name"] = unet_name

    # 8. 노드 134(LoraLoaderModelOnly) 수정 - lora_name 오버라이드
    if lora_name is not None and "134" in modified_workflow:
        modified_workflow["134"]["inputs"]["lora_name"] = lora_name

    # 8-1. 노드 143(LoraLoaderModelOnly, Lightning LoRA) 수정 - lightning_lora_name 오버라이드
    if lightning_lora_name is not None and "143" in modified_workflow:
        modified_workflow["143"]["inputs"]["lora_name"] = lightning_lora_name

    # 9. 노드 141(TextEncodeQwenImageEditPlus) 수정 - prompt 오버라이드
    if prompt is not None and "141" in modified_workflow:
        modified_workflow["141"]["inputs"]["prompt"] = prompt

    return modified_workflow


def modify_workflow_qwen_nunchaku_lora_hairstyle_edit(
    workflow: dict, image_path1: str, image_path2: str, gen_index: int,
    steps: int = None, cfg: float = None, scheduler: str = None,
    unet_name: str = None, lora_name: str = None,
    lightning_lora_name: str = None,
    prompt: str = None, output_dir: str = None,
) -> dict:
    """
    qwen_nunchaku_lora_hairstyle_edit 워크플로우를 수정합니다.

    노드 매핑 (기존 qwen_hairstyle_edit 대비):
    - LoadImage 1/2: 78 / 106 (동일)
    - KSampler: 3 (동일)
    - 모델 로더: 115 (NunchakuQwenImageDiTLoader, `model_name` 필드) — 기존 132 UNETLoader 대체
    - 헤어 LoRA: 117 (NunchakuQwenImageLoraLoader) — 기존 134 대체
    - Lightning LoRA: 132 (NunchakuQwenImageLoraLoader) — 기존 143 대체
    - Prompt: 133 (TextEncodeQwenImageEditPlus) — 기존 141 대체
    - SaveImageJpg 결과/입력/레퍼런스: 137 / 135 / 136 — 기존 138 / 139 / 140 대체

    Returns:
        수정된 워크플로우 딕셔너리
    """
    modified_workflow = json.loads(json.dumps(workflow))

    image1_filename = os.path.basename(image_path1)
    image1_name_without_ext = os.path.splitext(image1_filename)[0]
    image2_filename = os.path.basename(image_path2)
    image2_name_without_ext = os.path.splitext(image2_filename)[0]

    # 1. LoadImage (이미지 1 / 이미지 2)
    if "78" in modified_workflow:
        modified_workflow["78"]["inputs"]["image"] = image_path1
    if "106" in modified_workflow:
        modified_workflow["106"]["inputs"]["image"] = image_path2

    output_prefix = truncate_filename(f"{image1_name_without_ext}_{image2_name_without_ext}_{gen_index:04d}")

    # 2. SaveImageJpg (최종 결과: 137)
    if "137" in modified_workflow:
        prefix = os.path.join(output_dir, output_prefix) if output_dir else output_prefix
        modified_workflow["137"]["inputs"]["filename_prefix"] = prefix

    # 3. SaveImageJpg (입력 이미지: 135)
    if "135" in modified_workflow:
        prefix = os.path.join(output_dir, "input_image", output_prefix) if output_dir else os.path.join("input_image", output_prefix)
        modified_workflow["135"]["inputs"]["filename_prefix"] = prefix

    # 4. SaveImageJpg (레퍼런스 이미지: 136)
    if "136" in modified_workflow:
        prefix = os.path.join(output_dir, "reference_image", output_prefix) if output_dir else os.path.join("reference_image", output_prefix)
        modified_workflow["136"]["inputs"]["filename_prefix"] = prefix

    # 5. KSampler (노드 3) - seed 랜덤 + steps/cfg/scheduler 오버라이드
    if "3" in modified_workflow:
        modified_workflow["3"]["inputs"]["seed"] = random.randint(0, 2**64 - 1)
        if steps is not None:
            modified_workflow["3"]["inputs"]["steps"] = steps
        if cfg is not None:
            modified_workflow["3"]["inputs"]["cfg"] = cfg
        if scheduler is not None:
            modified_workflow["3"]["inputs"]["scheduler"] = scheduler

    # 6. NunchakuQwenImageDiTLoader (노드 115) - model_name 오버라이드 (unet_name 인자 재사용)
    if unet_name is not None and "115" in modified_workflow:
        modified_workflow["115"]["inputs"]["model_name"] = unet_name

    # 7. NunchakuQwenImageLoraLoader (노드 117, 헤어스타일 LoRA) - lora_name 오버라이드
    if lora_name is not None and "117" in modified_workflow:
        modified_workflow["117"]["inputs"]["lora_name"] = lora_name

    # 8. NunchakuQwenImageLoraLoader (노드 132, Lightning LoRA) - lightning_lora_name 오버라이드
    if lightning_lora_name is not None and "132" in modified_workflow:
        modified_workflow["132"]["inputs"]["lora_name"] = lightning_lora_name

    # 9. TextEncodeQwenImageEditPlus (노드 133) - prompt 오버라이드
    if prompt is not None and "133" in modified_workflow:
        modified_workflow["133"]["inputs"]["prompt"] = prompt

    return modified_workflow


def batch_request_qwen_hairstyle_edit(
    image_dir: str,
    workflow_path: str,
    comfyui_hosts: list[str],
    output_workflow_dir: str,
    output_dir: str,
    num_gens: int,
    workflow_type: str = "qwen_hairstyle_edit",
    force_request: bool = False,
    cookie: str = None,
    target_keywords: list[str] = None,
    steps: int = None,
    cfg: float = None,
    scheduler: str = None,
    unet_name: str = None,
    lora_name: str = None,
    lightning_lora_name: str = None,
    prompt: str = None,
) -> dict[str, str]:
    """
    qwen_hairstyle_edit / qwen_lora_hairstyle_edit 워크플로우를 위한 배치 요청 함수입니다.
    num_gens만큼 랜덤으로 2개의 이미지를 선택하여 워크플로우를 생성합니다.

    Args:
        image_dir: 이미지 파일들이 있는 디렉토리 경로 (콤마로 구분 가능)
        workflow_path: ComfyUI 워크플로우 JSON 파일 경로
        comfyui_hosts: ComfyUI 서버 호스트 목록 (ip:port 형식)
        output_workflow_dir: ComfyUI 워크플로우 JSON 파일 저장 디렉토리
        output_dir: 출력 이미지 디렉토리 경로
        num_gens: 생성할 워크플로우 수
        force_request: True면 무조건 요청, False면 output_dir에 파일 존재시 스킵
        steps: KSampler steps (None이면 워크플로우 원본값)
        cfg: KSampler cfg (None이면 워크플로우 원본값)
        unet_name: UNETLoader unet_name (None이면 워크플로우 원본값)
        lora_name: LoraLoaderModelOnly lora_name (None이면 워크플로우 원본값)
        lightning_lora_name: 노드 143 Lightning LoRA lora_name (None이면 워크플로우 원본값)
        prompt: TextEncodeQwenImageEditPlus prompt (None이면 워크플로우 원본값)

    Returns:
        이미지 경로와 prompt_id 매핑 딕셔너리
    """
    # 이미지 파일 목록 가져오기
    image_files = get_image_files(image_dir)

    if len(image_files) < 2:
        print(f"경고: 최소 2개의 이미지 파일이 필요합니다. 현재: {len(image_files)}개")
        return {}

    # target_keywords가 있으면 reference 후보를 키워드 매칭 파일로 제한
    if target_keywords:
        ref_files = [f for f in image_files if any(kw.lower() in os.path.basename(f).lower() for kw in target_keywords)]
        print(f"총 {len(image_files)}개의 이미지 파일 중 target_keywords 매칭: {len(ref_files)}개 (reference 후보)")
        if not ref_files:
            print(f"경고: target_keywords {target_keywords}에 매칭되는 파일이 없습니다.")
            return {}
    else:
        ref_files = image_files

    print(f"총 {len(image_files)}개의 이미지 파일을 발견했습니다.")
    print(f"{num_gens}개의 워크플로우를 생성합니다.")
    
    # 워크플로우 로드
    base_workflow = load_workflow(workflow_path)
    
    # 라운드로빈을 위한 호스트 순환자
    host_cycle = cycle(comfyui_hosts)
    
    # 결과 저장
    results = {}
    skipped_count = 0
    
    for gen_idx in range(num_gens):
        # 랜덤으로 이미지 선택: input은 전체에서, reference는 ref_files에서
        image_path1 = random.choice(image_files)
        image_path2 = random.choice(ref_files)
        while image_path2 == image_path1:
            image_path2 = random.choice(ref_files)
        
        # 파일명 생성을 위한 이름 추출
        image1_name = os.path.splitext(os.path.basename(image_path1))[0]
        image2_name = os.path.splitext(os.path.basename(image_path2))[0]
        output_prefix = truncate_filename(f"{image1_name}_{image2_name}_{gen_idx:04d}")
        
        # force_request가 False이면 output_dir에 파일 존재 여부 확인
        if not force_request:
            existing_files = glob.glob(os.path.join(output_dir, f"{output_prefix}.*"))
            if existing_files:
                skipped_count += 1
                print(f"[{gen_idx + 1}/{num_gens}] {output_prefix} -> 스킵 (이미 존재: {os.path.basename(existing_files[0])})")
                continue
        
        # 현재 호스트 선택 (라운드로빈)
        current_host = next(host_cycle)
        
        # 워크플로우 수정 (nunchaku 워크플로우는 노드 ID가 다르므로 별도 함수 사용)
        modify_fn = (
            modify_workflow_qwen_nunchaku_lora_hairstyle_edit
            if workflow_type == "qwen_nunchaku_lora_hairstyle_edit"
            else modify_workflow_qwen_hairstyle_edit
        )
        modified_workflow = modify_fn(
            workflow=base_workflow,
            image_path1=image_path1,
            image_path2=image_path2,
            gen_index=gen_idx,
            steps=steps,
            cfg=cfg,
            scheduler=scheduler,
            unet_name=unet_name,
            lora_name=lora_name,
            lightning_lora_name=lightning_lora_name,
            output_dir=output_dir,
            prompt=prompt,
        )
        
        if output_workflow_dir:
            # 워크플로우 저장
            workflow_filename = truncate_filename(output_prefix, ext=".json")
            json.dump(modified_workflow, open(os.path.join(output_workflow_dir, workflow_filename), 'w+'), indent=2, ensure_ascii=False)
        
        try:
            # ComfyUI에 요청
            prompt_id = queue_prompt(modified_workflow, current_host, cookie=cookie)
            results[output_prefix] = prompt_id
            print(f"[{gen_idx + 1}/{num_gens}] {output_prefix} -> {current_host} (prompt_id: {prompt_id})")
        except Exception as e:
            print(f"[{gen_idx + 1}/{num_gens}] {output_prefix} -> {current_host} 요청 실패: {e}")
            results[output_prefix] = None
    
    if skipped_count > 0:
        print(f"\n스킵된 워크플로우: {skipped_count}개")
    
    return results

batch_request_qwen_lightning_hairstyle_edit = batch_request_qwen_hairstyle_edit


def batch_request_to_comfyui(
    image_dir: str,
    workflow_path: str,
    workflow_type: str,
    comfyui_hosts: list[str],
    gender: str,
    output_workflow_dir: str,
    output_dir: str,
    force_request: bool = False,
    cookie: str = None,
) -> dict[str, str]:
    """
    이미지 파일들을 라운드로빈 방식으로 ComfyUI 서버에 요청합니다.
    
    Args:
        image_dir: 이미지 파일들이 있는 디렉토리 경로
        workflow_path: ComfyUI 워크플로우 JSON 파일 경로
        workflow_type: 워크플로우 타입
        comfyui_hosts: ComfyUI 서버 호스트 목록 (ip:port 형식)
        gender: 성별 (male 또는 female)
        output_workflow_dir: ComfyUI 워크플로우 JSON 파일 저장 디렉토리
        output_dir: 출력 이미지 디렉토리 경로
        force_request: True면 무조건 요청, False면 output_dir에 파일 존재시 스킵
    Returns:
        이미지 경로와 prompt_id 매핑 딕셔너리
    """
    # 이미지 파일 목록 가져오기
    image_files = get_image_files(image_dir)
    
    if not image_files:
        print(f"경고: {image_dir}에서 이미지 파일을 찾을 수 없습니다.")
        return {}
    
    print(f"총 {len(image_files)}개의 이미지 파일을 발견했습니다.")
    
    # 워크플로우 로드
    base_workflow = load_workflow(workflow_path)
    
    # 라운드로빈을 위한 호스트 순환자
    host_cycle = cycle(comfyui_hosts)
    
    # 결과 저장
    results = {}
    skipped_count = 0
    
    for idx, image_path in enumerate(image_files):
        # force_request가 False이면 output_dir에 파일 존재 여부 확인
        if not force_request:
            image_filename = os.path.basename(image_path)
            image_name_without_ext = os.path.splitext(image_filename)[0]
            # output_dir에서 동일한 파일명(확장자 무관)이 있는지 확인
            existing_files = glob.glob(os.path.join(output_dir, f"{image_name_without_ext}.*"))
            if existing_files:
                skipped_count += 1
                print(f"[{idx + 1}/{len(image_files)}] {image_filename} -> 스킵 (이미 존재: {os.path.basename(existing_files[0])})")
                continue
        
        # 현재 호스트 선택 (라운드로빈)
        current_host = next(host_cycle)
        
        # 워크플로우 수정
        modified_workflow = modify_workflow_for_image(
            workflow=base_workflow,
            workflow_type=workflow_type,
            image_path=image_path,
            gender=gender,
            output_dir=output_dir,
        )
        
        if output_workflow_dir:
            # 워크플로우 저장
            wf_filename = truncate_filename(os.path.basename(image_path), ext=".json")
            json.dump(modified_workflow, open(os.path.join(output_workflow_dir, wf_filename), 'w+'), indent=2, ensure_ascii=False)
        
        try:
            # ComfyUI에 요청
            prompt_id = queue_prompt(modified_workflow, current_host, cookie=cookie)
            results[image_path] = prompt_id
            print(f"[{idx + 1}/{len(image_files)}] {os.path.basename(image_path)} -> {current_host} (prompt_id: {prompt_id})")
        except Exception as e:
            print(f"[{idx + 1}/{len(image_files)}] {os.path.basename(image_path)} -> {current_host} 요청 실패: {e}")
            results[image_path] = None
    
    if skipped_count > 0:
        print(f"\n스킵된 이미지: {skipped_count}개")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='ComfyUI 대량 이미지 합성 요청 스크립트',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
사용 예시:
  python comfyui_batch_request.py --image_dir ./images --workflow ./workflow.json --hosts 192.168.1.100:8188 192.168.1.101:8188
        '''
    )
    
    parser.add_argument(
        '--image_dir',
        type=str,
        required=True,
        help='이미지 파일들이 있는 디렉토리 경로'
    )
    
    parser.add_argument(
        '--workflow_dir',
        type=str,
        default='./scripts/comfyui_workflows',
        help='ComfyUI 워크플로우 디렉토리 경로 (기본값: ./scripts/comfyui_workflows)'
    )
    
    parser.add_argument(
        '--workflow_type',
        type=str,
        default='random_face_change',
        choices=['random_face_change', 'random_background_change', 'random_camera_angle_move', 'random_cloth_change', 'qwen_hairstyle_edit', 'qwen_lightning_hairstyle_edit', 'qwen_lora_hairstyle_edit', 'qwen_nunchaku_lora_hairstyle_edit'],
        help='워크플로우 타입 (기본값: random_face_change)'
    )
    
    parser.add_argument(
        '--num_gens',
        type=int,
        default=100,
        help='생성할 워크플로우 수 (qwen_hairstyle_edit 전용, 기본값: 100)'
    )
    
    parser.add_argument(
        '--hosts',
        type=str,
        nargs='+',
        default=[
            '127.0.0.1:8188', '127.0.0.1:8189', '127.0.0.1:8190', '127.0.0.1:8191',
            '127.0.0.1:8192', '127.0.0.1:8193', '127.0.0.1:8194', '127.0.0.1:8195',
            '127.0.0.1:8196', '127.0.0.1:8197'
        ],
        dest='comfyui_hosts',
        metavar='IP:PORT',
        help='ComfyUI 서버 호스트 목록 (기본값: 127.0.0.1:8188~8197)'
    )
    
    parser.add_argument(
        '--gender',
        type=str,
        default='female',
        choices=['male', 'female'],
        help='성별 (기본값: female)'
    )

    #seed
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='랜덤 시드 (기본값: None, 매 실행마다 다른 결과)'
    )

    #output_workflow_dir
    parser.add_argument(
        '--output_workflow_dir',
        type=str,
        default=None,
        help='ComfyUI 워크플로우 JSON 파일 저장 디렉토리'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='출력 이미지 디렉토리 경로 (필수)'
    )
    
    parser.add_argument(
        '--force_request',
        action='store_true',
        default=False,
        help='True면 무조건 요청, False면 output_dir에 파일 존재시 스킵 (기본값: False)'
    )

    parser.add_argument(
        '--cookie',
        type=str,
        default=None,
        help='ComfyUI 인증 쿠키 (예: "session=abc123")'
    )

    parser.add_argument(
        '--target_keywords',
        type=str,
        nargs='+',
        default=None,
        help='reference 이미지 선정시 파일명에 포함되어야 할 키워드 목록 (예: --target_keywords 단발 숏컷)'
    )

    # qwen_hairstyle_edit / qwen_lightning_hairstyle_edit 전용 옵션
    parser.add_argument(
        '--steps',
        type=int,
        default=4,
        help='KSampler steps (qwen_hairstyle_edit 전용, 기본값: 워크플로우 원본값)'
    )
    parser.add_argument(
        '--cfg',
        type=float,
        default=1,
        help='KSampler cfg (qwen_hairstyle_edit 전용, 기본값: 워크플로우 원본값)'
    )
    parser.add_argument(
        '--scheduler',
        type=str,
        default=None,
        help='KSampler scheduler (qwen 전용, 기본값: None = 워크플로우 원본값)'
    )
    parser.add_argument(
        '--unet_name',
        type=str,
        default="qwen_image_edit_2511_fp8_e4m3fn_scaled_lightning_comfyui_4steps_v1.0.safetensors",
        help='UNETLoader unet_name (qwen_hairstyle_edit 전용, 기본값: 워크플로우 원본값)'
    )
    parser.add_argument(
        '--lora_name',
        type=str,
        default=None,
        help='LoraLoaderModelOnly lora_name (qwen_hairstyle_edit 전용, 기본값: 워크플로우 원본값)'
    )
    parser.add_argument(
        '--lightning_lora_name',
        type=str,
        default=None,
        help='노드 143 Lightning LoRA lora_name (qwen_lora_hairstyle_edit 전용, 기본값: 워크플로우 원본값)'
    )
    parser.add_argument(
        '--prompt',
        type=str,
        default="change only the hairstyle of the person in Image 1 to match the hairstyle of the person in Image 2.",
        help='TextEncodeQwenImageEditPlus prompt (qwen_hairstyle_edit 전용, 기본값: 워크플로우 원본값)'
    )
    
    args = parser.parse_args()

    if args.output_workflow_dir:
        os.makedirs(args.output_workflow_dir, exist_ok=True)
    
    # 입력 검증 (콤마로 구분된 디렉토리 각각 체크)
    for dir_path in [d.strip() for d in args.image_dir.split(',')]:
        if not os.path.isdir(dir_path):
            print(f"오류: 이미지 디렉토리를 찾을 수 없습니다: {dir_path}")
            return 1
    
    # output_dir 및 하위 폴더 생성
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "reference_image"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "input_image"), exist_ok=True)
    
    # 워크플로우 경로 조합
    workflow_path = f"{args.workflow_dir}/{args.workflow_type}.json"
    
    if not os.path.isfile(workflow_path):
        print(f"오류: 워크플로우 파일을 찾을 수 없습니다: {workflow_path}")
        return 1
    
    print("=" * 60)
    print("ComfyUI 대량 이미지 합성 요청")
    print("=" * 60)
    print(f"이미지 디렉토리: {args.image_dir}")
    print(f"워크플로우 파일: {workflow_path}")
    print(f"워크플로우 타입: {args.workflow_type}")
    print(f"성별: {args.gender}")
    print(f"출력 디렉토리: {args.output_dir}")
    print(f"강제 요청: {args.force_request}")
    print(f"ComfyUI 호스트: {', '.join(args.comfyui_hosts)}")
    if args.workflow_type in ["qwen_hairstyle_edit", "qwen_lightning_hairstyle_edit", "qwen_lora_hairstyle_edit", "qwen_nunchaku_lora_hairstyle_edit"]:
        if args.steps is not None:
            print(f"KSampler steps: {args.steps}")
        if args.cfg is not None:
            print(f"KSampler cfg: {args.cfg}")
        if args.scheduler is not None:
            print(f"KSampler scheduler: {args.scheduler}")
        if args.unet_name is not None:
            print(f"UNETLoader unet_name: {args.unet_name}")
        if args.lora_name is not None:
            print(f"LoraLoader lora_name: {args.lora_name}")
        if args.lightning_lora_name is not None:
            print(f"Lightning LoraLoader lora_name: {args.lightning_lora_name}")
        if args.prompt is not None:
            print(f"Prompt: {args.prompt}")
    print("=" * 60)

    random.seed(args.seed)
    
    # 배치 요청 실행
    if args.workflow_type in ["qwen_hairstyle_edit", "qwen_lightning_hairstyle_edit", "qwen_lora_hairstyle_edit", "qwen_nunchaku_lora_hairstyle_edit"]:
        results = batch_request_qwen_hairstyle_edit(
            image_dir=args.image_dir,
            workflow_path=workflow_path,
            comfyui_hosts=args.comfyui_hosts,
            output_workflow_dir=args.output_workflow_dir,
            output_dir=args.output_dir,
            num_gens=args.num_gens,
            workflow_type=args.workflow_type,
            force_request=args.force_request,
            cookie=args.cookie,
            target_keywords=args.target_keywords,
            steps=args.steps,
            cfg=args.cfg,
            scheduler=args.scheduler,
            unet_name=args.unet_name,
            lora_name=args.lora_name,
            lightning_lora_name=args.lightning_lora_name,
            prompt=args.prompt,
        )
    else:
        results = batch_request_to_comfyui(
            image_dir=args.image_dir,
            workflow_path=workflow_path,
            workflow_type=args.workflow_type,
            comfyui_hosts=args.comfyui_hosts,
            gender=args.gender,
            output_workflow_dir=args.output_workflow_dir,
            output_dir=args.output_dir,
            force_request=args.force_request,
            cookie=args.cookie
        )
    
    # 결과 요약
    success_count = sum(1 for v in results.values() if v is not None)
    fail_count = len(results) - success_count
    
    print("=" * 60)
    print(f"완료: 성공 {success_count}개, 실패 {fail_count}개")
    print("=" * 60)
    
    return 0 if fail_count == 0 else 1


if __name__ == '__main__':
    exit(main())
