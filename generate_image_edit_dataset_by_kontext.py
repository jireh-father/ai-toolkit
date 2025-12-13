import torch
import argparse
from diffusers import FluxKontextPipeline, QwenImageEditPlusPipeline
from PIL import Image
import numpy as np
import os
from pathlib import Path
import random

# 동적 프롬프트 생성을 위한 색상 리스트
BACKGROUND_COLORS = [
    "white", "black", "gray", "red", "blue", "green", "yellow", "orange",
    "pink", "purple", "beige", "brown", "navy", "sky blue", "mint",
    "ivory", "cream", "lavender", "coral", "teal"
]

# 동적 프롬프트 생성을 위한 실제 배경 리스트
REAL_BACKGROUNDS = [
    "ocean", "salon", "river", "cute room", "gorgeous room", "city", "night city",
    "beach", "sunset", "sunrise", "garden", "park",
    "library", "cafe", "restaurant", "office", "studio", "rooftop"
]


def generate_background_prompt():
    """배경 변경을 위한 동적 프롬프트 생성"""
    prompt_type = random.choice(["simple_color", "real"])
    
    if prompt_type == "simple_color":
        color = random.choice(BACKGROUND_COLORS)
        return f"change only background to {color} background"
    else:
        real_bg = random.choice(REAL_BACKGROUNDS)
        return f"change only background to {real_bg}"


# 카메라 방향 리스트
CAMERA_DIRECTIONS = ["left", "right"]


def generate_camera_angle_prompt():
    """카메라 각도 변경을 위한 동적 프롬프트 생성"""
    direction = random.choice(CAMERA_DIRECTIONS)
    return f"move the camera 5 degrees to the {direction}"


# 옷 색상 리스트
CLOTH_COLORS = [
    "white", "black", "gray", "red", "blue", "green", "yellow", "orange",
    "pink", "purple", "beige", "brown", "navy", "sky blue", "mint",
    "ivory", "cream", "lavender", "coral", "teal"
]

# 상의 종류 리스트
CLOTH_TOP_TYPES = [
    "t-shirt", "shirt", "blouse", "sweater", "hoodie", "cardigan",
    "jacket", "coat", "vest", "tank top", "polo shirt", "turtleneck",
    "crop top", "sweatshirt", "blazer"
]

# 상의 긴팔, 반팔
CLOTH_TOP_LENGTHS = [
    "long sleeved", "short sleeved"
]


def generate_cloth_prompt():
    """옷 변경을 위한 동적 프롬프트 생성"""
    color = random.choice(CLOTH_COLORS)
    cloth_type = random.choice(CLOTH_TOP_TYPES)
    cloth_length = random.choice(CLOTH_TOP_LENGTHS)
    return f"only change top to {cloth_length} {color} {cloth_type}"


PREFERED_KONTEXT_RESOLUTIONS = [
    (672, 1568),
    (688, 1504),
    (720, 1456),
    (752, 1392),
    (800, 1328),
    (832, 1248),
    (880, 1184),
    (944, 1104),
    (1024, 1024),
    (1104, 944),
    (1184, 880),
    (1248, 832),
    (1328, 800),
    (1392, 752),
    (1456, 720),
    (1504, 688),
    (1568, 672),
]

def convert_to_flux_kontext_image_scale(image):
    """PIL 이미지를 Kontext에 맞는 해상도로 리사이즈"""
    width = image.width
    height = image.height
    aspect_ratio = width / height
    
    # 가장 가까운 해상도 찾기
    _, target_width, target_height = min(
        (abs(aspect_ratio - w / h), w, h) 
        for w, h in PREFERED_KONTEXT_RESOLUTIONS
    )
    
    # center crop 계산
    old_aspect = width / height
    new_aspect = target_width / target_height
    
    if old_aspect > new_aspect:
        # 너비가 더 넓음 - 좌우 크롭
        new_width = int(height * new_aspect)
        left = (width - new_width) // 2
        image = image.crop((left, 0, left + new_width, height))
    elif old_aspect < new_aspect:
        # 높이가 더 높음 - 상하 크롭
        new_height = int(width / new_aspect)
        top = (height - new_height) // 2
        image = image.crop((0, top, width, top + new_height))
    
    # 리사이즈
    image = image.resize((target_width, target_height), Image.Resampling.LANCZOS)
    
    return image

def main():
    parser = argparse.ArgumentParser(description="FLUX Kontext 이미지 생성 (배치 처리)")
    parser.add_argument("--model_path", type=str, default="black-forest-labs/FLUX.1-Kontext-dev", help="로컬 모델 경로")
    parser.add_argument("--input_dir", type=str, default="inputs", help="입력 이미지 디렉토리")
    parser.add_argument("--resized_input_dir", type=str, default="resized_inputs", help="변환된 입력 이미지 저장 디렉토리")
    parser.add_argument("--output_dir", type=str, default="outputs", help="출력 이미지 저장 디렉토리")
    parser.add_argument("--prompt_file_path", type=str, default="image_edit_dataset_prompts.txt", help="프롬프트 텍스트 파일 경로 (라인별로 프롬프트 작성)")
    parser.add_argument("--label_prompt", type=str, default=None)#"change only hair to jelly perm hairstyle", help="레이블 프롬프트 (resized 이미지와 함께 txt 파일로 저장)")
    parser.add_argument("--guidance_scale", type=float, default=2.5, help="가이던스 스케일")
    parser.add_argument("--num_repeat", type=int, default=3, help="한 이미지당 생성할 반복 횟수")
    parser.add_argument("--lora_path", type=str, default=None, help="LoRA 가중치 전체 경로 (예: Kontext-Style/Line_lora/Line_lora_weights.safetensors 또는 로컬 경로)")
    parser.add_argument("--lora_scale", type=float, default=1.0, help="LoRA 적용 강도")
    # num_inference_steps
    parser.add_argument("--num_inference_steps", type=int, default=8, help="생성 단계 수")
    parser.add_argument("--dynamic_prompt_type", type=str, default=None, help="동적 프롬프트 타입 (None 또는 'background')")
    parser.add_argument("--model_type", type=str, default="flux_kontext", choices=["flux_kontext", "qwen_edit"], help="모델 타입 (flux_kontext 또는 qwen_edit)")
    
    args = parser.parse_args()
    
    # 동적 프롬프트 사용 여부에 따른 프롬프트 로딩
    use_dynamic_prompt = args.dynamic_prompt_type is not None
    prompts = []
    
    if not use_dynamic_prompt:
        # 기존 방식: 프롬프트 파일에서 읽기
        with open(args.prompt_file_path, 'r', encoding='utf-8') as f:
            prompts = [line.strip() for line in f if line.strip()]
        
        if not prompts:
            raise ValueError(f"프롬프트 파일이 비어있습니다: {args.prompt_file_path}")
        
        print(f"프롬프트 파일에서 {len(prompts)}개의 프롬프트를 로드했습니다.")
        
        # num_repeat이 프롬프트 개수보다 많으면 조정
        if args.num_repeat > len(prompts):
            print(f"경고: num_repeat({args.num_repeat})이 프롬프트 개수({len(prompts)})보다 많습니다.")
            args.num_repeat = len(prompts)
            print(f"num_repeat을 {args.num_repeat}로 조정합니다.")
    else:
        print(f"동적 프롬프트 모드 사용: {args.dynamic_prompt_type}")
    
    # 출력 디렉토리 생성
    os.makedirs(args.resized_input_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 모델 타입에 따른 모델 로드
    print(f"모델 로딩 중... (모델 타입: {args.model_type})")
    
    if args.model_type == "flux_kontext":
        # FLUX Kontext 모델 로드
        pipe = FluxKontextPipeline.from_pretrained(
            args.model_path, 
            torch_dtype=torch.bfloat16,
            local_files_only=True
        )
        pipe.to("cuda")
        print("FLUX Kontext 모델 로딩 완료!")

        pipe.load_lora_weights("alimama-creative/FLUX.1-Turbo-Alpha", adapter_name="turbo")
        
        # LoRA 로딩 (경로가 지정된 경우에만)
        if args.lora_path:
            print(f"LoRA 로딩 중: {args.lora_path}")
            
            lora_dir = os.path.dirname(args.lora_path)
            lora_filename = os.path.basename(args.lora_path)
            
            pipe.load_lora_weights(lora_dir, adapter_name="lora", weight_name=lora_filename, local_files_only=True)
            
            pipe.set_adapters(["turbo", "lora"], adapter_weights=[args.lora_scale, args.lora_scale])
            print(f"LoRA 로딩 완료! (scale: {args.lora_scale})")
        
        pipe.fuse_lora()
    
    elif args.model_type == "qwen_edit":
        # Qwen Image Edit 모델 로드
        qwen_model_path = "aidiffuser/Qwen-Image-Edit-2509"
        pipe = QwenImageEditPlusPipeline.from_pretrained(
            qwen_model_path,
            torch_dtype=torch.bfloat16,
            variant="Qwen-Image-Edit-2509_fp8_e4m3fn"
        )
        pipe.to("cuda")
        pipe.set_progress_bar_config(disable=True)
        print("Qwen Image Edit 모델 로딩 완료!")
        
        # Qwen Lightning LoRA 로드
        pipe.load_lora_weights(
            "lightx2v/Qwen-Image-Lightning", 
            weight_name="Qwen-Image-Edit-Lightning-4steps-V1.0-bf16.safetensors"
        )
        print("Qwen Lightning LoRA 로딩 완료!")
        
        # 추가 LoRA 로딩 (경로가 지정된 경우에만)
        if args.lora_path:
            print(f"추가 LoRA 로딩 중: {args.lora_path}")
            lora_dir = os.path.dirname(args.lora_path)
            lora_filename = os.path.basename(args.lora_path)
            pipe.load_lora_weights(lora_dir, weight_name=lora_filename, local_files_only=True)
            print(f"추가 LoRA 로딩 완료!")
    
    # 지원하는 이미지 확장자
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    
    # 입력 디렉토리의 모든 이미지 파일 찾기
    input_path = Path(args.input_dir)
    image_files = [f for f in input_path.iterdir() if f.suffix.lower() in image_extensions]
    
    total_iterations = len(image_files) * args.num_repeat
    print(f"\n총 {len(image_files)}개의 이미지를 각각 {args.num_repeat}번씩 처리합니다. (전체: {total_iterations}회)\n")
    
    current_iteration = 0
    for idx, image_file in enumerate(image_files, 1):
        print(f"[이미지 {idx}/{len(image_files)}] {image_file.name}")
        
        try:
            # 로컬 이미지 로드
            input_image = Image.open(image_file)
            
            # RGB로 변환 (RGBA 등 다른 모드 처리)
            if input_image.mode != 'RGB':
                input_image = input_image.convert('RGB')
            
            # Kontext에 맞는 해상도로 변환
            input_image = convert_to_flux_kontext_image_scale(input_image)
            
            print(f"  - 입력 이미지 크기: {input_image.width}x{input_image.height}")
            
            # 현재 이미지에 사용할 프롬프트 리스트 (중복 없이 랜덤 선택) - 동적 프롬프트가 아닌 경우에만
            if not use_dynamic_prompt:
                available_prompts = prompts.copy()
                random.shuffle(available_prompts)
                selected_prompts = available_prompts[:args.num_repeat]
            
            # num_repeat 만큼 반복
            for repeat_idx in range(args.num_repeat):
                current_iteration += 1
                print(f"  [{repeat_idx + 1}/{args.num_repeat}] (전체 진행: {current_iteration}/{total_iterations})")
                
                try:
                    # 동적 프롬프트 사용 여부에 따라 프롬프트 선택
                    if use_dynamic_prompt:
                        if args.dynamic_prompt_type == "background":
                            selected_prompt = generate_background_prompt()
                        elif args.dynamic_prompt_type == "camera_angle":
                            selected_prompt = generate_camera_angle_prompt()
                        elif args.dynamic_prompt_type == "cloth":
                            selected_prompt = generate_cloth_prompt()
                        else:
                            # 알 수 없는 동적 프롬프트 타입인 경우 기본 프롬프트 파일 사용
                            selected_prompt = selected_prompts[repeat_idx]
                    else:
                        # 미리 선택된 프롬프트 사용 (중복 없음)
                        selected_prompt = selected_prompts[repeat_idx]
                    print(f"    - 선택된 프롬프트: {selected_prompt}")
                    
                    # 파일명 생성 (확장자는 .jpg로 통일, 인덱스 추가)
                    output_filename = f"{image_file.stem}_{repeat_idx}.jpg"
                    
                    # 변환된 입력 이미지 저장
                    resized_filename = f"{image_file.stem}_{repeat_idx}.jpg"
                    resized_path = os.path.join(args.resized_input_dir, resized_filename)
                    input_image.save(resized_path, format='JPEG', quality=100)
                    print(f"    - 변환된 입력 이미지 저장: {resized_path}")
                    
                    # 레이블 프롬프트를 txt 파일로 저장
                    if args.label_prompt:
                        label_filename = f"{image_file.stem}_{repeat_idx}.txt"
                        label_path = os.path.join(args.resized_input_dir, label_filename)
                        with open(label_path, 'w', encoding='utf-8') as f:
                            f.write(args.label_prompt)
                        print(f"    - 레이블 텍스트 저장: {label_path}")
                        
                    
                    # 이미지 생성 (모델 타입에 따라 다른 파라미터 사용)
                    if args.model_type == "flux_kontext":
                        pipe_kwargs = {
                            "image": input_image,
                            "prompt": selected_prompt,
                            "guidance_scale": args.guidance_scale,
                            "width": input_image.width,
                            "height": input_image.height,
                            "num_inference_steps": args.num_inference_steps
                        }
                    elif args.model_type == "qwen_edit":
                        pipe_kwargs = {
                            "image": [input_image],
                            "prompt": selected_prompt,
                            "generator": torch.manual_seed(random.randint(0, 2**32 - 1)),
                            "true_cfg_scale": args.guidance_scale,
                            "negative_prompt": " ",
                            "num_inference_steps": args.num_inference_steps,
                            "guidance_scale": 1.0,
                            "num_images_per_prompt": 1,
                        }
                    
                    with torch.inference_mode():
                        image = pipe(**pipe_kwargs).images[0]
                    
                    # 이미지 저장 (jpg로 통일)
                    output_path = os.path.join(args.output_dir, output_filename)
                    image.save(output_path, format='JPEG', quality=100)
                    print(f"    - 생성된 이미지 저장: {output_path}")
                    
                except Exception as e:
                    print(f"    - 반복 {repeat_idx} 오류 발생: {str(e)}")
                    continue
            
            print()  # 이미지 간 구분을 위한 빈 줄
            
        except Exception as e:
            print(f"  - 이미지 로드 오류: {str(e)}\n")
            continue
    
    print(f"모든 이미지 처리 완료!")

if __name__ == "__main__":
    main()