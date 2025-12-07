import torch
import argparse
from diffusers import FluxKontextPipeline
from PIL import Image
import numpy as np
import os
from pathlib import Path
import random

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


def split_list_randomly(lst, n):
    """리스트를 n개의 그룹으로 랜덤하게 나눔"""
    shuffled = lst.copy()
    random.shuffle(shuffled)
    
    # 각 그룹의 기본 크기와 나머지
    base_size = len(shuffled) // n
    remainder = len(shuffled) % n
    
    result = []
    start = 0
    for i in range(n):
        # 앞쪽 그룹들에 나머지를 1개씩 더 배분
        size = base_size + (1 if i < remainder else 0)
        result.append(shuffled[start:start + size])
        start += size
    
    return result


def load_lora_to_pipe(pipe, lora_path, lora_scale):
    """LoRA를 파이프라인에 로드하고 fuse"""
    # turbo LoRA 로드
    pipe.load_lora_weights("alimama-creative/FLUX.1-Turbo-Alpha", adapter_name="turbo")
    
    if lora_path:
        lora_dir = os.path.dirname(lora_path)
        lora_filename = os.path.basename(lora_path)
        
        pipe.load_lora_weights(lora_dir, adapter_name="lora", weight_name=lora_filename, local_files_only=True)
        pipe.set_adapters(["turbo", "lora"], adapter_weights=[1.0, lora_scale])
    else:
        pipe.set_adapters(["turbo"], adapter_weights=[1.0])
    
    pipe.fuse_lora()


def unload_lora_from_pipe(pipe):
    """파이프라인에서 LoRA를 unfuse하고 unload"""
    pipe.unfuse_lora()
    pipe.unload_lora_weights()


def main():
    parser = argparse.ArgumentParser(description="FLUX Kontext 이미지 생성 (배치 처리, 다중 LoRA)")
    parser.add_argument("--model_path", type=str, default="black-forest-labs/FLUX.1-Kontext-dev", help="로컬 모델 경로")
    parser.add_argument("--input_dir", type=str, default="inputs", help="입력 이미지 디렉토리")
    parser.add_argument("--resized_input_dir", type=str, default="resized_inputs", help="변환된 입력 이미지 저장 디렉토리")
    parser.add_argument("--output_dir", type=str, default="outputs", help="출력 이미지 저장 디렉토리")
    parser.add_argument("--prompts", type=str, required=True, help="프롬프트들 (:::로 구분, lora_paths 개수와 동일해야 함)")
    parser.add_argument("--label_prompt", type=str, default="change only hair to jelly perm hairstyle", help="레이블 프롬프트 (resized 이미지와 함께 txt 파일로 저장)")
    parser.add_argument("--guidance_scale", type=float, default=2.5, help="가이던스 스케일")
    parser.add_argument("--num_repeat", type=int, default=3, help="한 이미지당 생성할 반복 횟수")
    parser.add_argument("--lora_paths", type=str, nargs='+', default=[], help="LoRA 가중치 경로들 (여러 개 지정 가능)")
    parser.add_argument("--lora_scale", type=float, default=1.0, help="LoRA 적용 강도")
    parser.add_argument("--num_inference_steps", type=int, default=8, help="생성 단계 수")
    parser.add_argument("--another_lora_when_repeat", default=False, action="store_true", help="반복 시 다른 LoRA 사용")
    #seed
    
    
    args = parser.parse_args()
    
    # LoRA 경로 검증
    if not args.lora_paths:
        raise ValueError("최소 1개 이상의 --lora_paths를 지정해야 합니다.")
    
    num_loras = len(args.lora_paths)
    print(f"총 {num_loras}개의 LoRA가 지정되었습니다:")
    for i, lora_path in enumerate(args.lora_paths):
        print(f"  [{i+1}] {lora_path}")
    
    # 프롬프트 파싱 (:::로 구분)
    prompts = [p.strip() for p in args.prompts.split(":::") if p.strip()]
    
    if not prompts:
        raise ValueError("프롬프트가 비어있습니다.")
    
    # prompts 개수와 lora_paths 개수 검증
    if len(prompts) != num_loras:
        raise ValueError(f"prompts 개수({len(prompts)})와 lora_paths 개수({num_loras})가 일치해야 합니다.")
    
    print(f"총 {len(prompts)}개의 프롬프트가 지정되었습니다:")
    for i, prompt in enumerate(prompts):
        print(f"  [{i+1}] {prompt}")
    
    # 출력 디렉토리 생성
    os.makedirs(args.resized_input_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 지원하는 이미지 확장자
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    
    # 입력 디렉토리의 모든 이미지 파일 찾기
    input_path = Path(args.input_dir)
    image_files = [f for f in input_path.iterdir() if f.suffix.lower() in image_extensions]
    
    if not image_files:
        raise ValueError(f"입력 디렉토리에 이미지가 없습니다: {args.input_dir}")
    
    print(f"\n총 {len(image_files)}개의 입력 이미지를 찾았습니다.")
    
    # 이미지 파일들을 LoRA 개수만큼 랜덤하게 나눔
    random.seed(args.seed)
    image_groups = split_list_randomly(image_files, num_loras)
    
    print(f"\n이미지 파일을 {num_loras}개의 그룹으로 나눴습니다:")
    for i, group in enumerate(image_groups):
        print(f"  [LoRA {i+1}] {len(group)}개 이미지")
    
    # 로컬 경로에서 모델 로드
    print("\n모델 로딩 중...")
    pipe = FluxKontextPipeline.from_pretrained(
        args.model_path, 
        torch_dtype=torch.bfloat16,
        local_files_only=True
    )
    pipe.to("cuda")
    print("모델 로딩 완료!")
    
    total_iterations = sum(len(group) for group in image_groups) * args.num_repeat
    current_iteration = 0
    
    # 현재 로드된 LoRA 인덱스 추적
    current_lora_idx = None
    
    # 바깥 루프: num_repeat
    for repeat_idx in range(args.num_repeat):
        print(f"\n{'='*60}")
        print(f"[반복 {repeat_idx + 1}/{args.num_repeat}]")
        print(f"{'='*60}")
        
        # 안쪽 루프: LoRA별로 처리
        for lora_idx, lora_path in enumerate(args.lora_paths):
            if args.another_lora_when_repeat:
                image_group_idx = lora_idx + repeat_idx
                if image_group_idx >= len(image_groups):
                    image_group_idx = 0
            else:
                image_group_idx = lora_idx
            image_group = image_groups[image_group_idx]
            
            if not image_group:
                print(f"\n[LoRA {lora_idx + 1}] 할당된 이미지가 없습니다. 건너뜁니다.")
                continue
            
            print(f"\n[LoRA {lora_idx + 1}/{num_loras}] {os.path.basename(lora_path)}")
            print(f"  - 처리할 이미지: {len(image_group)}개")
            
            # LoRA가 바뀌었으면 교체
            if current_lora_idx != lora_idx:
                if current_lora_idx is not None:
                    print(f"  - 기존 LoRA unfuse/unload 중...")
                    unload_lora_from_pipe(pipe)
                
                print(f"  - 새 LoRA 로딩 중: {lora_path}")
                load_lora_to_pipe(pipe, lora_path, args.lora_scale)
                print(f"  - LoRA 로딩 완료!")
                current_lora_idx = lora_idx
            
            # 해당 LoRA에 할당된 이미지들 처리
            for img_idx, image_file in enumerate(image_group):
                current_iteration += 1
                print(f"\n  [이미지 {img_idx + 1}/{len(image_group)}] {image_file.name} (전체 진행: {current_iteration}/{total_iterations})")
                
                try:
                    # 로컬 이미지 로드
                    input_image = Image.open(image_file)
                    
                    # RGB로 변환 (RGBA 등 다른 모드 처리)
                    if input_image.mode != 'RGB':
                        input_image = input_image.convert('RGB')
                    
                    # Kontext에 맞는 해상도로 변환
                    input_image = convert_to_flux_kontext_image_scale(input_image)
                    
                    print(f"    - 입력 이미지 크기: {input_image.width}x{input_image.height}")
                    
                    # 해당 LoRA에 맞는 프롬프트 사용
                    selected_prompt = prompts[lora_idx]
                    print(f"    - 사용 프롬프트: {selected_prompt}")
                    
                    # 파일명 생성 (LoRA 인덱스 + 반복 인덱스 포함)
                    output_filename = f"{image_file.stem}_lora{lora_idx}_{repeat_idx}.jpg"
                    
                    # 변환된 입력 이미지 저장
                    resized_filename = f"{image_file.stem}_lora{lora_idx}_{repeat_idx}.jpg"
                    resized_path = os.path.join(args.resized_input_dir, resized_filename)
                    input_image.save(resized_path, format='JPEG', quality=100)
                    print(f"    - 변환된 입력 이미지 저장: {resized_path}")
                    
                    # 레이블 프롬프트를 txt 파일로 저장
                    if args.label_prompt:
                        label_filename = f"{image_file.stem}_lora{lora_idx}_{repeat_idx}.txt"
                        label_path = os.path.join(args.resized_input_dir, label_filename)
                        with open(label_path, 'w', encoding='utf-8') as f:
                            f.write(args.label_prompt)
                        print(f"    - 레이블 텍스트 저장: {label_path}")
                    
                    # 이미지 생성
                    pipe_kwargs = {
                        "image": input_image,
                        "prompt": selected_prompt,
                        "guidance_scale": args.guidance_scale,
                        "width": input_image.width,
                        "height": input_image.height,
                        "num_inference_steps": args.num_inference_steps
                    }
                    
                    image = pipe(**pipe_kwargs).images[0]
                    
                    # 이미지 저장 (jpg로 통일)
                    output_path = os.path.join(args.output_dir, output_filename)
                    image.save(output_path, format='JPEG', quality=100)
                    print(f"    - 생성된 이미지 저장: {output_path}")
                    
                except Exception as e:
                    print(f"    - 오류 발생: {str(e)}")
                    continue
    
    print(f"\n모든 이미지 처리 완료!")


if __name__ == "__main__":
    main()
