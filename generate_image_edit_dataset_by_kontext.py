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

def main():
    parser = argparse.ArgumentParser(description="FLUX Kontext 이미지 생성 (배치 처리)")
    parser.add_argument("--model_path", type=str, default="black-forest-labs/FLUX.1-Kontext-dev", help="로컬 모델 경로")
    parser.add_argument("--input_dir", type=str, default="inputs", help="입력 이미지 디렉토리")
    parser.add_argument("--resized_input_dir", type=str, default="resized_inputs", help="변환된 입력 이미지 저장 디렉토리")
    parser.add_argument("--output_dir", type=str, default="outputs", help="출력 이미지 저장 디렉토리")
    parser.add_argument("--prompt_file_path", type=str, default="image_edit_dataset_prompts.txt", help="프롬프트 텍스트 파일 경로 (라인별로 프롬프트 작성)")
    parser.add_argument("--label_prompt", type=str, default="change only hair to jelly perm hairstyle", help="레이블 프롬프트 (resized 이미지와 함께 txt 파일로 저장)")
    parser.add_argument("--guidance_scale", type=float, default=2.5, help="가이던스 스케일")
    parser.add_argument("--num_repeat", type=int, default=3, help="한 이미지당 생성할 반복 횟수")
    
    args = parser.parse_args()
    
    # 프롬프트 파일 읽기
    with open(args.prompt_file_path, 'r', encoding='utf-8') as f:
        prompts = [line.strip() for line in f if line.strip()]
    
    if not prompts:
        raise ValueError(f"프롬프트 파일이 비어있습니다: {args.prompt_file_path}")
    
    print(f"프롬프트 파일에서 {len(prompts)}개의 프롬프트를 로드했습니다.")
    
    # 출력 디렉토리 생성
    os.makedirs(args.resized_input_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 로컬 경로에서 모델 로드 (local_files_only=True로 강제)
    print("모델 로딩 중...")
    pipe = FluxKontextPipeline.from_pretrained(
        args.model_path, 
        torch_dtype=torch.bfloat16,
        local_files_only=True
    )
    pipe.to("cuda")
    print("모델 로딩 완료!")
    
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
            
            # num_repeat 만큼 반복
            for repeat_idx in range(args.num_repeat):
                current_iteration += 1
                print(f"  [{repeat_idx + 1}/{args.num_repeat}] (전체 진행: {current_iteration}/{total_iterations})")
                
                try:
                    # 랜덤하게 프롬프트 선택
                    selected_prompt = random.choice(prompts)
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
                        
                    
                    # 이미지 생성
                    pipe_kwargs = {
                        "image": input_image,
                        "prompt": selected_prompt,
                        "guidance_scale": args.guidance_scale,
                        "width": input_image.width,
                        "height": input_image.height
                    }
                    
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