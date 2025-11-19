import torch
import argparse
from diffusers import FluxKontextPipeline
from PIL import Image
import numpy as np

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
    parser = argparse.ArgumentParser(description="FLUX Kontext 이미지 생성")
    parser.add_argument("--model_path", type=str, default="black-forest-labs/FLUX.1-Kontext-dev", help="로컬 모델 경로")
    parser.add_argument("--input_image", type=str, default="test.jpg", help="입력 이미지 경로")
    parser.add_argument("--output_image", type=str, default="output.png", help="출력 이미지 경로")
    parser.add_argument("--prompt", type=str, default="change only hair to bob hair", help="프롬프트")
    parser.add_argument("--guidance_scale", type=float, default=2.5, help="가이던스 스케일")
    parser.add_argument("--width", type=int, default=None, help="출력 이미지 너비")
    parser.add_argument("--height", type=int, default=None, help="출력 이미지 높이")
    
    args = parser.parse_args()
    
    # 로컬 경로에서 모델 로드 (local_files_only=True로 강제)
    pipe = FluxKontextPipeline.from_pretrained(
        args.model_path, 
        torch_dtype=torch.bfloat16,
        local_files_only=True
    )
    pipe.to("cuda")
    
    # 로컬 이미지 로드
    input_image = Image.open(args.input_image)
    
    # Kontext에 맞는 해상도로 변환
    input_image = convert_to_flux_kontext_image_scale(input_image)
    
    width = args.width or input_image.width
    height = args.height or input_image.height
    
    print(f"입력 이미지 크기: {input_image.width}x{input_image.height}")
    print(f"출력 이미지 크기: {width}x{height}")
    
    # 이미지 생성
    pipe_kwargs = {
        "image": input_image,
        "prompt": args.prompt,
        "guidance_scale": args.guidance_scale,
        "width": width,
        "height": height
    }
    
    image = pipe(**pipe_kwargs).images[0]
    
    # 이미지 저장
    image.save(args.output_image)
    print(f"이미지가 {args.output_image}에 저장되었습니다.")

if __name__ == "__main__":
    main()