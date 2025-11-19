import torch
import argparse
from diffusers import FluxKontextPipeline
from PIL import Image

def main():
    parser = argparse.ArgumentParser(description="FLUX Kontext 이미지 생성")
    parser.add_argument("--model_path", type=str, required=True, help="로컬 모델 경로")
    parser.add_argument("--input_image", type=str, required=True, help="입력 이미지 경로")
    parser.add_argument("--output_image", type=str, default="output.png", help="출력 이미지 경로")
    parser.add_argument("--prompt", type=str, default="Add a hat to the cat", help="프롬프트")
    parser.add_argument("--guidance_scale", type=float, default=2.5, help="가이던스 스케일")
    
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
    
    # 이미지 생성
    image = pipe(
        image=input_image,
        prompt=args.prompt,
        guidance_scale=args.guidance_scale
    ).images[0]
    
    # 이미지 저장
    image.save(args.output_image)
    print(f"이미지가 {args.output_image}에 저장되었습니다.")

if __name__ == "__main__":
    main()