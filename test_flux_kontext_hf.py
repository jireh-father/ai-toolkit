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

def lanczos(samples, width, height):
    images = [Image.fromarray(np.clip(255. * image.movedim(0, -1).cpu().numpy(), 0, 255).astype(np.uint8)) for image in samples]
    images = [image.resize((width, height), resample=Image.Resampling.LANCZOS) for image in images]
    images = [torch.from_numpy(np.array(image).astype(np.float32) / 255.0).movedim(-1, 0) for image in images]
    result = torch.stack(images)
    return result.to(samples.device, samples.dtype)

def upscale(samples, width, height, upscale_method, crop):
    orig_shape = tuple(samples.shape)
    if len(orig_shape) > 4:
        samples = samples.reshape(samples.shape[0], samples.shape[1], -1, samples.shape[-2], samples.shape[-1])
        samples = samples.movedim(2, 1)
        samples = samples.reshape(-1, orig_shape[1], orig_shape[-2], orig_shape[-1])
    if crop == "center":
        old_width = samples.shape[-1]
        old_height = samples.shape[-2]
        old_aspect = old_width / old_height
        new_aspect = width / height
        x = 0
        y = 0
        if old_aspect > new_aspect:
            x = round((old_width - old_width * (new_aspect / old_aspect)) / 2)
        elif old_aspect < new_aspect:
            y = round((old_height - old_height * (old_aspect / new_aspect)) / 2)
        s = samples.narrow(-2, y, old_height - y * 2).narrow(-1, x, old_width - x * 2)
    else:
        s = samples

    if upscale_method == "lanczos":
        out = lanczos(s, width, height)
    else:
        out = torch.nn.functional.interpolate(s, size=(height, width), mode=upscale_method)

    if len(orig_shape) == 4:
        return out

    out = out.reshape((orig_shape[0], -1, orig_shape[1]) + (height, width))
    return out.movedim(2, 1).reshape(orig_shape[:-2] + (height, width))

def convert_to_flux_kontext_image_scale(image):
    image = torch.from_numpy(np.array(image).astype(np.float32) / 255.0).movedim(-1, 0)
    width = image.shape[2]
    height = image.shape[1]
    aspect_ratio = width / height
    _, width, height = min((abs(aspect_ratio - w / h), w, h) for w, h in PREFERED_KONTEXT_RESOLUTIONS)
    image = upscale(image.movedim(-1, 1), width, height, "lanczos", "center").movedim(1, -1)
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
    
    input_image = convert_to_flux_kontext_image_scale(input_image)

    width = args.width or input_image.shape[2]
    height = args.height or input_image.shape[1]
    
    # 이미지 생성
    pipe_kwargs = {
        "image": input_image.movedim(1, -1),
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