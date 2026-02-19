"""VLM model loading, prompt building, and evaluation logic."""

import json
import logging
import re
from pathlib import Path
from typing import Any, Optional

import yaml
from PIL import Image

logger = logging.getLogger(__name__)

SCORE_FIELDS = [
    "hair_similarity_overall",
    "hair_color",
    "hair_length",
    "hair_texture",
    "hair_shape",
    "bangs_shape",
    "bangs_length",
    "hair_sharpness_vs_input",
    "hair_sharpness_vs_reference",
    "hair_detail",
    "non_hair_preservation",
    "naturalness",
    "face_shape_preservation",
    "face_color_preservation",
]

EVALUATION_PROMPT = """You are a strict image quality assessor for hair transfer models.

Above you see three labeled images: [INPUT], [REFERENCE], and [OUTPUT].
- INPUT = original person BEFORE hair editing.
- REFERENCE = target hairstyle (may be a different person, different angle).
- OUTPUT = result AFTER applying REFERENCE's hairstyle onto the INPUT person.

TASK: Compare carefully and score each criterion on 0-10 scale. Be strict and critical. Most real-world edits have flaws — a score of 7-8 should be reserved for genuinely good results.

CRITERIA (compare very carefully — look at details, not just overall impression):

1. hair_similarity_overall: Does OUTPUT's hairstyle match REFERENCE's hairstyle? Compare style, shape, volume, parting.
2. hair_color: Does OUTPUT's hair color match REFERENCE? Check highlights, roots, gradient, tone. If colors differ, score LOW.
3. hair_length: Does OUTPUT's hair length match REFERENCE? Short vs long is an obvious difference — score accordingly.
4. hair_texture: Does OUTPUT's hair texture match REFERENCE? Straight vs wavy vs curly vs permed. Wet vs dry hair is different texture.
5. hair_shape: Does OUTPUT's overall hair shape/silhouette match REFERENCE? Compare the full hairstyle shape, volume, layering, parting, and flow direction. The hair angle should match INPUT's head pose, but the style/shape should replicate REFERENCE.
6. bangs_shape: Does OUTPUT's bangs (fringe) shape match REFERENCE? Compare bangs style — straight-across, side-swept, curtain bangs, wispy, blunt, layered, or no bangs. If REFERENCE has no bangs and OUTPUT has no bangs, score 10. If one has bangs and the other does not, score very LOW.
7. bangs_length: Does OUTPUT's bangs length match REFERENCE? Compare how far the bangs extend — above eyebrows, eyebrow-level, eye-covering, cheekbone-length, or no bangs. If REFERENCE has no bangs and OUTPUT has no bangs, score 10. A mismatch in bangs length is a clear failure.
8. hair_sharpness_vs_input: Compare OUTPUT's hair sharpness/clarity against INPUT's hair. Is the OUTPUT hair at least as sharp and clear as INPUT? Look for blurriness, softness, loss of edge definition, or smearing in OUTPUT hair compared to INPUT hair. Score 10 if OUTPUT hair is equally or more sharp than INPUT. Score LOW if OUTPUT hair is noticeably blurrier, softer, or less defined than INPUT.
9. hair_sharpness_vs_reference: Compare OUTPUT's hair sharpness/clarity against REFERENCE's hair. Does OUTPUT's hair have similar sharpness and clarity as REFERENCE? Look for blurriness, loss of fine details, or degraded quality. Score 10 if OUTPUT hair matches REFERENCE sharpness. Score LOW if OUTPUT hair is significantly blurrier or less crisp than REFERENCE.
10. hair_detail: How well does OUTPUT express fine hair details? Evaluate strand-level detail, texture definition, highlights/shadows in individual strands, flyaway hairs, natural hair layering. Score 10 for photorealistic strand-level detail. Score LOW if hair looks flat, painted, plasticky, or lacks natural strand separation and micro-details.
11. non_hair_preservation: Is everything EXCEPT hair in OUTPUT identical to INPUT? Check face, eyes, skin, clothing, background, accessories. Any change = lower score.
12. naturalness: Does the hair edit look realistic? Check hair-face boundary, artifacts, color bleeding, lighting consistency, unnatural edges.
13. face_shape_preservation: Is OUTPUT's face shape EXACTLY identical to INPUT? Examine in extreme detail: jawline contour, chin shape, cheekbone width, forehead height/width, face outline symmetry, ear visibility. The face geometry must be pixel-level identical. ANY distortion, warping, slimming, widening, or reshaping of the face = score LOW. Even subtle changes to jaw angle or chin shape = deduct heavily.
14. face_color_preservation: Is OUTPUT's face/skin color EXACTLY identical to INPUT? Examine in extreme detail: skin tone, brightness, contrast, shadow patterns, under-eye area, lip color, complexion uniformity. Compare side-by-side very carefully. ANY color shift, brightening, darkening, smoothing, redness change, or tonal difference = score LOW. The face color must be indistinguishable from INPUT.
SCORING (be honest, not generous):
- 0-3: Major failure (wrong hairstyle, face changed, severe artifacts)
- 4-5: Poor (clearly visible problems, obvious mismatch)
- 6: Acceptable but flawed
- 7-8: Good with only minor issues
- 9-10: Near perfect (rare — reserve for truly excellent results)

Respond with ONLY a JSON object, no other text:
{"hair_similarity_overall":<0-10>,"hair_color":<0-10>,"hair_length":<0-10>,"hair_texture":<0-10>,"hair_shape":<0-10>,"bangs_shape":<0-10>,"bangs_length":<0-10>,"hair_sharpness_vs_input":<0-10>,"hair_sharpness_vs_reference":<0-10>,"hair_detail":<0-10>,"non_hair_preservation":<0-10>,"naturalness":<0-10>,"face_shape_preservation":<0-10>,"face_color_preservation":<0-10>,"reason":"<1-2 sentences>"}"""


def load_model_config(model_name: str) -> dict:
    """Load model configuration from models.yaml."""
    config_path = Path(__file__).parent.parent / "config" / "models.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if model_name not in config["models"]:
        available = ", ".join(config["models"].keys())
        raise ValueError(
            f"Unknown model '{model_name}'. Available: {available}"
        )
    return config["models"][model_name]


def _get_quantization_config(quantization: str):
    """Build BitsAndBytesConfig for the given quantization level."""
    import torch
    from transformers import BitsAndBytesConfig

    if quantization == "int4":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    elif quantization == "int8":
        return BitsAndBytesConfig(load_in_8bit=True)
    elif quantization == "fp16":
        return None
    else:
        raise ValueError(f"Unsupported quantization: {quantization}")


def load_vlm(
    model_name: str,
    quantization: str = "int4",
    device: str = "cuda:0",
    max_pixels: Optional[int] = None,
    min_pixels: Optional[int] = None,
    low_vram: bool = False,
) -> dict:
    """Load VLM model and processor.

    Args:
        model_name: model key from models.yaml
        quantization: "int4", "int8", or "fp16"
        device: target device (e.g. "cuda:0")
        max_pixels: max pixels per image for Qwen VL processor
                    (default: 401408 = 512*28*28, original default was 1003520)
        min_pixels: min pixels per image for Qwen VL processor
                    (default: 50176 = 64*28*28)
        low_vram: if True, use device_map="auto" for CPU offloading

    Returns dict with keys: model, processor, family, device
    """
    import torch
    from transformers import AutoProcessor, AutoModelForCausalLM

    if max_pixels is None:
        max_pixels = 512 * 28 * 28  # 401,408 (vs default 1,003,520)
    if min_pixels is None:
        min_pixels = 64 * 28 * 28   # 50,176

    model_config = load_model_config(model_name)
    hf_id = model_config["hf_id"]
    family = model_config["family"]

    logger.info(f"Loading model: {hf_id} (family={family}, quant={quantization})")
    if low_vram:
        logger.info("Low-VRAM mode: enabling CPU offloading with device_map='auto'")

    quant_config = _get_quantization_config(quantization)

    # Device mapping: use "auto" for CPU offloading in low-VRAM mode
    if low_vram:
        device_map = "auto"
    else:
        device_map = device

    load_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.float16,
        "device_map": device_map,
    }
    if quant_config is not None:
        load_kwargs["quantization_config"] = quant_config

    # Use memory-efficient attention (sdpa works on V100+, flash_attention_2 needs Ampere+)
    load_kwargs["attn_implementation"] = "sdpa"

    # Family-specific model class selection
    if family == "qwen2_vl":
        from transformers import AutoModelForVision2Seq
        model = AutoModelForVision2Seq.from_pretrained(
            hf_id, **load_kwargs
        )
        # Set min/max_pixels to control visual token count per image
        processor = AutoProcessor.from_pretrained(
            hf_id,
            trust_remote_code=True,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
        logger.info(
            f"Qwen VL processor: min_pixels={min_pixels}, max_pixels={max_pixels} "
            f"(max visual tokens/image={max_pixels // (28 * 28)})"
        )
    elif family == "internvl":
        model = AutoModelForCausalLM.from_pretrained(hf_id, **load_kwargs)
        processor = AutoProcessor.from_pretrained(hf_id, trust_remote_code=True)
    elif family == "minicpm":
        model = AutoModelForCausalLM.from_pretrained(hf_id, **load_kwargs)
        processor = AutoProcessor.from_pretrained(hf_id, trust_remote_code=True)
    elif family == "gemma":
        model = AutoModelForCausalLM.from_pretrained(hf_id, **load_kwargs)
        processor = AutoProcessor.from_pretrained(hf_id, trust_remote_code=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(hf_id, **load_kwargs)
        processor = AutoProcessor.from_pretrained(hf_id, trust_remote_code=True)

    model.eval()

    # Log VRAM usage after model load
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        logger.info(f"Model loaded — VRAM allocated: {allocated:.1f}GB, reserved: {reserved:.1f}GB")

    return {
        "model": model,
        "processor": processor,
        "family": family,
        "device": device,
        "model_name": model_name,
    }


def build_prompt(family: str) -> tuple[str, str]:
    """Build the system/user prompt based on model family.

    Returns: (system_prompt, user_text_prompt)
    """
    return ("", EVALUATION_PROMPT)


def _build_messages_qwen(
    images: tuple[Image.Image, Image.Image, Image.Image],
    prompt: str,
) -> list[dict]:
    """Build chat messages for Qwen2-VL / Qwen3-VL family.

    Places a text label before each image so the VLM clearly knows
    which image is INPUT, REFERENCE, and OUTPUT.
    """
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "[INPUT] Original person before hair editing:"},
                {"type": "image", "image": images[0]},
                {"type": "text", "text": "[REFERENCE] Target hairstyle to apply:"},
                {"type": "image", "image": images[1]},
                {"type": "text", "text": "[OUTPUT] Result after hair transfer:"},
                {"type": "image", "image": images[2]},
                {"type": "text", "text": prompt},
            ],
        }
    ]


def _build_messages_generic(
    images: tuple[Image.Image, Image.Image, Image.Image],
    prompt: str,
) -> list[dict]:
    """Build chat messages for generic models (InternVL, MiniCPM, Gemma).

    Places a text label before each image placeholder.
    """
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "[INPUT] Original person before hair editing:"},
                {"type": "image"},
                {"type": "text", "text": "[REFERENCE] Target hairstyle to apply:"},
                {"type": "image"},
                {"type": "text", "text": "[OUTPUT] Result after hair transfer:"},
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }
    ]


def _parse_json_response(text: str) -> Optional[dict]:
    """Extract and parse JSON from VLM response text."""
    # Try direct parse
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try extracting JSON from markdown code blocks
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Try finding first { ... } block
    match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    # Try finding nested { ... } block
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass

    return None


def _validate_scores(data: dict) -> Optional[dict]:
    """Validate that parsed JSON has all required score fields with valid values."""
    if not isinstance(data, dict):
        return None

    result = {}
    for field in SCORE_FIELDS:
        val = data.get(field)
        if val is None:
            return None
        try:
            score = int(val)
        except (ValueError, TypeError):
            try:
                score = int(float(val))
            except (ValueError, TypeError):
                return None
        if not (0 <= score <= 10):
            return None
        result[field] = score

    result["reason"] = str(data.get("reason", ""))
    return result


def evaluate_single(
    vlm: dict,
    images: tuple[Image.Image, Image.Image, Image.Image],
    max_retries: int = 3,
    max_new_tokens: int = 384,
) -> Optional[dict]:
    """Evaluate a single image triplet using the VLM.

    Args:
        vlm: dict from load_vlm()
        images: (input_img, reference_img, output_img)
        max_retries: maximum attempts on JSON parse failure
        max_new_tokens: max tokens to generate (default: 384, was 512)

    Returns:
        dict with score fields and reason, or None on complete failure
    """
    model = vlm["model"]
    processor = vlm["processor"]
    family = vlm["family"]

    _, user_prompt = build_prompt(family)

    import torch

    for attempt in range(max_retries):
        inputs = None
        generated_ids = None
        try:
            if family == "qwen2_vl":
                messages = _build_messages_qwen(images, user_prompt)
                from qwen_vl_utils import process_vision_info
                text = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                image_inputs, video_inputs = process_vision_info(messages)
                inputs = processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                ).to(model.device)
            else:
                messages = _build_messages_generic(images, user_prompt)
                text = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                inputs = processor(
                    text=[text],
                    images=list(images),
                    return_tensors="pt",
                ).to(model.device)

            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                )

            # Decode only the generated tokens (exclude input)
            input_len = inputs["input_ids"].shape[1]
            output_ids = generated_ids[:, input_len:]
            response_text = processor.batch_decode(
                output_ids, skip_special_tokens=True
            )[0]

            logger.debug(f"VLM response (attempt {attempt + 1}): {response_text[:200]}")

            parsed = _parse_json_response(response_text)
            if parsed is None:
                logger.warning(
                    f"JSON parse failed (attempt {attempt + 1}/{max_retries})"
                )
                continue

            validated = _validate_scores(parsed)
            if validated is None:
                logger.warning(
                    f"Score validation failed (attempt {attempt + 1}/{max_retries})"
                )
                continue

            return validated

        except Exception as e:
            logger.error(f"Evaluation error (attempt {attempt + 1}/{max_retries}): {e}")
            continue
        finally:
            # Free GPU memory after each attempt
            del inputs, generated_ids
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    logger.error("All evaluation attempts failed")
    return None


def evaluate_batch(
    vlm: dict,
    entries: list[dict],
    short_side: int = 512,
    max_retries: int = 3,
    max_new_tokens: int = 384,
) -> list[dict]:
    """Evaluate a batch of entries sequentially.

    Args:
        vlm: dict from load_vlm()
        entries: list of dicts with 'stem', 'input', 'reference', 'output' keys
        short_side: target short side for image resizing
        max_retries: max retries per evaluation
        max_new_tokens: max tokens to generate per evaluation

    Returns:
        list of result dicts with 'filename', 'scores', 'reason', 'error' keys
    """
    from .image_loader import load_image_triplet

    results = []
    for entry in entries:
        triplet = load_image_triplet(entry, short_side=short_side)
        if triplet is None:
            results.append({
                "filename": entry["stem"],
                "scores": None,
                "reason": "Failed to load images",
                "error": True,
            })
            continue

        scores = evaluate_single(vlm, triplet, max_retries=max_retries, max_new_tokens=max_new_tokens)
        if scores is None:
            results.append({
                "filename": entry["stem"],
                "scores": None,
                "reason": "VLM evaluation failed after retries",
                "error": True,
            })
        else:
            reason = scores.pop("reason", "")
            results.append({
                "filename": entry["stem"],
                "scores": scores,
                "reason": reason,
                "error": False,
            })

    return results


def is_pass(scores: dict, threshold: int = 7,
            threshold_hair: Optional[int] = None,
            threshold_preservation: Optional[int] = None,
            threshold_naturalness: Optional[int] = None,
            threshold_face: Optional[int] = None,
            threshold_angle: Optional[int] = None) -> bool:
    """Determine if scores pass the threshold criteria.

    For all fields: score < threshold → FAIL
    """
    if scores is None:
        return False

    hair_thresh = threshold_hair if threshold_hair is not None else threshold
    pres_thresh = threshold_preservation if threshold_preservation is not None else threshold
    nat_thresh = threshold_naturalness if threshold_naturalness is not None else threshold
    face_thresh = threshold_face if threshold_face is not None else threshold

    hair_fields = [
        "hair_similarity_overall", "hair_color", "hair_length", "hair_texture",
        "hair_shape", "bangs_shape", "bangs_length",
        "hair_sharpness_vs_input", "hair_sharpness_vs_reference", "hair_detail",
    ]
    for field in hair_fields:
        if scores.get(field, 0) < hair_thresh:
            return False

    if scores.get("non_hair_preservation", 0) < pres_thresh:
        return False

    if scores.get("naturalness", 0) < nat_thresh:
        return False

    face_fields = ["face_shape_preservation", "face_color_preservation"]
    for field in face_fields:
        if scores.get(field, 0) < face_thresh:
            return False

    return True
