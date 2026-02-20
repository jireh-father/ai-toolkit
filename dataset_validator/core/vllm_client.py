"""vLLM OpenAI-compatible API client for VLM evaluation."""

import base64
import io
import json
import logging
from typing import Optional

from PIL import Image

logger = logging.getLogger(__name__)


def _image_to_base64_url(image: Image.Image) -> str:
    """Convert PIL Image to base64 data URL for OpenAI API."""
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def _build_openai_messages(
    images: tuple[Image.Image, Image.Image, Image.Image],
    prompt: str,
) -> list[dict]:
    """Build OpenAI-compatible chat messages with base64-encoded images."""
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "[INPUT] Original person before hair editing:"},
                {"type": "image_url", "image_url": {"url": _image_to_base64_url(images[0])}},
                {"type": "text", "text": "[REFERENCE] Target hairstyle to apply:"},
                {"type": "image_url", "image_url": {"url": _image_to_base64_url(images[1])}},
                {"type": "text", "text": "[OUTPUT] Result after hair transfer:"},
                {"type": "image_url", "image_url": {"url": _image_to_base64_url(images[2])}},
                {"type": "text", "text": prompt},
            ],
        }
    ]


def evaluate_single_vllm(
    vllm_url: str,
    model_id: str,
    images: tuple[Image.Image, Image.Image, Image.Image],
    prompt: str,
    max_retries: int = 3,
) -> Optional[dict]:
    """Evaluate a single image triplet via vLLM OpenAI-compatible API.

    Args:
        vllm_url: vLLM server base URL (e.g. "http://localhost:8000")
        model_id: HuggingFace model ID served by vLLM
        images: (input_img, reference_img, output_img)
        prompt: evaluation prompt text
        max_retries: max retry attempts on parse failure

    Returns:
        dict with score fields and reason, or None on complete failure
    """
    from dataset_validator.core.evaluator import _parse_json_response, _validate_scores

    import requests

    messages = _build_openai_messages(images, prompt)
    api_url = f"{vllm_url.rstrip('/')}/v1/chat/completions"

    for attempt in range(max_retries):
        try:
            payload = {
                "model": model_id,
                "messages": messages,
                "max_completion_tokens": 512,
                "temperature": 0,
            }

            resp = requests.post(api_url, json=payload, timeout=120)
            resp.raise_for_status()

            data = resp.json()
            response_text = data["choices"][0]["message"]["content"]

            logger.debug(f"vLLM response (attempt {attempt + 1}): {response_text[:200]}")

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

        except requests.exceptions.ConnectionError:
            logger.error(
                f"Cannot connect to vLLM server at {vllm_url}. "
                f"Is the server running? (attempt {attempt + 1}/{max_retries})"
            )
            continue
        except Exception as e:
            logger.error(f"vLLM evaluation error (attempt {attempt + 1}/{max_retries}): {e}")
            continue

    logger.error("All vLLM evaluation attempts failed")
    return None


def check_server_health(vllm_url: str) -> bool:
    """Check if the vLLM server is reachable and ready."""
    import requests

    try:
        resp = requests.get(f"{vllm_url.rstrip('/')}/health", timeout=5)
        return resp.status_code == 200
    except Exception:
        return False
