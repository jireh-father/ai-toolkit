"""Ollama OpenAI-compatible API client for VLM evaluation."""

from __future__ import annotations

import base64
import io
import json
import logging
from pathlib import Path
from typing import Optional

import yaml
from PIL import Image

logger = logging.getLogger(__name__)


def _image_to_base64_url(image: Image.Image) -> str:
    """Convert PIL Image to base64 data URL for OpenAI API."""
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def _resolve_ollama_model(model_name: str) -> str:
    """Resolve model name to Ollama tag via models.yaml.

    If the model has an 'ollama_tag' field in config, use it.
    Otherwise, return the model_name as-is (user may pass Ollama tag directly).
    """
    config_path = Path(__file__).parent.parent / "config" / "models.yaml"
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        if model_name in config["models"]:
            return config["models"][model_name].get("ollama_tag", model_name)
    except Exception:
        pass
    return model_name


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


def evaluate_single_ollama(
    ollama_url: str,
    model_name: str,
    images: tuple[Image.Image, Image.Image, Image.Image],
    prompt: str,
    max_retries: int = 3,
) -> Optional[dict]:
    """Evaluate a single image triplet via Ollama OpenAI-compatible API.

    Args:
        ollama_url: Ollama server base URL (e.g. "http://localhost:11434")
        model_name: model key from models.yaml (resolved to Ollama tag)
        images: (input_img, reference_img, output_img)
        prompt: evaluation prompt text
        max_retries: max retry attempts on parse failure

    Returns:
        dict with score fields and reason, or None on complete failure
    """
    from dataset_validator.core.evaluator import _parse_json_response, _validate_scores

    import requests

    ollama_tag = _resolve_ollama_model(model_name)
    messages = _build_openai_messages(images, prompt)
    api_url = f"{ollama_url.rstrip('/')}/v1/chat/completions"

    for attempt in range(max_retries):
        try:
            payload = {
                "model": ollama_tag,
                "messages": messages,
                "max_completion_tokens": 512,
                "temperature": 0,
                "stream": False,
            }

            resp = requests.post(api_url, json=payload, timeout=1000)
            resp.raise_for_status()

            data = resp.json()
            response_text = data["choices"][0]["message"]["content"]

            logger.debug(f"Ollama response (attempt {attempt + 1}): {response_text[:200]}")

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
                f"Cannot connect to Ollama server at {ollama_url}. "
                f"Is Ollama running? (attempt {attempt + 1}/{max_retries})"
            )
            continue
        except Exception as e:
            logger.error(f"Ollama evaluation error (attempt {attempt + 1}/{max_retries}): {e}")
            continue

    logger.error("All Ollama evaluation attempts failed")
    return None


def check_server_health(ollama_url: str) -> bool:
    """Check if the Ollama server is reachable."""
    import requests

    try:
        resp = requests.get(f"{ollama_url.rstrip('/')}/api/tags", timeout=5)
        return resp.status_code == 200
    except Exception:
        return False
