"""Launch vLLM OpenAI-compatible server for dataset validation.

Usage:
    # Single GPU (V100 32GB)
    python dataset_validator/serve_vllm.py

    # Multi-GPU with tensor parallel
    python dataset_validator/serve_vllm.py --tensor-parallel-size 2

    # Custom model and port
    python dataset_validator/serve_vllm.py --model qwen3-vl-8b --port 8001
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


def _load_hf_id(model_name: str) -> str:
    """Resolve short model name to HuggingFace model ID via models.yaml."""
    config_path = Path(__file__).parent / "config" / "models.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if model_name in config["models"]:
        return config["models"][model_name]["hf_id"]

    # Treat as direct HF ID if not in config
    return model_name


def parse_args():
    parser = argparse.ArgumentParser(
        description="Launch vLLM server for Hair Transfer Dataset Validator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single GPU — Qwen3-VL-30B-A3B (default)
  python serve_vllm.py

  # 2-GPU tensor parallel
  python serve_vllm.py --tensor-parallel-size 2

  # Smaller model on single GPU
  python serve_vllm.py --model qwen3-vl-8b

  # Custom port
  python serve_vllm.py --port 8001
""",
    )

    parser.add_argument(
        "--model", type=str, default="qwen3-vl-30b-a3b",
        help="Model name from models.yaml or direct HF model ID "
             "(default: qwen3-vl-30b-a3b)",
    )
    parser.add_argument(
        "--host", type=str, default="0.0.0.0",
        help="Server host (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port", type=int, default=8000,
        help="Server port (default: 8000)",
    )
    parser.add_argument(
        "--tensor-parallel-size", type=int, default=1,
        help="Number of GPUs for tensor parallelism (default: 1)",
    )
    parser.add_argument(
        "--max-model-len", type=int, default=32768,
        help="Max context length (default: 32768)",
    )
    parser.add_argument(
        "--gpu-memory-utilization", type=float, default=0.9,
        help="Fraction of GPU memory to use (default: 0.9)",
    )
    parser.add_argument(
        "--dtype", type=str, default="auto",
        choices=["auto", "float16", "bfloat16", "half", "bf16"],
        help="Model dtype (default: auto). 'half' is alias for float16, 'bf16' is alias for bfloat16.",
    )
    parser.add_argument(
        "--quantization", type=str, default=None,
        choices=["awq", "gptq", "squeezellm", None],
        help="vLLM quantization method (default: None, uses model's native format)",
    )
    parser.add_argument(
        "--enable-expert-parallel", action="store_true",
        help="Enable MoE expert parallelism (recommended for MoE models)",
    )
    parser.add_argument(
        "--limit-mm-per-prompt", type=str, default='{"image": 6}',
        help='Limit multimodal inputs per prompt (default: {"image": 6})',
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.dtype == "half":
        args.dtype = "float16"
    elif args.dtype == "bf16":
        args.dtype = "bfloat16"

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    hf_id = _load_hf_id(args.model)
    logger.info(f"Launching vLLM server: {hf_id}")

    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", hf_id,
        "--host", args.host,
        "--port", str(args.port),
        "--tensor-parallel-size", str(args.tensor_parallel_size),
        "--max-model-len", str(args.max_model_len),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--dtype", args.dtype,
        "--trust-remote-code",
        "--limit-mm-per-prompt", args.limit_mm_per_prompt,
    ]

    if args.quantization:
        cmd.extend(["--quantization", args.quantization])

    if args.enable_expert_parallel:
        cmd.append("--enable-expert-parallel")

    logger.info(f"Command: {' '.join(cmd)}")
    logger.info(f"Server will be available at http://{args.host}:{args.port}")
    logger.info(f"Use with: python validate_dataset.py --backend vllm --vllm-url http://localhost:{args.port}")

    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        logger.info("Server stopped.")
    except FileNotFoundError:
        logger.error("vllm not found. Install with: pip install vllm")
        sys.exit(1)


if __name__ == "__main__":
    main()
