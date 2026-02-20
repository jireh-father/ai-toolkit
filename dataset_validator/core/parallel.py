"""Multi-GPU parallel evaluation workers."""

import logging
import multiprocessing as mp
from typing import Optional

from tqdm import tqdm

logger = logging.getLogger(__name__)


def _gpu_worker(
    gpu_id: int,
    entries: list[dict],
    model_name: str,
    quantization: str,
    short_side: int,
    max_retries: int,
    result_queue: mp.Queue,
    progress_queue: mp.Queue,
    low_vram: bool = False,
):
    """Worker process for a single GPU.

    Loads its own VLM instance and processes assigned entries.
    """
    import torch
    from dataset_validator.core.evaluator import load_vlm, evaluate_single, SCORE_FIELDS
    from dataset_validator.core.image_loader import load_image_triplet

    device = f"cuda:{gpu_id}"
    logger.info(f"[GPU {gpu_id}] Loading model on {device}...")

    try:
        vlm = load_vlm(
            model_name, quantization=quantization, device=device,
            low_vram=low_vram,
        )
    except Exception as e:
        logger.error(f"[GPU {gpu_id}] Failed to load model: {e}")
        for entry in entries:
            result_queue.put({
                "filename": entry["stem"],
                "scores": None,
                "reason": f"Model load failed on GPU {gpu_id}: {e}",
                "error": True,
            })
            progress_queue.put(1)
        return

    logger.info(f"[GPU {gpu_id}] Processing {len(entries)} entries...")

    for entry in entries:
        triplet = load_image_triplet(entry, short_side=short_side)
        if triplet is None:
            result_queue.put({
                "filename": entry["stem"],
                "scores": None,
                "reason": "Failed to load images",
                "error": True,
            })
        else:
            scores = evaluate_single(vlm, triplet, max_retries=max_retries)
            if scores is None:
                result_queue.put({
                    "filename": entry["stem"],
                    "scores": None,
                    "reason": "VLM evaluation failed after retries",
                    "error": True,
                })
            else:
                reason = scores.pop("reason", "")
                result_queue.put({
                    "filename": entry["stem"],
                    "scores": scores,
                    "reason": reason,
                    "error": False,
                })
        progress_queue.put(1)

    # Cleanup
    del vlm
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _distribute_entries(entries: list[dict], num_gpus: int) -> list[list[dict]]:
    """Distribute entries across GPUs in round-robin fashion."""
    chunks = [[] for _ in range(num_gpus)]
    for i, entry in enumerate(entries):
        chunks[i % num_gpus].append(entry)
    return chunks


def run_parallel_evaluation(
    entries: list[dict],
    model_name: str,
    quantization: str,
    num_gpus: int,
    short_side: int = 512,
    max_retries: int = 3,
    checkpoint_manager=None,
    checkpoint_interval: int = 100,
    low_vram: bool = False,
    backend: str = "local",
    vllm_url: str = "http://localhost:8000",
) -> list[dict]:
    """Run evaluation across multiple GPUs or via vLLM server.

    For backend="vllm", sends requests to external vLLM server (no local GPU needed).
    For backend="local" with num_gpus=1, runs directly without multiprocessing.
    For backend="local" with num_gpus>1, spawns worker processes.
    """
    if backend == "vllm":
        return _run_vllm(
            entries, model_name, short_side, max_retries,
            checkpoint_manager, checkpoint_interval, vllm_url,
        )

    if num_gpus <= 1:
        return _run_single_gpu(
            entries, model_name, quantization, short_side,
            max_retries, checkpoint_manager, checkpoint_interval,
            low_vram=low_vram,
        )

    return _run_multi_gpu(
        entries, model_name, quantization, num_gpus,
        short_side, max_retries, checkpoint_manager, checkpoint_interval,
        low_vram=low_vram,
    )


def _run_vllm(
    entries: list[dict],
    model_name: str,
    short_side: int,
    max_retries: int,
    checkpoint_manager,
    checkpoint_interval: int,
    vllm_url: str,
) -> list[dict]:
    """Run evaluation via vLLM OpenAI-compatible API (no local GPU needed)."""
    from dataset_validator.core.evaluator import load_model_config, build_prompt
    from dataset_validator.core.image_loader import load_image_triplet
    from dataset_validator.core.vllm_client import evaluate_single_vllm, check_server_health

    # Resolve HF model ID for vLLM API
    model_config = load_model_config(model_name)
    hf_id = model_config["hf_id"]
    _, user_prompt = build_prompt(model_config["family"])

    # Check server health
    if not check_server_health(vllm_url):
        logger.error(
            f"vLLM server not reachable at {vllm_url}. "
            f"Start it first: python dataset_validator/serve_vllm.py"
        )
        raise ConnectionError(f"vLLM server not reachable at {vllm_url}")

    logger.info(f"Connected to vLLM server at {vllm_url} (model: {hf_id})")

    results = []
    pbar = tqdm(entries, desc="Evaluating (vLLM)", unit="sample")

    for entry in pbar:
        triplet = load_image_triplet(entry, short_side=short_side)
        if triplet is None:
            result = {
                "filename": entry["stem"],
                "scores": None,
                "reason": "Failed to load images",
                "error": True,
            }
        else:
            scores = evaluate_single_vllm(
                vllm_url=vllm_url,
                model_id=hf_id,
                images=triplet,
                prompt=user_prompt,
                max_retries=max_retries,
            )
            if scores is None:
                result = {
                    "filename": entry["stem"],
                    "scores": None,
                    "reason": "vLLM evaluation failed after retries",
                    "error": True,
                }
            else:
                reason = scores.pop("reason", "")
                result = {
                    "filename": entry["stem"],
                    "scores": scores,
                    "reason": reason,
                    "error": False,
                }

        results.append(result)

        if checkpoint_manager is not None:
            checkpoint_manager.add_result(result)
            if checkpoint_manager.should_save(len(results)):
                checkpoint_manager.save()

        passed = sum(
            1 for r in results
            if not r["error"] and r["scores"] is not None
        )
        pbar.set_postfix(done=len(results), ok=passed)

    return results


def _run_single_gpu(
    entries: list[dict],
    model_name: str,
    quantization: str,
    short_side: int,
    max_retries: int,
    checkpoint_manager,
    checkpoint_interval: int,
    low_vram: bool = False,
) -> list[dict]:
    """Run evaluation on a single GPU (no multiprocessing)."""
    from dataset_validator.core.evaluator import load_vlm, evaluate_single
    from dataset_validator.core.image_loader import load_image_triplet

    vlm = load_vlm(
        model_name, quantization=quantization, device="cuda:0",
        low_vram=low_vram,
    )

    results = []
    pbar = tqdm(entries, desc="Evaluating", unit="sample")

    for entry in pbar:
        triplet = load_image_triplet(entry, short_side=short_side)
        if triplet is None:
            result = {
                "filename": entry["stem"],
                "scores": None,
                "reason": "Failed to load images",
                "error": True,
            }
        else:
            scores = evaluate_single(vlm, triplet, max_retries=max_retries)
            if scores is None:
                result = {
                    "filename": entry["stem"],
                    "scores": None,
                    "reason": "VLM evaluation failed after retries",
                    "error": True,
                }
            else:
                reason = scores.pop("reason", "")
                result = {
                    "filename": entry["stem"],
                    "scores": scores,
                    "reason": reason,
                    "error": False,
                }

        results.append(result)

        if checkpoint_manager is not None:
            checkpoint_manager.add_result(result)
            if checkpoint_manager.should_save(len(results)):
                checkpoint_manager.save()

        # Update progress bar postfix
        passed = sum(
            1 for r in results
            if not r["error"] and r["scores"] is not None
        )
        pbar.set_postfix(done=len(results), ok=passed)

    return results


def _run_multi_gpu(
    entries: list[dict],
    model_name: str,
    quantization: str,
    num_gpus: int,
    short_side: int,
    max_retries: int,
    checkpoint_manager,
    checkpoint_interval: int,
    low_vram: bool = False,
) -> list[dict]:
    """Run evaluation across multiple GPUs using multiprocessing."""
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    progress_queue = ctx.Queue()

    chunks = _distribute_entries(entries, num_gpus)

    processes = []
    for gpu_id in range(num_gpus):
        if not chunks[gpu_id]:
            continue
        p = ctx.Process(
            target=_gpu_worker,
            args=(
                gpu_id,
                chunks[gpu_id],
                model_name,
                quantization,
                short_side,
                max_retries,
                result_queue,
                progress_queue,
            ),
            kwargs={
                "low_vram": low_vram,
            },
        )
        p.start()
        processes.append(p)

    # Collect results with progress bar
    total = len(entries)
    results = []
    pbar = tqdm(total=total, desc="Evaluating (multi-GPU)", unit="sample")

    collected = 0
    while collected < total:
        # Wait for progress updates
        try:
            progress_queue.get(timeout=300)
            collected += 1
            pbar.update(1)
        except Exception:
            # Check if workers are still alive
            alive = any(p.is_alive() for p in processes)
            if not alive:
                break

        # Drain result queue
        while not result_queue.empty():
            result = result_queue.get_nowait()
            results.append(result)

            if checkpoint_manager is not None:
                checkpoint_manager.add_result(result)
                if checkpoint_manager.should_save(len(results)):
                    checkpoint_manager.save()

    pbar.close()

    # Drain remaining results
    while not result_queue.empty():
        result = result_queue.get_nowait()
        results.append(result)
        if checkpoint_manager is not None:
            checkpoint_manager.add_result(result)

    # Wait for all processes to finish
    for p in processes:
        p.join(timeout=30)

    return results
