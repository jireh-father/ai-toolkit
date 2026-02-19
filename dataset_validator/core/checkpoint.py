"""Checkpoint management for resumable evaluation."""

import json
import logging
import signal
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Manages checkpoint save/load for long-running evaluations."""

    def __init__(
        self,
        checkpoint_dir: Path,
        interval: int = 100,
        model_name: str = "",
        quantization: str = "int4",
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.interval = interval
        self.model_name = model_name
        self.quantization = quantization

        self.processed_files: set[str] = set()
        self.results: list[dict] = []
        self.total_count: int = 0
        self._save_requested = False
        self._original_sigint = None

        self._install_signal_handlers()

    def _install_signal_handlers(self):
        """Install Ctrl+C handler to save checkpoint before exit."""
        self._original_sigint = signal.getsignal(signal.SIGINT)

        def _handler(sig, frame):
            logger.info("Interrupt received — saving checkpoint before exit...")
            self._save_requested = True
            self.save()
            logger.info("Checkpoint saved. Exiting.")
            sys.exit(0)

        signal.signal(signal.SIGINT, _handler)

    def restore_signal_handlers(self):
        """Restore original signal handlers."""
        if self._original_sigint is not None:
            signal.signal(signal.SIGINT, self._original_sigint)

    def _checkpoint_path(self) -> Path:
        count = len(self.processed_files)
        return self.checkpoint_dir / f"checkpoint_{count:06d}.json"

    def _latest_checkpoint_path(self) -> Optional[Path]:
        """Find the most recent checkpoint file."""
        checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_*.json"))
        return checkpoints[-1] if checkpoints else None

    def should_save(self, processed_count: int) -> bool:
        """Check if a checkpoint should be saved at this count."""
        return processed_count > 0 and processed_count % self.interval == 0

    def save(self):
        """Save current state to a checkpoint file."""
        data = {
            "processed_files": sorted(self.processed_files),
            "processed_count": len(self.processed_files),
            "total_count": self.total_count,
            "results": self.results,
            "model": self.model_name,
            "quantization": self.quantization,
            "timestamp": datetime.now().isoformat(),
        }

        path = self._checkpoint_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info(
            f"Checkpoint saved: {path.name} "
            f"({len(self.processed_files)}/{self.total_count})"
        )

    def load(self) -> bool:
        """Load the latest checkpoint. Returns True if loaded successfully."""
        path = self._latest_checkpoint_path()
        if path is None:
            logger.info("No checkpoint found — starting from scratch")
            return False

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.processed_files = set(data.get("processed_files", []))
            self.results = data.get("results", [])
            self.total_count = data.get("total_count", 0)

            logger.info(
                f"Checkpoint loaded: {path.name} — "
                f"{len(self.processed_files)} already processed"
            )
            return True
        except Exception as e:
            logger.error(f"Failed to load checkpoint {path}: {e}")
            return False

    def add_result(self, result: dict):
        """Record a processed result."""
        self.results.append(result)
        self.processed_files.add(result["filename"])

    def is_processed(self, filename: str) -> bool:
        """Check if a file has already been processed."""
        return filename in self.processed_files

    def get_remaining(self, all_entries: list[dict]) -> list[dict]:
        """Filter out already-processed entries."""
        remaining = [
            e for e in all_entries
            if e["stem"] not in self.processed_files
        ]
        logger.info(
            f"Remaining to process: {len(remaining)} "
            f"(skipping {len(all_entries) - len(remaining)} already done)"
        )
        return remaining
