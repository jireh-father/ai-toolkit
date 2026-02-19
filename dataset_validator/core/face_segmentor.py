"""Face segmentation using facer (FaRL wrapper) for face parsing."""

import logging

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)

# celebm/448 label indices for face region mask
# Labels: 0:background, 1:neck, 2:face, 3:cloth, 4:rr, 5:lr, 6:rb, 7:lb,
#         8:re, 9:le, 10:nose, 11:imouth, 12:llip, 13:ulip, 14:hair,
#         15:eyeg, 16:hat, 17:earr, 18:neck_l
FACE_LABEL_INDICES = [2, 6, 7, 8, 9, 10, 11, 12, 13]
# face(skin), rb(right brow), lb(left brow), re(right eye), le(left eye),
# nose, imouth, llip(lower lip), ulip(upper lip)


class FaceSegmentor:
    """Face segmentation using facer's FaRL model."""

    def __init__(self, device: str = "cuda:0"):
        import facer

        self.device = torch.device(device)
        logger.info(f"Loading face detector (retinaface/mobilenet) on {device}")
        self.detector = facer.face_detector("retinaface/mobilenet", device=self.device)
        logger.info(f"Loading face parser (farl/celebm/448) on {device}")
        self.parser = facer.face_parser("farl/celebm/448", device=self.device)
        logger.info("FaceSegmentor ready")

    def _to_tensor(self, image: np.ndarray) -> torch.Tensor:
        """Convert HWC uint8 numpy array to BCHW uint8 tensor."""
        # image: (H, W, 3) uint8
        tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
        return tensor.to(self.device)

    def parse(self, image: np.ndarray) -> dict | None:
        """Run face detection + parsing on a single image.

        Args:
            image: HWC uint8 RGB numpy array

        Returns:
            dict with 'seg_logits' (1, C, H, W) tensor and 'label_names',
            or None if no face detected.
        """
        tensor = self._to_tensor(image)

        with torch.no_grad():
            faces = self.detector(tensor)

        if len(faces.get("rects", [])) == 0:
            return None

        with torch.no_grad():
            faces = self.parser(tensor, faces)

        seg_data = faces.get("seg")
        if seg_data is None:
            return None

        return {
            "seg_logits": seg_data["logits"],
            "label_names": seg_data["label_names"],
        }

    def get_face_mask(self, parse_result: dict) -> np.ndarray | None:
        """Extract boolean face mask from parsing result.

        Args:
            parse_result: output from self.parse()

        Returns:
            (H, W) boolean numpy array, or None if mask is empty.
        """
        logits = parse_result["seg_logits"]  # (1, C, H, W)
        seg_map = logits.argmax(dim=1).squeeze(0).cpu().numpy()  # (H, W)

        mask = np.isin(seg_map, FACE_LABEL_INDICES)

        if mask.sum() == 0:
            return None

        return mask

    def segment(self, image: np.ndarray) -> np.ndarray | None:
        """Convenience: parse image and return face mask directly.

        Args:
            image: HWC uint8 RGB numpy array

        Returns:
            (H, W) boolean numpy array, or None if no face detected.
        """
        result = self.parse(image)
        if result is None:
            return None
        return self.get_face_mask(result)
