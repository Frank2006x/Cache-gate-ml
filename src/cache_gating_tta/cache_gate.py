import torch

from src.cache_gating_tta.ssim import ssim
from src.cache_gating_tta.tent import Tent


class CacheGate:
    def __init__(self, tent: Tent, threshold: float = 0.95):
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0 and 1")

        self.tent = tent
        self.threshold = threshold
        self.reference = None

    def reset(self):
        self.reference = None
        self.tent.reset()

    def process(self, image, image_raw):
        """
        Process one frame.

        image: normalized model input, shape [1, C, H, W]
        image_raw: unnormalized image in [0, 1], same shape
        """
        if image.shape[0] != 1 or image_raw.shape[0] != 1:
            raise ValueError("CacheGate currently expects one frame at a time")

        if image.shape != image_raw.shape:
            raise ValueError("Normalized and raw inputs must have matching shapes")

        if image.device != image_raw.device:
            raise ValueError("Inputs must be on the same device")

        if not torch.isfinite(image_raw).all():
            raise ValueError("Raw image contains non-finite values")

        if image_raw.min() < 0 or image_raw.max() > 1:
            raise ValueError("Raw image must be in [0, 1]")

        score = None

        if self.reference is None:
            hit = False
        else:
            with torch.no_grad():
                score = ssim(image_raw, self.reference).item()
            hit = score >= self.threshold

        logits, entropy, backward_count = self.tent.adapt_step(
            image,
            adapt=not hit,
        )

        if not hit:
            self.reference = image_raw.detach().clone()

        return {
            "logits": logits,
            "cache_hit": hit,
            "ssim": score,
            "entropy": entropy,
            "backward_passes": backward_count,
        }
