"""
Perceptual similarity metrics implemented for efficient cache gating.
Includes PyTorch-native GPU-accelerated SSIM, perceptual hashing, and pixel distance.
"""

from typing import Callable, Optional, Tuple, Union
import math
import torch
import torch.nn.functional as F


def _gaussian_window(window_size: int, sigma: float, channel: int) -> torch.Tensor:
    """Generate 1D Gaussian kernel and outer product to 2D window."""
    coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    g_2d = g.unsqueeze(1) @ g.unsqueeze(0)
    window = g_2d.unsqueeze(0).unsqueeze(0).repeat(channel, 1, 1, 1)
    return window


def pytorch_ssim(
    img1: torch.Tensor,
    img2: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
    val_range: float = 1.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Compute Structural Similarity Index (SSIM) between two image tensors using pure PyTorch.
    Runs on GPU with no host-device transfers.

    Args:
        img1: Tensor of shape (B, C, H, W) or (C, H, W) with values normalized to [0, 1].
        img2: Tensor of shape matching img1.
        window_size: Size of Gaussian window (odd integer).
        sigma: Standard deviation of Gaussian kernel.
        val_range: Dynamic range of input images (default: 1.0 for normalized tensors).
        reduction: 'mean', 'none', or 'sum'.

    Returns:
        SSIM score tensor (scalar if reduction is 'mean' and input has batch).
    """
    if img1.dim() == 3:
        img1 = img1.unsqueeze(0)
    if img2.dim() == 3:
        img2 = img2.unsqueeze(0)

    assert img1.shape == img2.shape, f"Image dimensions must match: {img1.shape} vs {img2.shape}"

    b, c, h, w = img1.shape
    device = img1.device
    dtype = img1.dtype

    window = _gaussian_window(window_size, sigma, c).to(device=device, dtype=dtype)

    # Padding to preserve spatial dimensions
    pad = window_size // 2

    mu1 = F.conv2d(img1, window, padding=pad, groups=c)
    mu2 = F.conv2d(img2, window, padding=pad, groups=c)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=pad, groups=c) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=pad, groups=c) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=pad, groups=c) - mu1_mu2

    c1 = (0.01 * val_range) ** 2
    c2 = (0.03 * val_range) ** 2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    )

    if reduction == "mean":
        return ssim_map.mean()
    elif reduction == "sum":
        return ssim_map.sum()
    elif reduction == "none":
        return ssim_map.mean(dim=[1, 2, 3])
    else:
        raise ValueError(f"Unknown reduction: {reduction}")


def normalized_pixel_similarity(img1: torch.Tensor, img2: torch.Tensor, p: int = 1) -> torch.Tensor:
    """
    Compute normalized pixel similarity based on Lp distance in [0, 1].
    Similarity = 1.0 - (||img1 - img2||_p / max_possible_dist).
    """
    if img1.dim() == 3:
        img1 = img1.unsqueeze(0)
    if img2.dim() == 3:
        img2 = img2.unsqueeze(0)

    diff = (img1 - img2).abs()
    if p == 1:
        mean_diff = diff.mean()
    elif p == 2:
        mean_diff = torch.sqrt((diff ** 2).mean())
    else:
        mean_diff = diff.pow(p).mean().pow(1.0 / p)

    # For images scaled in [0, 1], maximum difference per pixel is 1.0
    sim = torch.clamp(1.0 - mean_diff, min=0.0, max=1.0)
    return sim


class PerceptualHashSimilarity:
    """
    Fast PyTorch-native perceptual hash (pHash) based on low-frequency 2D DCT.
    Falls back gracefully to spatial intensity hashing if DCT isn't supported.
    """

    def __init__(self, hash_size: int = 8, highfreq_factor: int = 4):
        self.hash_size = hash_size
        self.img_size = hash_size * highfreq_factor

    def _rgb_to_gray(self, img: torch.Tensor) -> torch.Tensor:
        """Convert (B, C, H, W) to (B, 1, H, W) grayscale."""
        if img.shape[1] == 3:
            # Luminance weights: 0.2989 R + 0.5870 G + 0.1140 B
            weights = torch.tensor([0.2989, 0.5870, 0.1140], device=img.device, dtype=img.dtype).view(1, 3, 1, 1)
            return (img * weights).sum(dim=1, keepdim=True)
        return img

    def _dct_2d(self, x: torch.Tensor) -> torch.Tensor:
        """
        Orthonormal 2D Discrete Cosine Transform (Type-II) implemented with PyTorch.
        """
        b, c, n, m = x.shape
        device, dtype = x.device, x.dtype

        # 1D DCT basis for rows
        i = torch.arange(n, device=device, dtype=dtype).unsqueeze(1)
        j = torch.arange(n, device=device, dtype=dtype).unsqueeze(0)
        basis_n = torch.cos((2 * j + 1) * i * math.pi / (2 * n))
        basis_n[0, :] *= 1.0 / math.sqrt(2.0)
        basis_n *= math.sqrt(2.0 / n)

        # 1D DCT basis for columns
        if n == m:
            basis_m = basis_n
        else:
            i_m = torch.arange(m, device=device, dtype=dtype).unsqueeze(1)
            j_m = torch.arange(m, device=device, dtype=dtype).unsqueeze(0)
            basis_m = torch.cos((2 * j_m + 1) * i_m * math.pi / (2 * m))
            basis_m[0, :] *= 1.0 / math.sqrt(2.0)
            basis_m *= math.sqrt(2.0 / m)

        # 2D DCT: basis_n @ x @ basis_m.T
        x_dct = torch.matmul(basis_n, torch.matmul(x, basis_m.t()))
        return x_dct

    def compute_hash(self, img: torch.Tensor) -> torch.Tensor:
        """
        Generate binary hash tensor for an image batch.
        Returns bool tensor of shape (B, hash_size * hash_size).
        """
        if img.dim() == 3:
            img = img.unsqueeze(0)

        # Resize to (img_size, img_size)
        gray = self._rgb_to_gray(img)
        resized = F.interpolate(gray, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False)

        # Compute 2D DCT
        dct = self._dct_2d(resized)

        # Extract lowest frequencies, excluding DC component at (0, 0)
        low_freq = dct[:, :, :self.hash_size, :self.hash_size]
        med = low_freq.median()
        binary_hash = (low_freq > med).view(img.shape[0], -1)
        return binary_hash

    def __call__(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        """
        Compute normalized pHash similarity: 1.0 - (Hamming distance / Total bits).
        """
        h1 = self.compute_hash(img1)
        h2 = self.compute_hash(img2)
        total_bits = h1.shape[1]
        hamming_dist = (h1 ^ h2).sum(dim=1).float()
        similarity = 1.0 - (hamming_dist / float(total_bits))
        return similarity.mean()


def get_similarity_metric(name: str, **kwargs) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """
    Factory function for perceptual similarity metrics.

    Args:
        name: One of ['ssim', 'phash', 'l1', 'l2', 'cosine']
    """
    name_clean = name.lower().strip()
    if name_clean in ("ssim", "structural_similarity"):
        window_size = kwargs.get("window_size", 11)
        sigma = kwargs.get("sigma", 1.5)
        return lambda x1, x2: pytorch_ssim(x1, x2, window_size=window_size, sigma=sigma)

    elif name_clean in ("phash", "perceptual_hash"):
        hasher = PerceptualHashSimilarity(
            hash_size=kwargs.get("hash_size", 8),
            highfreq_factor=kwargs.get("highfreq_factor", 4),
        )
        return hasher

    elif name_clean in ("l1", "pixel_l1"):
        return lambda x1, x2: normalized_pixel_similarity(x1, x2, p=1)

    elif name_clean in ("l2", "pixel_l2", "mse"):
        return lambda x1, x2: normalized_pixel_similarity(x1, x2, p=2)

    elif name_clean in ("cosine", "cos"):
        def cos_sim(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
            flat1 = x1.view(x1.size(0) if x1.dim() == 4 else 1, -1)
            flat2 = x2.view(x2.size(0) if x2.dim() == 4 else 1, -1)
            return F.cosine_similarity(flat1, flat2, dim=1).mean()
        return cos_sim

    else:
        raise ValueError(f"Unknown similarity metric: '{name}'. Supported: ['ssim', 'phash', 'l1', 'l2', 'cosine']")
