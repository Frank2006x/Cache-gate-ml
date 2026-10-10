import torch
import torch.nn.functional as F


def _gaussian_window(window_size, sigma, channels, device, dtype):
    coords = torch.arange(
        window_size, device=device, dtype=dtype
    ) - window_size // 2

    gaussian = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    gaussian = gaussian / gaussian.sum()

    window_2d = gaussian[:, None] * gaussian[None, :]
    window_2d = window_2d / window_2d.sum()

    return window_2d.expand(
        channels, 1, window_size, window_size
    ).contiguous()


def ssim(
    image_a,
    image_b,
    window_size=11,
    sigma=1.5,
    data_range=1.0,
):
    """
    Compute SSIM for RGB image batches.

    Inputs:
        image_a, image_b: [B, C, H, W], floating-point tensors.
        Values should be in [0, data_range].

    Returns:
        One SSIM score per image, with shape [B].
    """
    if image_a.shape != image_b.shape:
        raise ValueError("Input tensors must have identical shapes.")

    if image_a.ndim != 4:
        raise ValueError("Inputs must have shape [B, C, H, W].")

    if not image_a.is_floating_point() or not image_b.is_floating_point():
        raise TypeError("Inputs must be floating-point tensors.")

    if image_a.device != image_b.device:
        raise ValueError("Inputs must be on the same device.")

    if image_a.shape[-2] < window_size or image_a.shape[-1] < window_size:
        raise ValueError("Images must be at least as large as the SSIM window.")

    if not torch.isfinite(image_a).all() or not torch.isfinite(image_b).all():
        raise ValueError("Inputs must contain only finite values.")

    if image_a.min() < 0 or image_b.min() < 0:
        raise ValueError("Input values must be nonnegative.")

    if image_a.max() > data_range or image_b.max() > data_range:
        raise ValueError("Input values exceed the configured data range.")

    channels = image_a.shape[1]

    window = _gaussian_window(
        window_size,
        sigma,
        channels,
        image_a.device,
        image_a.dtype,
    )

    mu_a = F.conv2d(image_a, window, groups=channels)
    mu_b = F.conv2d(image_b, window, groups=channels)

    mu_a_sq = mu_a.square()
    mu_b_sq = mu_b.square()
    mu_ab = mu_a * mu_b

    sigma_a_sq = (
        F.conv2d(image_a * image_a, window, groups=channels)
        - mu_a_sq
    )
    sigma_b_sq = (
        F.conv2d(image_b * image_b, window, groups=channels)
        - mu_b_sq
    )
    sigma_ab = (
        F.conv2d(image_a * image_b, window, groups=channels)
        - mu_ab
    )

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    numerator = (2 * mu_ab + c1) * (2 * sigma_ab + c2)
    denominator = (
        (mu_a_sq + mu_b_sq + c1)
        * (sigma_a_sq + sigma_b_sq + c2)
    )

    score_map = numerator / denominator.clamp_min(1e-12)

    return score_map.mean(dim=(1, 2, 3))
