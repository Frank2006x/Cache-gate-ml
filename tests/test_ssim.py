import unittest

import torch

from src.cache_gating_tta.ssim import ssim


class TestSSIM(unittest.TestCase):
    def test_identical_images_score_near_one(self):
        torch.manual_seed(42)
        image = torch.rand(2, 3, 32, 32)

        score = ssim(image, image)

        self.assertTrue(torch.allclose(
            score,
            torch.ones_like(score),
            atol=1e-5,
        ))

    def test_different_images_have_lower_similarity(self):
        torch.manual_seed(42)
        image_a = torch.rand(2, 3, 32, 32)
        image_b = torch.rand(2, 3, 32, 32)

        score = ssim(image_a, image_b)

        self.assertTrue(torch.all(score < 0.5).item())

    def test_small_brightness_change_reduces_similarity(self):
        torch.manual_seed(42)
        image = torch.rand(2, 3, 32, 32) * 0.7 + 0.1
        brighter = (image + 0.05).clamp(0, 1)

        score = ssim(image, brighter)

        self.assertTrue(torch.all(score < 1.0).item())
        self.assertTrue(torch.all(score > 0.5).item())

    def test_shape_mismatch_raises(self):
        image_a = torch.rand(1, 3, 32, 32)
        image_b = torch.rand(1, 3, 31, 32)

        with self.assertRaises(ValueError):
            ssim(image_a, image_b)

    def test_unnormalized_range_is_required(self):
        image_a = torch.randn(1, 3, 32, 32)
        image_b = torch.randn(1, 3, 32, 32)

        with self.assertRaises(ValueError):
            ssim(image_a, image_b)


if __name__ == "__main__":
    unittest.main()
