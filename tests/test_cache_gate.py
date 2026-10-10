import unittest

import torch
import torch.nn as nn

from src.cache_gating_tta.cache_gate import CacheGate
from src.cache_gating_tta.tent import Tent, configure_tent


class TestCacheGate(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

        model = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(8, 4),
        )

        model, optimizer = configure_tent(model)
        self.tent = Tent(model, optimizer)
        self.gate = CacheGate(self.tent, threshold=0.95)

        self.image = torch.rand(1, 3, 32, 32)

    def test_first_frame_is_a_miss(self):
        result = self.gate.process(self.image, self.image)

        self.assertFalse(result["cache_hit"])
        self.assertEqual(result["backward_passes"], 1)

    def test_identical_frame_is_a_hit(self):
        self.gate.process(self.image, self.image)

        result = self.gate.process(self.image, self.image)

        self.assertTrue(result["cache_hit"])
        self.assertEqual(result["backward_passes"], 0)
        self.assertAlmostEqual(result["ssim"], 1.0, places=5)

    def test_changed_frame_causes_a_miss(self):
        self.gate.process(self.image, self.image)

        changed = 1.0 - self.image
        result = self.gate.process(changed, changed)

        self.assertFalse(result["cache_hit"])
        self.assertEqual(result["backward_passes"], 1)

    def test_reset_clears_cache(self):
        self.gate.process(self.image, self.image)
        self.gate.reset()

        result = self.gate.process(self.image, self.image)

        self.assertFalse(result["cache_hit"])
        self.assertEqual(result["backward_passes"], 1)


if __name__ == "__main__":
    unittest.main()
