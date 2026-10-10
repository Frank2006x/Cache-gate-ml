"""
Unit and integration tests for Cache-Gated TTA pipeline.
"""

import unittest
import torch
import torch.nn as nn
from gating.similarity import pytorch_ssim, PerceptualHashSimilarity, normalized_pixel_similarity, get_similarity_metric
from gating.cache_gate import CacheGate
from models.backbone import build_backbone
from models.source import Source
from models.tent import Tent
from models.eata import Eata
from models.sar import Sar
from models.cotta import Cotta
from models.dua import Dua
from models.memo import Memo
from utils.metrics import TTAMetricsTracker


class TestSimilarityMetrics(unittest.TestCase):
    def test_ssim_identical(self):
        x = torch.rand(1, 3, 32, 32)
        score = pytorch_ssim(x, x)
        self.assertAlmostEqual(score.item(), 1.0, places=4)

    def test_ssim_dissimilar(self):
        x1 = torch.zeros(1, 3, 32, 32)
        x2 = torch.ones(1, 3, 32, 32)
        score = pytorch_ssim(x1, x2)
        self.assertLess(score.item(), 0.1)

    def test_phash_identical(self):
        x = torch.rand(1, 3, 32, 32)
        hasher = PerceptualHashSimilarity()
        sim = hasher(x, x)
        self.assertAlmostEqual(sim.item(), 1.0, places=3)

    def test_factory(self):
        for name in ["ssim", "phash", "l1", "l2", "cosine"]:
            fn = get_similarity_metric(name)
            self.assertTrue(callable(fn))


class TestCacheGate(unittest.TestCase):
    def test_gating_decision_lifecycle(self):
        gate = CacheGate(metric_name="ssim", threshold=0.85, policy="keyframe")
        x_base = torch.rand(1, 3, 32, 32)

        # 1. First frame should trigger adaptation (cold cache warmup)
        d1 = gate.decide(x_base)
        self.assertTrue(d1.should_adapt)
        self.assertTrue(d1.is_warmup)
        self.assertFalse(d1.cache_hit)

        # 2. Identical frame should be identified as redundant (cache hit -> skip update)
        d2 = gate.decide(x_base)
        self.assertFalse(d2.should_adapt)
        self.assertTrue(d2.cache_hit)
        self.assertAlmostEqual(d2.similarity, 1.0, places=3)

        # 3. Micro-perturbed frame (e.g. slight noise) should also be recognized as redundant
        x_noisy = torch.clamp(x_base + torch.randn_like(x_base) * 0.01, 0.0, 1.0)
        d3 = gate.decide(x_noisy)
        self.assertFalse(d3.should_adapt)
        self.assertTrue(d3.cache_hit)
        self.assertGreater(d3.similarity, 0.85)

        # 4. Completely novel frame should trigger adaptation (cache miss)
        x_novel = torch.rand(1, 3, 32, 32)
        # Shift distribution significantly
        x_novel = 1.0 - x_base
        d4 = gate.decide(x_novel)
        self.assertTrue(d4.should_adapt)
        self.assertFalse(d4.cache_hit)


class TestTTAModels(unittest.TestCase):
    def setUp(self):
        self.backbone = build_backbone("resnet18", num_classes=10, pretrained=False, dataset="cifar10")

    def test_tent_trainable_params(self):
        tent = Tent(self.backbone, lr=1e-3)
        # Only BN weights and biases should require grad
        for nm, m in tent.model.named_modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                self.assertTrue(m.weight.requires_grad)
                self.assertTrue(m.bias.requires_grad)
            elif isinstance(m, nn.Conv2d):
                self.assertFalse(m.weight.requires_grad)

    def test_tent_forward_and_adapt(self):
        tent = Tent(self.backbone, lr=1e-3)
        x = torch.rand(2, 3, 32, 32)

        # Pure forward
        out_fwd = tent.forward(x)
        self.assertEqual(out_fwd.shape, (2, 10))

        # Adapt step
        out_adapt, metrics = tent.adapt_and_forward(x)
        self.assertEqual(out_adapt.shape, (2, 10))
        self.assertIn("loss", metrics)
        self.assertIn("entropy", metrics)

    def test_eata_adapt(self):
        eata = Eata(self.backbone, lr=1e-3, num_classes=10)
        x = torch.rand(2, 3, 32, 32)
        out, metrics = eata.adapt_and_forward(x)
        self.assertEqual(out.shape, (2, 10))
        self.assertIn("loss", metrics)

    def test_source_forward_and_adapt(self):
        source = Source(self.backbone)
        x = torch.rand(2, 3, 32, 32)
        out_fwd = source.forward(x)
        self.assertEqual(out_fwd.shape, (2, 10))
        out_adapt, metrics = source.adapt_and_forward(x)
        self.assertEqual(out_adapt.shape, (2, 10))
        self.assertEqual(metrics["loss"], 0.0)

    def test_cotta_adapt(self):
        cotta = Cotta(self.backbone, lr=1e-3)
        x = torch.rand(2, 3, 32, 32)
        out, metrics = cotta.adapt_and_forward(x)
        self.assertEqual(out.shape, (2, 10))
        self.assertIn("loss", metrics)

    def test_dua_adapt(self):
        dua = Dua(self.backbone, num_classes=10)
        x = torch.rand(2, 3, 32, 32)
        out, metrics = dua.adapt_and_forward(x)
        self.assertEqual(out.shape, (2, 10))
        self.assertIn("loss", metrics)

    def test_memo_adapt(self):
        memo = Memo(self.backbone, lr=1e-3, num_aug=3)
        x = torch.rand(1, 3, 32, 32)
        out, metrics = memo.adapt_and_forward(x)
        self.assertEqual(out.shape, (1, 10))
        self.assertIn("loss", metrics)
        self.assertIn("marginal_entropy", metrics)


class TestMetricsTracker(unittest.TestCase):
    def test_tracker_compute_savings(self):
        tracker = TTAMetricsTracker()
        # 10 frames: 4 adapted, 6 redundant (bypassed)
        for i in range(10):
            preds = torch.tensor([[1.0, 0.0]])
            target = torch.tensor([0])
            adapted = i < 4
            tracker.update(preds, target, adapted=adapted, cache_hit=not adapted)

        self.assertEqual(tracker.total_frames, 10)
        self.assertAlmostEqual(tracker.update_ratio, 0.4)
        self.assertAlmostEqual(tracker.cache_hit_ratio, 0.6)

        savings = tracker.compute_theoretical_compute_savings()
        # 6 bypassed backward passes out of 10
        self.assertGreater(savings["savings_percent"], 0.0)


if __name__ == "__main__":
    unittest.main()
