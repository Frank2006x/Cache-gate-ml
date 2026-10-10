import unittest

import torch

from src.cache_gating_tta.tent import (
    configure_tent,
    entropy_loss,
)


class TestTent(unittest.TestCase):
    def test_entropy_is_finite(self):
        logits = torch.randn(4, 10)
        loss = entropy_loss(logits)

        self.assertTrue(torch.isfinite(loss).item())
        self.assertEqual(loss.ndim, 0)

    def test_entropy_is_nonnegative(self):
        logits = torch.randn(4, 10)
        loss = entropy_loss(logits)

        self.assertGreaterEqual(loss.item(), 0.0)

    def test_only_batchnorm_affine_parameters_train(self):
        model = torch.nn.Sequential(
            torch.nn.Conv2d(3, 8, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(8),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(),
            torch.nn.Linear(8, 4),
        )

        model, optimizer = configure_tent(
            model,
            optimizer_type="adam",
            learning_rate=1e-3,
        )

        trainable = [
            name
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        ]

        self.assertEqual(
            set(trainable),
            {"1.weight", "1.bias"},
        )
        self.assertIsNotNone(optimizer)

    def test_adaptation_updates_batchnorm_parameters(self):
        torch.manual_seed(42)

        model = torch.nn.Sequential(
            torch.nn.Conv2d(3, 8, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(8),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(),
            torch.nn.Linear(8, 4),
        )

        model, optimizer = configure_tent(model)

        before = model[1].weight.detach().clone()
        images = torch.randn(8, 3, 16, 16)

        optimizer.zero_grad(set_to_none=True)
        loss = entropy_loss(model(images))
        loss.backward()
        optimizer.step()

        after = model[1].weight.detach()

        self.assertTrue(torch.isfinite(loss).item())
        self.assertFalse(torch.equal(before, after))


if __name__ == "__main__":
    unittest.main()
