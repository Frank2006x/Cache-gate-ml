"""
TENT: Test Entropy Minimization (Wang et al., ICLR 2021).
Adapts batch normalization affine parameters by minimizing test sample prediction entropy.
"""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from .base import BaseTTA


def softmax_entropy(logits: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Compute Shannon entropy H(p) = -sum(p * log(p)) over classes."""
    p = torch.softmax(logits, dim=-1)
    entropy = -(p * torch.log(p + eps)).sum(dim=-1)
    return entropy.mean()


class Tent(BaseTTA):
    """
    TENT: Fully Test-Time Adaptation by Entropy Minimization.

    Args:
        model: Base neural network backbone.
        lr: Learning rate for optimizer (default: 1e-3).
        opt_name: Optimizer name ('adam' or 'sgd').
        steps: Number of gradient steps per adaptation trigger (default: 1).
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-3,
        opt_name: str = "adam",
        steps: int = 1,
        **kwargs,
    ):
        super().__init__(model, lr=lr, **kwargs)
        self.opt_name = opt_name.lower().strip()
        self.steps = steps

        # Configure model: freeze weights, unfreeze BN affine parameters
        self.model = self.configure_model(self.model)
        self.params, self.param_names = self.collect_params(self.model)
        self.setup_optimizer()

    def setup_optimizer(self) -> None:
        """Setup optimizer for BN affine parameters."""
        if not self.params:
            self.optimizer = None
            return

        if self.opt_name == "adam":
            self.optimizer = optim.Adam(self.params, lr=self.lr, betas=(0.9, 0.999), weight_decay=0.0)
        elif self.opt_name == "sgd":
            self.optimizer = optim.SGD(self.params, lr=self.lr, momentum=0.9, weight_decay=0.0)
        else:
            raise ValueError(f"Unsupported optimizer: {self.opt_name}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pure inference pass without parameter updates.
        Runs without gradient tracking to conserve memory and latency.
        """
        with torch.no_grad():
            outputs = self.model(x)
        return outputs

    def adapt_and_forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Adapt BN parameters via entropy minimization on x and return predictions.

        Args:
            x: Test batch tensor of shape (B, C, H, W).

        Returns:
            outputs: Final prediction logits.
            metrics: Dictionary of adaptation metrics {'loss': entropy_val}.
        """
        if self.optimizer is None:
            return self.forward(x), {"loss": 0.0, "entropy": 0.0}

        outputs = None
        loss_val = 0.0

        for _ in range(self.steps):
            self.optimizer.zero_grad()
            outputs = self.model(x)
            loss = softmax_entropy(outputs)
            loss.backward()
            self.optimizer.step()
            loss_val = float(loss.detach().item())

        return outputs.detach(), {"loss": loss_val, "entropy": loss_val}
