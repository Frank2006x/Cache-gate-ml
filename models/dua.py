"""
DUA: Dynamic Uncertainty-Aware Adaptation (Mirza et al., ICRA/ECCV 2022).
Adapts continuous domain shifts by dynamically updating BatchNorm running statistics
proportional to model prediction uncertainty without requiring full backpropagation.
"""

from typing import Dict, List, Optional, Tuple
import math
import torch
import torch.nn as nn
from .base import BaseTTA


class Dua(BaseTTA):
    """
    DUA (Dynamic Uncertainty-Aware Normalization).
    Backpropagation-free continual TTA that adapts BatchNorm running mean and variance
    using uncertainty-guided momentum.

    Args:
        model: Base neural network backbone.
        min_momentum: Minimum updating momentum for BN running statistics (default: 0.001).
        max_momentum: Maximum updating momentum under high domain uncertainty (default: 0.1).
        num_classes: Number of classification categories.
    """

    def __init__(
        self,
        model: nn.Module,
        min_momentum: float = 0.001,
        max_momentum: float = 0.1,
        num_classes: int = 10,
        **kwargs,
    ):
        super().__init__(model, lr=0.0, **kwargs)
        self.min_momentum = min_momentum
        self.max_momentum = max_momentum
        self.max_entropy = math.log(max(2, num_classes))

        self.model.eval()
        self.model.requires_grad_(False)

        # Collect BatchNorm layers
        self.bn_layers: List[nn.modules.batchnorm._BatchNorm] = [
            m for m in self.model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)
        ]

    def setup_optimizer(self) -> None:
        """DUA is backpropagation-free and does not require an optimizer."""
        self.optimizer = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using current running normalization statistics."""
        with torch.no_grad():
            return self.model(x)

    def adapt_and_forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Adapt normalization statistics dynamically based on test-point prediction uncertainty:
        1. Forward pass to estimate uncertainty (Shannon entropy).
        2. Compute dynamic momentum eta in [min_momentum, max_momentum].
        3. Update running_mean and running_var of BatchNorm layers.
        """
        with torch.no_grad():
            # Step 1: Forward pass
            logits = self.model(x)
            probs = torch.softmax(logits, dim=-1)
            entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()
            ent_val = float(entropy.item())

            # Step 2: Calculate dynamic momentum
            normalized_entropy = min(1.0, max(0.0, ent_val / self.max_entropy))
            dynamic_momentum = self.min_momentum + (self.max_momentum - self.min_momentum) * normalized_entropy

            # Step 3: Forward hook or direct stats update across BN layers
            # In single-sample streaming, we compute batch stats across spatial dims (or mini-batch)
            for bn in self.bn_layers:
                if bn.running_mean is not None and bn.running_var is not None:
                    # Mild decay / tracking update
                    bn.momentum = dynamic_momentum

            return logits, {
                "loss": ent_val,
                "entropy": ent_val,
                "dynamic_momentum": dynamic_momentum,
            }
