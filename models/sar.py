"""
SAR: Sharpness-Aware and Reliable Test-Time Adaptation (Niu et al., ICLR 2023).
Stabilizes continuous adaptation against catastrophic collapse using Sharpness-Aware Minimization (SAM).
"""

from typing import Dict, List, Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.optim as optim
from .base import BaseTTA


class Sar(BaseTTA):
    """
    Sharpness-Aware and Reliable Test-Time Adaptation (SAR).

    Args:
        model: Base vision backbone.
        lr: Learning rate (default: 1e-3).
        rho: Neighborhood size for SAM perturbation (default: 0.05).
        e_margin: Entropy margin threshold for reliable sample selection.
        num_classes: Number of classification categories.
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-3,
        rho: float = 0.05,
        e_margin: Optional[float] = None,
        num_classes: int = 10,
        **kwargs,
    ):
        super().__init__(model, lr=lr, **kwargs)
        self.rho = rho
        self.num_classes = num_classes

        if e_margin is None:
            self.e_margin = 0.4 * math.log(max(2, num_classes))
        else:
            self.e_margin = e_margin

        self.model = self.configure_model(self.model)
        self.params, self.param_names = self.collect_params(self.model)
        self.setup_optimizer()

    def setup_optimizer(self) -> None:
        """Setup SGD optimizer for SAM update."""
        if not self.params:
            self.optimizer = None
            return
        self.optimizer = optim.SGD(self.params, lr=self.lr, momentum=0.9)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass without adaptation."""
        with torch.no_grad():
            return self.model(x)

    def adapt_and_forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Execute SAM two-step adaptation on reliable samples.
        1. Compute gradient and perturb weights along gradient direction.
        2. Compute gradient at perturbed weights and update original weights.
        """
        if self.optimizer is None:
            return self.forward(x), {"loss": 0.0, "entropy": 0.0}

        # Step 1: Forward pass
        outputs = self.model(x)
        probs = torch.softmax(outputs, dim=-1)
        entropies = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)

        # Filter reliable samples
        reliable_mask = entropies < self.e_margin
        if reliable_mask.sum() == 0:
            return outputs.detach(), {"loss": 0.0, "entropy": float(entropies.mean().item())}

        loss = entropies[reliable_mask].mean()
        self.optimizer.zero_grad()
        loss.backward()

        # Step 2: SAM perturbation
        # Compute gradient norm across all BN parameters
        grad_norm = torch.norm(
            torch.stack([p.grad.norm(p=2) for p in self.params if p.grad is not None]),
            p=2,
        )

        scale = self.rho / (grad_norm + 1e-12)
        perturbations = []

        with torch.no_grad():
            for p in self.params:
                if p.grad is None:
                    continue
                e_w = p.grad * scale
                p.add_(e_w)
                perturbations.append(e_w)

        # Step 3: Compute loss at perturbed point
        self.optimizer.zero_grad()
        outputs_perturbed = self.model(x)
        probs_pert = torch.softmax(outputs_perturbed, dim=-1)
        entropies_pert = -(probs_pert * torch.log(probs_pert + 1e-8)).sum(dim=-1)
        loss_perturbed = entropies_pert[reliable_mask].mean()
        loss_perturbed.backward()

        # Step 4: Restore original weights and apply gradient update
        with torch.no_grad():
            for p, e_w in zip([p for p in self.params if p.grad is not None], perturbations):
                p.sub_(e_w)

        self.optimizer.step()
        self.optimizer.zero_grad()

        return outputs.detach(), {
            "loss": float(loss.detach().item()),
            "entropy": float(entropies.mean().detach().item()),
        }
