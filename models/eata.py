"""
EATA: Efficient Anti-forgetting Test-time Adaptation (Niu et al., ICML 2022).
Implements active sample identification (reliable & non-redundant) and Fisher/weight anchoring.
"""

from typing import Dict, List, Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.optim as optim
from .base import BaseTTA


class Eata(BaseTTA):
    """
    EATA TTA implementation with reliable sample filtering and anti-forgetting regularization.

    Args:
        model: Base neural network backbone.
        lr: Learning rate for optimizer (default: 1e-3).
        fisher_alpha: Weight of Fisher / weight-anchor regularization (default: 2000.0 or 1.0).
        e_margin: Entropy margin threshold multiplier (default: 0.4 * ln(num_classes)).
        d_margin: Redundancy cosine similarity threshold (default: 0.05).
        num_classes: Number of classification categories.
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-3,
        fisher_alpha: float = 1.0,
        e_margin: Optional[float] = None,
        d_margin: float = 0.05,
        num_classes: int = 10,
        opt_name: str = "adam",
        **kwargs,
    ):
        super().__init__(model, lr=lr, **kwargs)
        self.fisher_alpha = fisher_alpha
        self.num_classes = num_classes
        self.opt_name = opt_name.lower().strip()

        # Reliable sample entropy threshold E_0
        if e_margin is None:
            self.e_margin = 0.4 * math.log(max(2, num_classes))
        else:
            self.e_margin = e_margin

        self.d_margin = d_margin

        # Configure trainable BN parameters and anchor weights
        self.model = self.configure_model(self.model)
        self.params, self.param_names = self.collect_params(self.model)

        # Store anchor parameters for anti-forgetting regularization
        self.anchor_params = {name: p.detach().clone() for name, p in zip(self.param_names, self.params)}

        # Moving average prediction for redundancy filtering
        self.current_model_probs: Optional[torch.Tensor] = None

        self.setup_optimizer()

    def setup_optimizer(self) -> None:
        """Setup optimizer for BN parameters."""
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
        """Forward pass without parameter updates."""
        with torch.no_grad():
            outputs = self.model(x)
        return outputs

    def adapt_and_forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Adapt BN parameters on reliable, non-redundant samples with anti-forgetting penalty.
        """
        if self.optimizer is None:
            return self.forward(x), {"loss": 0.0, "entropy": 0.0, "adapted_samples": 0}

        self.optimizer.zero_grad()
        outputs = self.model(x)
        probs = torch.softmax(outputs, dim=-1)

        # 1. Compute per-sample entropy
        entropies = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)

        # 2. Reliable sample filter: H(p) < E_0
        reliable_mask = entropies < self.e_margin

        # 3. Non-redundant sample filter: cosine similarity with moving average
        if self.current_model_probs is not None:
            cos_sim = torch.cosine_similarity(probs, self.current_model_probs.unsqueeze(0), dim=-1)
            non_redundant_mask = cos_sim < (1.0 - self.d_margin)
            filter_mask = reliable_mask & non_redundant_mask
        else:
            filter_mask = reliable_mask

        # Update running prediction moving average
        with torch.no_grad():
            mean_probs = probs.mean(dim=0)
            if self.current_model_probs is None:
                self.current_model_probs = mean_probs
            else:
                self.current_model_probs = 0.9 * self.current_model_probs + 0.1 * mean_probs

        num_adapted = int(filter_mask.sum().item())

        if num_adapted > 0:
            # Weighted entropy loss on selected samples
            weights = 1.0 / torch.exp(torch.clamp(entropies[filter_mask] - self.e_margin, min=-5.0, max=5.0))
            entropy_loss = (weights * entropies[filter_mask]).mean()

            # Anti-forgetting regularization: anchor to source weights
            fisher_loss = 0.0
            if self.fisher_alpha > 0:
                for name, p in zip(self.param_names, self.params):
                    anchor = self.anchor_params[name]
                    fisher_loss = fisher_loss + (p - anchor).pow(2).sum()

            total_loss = entropy_loss + self.fisher_alpha * fisher_loss
            total_loss.backward()
            self.optimizer.step()

            loss_val = float(total_loss.detach().item())
            ent_val = float(entropies[filter_mask].mean().detach().item())
        else:
            loss_val = 0.0
            ent_val = float(entropies.mean().detach().item())

        return outputs.detach(), {
            "loss": loss_val,
            "entropy": ent_val,
            "adapted_samples": float(num_adapted),
        }
