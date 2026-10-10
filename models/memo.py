"""
MEMO: Test-Time Robustness via Marginal Entropy Minimization with One Test Point (Zhang et al., NeurIPS 2022).
Adapts the model on a single test point by augmenting it into multiple views and minimizing
the marginal entropy of the ensemble prediction.
"""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.functional as TF
from .base import BaseTTA


class Memo(BaseTTA):
    """
    MEMO (Marginal Entropy Minimization with One test point).

    Args:
        model: Base neural network backbone.
        lr: Learning rate for adaptation optimizer (default: 1e-3).
        num_aug: Number of augmented views generated per test point (default: 4).
        opt_name: Optimizer name ('adam' or 'sgd').
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-3,
        num_aug: int = 4,
        opt_name: str = "adam",
        **kwargs,
    ):
        super().__init__(model, lr=lr, **kwargs)
        self.num_aug = max(2, num_aug)
        self.opt_name = opt_name.lower().strip()

        # Configure trainable BN parameters
        self.model = self.configure_model(self.model)
        self.params, self.param_names = self.collect_params(self.model)
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

    def _generate_augmented_views(self, x: torch.Tensor) -> torch.Tensor:
        """
        Generate num_aug augmented views of the test point x.
        x shape: (1, C, H, W)
        Returns tensor of shape (num_aug, C, H, W).
        """
        views = [x.clone()]  # First view is original unaugmented image

        for _ in range(1, self.num_aug):
            aug_x = x.clone()
            # Random horizontal flip
            if torch.rand(1).item() > 0.5:
                aug_x = TF.hflip(aug_x)
            # Random slight translation (-2 to +2 pixels)
            dx = int(torch.randint(-2, 3, (1,)).item())
            dy = int(torch.randint(-2, 3, (1,)).item())
            if dx != 0 or dy != 0:
                aug_x = TF.affine(aug_x, angle=0, translate=[dx, dy], scale=1.0, shear=0)
            # Mild brightness jitter
            factor = 1.0 + (torch.rand(1).item() - 0.5) * 0.2
            aug_x = torch.clamp(aug_x * factor, 0.0, 1.0)
            views.append(aug_x)

        return torch.cat(views, dim=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward pass on the original test point."""
        with torch.no_grad():
            return self.model(x)

    def adapt_and_forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Execute MEMO adaptation:
        1. Augment single test point into multiple views.
        2. Compute marginal predictive distribution p_bar = mean(softmax(logits_k)).
        3. Minimize Shannon entropy of the marginal distribution H(p_bar).
        4. Return prediction on original sample.
        """
        if self.optimizer is None:
            return self.forward(x), {"loss": 0.0}

        # Step 1: Generate augmented views
        x_aug = self._generate_augmented_views(x)

        # Step 2: Forward pass across all views
        self.optimizer.zero_grad()
        logits_aug = self.model(x_aug)
        probs_aug = torch.softmax(logits_aug, dim=-1)

        # Step 3: Marginal distribution across augmentations
        marginal_prob = probs_aug.mean(dim=0, keepdim=True)  # (1, num_classes)

        # Step 4: Marginal entropy loss H(p_bar) = -sum(p_bar * log(p_bar))
        marginal_entropy = -(marginal_prob * torch.log(marginal_prob + 1e-8)).sum(dim=-1).mean()
        marginal_entropy.backward()
        self.optimizer.step()

        # Step 5: Final forward pass on original sample
        with torch.no_grad():
            out_final = self.model(x)

        loss_val = float(marginal_entropy.detach().item())
        return out_final.detach(), {"loss": loss_val, "marginal_entropy": loss_val}
