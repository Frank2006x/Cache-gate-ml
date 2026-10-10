"""
Base class and interface for Test-Time Adaptation (TTA) algorithms.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
import copy
import torch
import torch.nn as nn


class BaseTTA(nn.Module, ABC):
    """
    Abstract base wrapper for test-time adaptation models.
    Enforces clean separation between pure inference and adaptation.
    """

    def __init__(self, model: nn.Module, lr: float = 1e-3, **kwargs):
        super().__init__()
        self.model = model
        self.lr = lr
        # Keep a clean copy of the initial state dict to support stream reset
        self.source_state = copy.deepcopy(self.model.state_dict())
        self.optimizer: Optional[torch.optim.Optimizer] = None

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward pass without updating model parameters."""
        pass

    @abstractmethod
    def adapt_and_forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Execute adaptation step on incoming batch/frame x, then return logits and loss metrics.

        Returns:
            logits: Output classification logits tensor.
            metrics: Dictionary containing loss components (e.g. {'loss': 0.42, 'entropy': 0.38}).
        """
        pass

    def reset(self) -> None:
        """Reset model parameters back to source pretrained weights."""
        self.model.load_state_dict(copy.deepcopy(self.source_state))
        self.setup_optimizer()

    @abstractmethod
    def setup_optimizer(self) -> None:
        """Configure optimizer for trainable adaptation parameters."""
        pass

    @staticmethod
    def collect_params(model: nn.Module) -> Tuple[List[nn.Parameter], List[str]]:
        """
        Collect BatchNorm scale and shift affine parameters (gamma, beta) for adaptation.
        Leaves all convolutional and linear weights frozen.
        """
        params = []
        names = []
        for nm, m in model.named_modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.GroupNorm, nn.LayerNorm)):
                for np, p in m.named_parameters():
                    if np in ["weight", "bias"]:
                        params.append(p)
                        names.append(f"{nm}.{np}")
        return params, names

    @staticmethod
    def configure_model(model: nn.Module, use_running_stats: bool = True) -> nn.Module:
        """
        Freeze all model parameters except normalization layer affine parameters (gamma, beta).
        For online single-sample or small-batch streaming, set normalization layers to eval mode
        so they use running mean and variance while keeping affine parameters trainable.
        """
        model.eval()
        model.requires_grad_(False)
        for m in model.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.GroupNorm, nn.LayerNorm)):
                m.requires_grad_(True)
                if use_running_stats:
                    m.eval()
                else:
                    m.train()
        return model
