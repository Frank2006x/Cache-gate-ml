"""
Source model baseline without adaptation (pure zero-update inference).
Conforms to BaseTTA interface.
"""

from typing import Dict, Tuple
import torch
import torch.nn as nn
from .base import BaseTTA


class Source(BaseTTA):
    """
    Source Baseline: Evaluates pretrained model without any parameter adaptation.
    Serves as the zero-adaptation anchor benchmark.
    """

    def __init__(self, model: nn.Module, **kwargs):
        super().__init__(model, lr=0.0, **kwargs)
        self.model.eval()
        self.model.requires_grad_(False)

    def setup_optimizer(self) -> None:
        self.optimizer = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.model(x)

    def adapt_and_forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Source model performs pure forward inference with zero updates."""
        outputs = self.forward(x)
        return outputs, {"loss": 0.0}
