"""
CoTTA: Continual Test-Time Adaptation (Wang et al., CVPR 2022).
Addresses long-term error accumulation and catastrophic forgetting using:
1. Mean Teacher with Exponential Moving Average (EMA).
2. Teacher-Student cross-entropy consistency.
3. Stochastic parameter restoration to initial source weights.
"""

from typing import Dict, List, Optional, Tuple
import copy
import random
import torch
import torch.nn as nn
import torch.optim as optim
from .base import BaseTTA


class Cotta(BaseTTA):
    """
    CoTTA (Continual Test-Time Adaptation).

    Args:
        model: Base neural network backbone.
        lr: Learning rate for student optimizer (default: 1e-3).
        mt_alpha: Momentum coefficient for EMA teacher update (default: 0.999).
        rst_m: Stochastic restoration probability (default: 0.01).
        temperature: Softmax temperature for teacher predictions (default: 1.0).
        opt_name: Optimizer name ('adam' or 'sgd').
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-3,
        mt_alpha: float = 0.999,
        rst_m: float = 0.01,
        temperature: float = 1.0,
        opt_name: str = "adam",
        **kwargs,
    ):
        super().__init__(model, lr=lr, **kwargs)
        self.mt_alpha = mt_alpha
        self.rst_m = rst_m
        self.temperature = temperature
        self.opt_name = opt_name.lower().strip()

        # Configure student model
        self.model = self.configure_model(self.model)
        self.params, self.param_names = self.collect_params(self.model)

        # Create Mean Teacher as an EMA copy of student
        self.model_teacher = copy.deepcopy(self.model)
        self.model_teacher.eval()
        self.model_teacher.requires_grad_(False)

        # Retain initial source parameter tensors for stochastic restoration
        self.source_params = {
            name: p.detach().clone() for name, p in zip(self.param_names, self.params)
        }

        self.setup_optimizer()

    def setup_optimizer(self) -> None:
        """Setup optimizer for trainable BN parameters."""
        if not self.params:
            self.optimizer = None
            return

        if self.opt_name == "adam":
            self.optimizer = optim.Adam(self.params, lr=self.lr, betas=(0.9, 0.999), weight_decay=0.0)
        elif self.opt_name == "sgd":
            self.optimizer = optim.SGD(self.params, lr=self.lr, momentum=0.9, weight_decay=0.0)
        else:
            raise ValueError(f"Unsupported optimizer: {self.opt_name}")

    def reset(self) -> None:
        """Reset student and teacher to source checkpoint."""
        super().reset()
        self.model_teacher = copy.deepcopy(self.model)
        self.model_teacher.eval()
        self.model_teacher.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using the robust EMA teacher network."""
        with torch.no_grad():
            return self.model_teacher(x)

    def _update_ema_teacher(self) -> None:
        """Update teacher network weights via exponential moving average."""
        with torch.no_grad():
            for p_student, p_teacher in zip(self.model.parameters(), self.model_teacher.parameters()):
                p_teacher.data.mul_(self.mt_alpha).add_(p_student.data, alpha=1.0 - self.mt_alpha)

    def _stochastic_restoration(self) -> None:
        """With probability rst_m, reset weights back to initial source values."""
        if self.rst_m <= 0:
            return

        with torch.no_grad():
            for name, p in zip(self.param_names, self.params):
                mask = (torch.rand_like(p) < self.rst_m).float()
                src_val = self.source_params[name]
                p.data.copy_(mask * src_val + (1.0 - mask) * p.data)

    def adapt_and_forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Execute CoTTA consistency adaptation:
        1. Teacher generates soft pseudo-labels.
        2. Student minimizes cross-entropy with teacher.
        3. EMA teacher update and stochastic parameter restoration.
        """
        if self.optimizer is None:
            return self.forward(x), {"loss": 0.0}

        # Step 1: Teacher forward prediction
        with torch.no_grad():
            teacher_logits = self.model_teacher(x)
            teacher_probs = torch.softmax(teacher_logits / self.temperature, dim=-1)

        # Step 2: Student forward pass
        self.optimizer.zero_grad()
        student_logits = self.model(x)
        student_log_probs = torch.log_softmax(student_logits, dim=-1)

        # Step 3: Cross-entropy consistency loss
        loss = -(teacher_probs * student_log_probs).sum(dim=-1).mean()
        loss.backward()
        self.optimizer.step()

        # Step 4: EMA Teacher update
        self._update_ema_teacher()

        # Step 5: Stochastic parameter restoration
        self._stochastic_restoration()

        loss_val = float(loss.detach().item())
        return teacher_logits.detach(), {"loss": loss_val, "consistency_loss": loss_val}
