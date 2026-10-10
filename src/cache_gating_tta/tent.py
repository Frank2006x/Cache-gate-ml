from copy import deepcopy
import torch
import torch.nn as nn


def collect_params(model):
    """Collect the affine scale + shift parameters from batch norm layers."""
    params = []
    names = []
    for nm, m in model.named_modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            for np, p in m.named_parameters():
                if np in ["weight", "bias"]:
                    params.append(p)
                    names.append(f"{nm}.{np}")
    return params, names


def configure_tent(model, optimizer_type="adam", learning_rate=1e-3, weight_decay=0.0):
    """Configure model for TENT: set BN layers to use batch stats and enable gradients on affine params."""
    model.train()
    for param in model.parameters():
        param.requires_grad_(False)

    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            if module.affine:
                module.weight.requires_grad_(True)
                module.bias.requires_grad_(True)
            module.track_running_stats = False
            

    params, _ = collect_params(model)
    if not params:
        raise ValueError("No BatchNorm affine parameters found for TENT adaptation.")

    if optimizer_type.lower() == "adam":
        optimizer = torch.optim.Adam(params, lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_type.lower() == "sgd":
        optimizer = torch.optim.SGD(params, lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

    return model, optimizer


def softmax_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Compute per-sample Shannon entropy from logits."""
    return -(logits.softmax(dim=1) * logits.log_softmax(dim=1)).sum(dim=1)


def entropy_loss(logits: torch.Tensor) -> torch.Tensor:
    """Mean entropy loss over a batch."""
    return softmax_entropy(logits).mean()


def copy_model_and_optimizer(model, optimizer):
    """Copy model and optimizer states for resetting."""
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    return model_state, optimizer_state


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore model and optimizer states."""
    model.load_state_dict(model_state)
    optimizer.load_state_dict(optimizer_state)


class Tent(nn.Module):
    """Tent wrapper class for Test-Time Adaptation."""

    def __init__(self, model, optimizer, steps=1, episodic=False):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        assert steps > 0, "Tent requires at least 1 adaptation step per batch"
        self.episodic = episodic

        self.model_state, self.optimizer_state = copy_model_and_optimizer(
            self.model, self.optimizer
        )

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise RuntimeError("Cannot reset without saved state.")
        load_model_and_optimizer(
            self.model, self.optimizer, self.model_state, self.optimizer_state
        )

    def adapt_step(self, x, adapt=True):
        if self.episodic:
            self.reset()

        self.model.train()
        backward_count = 0

        if adapt:
            for _ in range(self.steps):
                self.optimizer.zero_grad(set_to_none=True)
                outputs = self.model(x)
                loss = entropy_loss(outputs)
                loss.backward()
                self.optimizer.step()
                backward_count += 1
        else:
            with torch.no_grad():
                outputs = self.model(x)
                loss = entropy_loss(outputs)

        return outputs.detach(), loss.item(), backward_count

    def forward(self, x):
        outputs, _, _ = self.adapt_step(x, adapt=True)
        return outputs


def tent_adapt_step(model, optimizer, images):
    """Backwards-compatible standalone function for single-step adaptation."""
    model.train()
    optimizer.zero_grad(set_to_none=True)

    logits = model(images)
    loss = entropy_loss(logits)

    loss.backward()
    optimizer.step()

    return logits.detach(), loss.item()
