"""
Vision backbone architectures and model loader utilities.
Supports standard torchvision ResNet models customized for CIFAR-10 and ImageNet.
"""

from typing import Optional
import torch
import torch.nn as nn
import torchvision.models as models


def modify_resnet_for_cifar(model: nn.Module) -> nn.Module:
    """
    Modify a standard ImageNet ResNet for 32x32 CIFAR images:
    - Replace 7x7 stride-2 conv1 with 3x3 stride-1 conv1
    - Remove the initial max pooling layer (replace with identity)
    """
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model


def build_backbone(
    name: str = "resnet18",
    num_classes: int = 10,
    pretrained: bool = False,
    dataset: str = "cifar10",
) -> nn.Module:
    """
    Build and return a vision backbone model.

    Args:
        name: Backbone architecture name ('resnet18', 'resnet34', 'resnet50').
        num_classes: Number of output classification classes.
        pretrained: Whether to load torchvision pre-trained ImageNet weights.
        dataset: Target dataset ('cifar10', 'cifar100', 'imagenet').
    """
    name_clean = name.lower().strip()

    if name_clean == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
    elif name_clean == "resnet34":
        weights = models.ResNet34_Weights.DEFAULT if pretrained else None
        model = models.resnet34(weights=weights)
    elif name_clean == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
    else:
        raise ValueError(f"Unsupported backbone: '{name}'. Choose from ['resnet18', 'resnet34', 'resnet50'].")

    is_cifar = "cifar" in dataset.lower()
    if is_cifar:
        model = modify_resnet_for_cifar(model)

    # Replace classifier head if num_classes differs from ImageNet 1000
    if model.fc.out_features != num_classes:
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model
