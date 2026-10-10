import torch
from torchvision.models import resnet50, ResNet50_Weights


def load_model(device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    weights = ResNet50_Weights.IMAGENET1K_V2
    model = resnet50(weights=weights)
    model = model.to(device)
    model.eval()

    return model, weights.transforms(), device
