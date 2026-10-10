import torch
from torchvision.transforms.functional import gaussian_blur

from src.cache_gating_tta.dataset import ImageNetCorruptionDataset
from src.cache_gating_tta.ssim import ssim


dataset = ImageNetCorruptionDataset(root="data/brightness/1")

first_index = 0
first_class = dataset.samples[first_index][1]

second_index = next(
    i for i, (_, class_id) in enumerate(dataset.samples)
    if class_id != first_class
)

same_class_index = next(
    i for i, (_, class_id) in enumerate(dataset.samples)
    if i != first_index and class_id == first_class
)

image = dataset[first_index]["image_raw"].unsqueeze(0)
different = dataset[second_index]["image_raw"].unsqueeze(0)
same_class = dataset[same_class_index]["image_raw"].unsqueeze(0)


def score(a, b):
    return ssim(a, b).item()


print("SSIM calibration")
print("-" * 45)
print(f"Identical image:       {score(image, image):.4f}")

for delta in (0.01, 0.02, 0.05, 0.10):
    changed = (image + delta).clamp(0, 1)
    print(f"Brightness +{delta:.2f}:     {score(image, changed):.4f}")

blurred = gaussian_blur(image, kernel_size=9, sigma=2.0)
print(f"Gaussian blur:         {score(image, blurred):.4f}")
print(f"Different class:       {score(image, different):.4f}")
print(f"Same class, other img: {score(image, same_class):.4f}")
