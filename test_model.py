import torch
from torch.utils.data import DataLoader

from src.cache_gating_tta.dataset import ImageNetCorruptionDataset
from src.cache_gating_tta.model import load_model

dataset = ImageNetCorruptionDataset(
    root="data/brightness/1",
    max_images=16,
)

model, _, device = load_model()

loader = DataLoader(dataset, batch_size=4, shuffle=False)

batch = next(iter(loader))
images = batch["image"].to(device)

with torch.no_grad():
    logits = model(images)

predictions = logits.argmax(dim=1)

print("Device:", device)
print("Output shape:", logits.shape)
print("Predicted class indices:", predictions.tolist())
print("Ground-truth synset IDs:", batch["class_id"])
