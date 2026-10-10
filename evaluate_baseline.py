import json
import random
import time
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from src.cache_gating_tta.dataset import ImageNetCorruptionDataset
from src.cache_gating_tta.model import load_model


SEED = 42
MAX_CLASSES = 100
IMAGES_PER_CLASS = 5
BATCH_SIZE = 32

random.seed(SEED)

mapping_path = Path("data/metadata/imagenet_class_index.json")

with mapping_path.open() as f:
    raw_mapping = json.load(f)

wnid_to_idx = {
    value[0]: int(index)
    for index, value in raw_mapping.items()
}

dataset = ImageNetCorruptionDataset(root="data/brightness/1")

by_class = defaultdict(list)

for index, (_, class_id) in enumerate(dataset.samples):
    if class_id in wnid_to_idx:
        by_class[class_id].append(index)

classes = sorted(by_class)[:MAX_CLASSES]
selected_indices = []

for class_id in classes:
    indices = by_class[class_id]
    random.shuffle(indices)
    selected_indices.extend(indices[:IMAGES_PER_CLASS])

subset = Subset(dataset, selected_indices)
loader = DataLoader(
    subset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available(),
)

model, _, device = load_model()
model.eval()

correct = 0
total = 0
start = time.perf_counter()

with torch.inference_mode():
    for batch in loader:
        images = batch["image"].to(device)
        targets = torch.tensor(
            [wnid_to_idx[c] for c in batch["class_id"]],
            device=device,
        )

        predictions = model(images).argmax(dim=1)
        correct += (predictions == targets).sum().item()
        total += targets.numel()

if device == "cuda":
    torch.cuda.synchronize()

elapsed = time.perf_counter() - start

print(f"Device: {device}")
print(f"Classes evaluated: {len(classes)}")
print(f"Images evaluated: {total}")
print(f"Top-1 accuracy: {100 * correct / total:.2f}%")
print(f"Evaluation time: {elapsed:.2f} seconds")
print(f"Throughput: {total / elapsed:.2f} images/second")
