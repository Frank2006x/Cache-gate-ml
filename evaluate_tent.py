import json
import random
import time
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from src.cache_gating_tta.dataset import ImageNetCorruptionDataset
from src.cache_gating_tta.model import load_model
from src.cache_gating_tta.tent import configure_tent, tent_adapt_step


SEED = 42
MAX_CLASSES = 100
IMAGES_PER_CLASS = 5
BATCH_SIZE = 32

random.seed(SEED)
torch.manual_seed(SEED)

with open("data/metadata/imagenet_class_index.json") as f:
    mapping = json.load(f)

wnid_to_idx = {
    value[0]: int(index)
    for index, value in mapping.items()
}

dataset = ImageNetCorruptionDataset(root="data/brightness/1")
by_class = defaultdict(list)

for index, (_, class_id) in enumerate(dataset.samples):
    if class_id in wnid_to_idx:
        by_class[class_id].append(index)

classes = sorted(by_class)[:MAX_CLASSES]
selected_indices = []

for class_id in classes:
    indices = by_class[class_id].copy()
    random.shuffle(indices)
    selected_indices.extend(indices[:IMAGES_PER_CLASS])

# Mix classes within batches for a more representative adaptation stream.

loader = DataLoader(
    Subset(dataset, selected_indices),
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available(),
)

model, _, device = load_model()
model, optimizer = configure_tent(model)

correct = 0
total = 0
adaptation_steps = 0
entropy_sum = 0.0

if device == "cuda":
    torch.cuda.synchronize()

start = time.perf_counter()

for batch in loader:
    images = batch["image"].to(device)
    targets = torch.tensor(
        [wnid_to_idx[c] for c in batch["class_id"]],
        device=device,
    )

    # Score the current batch, then adapt for subsequent batches.
    logits, loss = tent_adapt_step(model, optimizer, images)

    predictions = logits.argmax(dim=1)
    correct += (predictions == targets).sum().item()
    total += targets.numel()
    entropy_sum += loss
    adaptation_steps += 1

if device == "cuda":
    torch.cuda.synchronize()

elapsed = time.perf_counter() - start

print(f"Device: {device}")
print(f"Images evaluated: {total}")
print(f"Top-1 accuracy: {100 * correct / total:.2f}%")
print(f"Adaptation steps: {adaptation_steps}")
print(f"Mean batch entropy: {entropy_sum / adaptation_steps:.4f}")
print(f"Evaluation time: {elapsed:.2f} seconds")
print(f"Throughput: {total / elapsed:.2f} images/second")
