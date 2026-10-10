import copy
import json
import random
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from src.cache_gating_tta.dataset import (
    ImageNetCorruptionDataset,
    load_wnid_to_idx,
)
from src.cache_gating_tta.model import load_model
from src.cache_gating_tta.tent import Tent, configure_tent, entropy_loss
from src.cache_gating_tta.cache_gate import CacheGate


SEED = 42
NUM_BASE_IMAGES = 8
REPEATS = 4
THRESHOLD = 0.95
LEARNING_RATE = 1e-3

random.seed(SEED)
torch.manual_seed(SEED)

device = "cuda" if torch.cuda.is_available() else "cpu"


def synchronize():
    if device == "cuda":
        torch.cuda.synchronize()


# ---------------------------------------------------------
# Load and select deterministic samples
# ---------------------------------------------------------
dataset = ImageNetCorruptionDataset(root="data/brightness/1")
wnid_to_idx = load_wnid_to_idx()

by_class = {}
for index, (_, class_id) in enumerate(dataset.samples):
    if class_id in wnid_to_idx:
        by_class.setdefault(class_id, []).append(index)

class_ids = sorted(by_class)
rng = random.Random(SEED)
rng.shuffle(class_ids)
class_ids = class_ids[:NUM_BASE_IMAGES]

selected = [by_class[class_id][0] for class_id in class_ids]
loader = DataLoader(
    Subset(dataset, selected),
    batch_size=1,
    shuffle=False,
    num_workers=0,
)

base_frames = []
for batch in loader:
    base_frames.append({
        "raw": batch["image_raw"],
        "image": batch["image"],
        "target": wnid_to_idx[batch["class_id"][0]],
        "path": batch["path"][0],
    })


# ---------------------------------------------------------
# Construct the controlled stream
# Each source image has 4 brightness levels.
# These remain synthetic variants, not a natural video stream.
# ---------------------------------------------------------
stream = []

mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

for frame in base_frames:
    for delta in (0.0, 0.01, 0.02, 0.05):
        raw = (frame["raw"] + delta).clamp(0, 1)
        normalized = (raw - mean) / std

        stream.append({
            "raw": raw,
            "image": normalized,
            "target": frame["target"],
            "source": frame["path"],
            "brightness_delta": delta,
        })


# ---------------------------------------------------------
# Preserve identical pretrained initialization
# ---------------------------------------------------------
source_model, _, device = load_model()
initial_state = copy.deepcopy(source_model.state_dict())
del source_model


def make_source_model():
    model, _, _ = load_model(device=device)
    model.load_state_dict(initial_state)
    model.eval()
    return model


def make_tent():
    model = make_source_model()
    model, optimizer = configure_tent(
        model,
        optimizer_type="adam",
        learning_rate=LEARNING_RATE,
    )
    return Tent(model, optimizer, steps=1, episodic=False)


def run_source_only():
    model = make_source_model()
    correct = 0
    entropy_total = 0.0

    synchronize()
    start = time.perf_counter()

    with torch.inference_mode():
        for frame in stream:
            image = frame["image"].to(device)
            target = frame["target"]

            logits = model(image)
            correct += int(logits.argmax(dim=1).item() == target)
            entropy_total += entropy_loss(logits).item()

    synchronize()
    elapsed = time.perf_counter() - start

    return {
        "method": "source_only",
        "images": len(stream),
        "accuracy_percent": 100 * correct / len(stream),
        "mean_entropy": entropy_total / len(stream),
        "backward_passes": 0,
        "cache_hits": 0,
        "cache_hit_ratio_percent": 0.0,
        "elapsed_seconds": elapsed,
        "throughput_images_per_second": len(stream) / elapsed,
    }


def run_tent():
    tent = make_tent()
    correct = 0
    entropy_total = 0.0
    backward_passes = 0

    synchronize()
    start = time.perf_counter()

    for frame in stream:
        image = frame["image"].to(device)
        target = frame["target"]

        logits, entropy, backwards = tent.adapt_step(
            image,
            adapt=True,
        )

        correct += int(logits.argmax(dim=1).item() == target)
        entropy_total += entropy
        backward_passes += backwards

    synchronize()
    elapsed = time.perf_counter() - start

    return {
        "method": "tent_every_frame",
        "images": len(stream),
        "accuracy_percent": 100 * correct / len(stream),
        "mean_entropy": entropy_total / len(stream),
        "backward_passes": backward_passes,
        "cache_hits": 0,
        "cache_hit_ratio_percent": 0.0,
        "elapsed_seconds": elapsed,
        "throughput_images_per_second": len(stream) / elapsed,
    }


def run_gated_tent():
    tent = make_tent()
    gate = CacheGate(tent, threshold=THRESHOLD)

    correct = 0
    entropy_total = 0.0
    backward_passes = 0
    hits = 0

    synchronize()
    start = time.perf_counter()

    for frame in stream:
        image = frame["image"].to(device)
        raw = frame["raw"].to(device)
        target = frame["target"]

        result = gate.process(image, raw)
        logits = result["logits"]

        correct += int(logits.argmax(dim=1).item() == target)
        entropy_total += result["entropy"]
        backward_passes += result["backward_passes"]
        hits += int(result["cache_hit"])

    synchronize()
    elapsed = time.perf_counter() - start

    return {
        "method": "cache_gated_tent",
        "images": len(stream),
        "accuracy_percent": 100 * correct / len(stream),
        "mean_entropy": entropy_total / len(stream),
        "backward_passes": backward_passes,
        "cache_hits": hits,
        "cache_hit_ratio_percent": 100 * hits / len(stream),
        "elapsed_seconds": elapsed,
        "throughput_images_per_second": len(stream) / elapsed,
    }


results = [
    run_source_only(),
    run_tent(),
    run_gated_tent(),
]

output_path = Path("results")
output_path.mkdir(exist_ok=True)

with (output_path / "cache_gate_pilot.json").open("w") as f:
    json.dump(results, f, indent=2)

print(f"\nDevice: {device}")
print(f"Stream length: {len(stream)}")
print(f"Base images: {len(base_frames)}")
print(f"SSIM threshold: {THRESHOLD}")

for result in results:
    print("\n", result)

tent_result = results[1]
gated_result = results[2]

if tent_result["backward_passes"]:
    saved = 100 * (
        1 - gated_result["backward_passes"]
        / tent_result["backward_passes"]
    )
    print(f"\nBackward-pass reduction: {saved:.2f}%")

print(f"Saved results to {output_path / 'cache_gate_pilot.json'}")
