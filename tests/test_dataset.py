from torch.utils.data import DataLoader

from src.cache_gating_tta.dataset import ImageNetCorruptionDataset

dataset = ImageNetCorruptionDataset(
    root="data/brightness/1",
    max_images=16,
)

loader = DataLoader(dataset, batch_size=4, shuffle=False)
batch = next(iter(loader))

print("Dataset size:", len(dataset))
print("Normalized batch:", batch["image"].shape)
print("Raw batch:", batch["image_raw"].shape)
print("Pixel range:", batch["image_raw"].min().item(),
      batch["image_raw"].max().item())
print("Class IDs:", batch["class_id"])
print("First path:", batch["path"][0])
