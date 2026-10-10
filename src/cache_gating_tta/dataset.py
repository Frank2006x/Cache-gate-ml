import json
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def load_wnid_to_idx(mapping_path="data/metadata/imagenet_class_index.json"):
    path = Path(mapping_path)
    if not path.is_file():
        raise FileNotFoundError(f"Mapping file not found at {path.resolve()}")
    with path.open() as f:
        mapping = json.load(f)
    return {v[0]: int(k) for k, v in mapping.items()}


class ImageNetCorruptionDataset(Dataset):
    def __init__(
        self,
        root="data/brightness/1",
        max_images=None,
        image_size=224,
        transform=None,
    ):
        self.root = Path(root)

        if not self.root.is_dir():
            raise FileNotFoundError(
                f"Dataset directory not found: {self.root.resolve()}"
            )

        self.samples = []

        for class_dir in sorted(self.root.iterdir()):
            if not class_dir.is_dir():
                continue

            for path in sorted(class_dir.glob("*")):
                if path.suffix.lower() in {".jpeg", ".jpg", ".png"}:
                    self.samples.append((path, class_dir.name))

        if max_images is not None:
            self.samples = self.samples[:max_images]

        if not self.samples:
            raise RuntimeError(f"No images found in {self.root.resolve()}")

        if transform is not None:
            self.transform = transform
            self.raw_transform = None
            self.normalize = None
        else:
            self.transform = None
            self.raw_transform = transforms.Compose([
                transforms.Resize(232, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
            ])
            self.normalize = transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, class_id = self.samples[index]

        with Image.open(path) as image:
            image = image.convert("RGB")
            if self.transform is not None:
                norm_tensor = self.transform(image)
                raw_tensor = norm_tensor
            else:
                raw_tensor = self.raw_transform(image)
                norm_tensor = self.normalize(raw_tensor)

        return {
            "image": norm_tensor,
            "image_raw": raw_tensor,
            "class_id": class_id,
            "path": str(path),
        }


