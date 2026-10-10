from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class ImageNetCorruptionDataset(Dataset):
    def __init__(
        self,
        root="data/brightness/1",
        max_images=None,
        image_size=224,
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

        self.transform = transforms.Compose([
            transforms.Resize(256),
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
            image = self.transform(image)

        return {
            "image": self.normalize(image),
            "image_raw": image,
            "class_id": class_id,
            "path": str(path),
        }
