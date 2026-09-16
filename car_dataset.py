

from pathlib import Path

from PIL import Image, ImageFile
from torch.utils.data import Dataset

from class_config import find_class_directory

ImageFile.LOAD_TRUNCATED_IMAGES = True

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def is_readable_image(path):
    
    try:
        CarDataset._open_rgb(path).load()
        return True
    except Exception:
        return False


class CarDataset(Dataset):
    

    def __init__(self, base_dir, classes, transform=None, split_name=""):
        self.image_paths = []
        self.targets = []
        self.classes = classes
        self.transform = transform
        skipped = 0

        for cls_idx, cls in enumerate(classes):
            cls_dir = find_class_directory(base_dir, cls)
            if cls_dir is None:
                continue
            for img_path in cls_dir.glob("*.*"):
                if img_path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
                if not is_readable_image(img_path):
                    skipped += 1
                    continue
                self.image_paths.append(str(img_path))
                self.targets.append(cls_idx)

        label = f" [{split_name}]" if split_name else ""
        if skipped:
            print(f"  !{label} {skipped} bozuk/truncated gorsel atlandi")

        if len(self.image_paths) == 0:
            raise RuntimeError(f"Gecerli gorsel bulunamadi: {base_dir}")

    def __len__(self):
        return len(self.image_paths)

    @staticmethod
    def _open_rgb(path):
        with Image.open(path) as img:
            if img.mode in ("P", "RGBA", "LA"):
                img = img.convert("RGBA")
            return img.convert("RGB")

    def __getitem__(self, idx):
        last_error = None
        for _ in range(8):
            try:
                img = self._open_rgb(self.image_paths[idx])
                if self.transform:
                    img = self.transform(img)
                return img, self.targets[idx]
            except OSError as exc:
                last_error = exc
                idx = (idx + 1) % len(self.image_paths)
        raise OSError(f"Gorsel okunamadi (8 deneme): {last_error}")
