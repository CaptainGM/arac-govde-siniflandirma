"""Egitim/tahmin — preprocess modu model_info.json ile eslesmeli."""

from PIL import Image
import torchvision.transforms as transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
INPUT_SIZE = 224
LETTERBOX_FILL = (114, 114, 114)

# Kayitli model hangi modla egitildiyse app aynisini kullanmali
PREPROCESS_STRETCH = "stretch"      # Resize(224,224) — mevcut best_model.pt bununla
PREPROCESS_LETTERBOX = "letterbox"  # Oran korunur — yeni egitimler icin onerilen
DEFAULT_TRAIN_PREPROCESS = PREPROCESS_LETTERBOX


class LetterboxToSquare:
    def __init__(self, size=INPUT_SIZE, fill=LETTERBOX_FILL):
        self.size = size
        self.fill = fill

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        scale = self.size / max(w, h)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        img = img.resize((new_w, new_h), Image.Resampling.BILINEAR)
        canvas = Image.new("RGB", (self.size, self.size), self.fill)
        canvas.paste(img, ((self.size - new_w) // 2, (self.size - new_h) // 2))
        return canvas


def _normalize_block():
    return [
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]


def build_train_transform(preprocess=DEFAULT_TRAIN_PREPROCESS):
    geom = (
        [LetterboxToSquare(INPUT_SIZE)]
        if preprocess == PREPROCESS_LETTERBOX
        else [transforms.Resize((INPUT_SIZE, INPUT_SIZE))]
    )
    return transforms.Compose(
        geom
        + [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.2, hue=0.05),
        ]
        + _normalize_block()
        + [transforms.RandomErasing(p=0.15, scale=(0.02, 0.15))]
    )


def build_aggressive_train_transform(preprocess=DEFAULT_TRAIN_PREPROCESS):
    """Daha agresif augmentasyon — sedan/hatchback/station_wagon ayrimi icin."""
    geom = (
        [LetterboxToSquare(INPUT_SIZE)]
        if preprocess == PREPROCESS_LETTERBOX
        else [transforms.Resize((INPUT_SIZE, INPUT_SIZE))]
    )
    return transforms.Compose(
        geom
        + [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomResizedCrop(INPUT_SIZE, scale=(0.65, 1.0), ratio=(0.8, 1.2)),
            transforms.RandomPerspective(distortion_scale=0.15, p=0.3),
            transforms.RandomRotation(degrees=18),
            transforms.RandomAffine(degrees=0, translate=(0.15, 0.15), scale=(0.85, 1.15)),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.25, hue=0.08),
        ]
        + _normalize_block()
        + [transforms.RandomErasing(p=0.2, scale=(0.02, 0.2))]
    )


def build_eval_transform(preprocess=DEFAULT_TRAIN_PREPROCESS):
    geom = (
        [LetterboxToSquare(INPUT_SIZE)]
        if preprocess == PREPROCESS_LETTERBOX
        else [transforms.Resize((INPUT_SIZE, INPUT_SIZE))]
    )
    return transforms.Compose(geom + _normalize_block())


def resolve_preprocess(model_info=None, explicit=None):
    """Checkpoint ile uyumlu mod (eski modeller: stretch)."""
    if explicit in (PREPROCESS_STRETCH, PREPROCESS_LETTERBOX):
        return explicit
    if model_info and model_info.get("preprocess") in (
        PREPROCESS_STRETCH,
        PREPROCESS_LETTERBOX,
    ):
        return model_info["preprocess"]
    return PREPROCESS_STRETCH
