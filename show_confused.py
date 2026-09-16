import torch
import torch.nn as nn
import torchvision.models as models
from car_dataset import CarDataset
from transforms_config import build_eval_transform, resolve_preprocess
from class_config import get_canonical_classes
from pathlib import Path
import json
import shutil

device = torch.device('cpu')
classes = get_canonical_classes()

model = models.resnet50(weights=None)
model.fc = nn.Sequential(
    nn.Linear(2048, 1024), nn.BatchNorm1d(1024), nn.ReLU(), nn.Dropout(0.5),
    nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.4),
    nn.Linear(512, len(classes)),
)
checkpoint = torch.load('models/best_model.pt', map_location=device, weights_only=False)
model.load_state_dict(checkpoint)
model.eval()

info = {}
if Path('models/model_info.json').exists():
    with open('models/model_info.json') as f:
        info = json.load(f)
transform = build_eval_transform(resolve_preprocess(info))

test_ds = CarDataset('test', classes, transform=transform, split_name='test')

# Tüm yanlış tahminleri topla
misclassified = []
for i in range(len(test_ds)):
    img, label = test_ds[i]
    with torch.no_grad():
        output = model(img.unsqueeze(0))
        pred = output.argmax().item()
    if pred != label:
        misclassified.append({
            'path': test_ds.image_paths[i],
            'true': classes[label],
            'pred': classes[pred]
        })

print(f'Toplam yanlış: {len(misclassified)} / {len(test_ds)}')

# Sadece SEDAN/HATCHBACK/STATION_WAGON karışıkları
problem_pairs = [
    ('HATCHBACK', 'SEDAN'), ('HATCHBACK', 'SUV'),
    ('SEDAN', 'HATCHBACK'), ('STATION_WAGON', 'SEDAN')
]

check_dir = Path('confused_images')
check_dir.mkdir(exist_ok=True)

for m in misclassified:
    if (m['true'], m['pred']) in problem_pairs:
        src = Path(m['path'])
        if src.exists():
            dst = check_dir / f"{m['true']}_to_{m['pred']}_{src.name}"
            shutil.copy(src, dst)
            print(f"Kopyalandı: {m['true']} → {m['pred']}: {src.name}")

print(f'\n📁 {check_dir} klasörüne bak. Karışan resimler orada.')