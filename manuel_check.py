import pandas as pd
from pathlib import Path
import shutil

df = pd.read_csv('reports/suspicious_labels.csv')
sedan_to_hatch = df[(df['true_class'] == 'SEDAN') & (df['predicted_class'] == 'HATCHBACK')]
sedan_to_hatch = sedan_to_hatch[sedan_to_hatch['confidence'].astype(float) >= 0.85]

check_dir = Path('manual_check_sedan')
check_dir.mkdir(exist_ok=True)

for i, row in sedan_to_hatch.iterrows():
    src = Path(row['path'])
    if src.exists():
        shutil.copy(src, check_dir / f"{row['confidence']}_{src.name}")

print(f"{len(sedan_to_hatch)} resim kopyalandı. manual_check_sedan klasörüne bak.")