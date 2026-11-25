"""Create a tiny fake dataset with the Kaggle C-NMC folder layout (smoke tests only)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

root = Path(sys.argv[1])
rng = np.random.default_rng(0)


def cell(label):
    img = np.zeros((96, 96, 3), np.uint8)
    yy, xx = np.mgrid[:96, :96]
    r = 30 if label else 22
    m = (yy - 48) ** 2 + (xx - 48) ** 2 < r ** 2
    img[m] = (rng.integers(100, 160), rng.integers(40, 90), rng.integers(120, 200))
    return Image.fromarray(img)


tr = root / "C-NMC_Leukemia" / "training_data"
for k, subj in enumerate([f"{i}" for i in range(1, 7)] + [f"H{i}" for i in range(1, 6)]):
    label = 0 if subj.startswith("H") else 1
    d = tr / f"fold_{k % 3}" / ("hem" if label == 0 else "all")
    d.mkdir(parents=True, exist_ok=True)
    for n in range(4):
        cell(label).save(d / f"UID_{subj}_{n + 1}_1_{'hem' if label == 0 else 'all'}.bmp")

va = root / "C-NMC_Leukemia" / "validation_data"
img_dir = va / "C-NMC_test_prelim_phase_data"
img_dir.mkdir(parents=True, exist_ok=True)
rows, idx = [], 1
for subj in [f"{i}" for i in range(50, 55)] + [f"H{i}" for i in range(50, 55)]:
    label = 0 if subj.startswith("H") else 1
    for n in range(3):
        cell(label).save(img_dir / f"{idx}.bmp")
        rows.append({"Patient_ID": f"UID_{subj}_{n + 1}_1_{'hem' if label == 0 else 'all'}.bmp",
                     "new_names": f"{idx}.bmp", "labels": label})
        idx += 1
pd.DataFrame(rows).to_csv(va / "C-NMC_test_prelim_phase_data_labels.csv", index=False)
print("fake data at", root / "C-NMC_Leukemia")
