"""Datasets and transforms.

Training augmentation (applied only to training images, inside each fold):
  Resize 224x224; RandomHorizontalFlip p=0.5; RandomRotation +-15 deg (p=1,
  angle uniform in [-15, 15]); ColorJitter brightness and contrast factors
  uniform in [0.9, 1.1]; ImageNet normalisation.

Evaluation: Resize 224x224 + ImageNet normalisation only.

Test-time augmentation (TTA), K = 3 DETERMINISTIC views per image (the original
submission used a random ColorJitter in the third view, which made results
non-reproducible):
  t1 identity, t2 horizontal flip, t3 brightness x1.2 and contrast x1.2.
Softmax probabilities of the K views are averaged (TTA-based probability
averaging).
"""
from pathlib import Path

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
SIZE = 224


def train_transform():
    return T.Compose([
        T.Resize((SIZE, SIZE)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(15),
        T.ColorJitter(brightness=0.1, contrast=0.1),
        T.ToTensor(), T.Normalize(MEAN, STD),
    ])


def eval_transform():
    return T.Compose([T.Resize((SIZE, SIZE)), T.ToTensor(), T.Normalize(MEAN, STD)])


class _Fixed:
    def __init__(self, fn):
        self.fn = fn

    def __call__(self, img):
        return self.fn(img)


def _bright_contrast(im):
    return TF.adjust_contrast(TF.adjust_brightness(im, 1.2), 1.2)


def tta_transforms():
    base = [T.Resize((SIZE, SIZE))]
    tail = [T.ToTensor(), T.Normalize(MEAN, STD)]
    return [
        T.Compose(base + tail),
        T.Compose(base + [_Fixed(TF.hflip)] + tail),
        T.Compose(base + [_Fixed(_bright_contrast)] + tail),
    ]


class CellDataset(Dataset):
    """Rows of an index CSV (columns: path, label, subject). ``views`` is either
    one transform or a list of transforms (TTA); with a list, the item is a
    tensor of shape (K, 3, H, W)."""

    def __init__(self, df, data_root, views):
        self.paths = [str(Path(data_root) / p) for p in df["path"]]
        self.labels = df["label"].astype(int).tolist()
        self.views = views

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        if isinstance(self.views, (list, tuple)):
            x = torch.stack([t(img) for t in self.views])
        else:
            x = self.views(img)
        return x, self.labels[i], i
