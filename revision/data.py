"""Datasets and transforms.

Training augmentation (applied only to training images, inside each fold):
  Resize 224x224; RandomHorizontalFlip p=0.5; RandomRotation +-15 deg (p=1,
  angle uniform in [-15, 15]); ColorJitter brightness and contrast factors
  uniform in [0.9, 1.1]; ImageNet normalisation.

Evaluation: Resize 224x224 + ImageNet normalisation only.

Improvement variants (revision/run_variants.py, pre-specified before looking at
any target result):
  aug="strong"   horizontal and vertical flips; RandomAffine rotation +-15 deg,
                 scale 0.8-1.2 (cell-size cue); ColorJitter brightness 0.3,
                 contrast 0.3, saturation 0.2, hue 0.04.
  standardize    cell-wise colour standardisation: the RGB mean and SD of the
                 cell pixels of every image (background = near-black pixels) are
                 mapped to reference values computed from SOURCE-phase images
                 only (color_reference.json). Applied identically to training,
                 evaluation and TTA images.

Test-time augmentation (TTA), K = 3 DETERMINISTIC views per image (the original
submission used a random ColorJitter in the third view, which made results
non-reproducible):
  t1 identity, t2 horizontal flip, t3 brightness x1.2 and contrast x1.2.
Softmax probabilities of the K views are averaged (TTA-based probability
averaging).
"""
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
SIZE = 224


BG_THRESHOLD = 10  # a pixel belongs to the cell if its brightest channel exceeds this (0-255)


def cell_mask(arr):
    return arr.max(axis=2) > BG_THRESHOLD


def cell_color_stats(img):
    """Per-channel mean and SD (0-255) of the cell pixels of a PIL image."""
    a = np.asarray(img, dtype=np.float32)
    m = cell_mask(a)
    if m.sum() < 10:
        return None
    px = a[m]
    return px.mean(0), px.std(0)


class CellStandardize:
    """Map the RGB mean/SD of the cell pixels to fixed reference values; the
    black background is left unchanged. Picklable (DataLoader workers)."""

    def __init__(self, ref_mean, ref_sd):
        self.ref_mean = np.asarray(ref_mean, dtype=np.float32)
        self.ref_sd = np.asarray(ref_sd, dtype=np.float32)

    def __call__(self, img):
        a = np.asarray(img, dtype=np.float32)
        m = cell_mask(a)
        if m.sum() < 10:
            return img
        px = a[m]
        mu, sd = px.mean(0), px.std(0) + 1e-6
        a[m] = np.clip((px - mu) / sd * self.ref_sd + self.ref_mean, 1, 255)
        return Image.fromarray(a.astype(np.uint8))


def _pre(std):
    return [CellStandardize(*std)] if std is not None else []


def train_transform(aug="basic", std=None):
    if aug == "strong":
        geo = [T.RandomHorizontalFlip(p=0.5), T.RandomVerticalFlip(p=0.5),
               T.RandomAffine(degrees=15, scale=(0.8, 1.2)),
               T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.04)]
    else:
        geo = [T.RandomHorizontalFlip(p=0.5), T.RandomRotation(15),
               T.ColorJitter(brightness=0.1, contrast=0.1)]
    return T.Compose(_pre(std) + [T.Resize((SIZE, SIZE))] + geo + [T.ToTensor(), T.Normalize(MEAN, STD)])


def eval_transform(std=None):
    return T.Compose(_pre(std) + [T.Resize((SIZE, SIZE)), T.ToTensor(), T.Normalize(MEAN, STD)])


class _Fixed:
    def __init__(self, fn):
        self.fn = fn

    def __call__(self, img):
        return self.fn(img)


def _bright_contrast(im):
    return TF.adjust_contrast(TF.adjust_brightness(im, 1.2), 1.2)


def tta_transforms(std=None):
    base = _pre(std) + [T.Resize((SIZE, SIZE))]
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


def compute_color_reference(df, data_root, max_images=3000, seed=0):
    """Reference cell-pixel RGB mean/SD from a (source-phase) index: the
    average of the per-image statistics over a fixed random sample."""
    d = df.sample(n=min(max_images, len(df)), random_state=seed)
    means, sds = [], []
    for p in d["path"]:
        st = cell_color_stats(Image.open(Path(data_root) / p).convert("RGB"))
        if st is not None:
            means.append(st[0])
            sds.append(st[1])
    return np.mean(means, 0).tolist(), np.mean(sds, 0).tolist()
