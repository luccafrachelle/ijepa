"""ImageNet-100 data: datasets, internal val split, I-JEPA transforms.

Label mapping follows SPHERE-JEPA (dataloaders/utils.py HFImages): the HF
ClassLabel integer is used as-is and images are converted to RGB.
For source='folder', torchvision ImageFolder semantics (sorted class dirs).
"""
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from torchvision import transforms as T

DATASET_ID = "clane9/imagenet-100"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class HFImages(Dataset):
    """Same semantics as SPHERE-JEPA HFImages: (RGB PIL image, int label)."""

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset[int(index)]
        return sample["image"].convert("RGB"), int(sample["label"])

    @property
    def labels(self):
        return np.asarray(self.dataset["label"])


class TransformDataset(Dataset):
    def __init__(self, ds, transform):
        self.ds = ds
        self.transform = transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        image, label = self.ds[index]
        return self.transform(image), int(label)


def load_imagenet100(root="data/imagenet100", source="hf"):
    """Return {'train': ds, 'val': ds} (official validation split as 'val')."""
    if source == "hf":
        from datasets import load_dataset
        ds = load_dataset(DATASET_ID, cache_dir=str(root))
        return {"train": HFImages(ds["train"]), "val": HFImages(ds["validation"])}
    if source == "folder":
        from torchvision.datasets import ImageFolder
        root = Path(root)
        train, val = ImageFolder(root / "train"), ImageFolder(root / "val")
        if len(train.classes) != 100:
            raise ValueError("ImageNet100 must contain exactly 100 training classes.")
        if val.class_to_idx != train.class_to_idx:
            raise ValueError("Class mappings differ between splits.")
        return {"train": _RGBFolder(train), "val": _RGBFolder(val)}
    raise ValueError("source must be 'hf' or 'folder'")


class _RGBFolder(Dataset):
    def __init__(self, ds):
        self.ds = ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        img, y = self.ds[i]
        return img.convert("RGB"), y


def split_train(train_ds, val_ids_path="data/val_interno.txt"):
    """-> (train_sub, val_interno) as Subsets using the saved internal-val ids."""
    ids = np.loadtxt(val_ids_path, dtype=np.int64).reshape(-1)
    mask = np.zeros(len(train_ds), dtype=bool)
    mask[ids] = True
    return Subset(train_ds, np.flatnonzero(~mask).tolist()), Subset(train_ds, ids.tolist())


def ijepa_train_transform(size=224, scale=(0.3, 1.0), hflip=False):
    """Official I-JEPA make_transforms with in1k_vith14_ep300.yaml values:
    RandomResizedCrop(224, scale=(0.3,1.0)) (default ratio 3/4..4/3, bilinear),
    horizontal flip OFF (use_horizontal_flip: false), no color distortion/blur,
    ToTensor + ImageNet normalize."""
    tl = [T.RandomResizedCrop(size, scale=scale)]
    if hflip:
        tl.append(T.RandomHorizontalFlip())
    tl += [T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return T.Compose(tl)


def eval_transform(size=224):
    """Same as SPHERE-JEPA ImageNet100DataModule evaluation transform."""
    return T.Compose([T.Resize(size), T.CenterCrop(size), T.ToTensor(),
                      T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])


def collapse_subset_ids(train_len, n=10000, seed=0):
    """Fixed sorted random subset of train indices for collapse metrics."""
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(train_len, size=min(n, train_len), replace=False))
