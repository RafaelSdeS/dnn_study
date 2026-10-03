import random
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from torchvision.datasets.folder import default_loader
import torch

from .config import DataConfig

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
TINY_IMAGENET_TEST_SIZE = 10_000  # the official val split: 50 labelled images per class


def _seed_worker(worker_id: int) -> None:
    """PyTorch's reproducibility-notes pattern: each worker's torch seed (base seed drawn from the seeded main RNG,
    + worker id, new every epoch) also seeds `random` and numpy, so any transform drawing from them is reproducible
    too. The torchvision transforms used here draw from torch alone."""
    seed = torch.initial_seed() % 2 ** 32
    random.seed(seed)
    np.random.seed(seed)


def _val_transform(img_size: int):
    return transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class _TinyImageNetVal(Dataset):
    """Tiny ImageNet's official val split: val/images/*.JPEG labelled by val/val_annotations.txt (file, wnid, box)."""

    def __init__(self, root: Path, class_to_idx: dict, transform):
        rows = (line.split("\t") for line in (root / "val_annotations.txt").read_text().splitlines())
        self.samples = [(root / "images" / f, class_to_idx[wnid]) for f, wnid, *_ in rows]
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        return self.transform(default_loader(path)), label


def create_test_loader(cfg: DataConfig, train_ds: Subset) -> DataLoader:
    """Held-out test set: Tiny ImageNet's official val split (its test labels are withheld), next to the train/ dir
    cfg.dataset_path points at. The 90/10 split of train/ stays the validation set that picks the best epoch, so
    the reported numbers come from images no selection ever saw (Cawley & Talbot, JMLR 2010) -- and from the same
    10k images for every seed, while the 90/10 split moves with cfg.seed. train_ds: create_imagenet_loaders' train
    Subset, whose ImageFolder's class_to_idx labels the test images."""
    ds = _TinyImageNetVal(Path(cfg.dataset_path).parent / "val", train_ds.dataset.class_to_idx, _val_transform(cfg.img_size))
    assert len(ds) == TINY_IMAGENET_TEST_SIZE, len(ds)
    return DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=cfg.pin_memory)


def create_imagenet_loaders(cfg: DataConfig, persistent_workers: bool = False):
    """
    Build train/val ImageFolder DataLoaders from a single ImageNet-style
    directory using a seeded deterministic split.

    Returns (train_ds, val_ds, train_loader, val_loader).
    train_ds / val_ds are Subset objects — use len() or pass to a second
    DataLoader (e.g. CPU-side for INT8 evaluation).
    """
    if cfg.train_aug == "crop_flip_autoaug":
        # the standard small-image baseline -- pad 4 px, random crop, horizontal flip (He et al. 2016) -- then
        # AutoAugment's ImageNet policy (Cubuk et al. 2019), which applies "in addition to the baseline pre-processing"
        geometric = [transforms.RandomCrop(cfg.img_size, padding=4), transforms.RandomHorizontalFlip(p=0.5)]
    else:
        assert cfg.train_aug == "legacy", cfg.train_aug
        geometric = [
            transforms.RandomResizedCrop(cfg.img_size, scale=(0.7, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15, interpolation=transforms.InterpolationMode.BICUBIC),
        ]
    transform_train = transforms.Compose(geometric + [
        transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.IMAGENET),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    transform_val = _val_transform(cfg.img_size)

    train_full = datasets.ImageFolder(cfg.dataset_path, transform=transform_train)
    val_full = datasets.ImageFolder(cfg.dataset_path, transform=transform_val)
    assert train_full.classes == val_full.classes

    n_total = len(train_full)
    perm = torch.randperm(n_total, generator=torch.Generator().manual_seed(cfg.seed)).tolist()
    n_train = int(cfg.train_val_split * n_total)

    train_ds = Subset(train_full, perm[:n_train])
    val_ds = Subset(val_full, perm[n_train:])

    use_pw = persistent_workers and cfg.num_workers > 0
    worker_init = _seed_worker if cfg.num_workers > 0 else None
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        persistent_workers=use_pw,
        drop_last=True,
        worker_init_fn=worker_init,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        persistent_workers=use_pw,
        worker_init_fn=worker_init,
    )

    return train_ds, val_ds, train_loader, val_loader
