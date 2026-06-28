from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.transforms import AutoAugmentPolicy
import torch

from .config import DataConfig

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def create_imagenet_loaders(cfg: DataConfig, persistent_workers: bool = False):
    """
    Build train/val ImageFolder DataLoaders from a single ImageNet-style
    directory using a seeded deterministic split.

    Returns (train_ds, val_ds, train_loader, val_loader).
    train_ds / val_ds are Subset objects — use len() or pass to a second
    DataLoader (e.g. CPU-side for INT8 evaluation).
    """
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(
            cfg.img_size,
            scale=(0.7, 1.0),
            interpolation=transforms.InterpolationMode.BICUBIC,
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(
            degrees=15,
            interpolation=transforms.InterpolationMode.BICUBIC,
        ),
        transforms.AutoAugment(policy=AutoAugmentPolicy.IMAGENET),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    transform_val = transforms.Compose([
        transforms.Resize(cfg.img_size),
        transforms.CenterCrop(cfg.img_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    train_full = datasets.ImageFolder(cfg.dataset_path, transform=transform_train)
    val_full = datasets.ImageFolder(cfg.dataset_path, transform=transform_val)
    assert train_full.classes == val_full.classes

    n_total = len(train_full)
    perm = torch.randperm(n_total, generator=torch.Generator().manual_seed(cfg.seed)).tolist()
    n_train = int(cfg.train_val_split * n_total)

    train_ds = Subset(train_full, perm[:n_train])
    val_ds = Subset(val_full, perm[n_train:])

    use_pw = persistent_workers and cfg.num_workers > 0
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        persistent_workers=use_pw,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        persistent_workers=use_pw,
    )

    return train_ds, val_ds, train_loader, val_loader
