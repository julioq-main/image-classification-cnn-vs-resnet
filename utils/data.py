"""
DataLoader construction for image classification with configurable augmentations.
"""
import logging
from functools import partial

import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

from utils.seed import worker_init_fn as _worker_init_fn

# Maps config names to torchvision transform classes
AUGMENTATION_REGISTRY = {
    "RandomResizedCrop": transforms.RandomResizedCrop,
    "RandomHorizontalFlip": transforms.RandomHorizontalFlip,
    "RandomVerticalFlip": transforms.RandomVerticalFlip,
    "ColorJitter": transforms.ColorJitter,
    "RandomRotation": transforms.RandomRotation,
    "RandomGrayscale": transforms.RandomGrayscale,
}

logger = logging.getLogger(__name__)


def build_transforms(
        mean: list[float],
        std: list[float],
        image_size: int,
        resize_size: int,
        augmentations: list[dict] | None = None,
    ) -> tuple[transforms.Compose, transforms.Compose]:
    """
    Build train and validation/test transform pipelines.

    Parameters
    ----------
    mean : list of float
        Per-channel mean values used for normalization.
    std : list float
        Per-channel standard deviation values used for normalization.
    image_size : int
        Final crop size passed to ``CenterCrop`` in the val/test pipeline
        and used as the target size in augmentations that require it.
    resize_size : int
        Size passed to ``Resize`` before center-cropping in the val/test pipeline.
    augmentations : list of dict, optional
        List of augmentation configs, each with a ``name`` key and an optional
        ``params`` dict. Names must be present in ``AUGMENTATION_REGISTRY``.

    Returns
    -------
    train_transform : transforms.Compose
        Pipeline with optional augmentations, ToTensor, and Normalize.
    test_val_transform : transforms.Compose
        Pipeline with Resize, CenterCrop, ToTensor, and Normalize.

    Raises
    ------
    ValueError
        If any augmentation name is not found in ``AUGMENTATION_REGISTRY``.

    Notes
    -----
    The val/test pipeline applies a fixed Resize → CenterCrop sequence to ensure
    consistent input geometry during evaluation. The train pipeline omits this
    to allow augmentations such as ``RandomResizedCrop`` to control the crop.

    Examples
    --------
    >>> train_tf, val_tf = build_transforms(
    ...     mean=[0.485, 0.456, 0.406],
    ...     std=[0.229, 0.224, 0.225],
    ...     image_size=224,
    ...     resize_size=256,
    ...     augmentations=[{"name": "RandomHorizontalFlip", "params": {"p": 0.5}}],
    ... )     
    """
    aug_list = []

    if augmentations:
        for aug in augmentations:
            name = aug["name"]
            params = aug.get("params", {})
            
            if name not in AUGMENTATION_REGISTRY:
                raise ValueError(
                    f"Unknown augmentation '{name}'."
                    f"Available: {list(AUGMENTATION_REGISTRY)}"
                )
            
            aug_list.append(AUGMENTATION_REGISTRY[name](**params))
    else:
        aug_list.append(transforms.Resize(resize_size))
        aug_list.append(transforms.CenterCrop(image_size))

    train_transform = transforms.Compose([
        *aug_list,
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    test_val_transform = transforms.Compose([
        transforms.Resize(resize_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    return train_transform, test_val_transform


def get_dataloader(cfg: dict, seed: int | None) -> dict[str, DataLoader]:
    """
    Build and return train, validation, and test DataLoaders from a config dict.

    Parameters
    ----------
    cfg : dict
        Configuration dictionary with the following keys:

        - ``mean`` : list of float 
            Per-channel normalization mean.
        - ``std`` : list of float 
            Per-channel normalization std.
        - ``batch_size`` : int
            Number of samples per batch.
        - ``train_dir`` : str
            Root directory for the training set.
        - ``val_dir`` : str
            Root directory for the validation set.
        - ``test_dir`` : str
            Root directory for the test set.
        - ``num_workers`` : int, default=0
            Number of DataLoader worker processes.
        - ``drop_last`` : bool, default=False
            Drop the last incomplete batch during training.
        - ``augmentations`` : list of dict, optional
            Augmentation configs passed to ``build_transforms``.
        - ``image_size`` : int, default=224
            Final crop size.
        - ``resize_size`` : int, default=256
            Resize size before center crop.

    seed : int or None
        Seed used for the train DataLoader's ``generator`` and ``worker_init_fn``.
        If None, no generator is set and workers are not seeded.

    Returns
    -------
    dict of DataLoader
        - ``train_loader``: Dataloader 
            DataLoader with shuffling and optional drop_last.
        - ``val_loader``: DataLoader
            DataLoader without shuffling.
        - ``test_loader``: DataLoader 
            DataLoader without shuffling.

    Notes
    -----
    ``pin_memory`` is enabled automatically when a CUDA device is available.
    ``persistent_workers`` is enabled when ``num_workers > 0`` to avoid
    worker respawn overhead between epochs.
    """
    mean = cfg["mean"]
    std = cfg["std"]
    batch_size = cfg["batch_size"]
    num_workers = cfg.get("num_workers", 0)
    drop_last = cfg.get("drop_last", False)
    augmentations = cfg.get("augmentations", None)
    image_size = cfg.get("image_size", 224)
    resize_size = cfg.get("resize_size", 256)

    train_transform, test_val_transform = build_transforms(
        mean, std, image_size, resize_size, augmentations
    )

    train_dataset = datasets.ImageFolder(root=cfg["train_dir"], transform=train_transform)
    test_dataset = datasets.ImageFolder(root=cfg["test_dir"], transform=test_val_transform)
    val_dataset = datasets.ImageFolder(root=cfg["val_dir"], transform=test_val_transform)

    pin_memory = torch.cuda.is_available()
    worker_init_fn = (partial(_worker_init_fn, seed=seed) if seed is not None else None)
    generator = torch.Generator().manual_seed(seed) if seed is not None else None
    persistent_workers = num_workers >  0

    common_args = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "worker_init_fn": worker_init_fn,
        "persistent_workers": persistent_workers,
    }

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        generator=generator,
        drop_last=drop_last,
        **common_args,
    )
    
    test_loader = DataLoader(test_dataset, shuffle=False, **common_args)
    val_loader = DataLoader(val_dataset, shuffle=False, **common_args)
    
    logger.info(
        f"Dataloaders ready — train: {len(train_dataset)} samples, "
        f"val: {len(val_dataset)} samples, test: {len(test_dataset)} samples"
    )

    return {
        "train_loader": train_loader,
        "test_loader": test_loader,
        "val_loader": val_loader,
    }