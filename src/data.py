"""Dataset loading, class-label mapping, and tensor (de)normalization helpers."""

import json
import os
import zipfile

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from . import config


def extract_dataset(zip_path: str = config.DATASET_ZIP,
                    dataset_dir: str = config.DATASET_DIR) -> str:
    """Unzip the test dataset if it has not been extracted yet."""
    if not os.path.exists(dataset_dir):
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall("./TestDataSet")
        print("Dataset extracted successfully.")
    else:
        print("Dataset already exists.")
    return dataset_dir


def get_transforms() -> transforms.Compose:
    """ToTensor + ImageNet normalization."""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
    ])


def load_idx_to_class(labels_file: str = config.LABELS_FILE) -> dict:
    """Parse ``labels_list.json`` ("401: accordion") into {index: class_name}."""
    with open(labels_file, "r") as f:
        class_list = json.load(f)
    idx_to_class = {}
    for item in class_list:
        index, name = item.split(": ")
        idx_to_class[int(index)] = name
    return idx_to_class


def load_dataset(dataset_dir: str = config.DATASET_DIR):
    """Return an ``ImageFolder`` dataset with normalization applied."""
    return torchvision.datasets.ImageFolder(root=dataset_dir, transform=get_transforms())


def get_dataloader(dataset, batch_size: int = config.BATCH_SIZE) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def denormalize(tensor: torch.Tensor):
    """Invert ImageNet normalization and return a PIL image (for visualization)."""
    tensor = tensor.clone().cpu()
    for t, m, s in zip(tensor, config.IMAGENET_MEAN, config.IMAGENET_STD):
        t.mul_(s).add_(m)
    return transforms.ToPILImage()(tensor.clamp(0, 1))
