"""Generate and persist adversarial test sets from a clean dataloader."""

import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from . import config


def generate_adversarial_set(model, loader, attack_fn, save_path: str, desc: str, **attack_kwargs):
    """Run ``attack_fn`` over ``loader`` and save a {adv_images, true_labels} dict.

    Returns a ``DataLoader`` over the generated adversarial examples.
    """
    adv_images, true_labels = [], []
    for images, labels in tqdm(loader, desc=desc):
        images, labels = images.to(config.DEVICE), labels.to(config.DEVICE)
        perturbed = attack_fn(model, images, labels, **attack_kwargs)
        adv_images.append(perturbed.cpu())
        true_labels.append(labels.cpu())

    adv_images = torch.cat(adv_images)
    true_labels = torch.cat(true_labels)

    torch.save({"adv_images": adv_images, "true_labels": true_labels}, save_path)

    adv_dataset = TensorDataset(adv_images, true_labels)
    return DataLoader(adv_dataset, batch_size=config.BATCH_SIZE, shuffle=False)


def load_adversarial_dataset(path: str) -> DataLoader:
    """Load a saved adversarial set, tolerating both dict and TensorDataset formats."""
    data = torch.load(path)
    if isinstance(data, dict):
        images, labels = data["adv_images"], data["true_labels"]
    elif isinstance(data, TensorDataset):
        images, labels = data.tensors
    else:
        raise ValueError(f"Unsupported dataset format in {path}")

    if not isinstance(labels, torch.Tensor):
        labels = torch.tensor(labels)

    return DataLoader(TensorDataset(images, labels), batch_size=config.BATCH_SIZE, shuffle=False)
