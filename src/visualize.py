"""Side-by-side visualization of clean vs. adversarial images and predictions."""

import random

import matplotlib.pyplot as plt
import torch

from . import config
from .data import denormalize


@torch.no_grad()
def _predict(model, tensor):
    return model(tensor.unsqueeze(0).to(config.DEVICE)).argmax().item()


def visualize_attack(model, clean_dataset, adv_dataset, idx_to_class,
                     num_samples: int = 5, title: str = "Adversarial",
                     save_path: str = None):
    """Plot ``num_samples`` clean/adversarial pairs with their predicted labels.

    Only samples whose true/clean/adversarial classes are all known are shown.
    """
    valid_indices = []
    for idx in range(len(adv_dataset)):
        adv_img, label = adv_dataset[idx]
        label_int = label.item() + config.LABEL_OFFSET
        true_class = idx_to_class.get(label_int)
        orig_class = idx_to_class.get(_predict(model, clean_dataset[idx][0]))
        adv_class = idx_to_class.get(_predict(model, adv_img))
        if None not in (true_class, orig_class, adv_class):
            valid_indices.append(idx)

    sample_indices = random.sample(valid_indices, min(num_samples, len(valid_indices)))
    n = len(sample_indices)

    plt.figure(figsize=(10, 3 * n))
    for i, idx in enumerate(sample_indices):
        adv_img, label = adv_dataset[idx]
        label_int = label.item() + config.LABEL_OFFSET
        orig_class = idx_to_class.get(_predict(model, clean_dataset[idx][0]), "Unknown")
        adv_class = idx_to_class.get(_predict(model, adv_img), "Unknown")
        true_class = idx_to_class.get(label_int, "Unknown")

        plt.subplot(n, 2, 2 * i + 1)
        plt.imshow(denormalize(clean_dataset[idx][0]))
        plt.title(f"Original\nTrue: {true_class}\nPred: {orig_class}")
        plt.axis("off")

        plt.subplot(n, 2, 2 * i + 2)
        plt.imshow(denormalize(adv_img))
        plt.title(f"{title}\nPred: {adv_class}")
        plt.axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()
    plt.close()
