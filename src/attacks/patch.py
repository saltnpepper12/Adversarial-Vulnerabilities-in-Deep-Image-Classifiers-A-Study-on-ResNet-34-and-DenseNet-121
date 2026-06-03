"""Localized patch attack (inspired by Brown et al., 2017).

Perturbations are confined to a randomly placed ``patch_size`` x ``patch_size``
region and optimized (targeted, toward the model's least-likely class) with a
larger epsilon to compensate for the smaller attack surface.
"""

import torch
import torch.nn as nn

from .. import config


def patch_attack(model, images, labels,
                 epsilon: float = 0.5, patch_size: int = 32,
                 num_iter: int = 50, alpha: float = 0.1):
    """Generate patch adversarial examples for a normalized batch.

    ``labels`` is accepted for API symmetry with the other attacks but is not
    used: the target is the model's least-likely predicted class.
    """
    mean = torch.tensor(config.IMAGENET_MEAN).view(1, 3, 1, 1).to(config.DEVICE)
    std = torch.tensor(config.IMAGENET_STD).view(1, 3, 1, 1).to(config.DEVICE)

    images = images.to(config.DEVICE)
    batch_size, _, h, w = images.shape
    delta = torch.zeros_like(images).to(config.DEVICE)
    masks = torch.zeros_like(images)

    # Random patch position per image.
    xs = torch.randint(0, h - patch_size, (batch_size,))
    ys = torch.randint(0, w - patch_size, (batch_size,))
    for i in range(batch_size):
        masks[i, :, xs[i]:xs[i] + patch_size, ys[i]:ys[i] + patch_size] = 1

    delta.data.uniform_(-epsilon, epsilon).mul_(masks)

    # Targeted attack toward each image's least-likely class.
    with torch.no_grad():
        outputs = model(images)
        _, target_labels = torch.topk(-outputs, 1)
        target_labels = target_labels.squeeze()

    adv_images = images
    for _ in range(num_iter):
        delta.requires_grad = True
        adv_images = torch.clamp(images + delta, (0 - mean) / std, (1 - mean) / std)

        outputs = model(adv_images)
        loss = nn.CrossEntropyLoss()(outputs, target_labels)

        model.zero_grad()
        loss.backward()

        # Descend toward the target class, constrained to the patch region.
        delta.data = delta.data - alpha * delta.grad.sign() * masks
        delta.data = torch.clamp(delta.data, -epsilon, epsilon) * masks

    return adv_images.detach()
