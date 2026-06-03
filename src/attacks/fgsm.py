"""Fast Gradient Sign Method (Goodfellow et al., 2014).

Single-step, L-infinity bounded attack:

    x_adv = x + epsilon * sign(grad_x L(x, y))
"""

import torch
import torch.nn as nn

from .. import config


def fgsm_attack(model, images, labels, epsilon: float = 0.02):
    """Generate FGSM adversarial examples for a normalized batch.

    Perturbations are clipped so the de-normalized pixels stay within [0, 1].
    """
    mean, std = config.IMAGENET_MEAN, config.IMAGENET_STD

    images = images.clone().detach().to(config.DEVICE).float()
    labels = labels.clone().detach().to(config.DEVICE) + config.LABEL_OFFSET
    images.requires_grad = True

    outputs = model(images)
    loss = nn.CrossEntropyLoss()(outputs, labels)

    model.zero_grad()
    loss.backward()
    data_grad = images.grad.data

    perturbed = images + epsilon * data_grad.sign()

    # Keep pixels valid in normalized space.
    for c in range(3):
        lower = (0 - mean[c]) / std[c]
        upper = (1 - mean[c]) / std[c]
        perturbed[:, c, :, :] = torch.clamp(perturbed[:, c, :, :], lower, upper)

    return perturbed.detach()
