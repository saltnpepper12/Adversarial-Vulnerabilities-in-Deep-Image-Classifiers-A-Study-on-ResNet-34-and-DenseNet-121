"""Projected Gradient Descent (Madry et al., 2017).

Iterative FGSM with random start, projecting the perturbation back into the
L-infinity epsilon-ball after every step:

    delta_{t+1} = Clip_epsilon( delta_t + alpha * sign(grad_x L(x + delta_t, y)) )
"""

import torch
import torch.nn as nn

from .. import config


def pgd_attack(model, images, labels,
               epsilon: float = 0.02, alpha: float = 0.002, num_iter: int = 20):
    """Generate PGD adversarial examples for a normalized batch."""
    mean, std = config.IMAGENET_MEAN, config.IMAGENET_STD

    images = images.to(config.DEVICE)
    labels = labels.to(config.DEVICE) + config.LABEL_OFFSET
    orig_images = images.clone().detach()

    delta = torch.zeros_like(images).to(config.DEVICE)
    delta.uniform_(-epsilon, epsilon)

    for _ in range(num_iter):
        delta.requires_grad = True
        outputs = model(orig_images + delta)
        loss = nn.CrossEntropyLoss()(outputs, labels)

        model.zero_grad()
        loss.backward()
        grad = delta.grad.detach()

        # Ascend, then project back into the epsilon-ball.
        delta.data = delta.data + alpha * grad.sign()
        delta.data = torch.clamp(delta.data, -epsilon, epsilon)

        # Keep de-normalized pixels valid in [0, 1].
        for c in range(3):
            lower = (0 - mean[c]) / std[c] - orig_images[:, c, :, :]
            upper = (1 - mean[c]) / std[c] - orig_images[:, c, :, :]
            delta[:, c, :, :].data = torch.clamp(delta[:, c, :, :], min=lower, max=upper)

    return (orig_images + delta).detach()
