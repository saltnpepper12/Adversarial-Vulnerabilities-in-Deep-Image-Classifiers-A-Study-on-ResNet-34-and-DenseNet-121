"""Top-1 / Top-5 accuracy evaluation with ImageNet label-offset handling."""

import torch

from . import config


@torch.no_grad()
def evaluate(model, dataloader, label_offset: int = config.LABEL_OFFSET):
    """Compute Top-1 and Top-5 accuracy.

    Folder labels are 0-99; they are offset by ``label_offset`` to match the
    pretrained model's ImageNet class indices (401-500).
    """
    top1_correct = 0
    top5_correct = 0
    total = 0
    for images, labels in dataloader:
        images, labels = images.to(config.DEVICE), labels.to(config.DEVICE)
        outputs = model(images)
        _, preds = torch.topk(outputs, 5, dim=1)

        adjusted_labels = labels + label_offset
        correct = preds.eq(adjusted_labels.view(-1, 1))

        top1_correct += correct[:, 0].sum().item()
        top5_correct += correct.any(dim=1).sum().item()
        total += labels.size(0)

    return top1_correct / total, top5_correct / total
