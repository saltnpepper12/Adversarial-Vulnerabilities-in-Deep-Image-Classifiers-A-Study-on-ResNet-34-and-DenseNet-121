"""Pretrained model factory (ResNet-34 source model, DenseNet-121 transfer target)."""

import torchvision

from . import config


def load_resnet34():
    """ResNet-34 pretrained on ImageNet-1K — the white-box target model."""
    model = torchvision.models.resnet34(weights="IMAGENET1K_V1").to(config.DEVICE)
    model.eval()
    return model


def load_densenet121():
    """DenseNet-121 pretrained on ImageNet-1K — used for transferability tests."""
    model = torchvision.models.densenet121(weights="IMAGENET1K_V1").to(config.DEVICE)
    model.eval()
    return model
