"""Central configuration: paths, normalization constants, and attack defaults.

The evaluation dataset is a curated subset of ImageNet-1K containing 500 images
across 100 classes. Those 100 classes correspond to ImageNet indices 401-500, so
the folder labels produced by ``ImageFolder`` (0-99) are offset by ``LABEL_OFFSET``
to recover the true ImageNet class indices used by the pretrained models.
"""

from dataclasses import dataclass

import torch


# --- Device ---------------------------------------------------------------
def get_device() -> torch.device:
    """Return CUDA if available, else MPS (Apple Silicon), else CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = get_device()

# --- Data -----------------------------------------------------------------
# ImageNet normalization statistics (the pretrained models expect these).
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# The 100-class subset maps to ImageNet indices 401..500.
LABEL_OFFSET = 401

# Default file locations (relative to the repo root).
DATASET_ZIP = "./TestDataSet.zip"
DATASET_DIR = "./TestDataSet/TestDataSet"
LABELS_FILE = "./labels_list.json"

BATCH_SIZE = 32


# --- Attack hyper-parameters ----------------------------------------------
@dataclass
class FGSMConfig:
    epsilon: float = 0.02


@dataclass
class PGDConfig:
    epsilon: float = 0.02
    alpha: float = 0.002  # step size, epsilon / 10
    num_iter: int = 20


@dataclass
class PatchConfig:
    epsilon: float = 0.5
    patch_size: int = 32
    num_iter: int = 50
    alpha: float = 0.1


# Where generated adversarial sets are written.
ADV_SET_PATHS = {
    "fgsm": "Adversarial_Test_Set_1.pt",
    "pgd": "Adversarial_Test_Set_2.pt",
    "patch": "Adversarial_Test_Set_3.pt",
}
