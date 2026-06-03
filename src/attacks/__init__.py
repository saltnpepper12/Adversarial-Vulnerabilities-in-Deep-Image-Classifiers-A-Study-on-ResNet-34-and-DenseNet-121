"""Adversarial attack implementations: FGSM, PGD, and localized patch attacks."""

from .fgsm import fgsm_attack
from .pgd import pgd_attack
from .patch import patch_attack

__all__ = ["fgsm_attack", "pgd_attack", "patch_attack"]
