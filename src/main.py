"""Command-line entry point reproducing the five experiment tasks.

Tasks
-----
1. baseline  : evaluate clean ResNet-34 accuracy
2. fgsm      : generate + evaluate FGSM adversarial set
3. pgd       : generate + evaluate PGD adversarial set
4. patch     : generate + evaluate patch adversarial set
5. transfer  : evaluate all sets against DenseNet-121
all          : run every task in sequence (default)

Usage
-----
    python -m src.main                 # run everything
    python -m src.main pgd --visualize # one task, with figure output
"""

import argparse
import os

from . import config
from .attacks import fgsm_attack, patch_attack, pgd_attack
from .data import extract_dataset, get_dataloader, load_dataset, load_idx_to_class
from .evaluate import evaluate
from .generate import generate_adversarial_set, load_adversarial_dataset
from .models import load_densenet121, load_resnet34
from .visualize import visualize_attack


def _pct(t1, t5):
    return f"Top-1: {t1 * 100:.2f}%, Top-5: {t5 * 100:.2f}%"


def run_baseline(ctx):
    print("\n=== Task 1: Baseline (clean) ResNet-34 ===")
    t1, t5 = evaluate(ctx["model"], ctx["loader"])
    print(_pct(t1, t5))
    ctx["baseline_top1"] = t1
    return t1, t5


def run_fgsm(ctx, visualize=False):
    print("\n=== Task 2: FGSM Attack ===")
    cfg = config.FGSMConfig()
    adv_loader = generate_adversarial_set(
        ctx["model"], ctx["loader"], fgsm_attack,
        config.ADV_SET_PATHS["fgsm"], "Generating FGSM examples",
        epsilon=cfg.epsilon,
    )
    t1, t5 = evaluate(ctx["model"], adv_loader)
    print(_pct(t1, t5))
    if visualize:
        visualize_attack(ctx["model"], ctx["dataset"], adv_loader.dataset,
                         ctx["idx_to_class"], num_samples=5, title="FGSM",
                         save_path="assets/fgsm_examples.png")
    return t1, t5


def run_pgd(ctx, visualize=False):
    print("\n=== Task 3: PGD Attack ===")
    cfg = config.PGDConfig()
    adv_loader = generate_adversarial_set(
        ctx["model"], ctx["loader"], pgd_attack,
        config.ADV_SET_PATHS["pgd"], "Generating PGD examples",
        epsilon=cfg.epsilon, alpha=cfg.alpha, num_iter=cfg.num_iter,
    )
    t1, t5 = evaluate(ctx["model"], adv_loader)
    print(_pct(t1, t5))
    if "baseline_top1" in ctx:
        print(f"Accuracy drop: {ctx['baseline_top1'] * 100 - t1 * 100:.2f}pp (Top-1)")
    if visualize:
        visualize_attack(ctx["model"], ctx["dataset"], adv_loader.dataset,
                         ctx["idx_to_class"], num_samples=3, title="PGD",
                         save_path="assets/pgd_examples.png")
    return t1, t5


def run_patch(ctx, visualize=False):
    print("\n=== Task 4: Patch Attack ===")
    cfg = config.PatchConfig()
    adv_loader = generate_adversarial_set(
        ctx["model"], ctx["loader"], patch_attack,
        config.ADV_SET_PATHS["patch"], "Generating patch examples",
        epsilon=cfg.epsilon, patch_size=cfg.patch_size,
        num_iter=cfg.num_iter, alpha=cfg.alpha,
    )
    t1, t5 = evaluate(ctx["model"], adv_loader)
    print(_pct(t1, t5))
    if visualize:
        visualize_attack(ctx["model"], ctx["dataset"], adv_loader.dataset,
                         ctx["idx_to_class"], num_samples=3, title="Patch",
                         save_path="assets/patch_examples.png")
    return t1, t5


def run_transfer(ctx):
    print("\n=== Task 5: Transferability to DenseNet-121 ===")
    transfer_model = load_densenet121()
    sets = {"Original": ctx["loader"]}
    for name, key in [("FGSM", "fgsm"), ("PGD", "pgd"), ("Patch", "patch")]:
        path = config.ADV_SET_PATHS[key]
        if os.path.exists(path):
            sets[name] = load_adversarial_dataset(path)
        else:
            print(f"  (skipping {name}: {path} not found — run that task first)")

    print("{:<10} {:<10} {:<10}".format("Dataset", "Top-1", "Top-5"))
    print("-" * 32)
    for name, dl in sets.items():
        t1, t5 = evaluate(transfer_model, dl)
        print("{:<10} {:<9.2f}% {:<9.2f}%".format(name, t1 * 100, t5 * 100))


def build_context():
    extract_dataset()
    dataset = load_dataset()
    return {
        "model": load_resnet34(),
        "dataset": dataset,
        "loader": get_dataloader(dataset),
        "idx_to_class": load_idx_to_class(),
    }


def main():
    parser = argparse.ArgumentParser(description="Adversarial attacks on deep image classifiers.")
    parser.add_argument(
        "task", nargs="?", default="all",
        choices=["all", "baseline", "fgsm", "pgd", "patch", "transfer"],
        help="Which experiment task to run (default: all).",
    )
    parser.add_argument("--visualize", action="store_true",
                        help="Save clean/adversarial comparison figures to assets/.")
    args = parser.parse_args()

    print(f"Using device: {config.DEVICE}")
    ctx = build_context()

    if args.task in ("all", "baseline"):
        run_baseline(ctx)
    if args.task in ("all", "fgsm"):
        run_fgsm(ctx, visualize=args.visualize)
    if args.task in ("all", "pgd"):
        run_pgd(ctx, visualize=args.visualize)
    if args.task in ("all", "patch"):
        run_patch(ctx, visualize=args.visualize)
    if args.task in ("all", "transfer"):
        run_transfer(ctx)


if __name__ == "__main__":
    main()
