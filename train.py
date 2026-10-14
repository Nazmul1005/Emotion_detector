"""
train.py
========
Training script for EmotionCNN on the FER-2013 dataset.

Dataset setup (one-time)
────────────────────────
1. Download from Kaggle:
       https://www.kaggle.com/datasets/msambare/fer2013
2. After extraction, the folder layout should be:

       data/
       └── fer2013/
           ├── train/
           │   ├── angry/
           │   ├── disgust/
           │   ├── fear/
           │   ├── happy/
           │   ├── neutral/
           │   ├── sad/
           │   └── surprise/
           └── test/
               ├── angry/
               └── ...

   Each sub-folder contains .jpg/.png face images (48×48 grayscale).

Usage
─────
    python train.py                          # default settings
    python train.py --data data/fer2013      # custom dataset path
    python train.py --epochs 60 --batch 64  # custom hyperparams
    python train.py --resume checkpoints/best.pth  # resume training

MPS acceleration
────────────────
    All tensors are moved to the MPS device (Apple Silicon GPU) via
    `tensor.to(device)` before every forward pass. The `torch.no_grad()`
    context is used during validation to skip gradient tracking on MPS,
    reducing memory pressure and speeding up evaluation.

    Note: MPS does NOT support `torch.cuda.amp` (CUDA-only). Standard
    float32 precision is used, which is already fast on Apple Silicon.
"""

import argparse
import copy
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms

from model import EMOTION_LABELS, EmotionCNN, get_device


# ──────────────────────────────────────────────────────────────────
# Constants / defaults
# ──────────────────────────────────────────────────────────────────
DEFAULT_DATA_DIR   = "data/fer2013"
DEFAULT_EPOCHS     = 60
DEFAULT_BATCH_SIZE = 64
DEFAULT_LR         = 1e-3
DEFAULT_WEIGHT_DECAY = 1e-4
IMG_SIZE           = 48           # FER-2013 images are 48×48
NUM_WORKERS        = 0            # MPS: keep 0 to avoid macOS fork issues
CHECKPOINT_DIR     = Path("checkpoints")
HISTORY_FILE       = CHECKPOINT_DIR / "training_history.json"

# Normalisation (grayscale mean/std derived from FER-2013 train split)
MEAN = (0.5,)
STD  = (0.5,)


# ──────────────────────────────────────────────────────────────────
# Data transforms
# ──────────────────────────────────────────────────────────────────
def build_transforms() -> dict[str, transforms.Compose]:
    """
    Build separate augmentation pipelines for train and validation.

    Train augmentations:
        • Random horizontal flip   — faces are left-right symmetric
        • Random rotation ±10°     — slight pose variation
        • Random affine shift/scale — mild spatial jitter
        • Color jitter on brightness/contrast — lighting variation
        • Normalise to [-1, 1]

    Validation: deterministic center-crop + normalise only.
    """
    train_tf = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.1, 0.1),
            scale=(0.9, 1.1),
        ),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    val_tf = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    return {"train": train_tf, "val": val_tf}


# ──────────────────────────────────────────────────────────────────
# Dataset loading
# ──────────────────────────────────────────────────────────────────
def load_datasets(
    data_dir: str,
    transforms: dict,
) -> tuple[datasets.ImageFolder, datasets.ImageFolder]:
    """
    Load FER-2013 train and test splits as ImageFolder datasets.

    Validates that the expected sub-directories exist and prints
    per-class sample counts to help diagnose class-imbalance issues.

    Args:
        data_dir:   Root directory containing 'train/' and 'test/' sub-dirs.
        transforms: Dict with 'train' and 'val' transform pipelines.

    Returns:
        (train_dataset, val_dataset)
    """
    root = Path(data_dir)
    train_dir = root / "train"
    test_dir  = root / "test"

    for split_dir in (train_dir, test_dir):
        if not split_dir.exists():
            sys.exit(
                f"\n[Error] Dataset split not found: {split_dir}\n"
                "        Please download FER-2013 from Kaggle and extract it.\n"
                "        See the module docstring for the expected folder layout."
            )

    train_ds = datasets.ImageFolder(str(train_dir), transform=transforms["train"])
    val_ds   = datasets.ImageFolder(str(test_dir),  transform=transforms["val"])

    # Print class distribution
    print("\n[Dataset] Class distribution (train)")
    print(f"  {'Class':<12} {'Samples':>8}")
    print("  " + "-" * 22)
    for cls, idx in train_ds.class_to_idx.items():
        count = sum(1 for _, label in train_ds.samples if label == idx)
        print(f"  {cls:<12} {count:>8,}")
    print(f"  {'TOTAL':<12} {len(train_ds):>8,}")
    print(f"\n[Dataset] Validation samples : {len(val_ds):,}\n")

    return train_ds, val_ds


def build_balanced_sampler(dataset: datasets.ImageFolder) -> WeightedRandomSampler:
    """
    Create a WeightedRandomSampler to handle FER-2013 class imbalance.

    FER-2013 is heavily imbalanced (Happy ~8,989 vs Disgust ~547).
    Oversampling rare classes during training prevents the model
    from collapsing to a majority-class predictor.

    Returns:
        A sampler that draws each class with equal expected frequency.
    """
    targets  = [label for _, label in dataset.samples]
    n_classes = len(dataset.classes)
    class_counts = [0] * n_classes
    for t in targets:
        class_counts[t] += 1

    # Weight per sample = inverse of its class frequency
    weights = [1.0 / class_counts[t] for t in targets]
    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True,
    )
    return sampler


def build_loaders(
    train_ds: datasets.ImageFolder,
    val_ds: datasets.ImageFolder,
    batch_size: int,
) -> tuple[DataLoader, DataLoader]:
    """Build DataLoaders with balanced sampling for the train split."""
    sampler = build_balanced_sampler(train_ds)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,          # replaces shuffle=True
        num_workers=NUM_WORKERS,  # 0 avoids macOS multiprocessing issues with MPS
        pin_memory=False,         # pin_memory is CUDA-only, must be False for MPS
        drop_last=True,           # avoids tiny final batches on MPS
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=False,
        drop_last=False,
    )
    return train_loader, val_loader


# ──────────────────────────────────────────────────────────────────
# Training utilities
# ──────────────────────────────────────────────────────────────────
def run_epoch(
    model: EmotionCNN,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer | None,
    device: torch.device,
    is_train: bool,
) -> tuple[float, float]:
    """
    Run one full pass (train or validate) over *loader*.

    MPS usage:
        Each batch is moved to the MPS device with `.to(device)`.
        During validation, `torch.no_grad()` prevents MPS from
        allocating gradient buffers, significantly reducing GPU memory.

    Args:
        model:     EmotionCNN on *device*.
        loader:    DataLoader for the split.
        criterion: Loss function (CrossEntropyLoss).
        optimizer: Optimizer (None during validation).
        device:    Compute device (mps / cuda / cpu).
        is_train:  If True, run backward pass and optimizer step.

    Returns:
        (avg_loss, accuracy) for this epoch.
    """
    model.train() if is_train else model.eval()
    total_loss    = 0.0
    correct       = 0
    total_samples = 0

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for images, labels in loader:
            # ── Move batch to MPS / CUDA / CPU ────────────────────
            images = images.to(device)   # (B, 1, 48, 48) → MPS
            labels = labels.to(device)   # (B,)            → MPS

            # ── Forward pass ──────────────────────────────────────
            logits = model(images)                   # (B, 7)
            loss   = criterion(logits, labels)

            # ── Backward pass (train only) ─────────────────────────
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                # Gradient clipping prevents exploding gradients on MPS
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

            # ── Accumulate stats ──────────────────────────────────
            batch_size     = labels.size(0)
            total_loss    += loss.item() * batch_size
            preds          = logits.argmax(dim=1)
            correct       += (preds == labels).sum().item()
            total_samples += batch_size

    avg_loss = total_loss / max(total_samples, 1)
    accuracy = correct   / max(total_samples, 1)
    return avg_loss, accuracy


# ──────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ──────────────────────────────────────────────────────────────────
def save_checkpoint(
    model: EmotionCNN,
    optimizer: optim.Optimizer,
    scheduler,
    epoch: int,
    val_acc: float,
    is_best: bool,
    history: list[dict],
) -> None:
    """
    Save a full training checkpoint.

    Two files are maintained:
        last.pth  — latest epoch (for resuming)
        best.pth  — snapshot with highest validation accuracy
    """
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    state = {
        "epoch":          epoch,
        "val_acc":        val_acc,
        "model_state":    model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
    }

    last_path = CHECKPOINT_DIR / "last.pth"
    torch.save(state, last_path)

    if is_best:
        best_path = CHECKPOINT_DIR / "best.pth"
        torch.save(state, best_path)
        print(f"  [Checkpoint] 🏆 New best saved → {best_path}  (val_acc={val_acc:.4f})")

    # Save training curves as JSON (for plotting / inspection)
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)


def load_checkpoint(
    path: str,
    model: EmotionCNN,
    optimizer: optim.Optimizer,
    scheduler,
    device: torch.device,
) -> int:
    """
    Load a checkpoint and return the epoch to resume from.

    Args:
        path: Path to the .pth checkpoint file.

    Returns:
        start_epoch: The epoch number to resume from (0-indexed).
    """
    state = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state"])
    optimizer.load_state_dict(state["optimizer_state"])
    scheduler.load_state_dict(state["scheduler_state"])
    start_epoch = state["epoch"] + 1
    print(
        f"[Resume] Loaded checkpoint from epoch {state['epoch']}  "
        f"(best val_acc={state['val_acc']:.4f})"
    )
    return start_epoch


# ──────────────────────────────────────────────────────────────────
# Main training function
# ──────────────────────────────────────────────────────────────────
def train(args: argparse.Namespace) -> None:
    """End-to-end training pipeline for EmotionCNN on FER-2013."""

    # ── 1. Device (MPS preferred) ─────────────────────────────────
    device = get_device()

    # ── 2. Data ───────────────────────────────────────────────────
    tfs = build_transforms()
    train_ds, val_ds = load_datasets(args.data, tfs)
    train_loader, val_loader = build_loaders(train_ds, val_ds, args.batch)

    # ── 3. Model ──────────────────────────────────────────────────
    model = EmotionCNN(num_classes=len(EMOTION_LABELS)).to(device)

    # ── 4. Loss & Optimiser ───────────────────────────────────────
    # Label smoothing (0.1) helps generalisation on FER-2013 noise
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
    )

    # Cosine annealing reduces LR smoothly over all epochs
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-6,
    )

    # ── 5. Optionally resume from checkpoint ──────────────────────
    start_epoch = 0
    best_val_acc = 0.0
    history: list[dict] = []

    if args.resume and Path(args.resume).exists():
        start_epoch = load_checkpoint(
            args.resume, model, optimizer, scheduler, device
        )
        # Recover history for appending
        if HISTORY_FILE.exists():
            with open(HISTORY_FILE) as f:
                history = json.load(f)
            best_val_acc = max(h["val_acc"] for h in history) if history else 0.0

    # ── 6. Training loop ──────────────────────────────────────────
    print("=" * 62)
    print(f"  Training EmotionCNN on FER-2013")
    print(f"  Device      : {device}")
    print(f"  Epochs      : {args.epochs}  (starting from {start_epoch})")
    print(f"  Batch size  : {args.batch}")
    print(f"  LR          : {args.lr}  (cosine decay to 1e-6)")
    print(f"  Weight decay: {args.weight_decay}")
    print(f"  Output dir  : {CHECKPOINT_DIR}/")
    print("=" * 62)

    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.perf_counter()

        # ── Train ─────────────────────────────────────────────────
        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, optimizer, device, is_train=True
        )

        # ── Validate ──────────────────────────────────────────────
        val_loss, val_acc = run_epoch(
            model, val_loader, criterion, None, device, is_train=False
        )

        scheduler.step()
        epoch_time   = time.perf_counter() - epoch_start
        current_lr   = scheduler.get_last_lr()[0]
        is_best      = val_acc > best_val_acc
        best_val_acc = max(best_val_acc, val_acc)

        # ── Log ───────────────────────────────────────────────────
        record = {
            "epoch":      epoch,
            "train_loss": round(train_loss, 6),
            "train_acc":  round(train_acc,  6),
            "val_loss":   round(val_loss,   6),
            "val_acc":    round(val_acc,    6),
            "lr":         round(current_lr, 8),
            "time_s":     round(epoch_time, 2),
        }
        history.append(record)

        best_marker = " ← best" if is_best else ""
        print(
            f"  Epoch [{epoch + 1:>3}/{args.epochs}]  "
            f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  "
            f"lr={current_lr:.2e}  ({epoch_time:.1f}s){best_marker}"
        )

        # ── Save checkpoint ───────────────────────────────────────
        save_checkpoint(
            model, optimizer, scheduler, epoch, val_acc, is_best, history
        )

    print("\n" + "=" * 62)
    print(f"  Training complete!")
    print(f"  Best validation accuracy : {best_val_acc * 100:.2f}%")
    print(f"  Best weights saved at    : {CHECKPOINT_DIR / 'best.pth'}")
    print(f"  Training history saved at: {HISTORY_FILE}")
    print("=" * 62)
    print(
        f"\n  Run inference with:\n"
        f"    python main.py --weights {CHECKPOINT_DIR / 'best.pth'}\n"
    )


# ──────────────────────────────────────────────────────────────────
# CLI argument parsing
# ──────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train EmotionCNN on FER-2013 (Apple MPS optimised)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data", "-d",
        type=str,
        default=DEFAULT_DATA_DIR,
        help="Path to FER-2013 root directory (must contain train/ and test/).",
    )
    parser.add_argument(
        "--epochs", "-e",
        type=int,
        default=DEFAULT_EPOCHS,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch", "-b",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Batch size per training step.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=DEFAULT_LR,
        help="Initial learning rate for AdamW.",
    )
    parser.add_argument(
        "--weight-decay", "--wd",
        dest="weight_decay",
        type=float,
        default=DEFAULT_WEIGHT_DECAY,
        help="L2 regularisation weight decay for AdamW.",
    )
    parser.add_argument(
        "--resume", "-r",
        type=str,
        default=None,
        help="Path to a checkpoint .pth file to resume training from.",
    )
    return parser.parse_args()


# ──────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    args = parse_args()
    train(args)
