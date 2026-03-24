import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import yaml
import time
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from src.dataset import CheXpertDataset, PATHOLOGY_COLS


def verify(config_path: str = "configs/config.yaml"):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    data_root = cfg["data_root"]
    train_csv = str(Path(data_root) / "train.csv")
    valid_csv = str(Path(data_root) / "valid.csv")

    print("\n" + "="*55)
    print("  DataLoader Verification")
    print("="*55)

    # --- build datasets ---
    train_ds = CheXpertDataset(
        csv_path=train_csv,
        data_root=data_root,
        split="train",
        image_size=cfg["image_size"],
        uncertainty_policy=cfg["uncertainty_policy"],
        frontal_only=cfg["frontal_only"],
    )
    val_ds = CheXpertDataset(
        csv_path=valid_csv,
        data_root=data_root,
        split="val",
        image_size=cfg["image_size"],
        uncertainty_policy=cfg["uncertainty_policy"],
        frontal_only=cfg["frontal_only"],
    )

    # --- build dataloaders ---
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        pin_memory=False,   # MPS compatibility
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=False,
    )

    # --- fetch one batch ---
    print("\n  Fetching first train batch...")
    t0 = time.time()
    images, labels = next(iter(train_loader))
    elapsed = time.time() - t0

    print(f"  Batch fetch time   : {elapsed:.2f}s")
    print(f"  Image tensor shape : {images.shape}")   # (B, 3, H, W)
    print(f"  Image dtype        : {images.dtype}")
    print(f"  Image min/max      : {images.min():.3f} / {images.max():.3f}")
    print(f"  Label tensor shape : {labels.shape}")   # (B, 14)
    print(f"  Label dtype        : {labels.dtype}")
    print(f"  Unique label values: {labels.unique().tolist()}")

    # --- label stats for this batch ---
    print(f"\n  Per-label positive rate in this batch:")
    for i, col in enumerate(PATHOLOGY_COLS):
        pos_rate = (labels[:, i] == 1).float().mean().item()
        print(f"    {col:<35} {pos_rate*100:>5.1f}%")

    # --- val batch ---
    print("\n  Fetching first val batch...")
    val_images, val_labels = next(iter(val_loader))
    print(f"  Val image shape    : {val_images.shape}")
    print(f"  Val label shape    : {val_labels.shape}")

    # --- save sample batch grid ---
    _save_batch_grid(images, labels, "outputs/batch_sample.png")
    print("\n  Verification complete ✅")


def _save_batch_grid(images, labels, save_path: str):
    """Save first 8 images from a batch as a grid with label names."""
    import torchvision.utils as vutils

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    n = min(8, images.shape[0])
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    fig.suptitle("Batch Sample — Augmented Training Images", fontsize=12)

    for i, ax in enumerate(axes.flat):
        if i >= n:
            ax.axis("off")
            continue
        img = images[i] * std + mean          # denormalize
        img = img.permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)
        # convert to grayscale-ish for X-rays
        gray = img.mean(axis=2)
        ax.imshow(gray, cmap="gray")
        # find positive labels
        pos_labels = [PATHOLOGY_COLS[j] for j in range(14) if labels[i, j] == 1.0]
        title = "\n".join(pos_labels[:2]) if pos_labels else "No Finding"
        ax.set_title(title, fontsize=7)
        ax.axis("off")

    plt.tight_layout()
    Path(save_path).parent.mkdir(exist_ok=True)
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"  [Saved] Batch grid → {save_path}")


if __name__ == "__main__":
    verify()