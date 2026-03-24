import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch


PATHOLOGY_COLS = [
    "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly",
    "Lung Opacity", "Lung Lesion", "Edema", "Consolidation",
    "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion",
    "Pleural Other", "Fracture", "Support Devices"
]

COMPETITION_LABELS = [
    "Atelectasis", "Cardiomegaly", "Consolidation",
    "Edema", "Pleural Effusion"
]


def inspect_dataset(data_root: str):
    root = Path(data_root)
    train_csv = root / "train.csv"
    valid_csv = root / "valid.csv"

    assert train_csv.exists(), f"train.csv not found at {train_csv}"
    assert valid_csv.exists(), f"valid.csv not found at {valid_csv}"

    train_df = pd.read_csv(train_csv)
    valid_df = pd.read_csv(valid_csv)

    print(f"\n{'='*55}")
    print(f"  CheXpert Dataset Inspection")
    print(f"{'='*55}")
    print(f"  Train samples : {len(train_df):,}")
    print(f"  Valid samples : {len(valid_df):,}")
    print(f"  CSV columns   : {list(train_df.columns)}\n")

    # --- label value counts per pathology ---
    print(f"{'Label':<35} {'NaN':>7} {'0 (neg)':>9} {'1 (pos)':>9} {'-1 (unc)':>10} {'Pos%':>7}")
    print("-" * 80)
    for col in PATHOLOGY_COLS:
        if col not in train_df.columns:
            continue
        s = train_df[col]
        n_nan  = s.isna().sum()
        n_neg  = (s == 0).sum()
        n_pos  = (s == 1).sum()
        n_unc  = (s == -1).sum()
        total  = n_neg + n_pos + n_unc
        pos_pct = 100 * n_pos / total if total > 0 else 0
        print(f"  {col:<33} {n_nan:>7,} {n_neg:>9,} {n_pos:>9,} {n_unc:>10,} {pos_pct:>6.1f}%")

    # --- bar chart ---
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)

    pos_rates = []
    labels_found = []
    for col in PATHOLOGY_COLS:
        if col not in train_df.columns:
            continue
        s = train_df[col]
        total = (s == 0).sum() + (s == 1).sum() + (s == -1).sum()
        pos_rates.append(100 * (s == 1).sum() / total if total > 0 else 0)
        labels_found.append(col)

    fig, ax = plt.subplots(figsize=(13, 5))
    colors = ["#e74c3c" if l in COMPETITION_LABELS else "#3498db" for l in labels_found]
    bars = ax.bar(labels_found, pos_rates, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_title("CheXpert — Positive Label Rate per Pathology (Train Set)", fontsize=13, pad=12)
    ax.set_ylabel("Positive Rate (%)")
    ax.set_ylim(0, max(pos_rates) * 1.2)
    ax.set_xticklabels(labels_found, rotation=40, ha="right", fontsize=9)
    for bar, val in zip(bars, pos_rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=7.5)
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(color="#e74c3c", label="Competition label"),
        Patch(color="#3498db", label="Other pathology")
    ], fontsize=9)
    plt.tight_layout()
    save_path = out_dir / "label_distribution.png"
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"\n  [Saved] Label distribution chart → {save_path}")

    # --- sample X-ray grid ---
    _save_sample_grid(train_df, root)


def _save_sample_grid(df: pd.DataFrame, root: Path):
    import random
    from PIL import Image

    out_dir = Path("outputs")
    sampled = df.sample(9, random_state=42)

    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    fig.suptitle("CheXpert — Sample X-rays (Train Set)", fontsize=13)

    for ax, (_, row) in zip(axes.flat, sampled.iterrows()):
        img_path = root.parent / row["Path"]  # Path in CSV is relative to dataset root
        if not img_path.exists():
            # fallback: try relative to project root
            img_path = Path(row["Path"])
        if img_path.exists():
            img = Image.open(img_path).convert("L")
            ax.imshow(img, cmap="gray")
        else:
            ax.text(0.5, 0.5, "Not found", ha="center", va="center")
        # find first positive label
        label_str = "No Finding"
        for col in PATHOLOGY_COLS:
            if col in row and row[col] == 1.0:
                label_str = col
                break
        ax.set_title(label_str, fontsize=8)
        ax.axis("off")

    plt.tight_layout()
    save_path = out_dir / "sample_xrays.png"
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"  [Saved] Sample X-ray grid        → {save_path}")


def compute_pos_weights(csv_path: str, uncertainty_policy: str = "ones") -> torch.Tensor:
    """
    Compute per-label positive class weights for BCEWithLogitsLoss.
    pos_weight[i] = num_negative[i] / num_positive[i]
    Higher weight = rarer positive class = penalize misses more.
    """

    df = pd.read_csv(csv_path)

    print(f"\n{'='*55}")
    print(f"  Class Weight Computation (policy={uncertainty_policy})")
    print(f"{'='*55}")
    print(f"  {'Label':<35} {'Pos':>8} {'Neg':>8} {'Weight':>8}")
    print(f"  {'-'*62}")

    weights = []
    for col in PATHOLOGY_COLS:
        if col not in df.columns:
            weights.append(1.0)
            continue

        s = df[col].copy()
        s = s.fillna(0.0)  # NaN → negative

        # apply uncertainty policy
        if uncertainty_policy == "ones":
            s = s.replace(-1.0, 1.0)
        elif uncertainty_policy == "zeros":
            s = s.replace(-1.0, 0.0)
        else:  # ignore: treat as negative for weight computation
            s = s.replace(-1.0, 0.0)

        n_pos = (s == 1.0).sum()
        n_neg = (s == 0.0).sum()

        if n_pos == 0:
            w = 1.0
        else:
            w = n_neg / n_pos

        weights.append(w)
        print(f"  {col:<35} {n_pos:>8,} {n_neg:>8,} {w:>8.2f}")

    weight_tensor = torch.tensor(weights, dtype=torch.float32)
    print(f"\n  Min weight: {weight_tensor.min():.2f} | "
          f"Max weight: {weight_tensor.max():.2f} | "
          f"Mean weight: {weight_tensor.mean():.2f}")
    return weight_tensor


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inspect", action="store_true")
    parser.add_argument("--weights", action="store_true")
    parser.add_argument("--data_root", type=str, default="data/chexpert")
    args = parser.parse_args()

    if args.inspect:
        inspect_dataset(args.data_root)
    if args.weights:
        compute_pos_weights(
            csv_path=str(Path(args.data_root) / "train.csv"),
            uncertainty_policy="ones"
        )