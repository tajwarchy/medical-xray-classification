import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import json
import yaml
from torch.utils.data import DataLoader
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    average_precision_score,
    confusion_matrix, f1_score
)

from src.dataset import CheXpertDataset, PATHOLOGY_COLS, COMPETITION_LABELS
from src.model import CheXpertModel
from src.utils import compute_pos_weights


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ── inference ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference(model, loader, device):
    """Run full inference, return (labels_np, probs_np)."""
    model.eval()
    all_labels, all_probs = [], []

    for images, labels in loader:
        images = images.to(device)
        logits, _ = model(images)
        probs = torch.sigmoid(logits).cpu().numpy()
        all_labels.append(labels.numpy())
        all_probs.append(probs)

    return (
        np.concatenate(all_labels, axis=0),   # (N, 14)
        np.concatenate(all_probs,  axis=0),   # (N, 14)
    )


# ── threshold optimization ────────────────────────────────────────────────────

def find_optimal_threshold(y_true, y_prob):
    """
    Find threshold maximizing Youden's J = Sensitivity + Specificity - 1.
    Returns (threshold, sensitivity, specificity).
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    specificity = 1 - fpr
    j_scores = tpr + specificity - 1
    best_idx = np.argmax(j_scores)
    return float(thresholds[best_idx]), float(tpr[best_idx]), float(specificity[best_idx])


# ── per-label metrics ─────────────────────────────────────────────────────────

def compute_all_metrics(labels_np, probs_np):
    """
    Compute AUC, AP, sensitivity, specificity, F1 for each label.
    Returns a list of dicts, one per label.
    """
    results = []
    thresholds = {}

    for i, col in enumerate(PATHOLOGY_COLS):
        y_true = labels_np[:, i]
        y_prob = probs_np[:, i]

        if len(np.unique(y_true)) < 2:
            results.append({
                "label": col, "auc": float("nan"), "ap": float("nan"),
                "sensitivity": float("nan"), "specificity": float("nan"),
                "f1": float("nan"), "threshold": 0.5,
                "competition": col in COMPETITION_LABELS
            })
            thresholds[col] = 0.5
            continue

        auc = roc_auc_score(y_true, y_prob)
        ap  = average_precision_score(y_true, y_prob)
        thresh, sens, spec = find_optimal_threshold(y_true, y_prob)
        y_pred = (y_prob >= thresh).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        thresholds[col] = thresh
        results.append({
            "label"      : col,
            "auc"        : auc,
            "ap"         : ap,
            "sensitivity": sens,
            "specificity": spec,
            "f1"         : f1,
            "threshold"  : thresh,
            "competition": col in COMPETITION_LABELS,
        })

    return results, thresholds


# ── plots ─────────────────────────────────────────────────────────────────────

def plot_auroc_curves(labels_np, probs_np, out_dir: Path):
    """Plot AUROC curves for all 14 labels, highlight competition labels."""
    fig, ax = plt.subplots(figsize=(11, 8))

    colors_comp  = plt.cm.Reds(np.linspace(0.4, 0.9, len(COMPETITION_LABELS)))
    colors_other = plt.cm.Blues(np.linspace(0.3, 0.7, len(PATHOLOGY_COLS) - len(COMPETITION_LABELS)))

    comp_idx  = 0
    other_idx = 0

    for i, col in enumerate(PATHOLOGY_COLS):
        y_true = labels_np[:, i]
        y_prob = probs_np[:, i]
        if len(np.unique(y_true)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)

        if col in COMPETITION_LABELS:
            ax.plot(fpr, tpr, color=colors_comp[comp_idx],
                    linewidth=2.5, label=f"{col} (AUC={auc:.3f})")
            comp_idx += 1
        else:
            ax.plot(fpr, tpr, color=colors_other[other_idx],
                    linewidth=1.0, alpha=0.6, linestyle="--",
                    label=f"{col} (AUC={auc:.3f})")
            other_idx += 1

    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.5, label="Random")
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=12)
    ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=12)
    ax.set_title("AUROC Curves — CheXpert Validation Set", fontsize=14, pad=12)
    ax.legend(loc="lower right", fontsize=7.5, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    save_path = out_dir / "auroc_curves.png"
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  [Saved] AUROC curves → {save_path}")


def plot_metrics_bar(results, out_dir: Path):
    """Bar chart of AUC, Sensitivity, Specificity per label."""
    labels  = [r["label"] for r in results if not np.isnan(r["auc"])]
    aucs    = [r["auc"]         for r in results if not np.isnan(r["auc"])]
    sens    = [r["sensitivity"] for r in results if not np.isnan(r["auc"])]
    specs   = [r["specificity"] for r in results if not np.isnan(r["auc"])]

    x = np.arange(len(labels))
    w = 0.25

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(x - w, aucs, w, label="AUC",         color="#2ecc71", edgecolor="white")
    ax.bar(x,     sens, w, label="Sensitivity",  color="#3498db", edgecolor="white")
    ax.bar(x + w, specs, w, label="Specificity", color="#e74c3c", edgecolor="white")
    ax.axhline(y=0.85, color="black", linestyle="--", alpha=0.5, linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_title("Per-Label Clinical Metrics — Validation Set", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2, axis="y")

    plt.tight_layout()
    save_path = out_dir / "metrics_bar.png"
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  [Saved] Metrics bar chart → {save_path}")


def plot_confusion_matrices(labels_np, probs_np, thresholds, out_dir: Path):
    """3x5 grid of confusion matrices for all 14 labels (skip last if odd)."""
    n = len(PATHOLOGY_COLS)
    fig, axes = plt.subplots(3, 5, figsize=(16, 10))
    fig.suptitle("Confusion Matrices at Optimal Threshold", fontsize=13)

    for i, (col, ax) in enumerate(zip(PATHOLOGY_COLS, axes.flat)):
        y_true = labels_np[:, i]
        y_prob = probs_np[:, i]
        thresh = thresholds.get(col, 0.5)
        y_pred = (y_prob >= thresh).astype(int)

        if len(np.unique(y_true)) < 2:
            ax.axis("off")
            continue

        cm = confusion_matrix(y_true, y_pred)
        im = ax.imshow(cm, interpolation="nearest",
                       cmap="Blues" if col not in COMPETITION_LABELS else "Reds")
        ax.set_title(col, fontsize=7,
                     fontweight="bold" if col in COMPETITION_LABELS else "normal")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["Neg", "Pos"], fontsize=7)
        ax.set_yticklabels(["Neg", "Pos"], fontsize=7)
        for r in range(2):
            for c in range(2):
                ax.text(c, r, str(cm[r, c]), ha="center", va="center",
                        fontsize=9, color="white" if cm[r, c] > cm.max()/2 else "black")

    # hide unused axes
    for ax in axes.flat[n:]:
        ax.axis("off")

    plt.tight_layout()
    save_path = out_dir / "confusion_matrices.png"
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"  [Saved] Confusion matrices → {save_path}")


# ── clinical report ───────────────────────────────────────────────────────────

def generate_clinical_report(results, thresholds, labels_np, out_dir: Path, cfg: dict):
    """Auto-generate a plain-text clinical evaluation report."""
    lines = []
    lines.append("=" * 65)
    lines.append("  CheXpert Clinical Evaluation Report")
    lines.append("=" * 65)
    lines.append(f"  Model         : DenseNet-121 + Channel Attention")
    lines.append(f"  Image size    : {cfg['image_size']}x{cfg['image_size']}")
    lines.append(f"  Val samples   : {labels_np.shape[0]}")
    lines.append(f"  Unc. policy   : {cfg['uncertainty_policy']}")
    lines.append("")
    lines.append("  Competition Labels (CheXpert 5):")
    lines.append(f"  {'Label':<28} {'AUC':>6} {'Sens':>6} {'Spec':>6} {'F1':>6} {'Thresh':>8}")
    lines.append("  " + "-" * 60)

    comp_aucs = []
    for r in results:
        if r["label"] not in COMPETITION_LABELS:
            continue
        auc  = r["auc"]
        sens = r["sensitivity"]
        spec = r["specificity"]
        f1   = r["f1"]
        thr  = r["threshold"]
        flag = " ◄" if not np.isnan(auc) and auc >= 0.85 else ""
        lines.append(f"  {r['label']:<28} {auc:>6.3f} {sens:>6.3f} "
                     f"{spec:>6.3f} {f1:>6.3f} {thr:>8.3f}{flag}")
        if not np.isnan(auc):
            comp_aucs.append(auc)

    mean_comp = np.mean(comp_aucs) if comp_aucs else float("nan")
    lines.append("")
    lines.append(f"  Mean Competition AUC : {mean_comp:.4f}")
    lines.append(f"  Target (≥0.85)       : {'✅ ACHIEVED' if mean_comp >= 0.85 else '❌ NOT YET'}")
    lines.append("")
    lines.append("  All Labels:")
    lines.append(f"  {'Label':<28} {'AUC':>6} {'Sens':>6} {'Spec':>6} {'F1':>6}")
    lines.append("  " + "-" * 55)
    for r in results:
        auc  = r["auc"]
        sens = r["sensitivity"]
        spec = r["specificity"]
        f1   = r["f1"]
        lines.append(f"  {r['label']:<28} {auc:>6.3f} {sens:>6.3f} "
                     f"{spec:>6.3f} {f1:>6.3f}")
    lines.append("")
    lines.append("=" * 65)

    report_str = "\n".join(lines)
    print("\n" + report_str)

    save_path = out_dir / "reports" / "evaluation_report.txt"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text(report_str)
    print(f"\n  [Saved] Clinical report → {save_path}")

    return mean_comp


# ── main ──────────────────────────────────────────────────────────────────────

def evaluate(config_path: str = "configs/config.yaml"):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device   = get_device()
    out_dir  = Path(cfg["output_dir"])
    ckpt_dir = Path(cfg["checkpoint_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  CheXpert Evaluation — Device: {device}")
    print(f"{'='*60}")

    # --- dataset ---
    val_ds = CheXpertDataset(
        csv_path=f"{cfg['data_root']}/valid.csv",
        data_root=cfg["data_root"],
        split="val",
        image_size=cfg["image_size"],
        uncertainty_policy=cfg["uncertainty_policy"],
        frontal_only=cfg["frontal_only"],
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=False,
    )

    # --- model ---
    model = CheXpertModel(
        num_classes=cfg["num_classes"],
        pretrained=False,
        use_attention=cfg["use_attention"],
    ).to(device)

    ckpt = torch.load(ckpt_dir / "best_model.pth", map_location=device)
    model.load_state_dict(ckpt["model_state"])
    print(f"  Loaded checkpoint — epoch {ckpt['epoch']}, "
          f"best AUC {ckpt['best_auc']:.4f}")

    # --- inference ---
    print(f"\n  Running inference on {len(val_ds)} validation samples...")
    labels_np, probs_np = run_inference(model, val_loader, device)
    print(f"  Done. Labels shape: {labels_np.shape}, Probs shape: {probs_np.shape}")

    # --- metrics ---
    print(f"\n  Computing metrics...")
    results, thresholds = compute_all_metrics(labels_np, probs_np)

    # save thresholds
    thresh_path = out_dir / "thresholds.json"
    with open(thresh_path, "w") as f:
        json.dump(thresholds, f, indent=2)
    print(f"  [Saved] Thresholds → {thresh_path}")

    # save metrics CSV
    metrics_df = pd.DataFrame(results)
    csv_path = out_dir / "metrics.csv"
    metrics_df.to_csv(csv_path, index=False)
    print(f"  [Saved] Metrics CSV → {csv_path}")

    # --- plots ---
    print(f"\n  Generating plots...")
    plot_auroc_curves(labels_np, probs_np, out_dir)
    plot_metrics_bar(results, out_dir)
    plot_confusion_matrices(labels_np, probs_np, thresholds, out_dir)

    # --- clinical report ---
    mean_auc = generate_clinical_report(results, thresholds, labels_np, out_dir, cfg)

    print(f"\n  Evaluation complete ✅")
    print(f"  Mean competition AUC: {mean_auc:.4f}")


if __name__ == "__main__":
    evaluate()