import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import numpy as np
import cv2
import yaml
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
from torch.utils.data import DataLoader

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from src.dataset import CheXpertDataset, PATHOLOGY_COLS, COMPETITION_LABELS
from src.model import CheXpertModel


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(cfg: dict, device):
    """Load best checkpoint."""
    model = CheXpertModel(
        num_classes=cfg["num_classes"],
        pretrained=False,
        use_attention=cfg["use_attention"],
    ).to(device)
    ckpt_path = Path(cfg["checkpoint_dir"]) / "best_model.pth"
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"  Loaded checkpoint — AUC {ckpt['best_auc']:.4f}")
    return model


def get_gradcam_target_layer(model):
    """
    Hook into the last DenseNet dense block.
    This is the deepest semantic feature map before pooling.
    """
    return [model.features.denseblock4.denselayer16.conv2]


# ── Grad-CAM generation ───────────────────────────────────────────────────────

def generate_gradcam(model, image_tensor, target_label_idx, device):
    """
    Generate Grad-CAM heatmap for a single image and target label.

    Args:
        model           : CheXpertModel (eval mode)
        image_tensor    : (1, 3, H, W) normalized tensor on CPU
        target_label_idx: int, index into PATHOLOGY_COLS
        device          : torch device

    Returns:
        heatmap_np : (H, W) float32 in [0, 1]
    """
    target_layers = get_gradcam_target_layer(model)

    # GradCAM needs CPU or CUDA — MPS works via cpu fallback
    cam = GradCAM(model=model, target_layers=target_layers)
    targets = [ClassifierOutputTarget(target_label_idx)]

    img = image_tensor.to(device)
    heatmap = cam(input_tensor=img, targets=targets)   # (1, H, W)
    return heatmap[0]   # (H, W)


def overlay_heatmap(image_np, heatmap_np, alpha=0.45):
    """
    Overlay Grad-CAM heatmap on original X-ray image.

    Args:
        image_np  : (H, W, 3) uint8 RGB image
        heatmap_np: (H, W) float32 in [0, 1]
        alpha     : blending factor

    Returns:
        overlay: (H, W, 3) uint8
    """
    heatmap_resized = cv2.resize(heatmap_np, (image_np.shape[1], image_np.shape[0]))
    heatmap_uint8   = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_rgb     = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    overlay = (alpha * heatmap_rgb + (1 - alpha) * image_np).astype(np.uint8)
    return overlay


def denormalize(tensor):
    """Convert normalized tensor back to uint8 RGB numpy array."""
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    img  = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    img  = std * img + mean
    img  = np.clip(img, 0, 1)
    return (img * 255).astype(np.uint8)


# ── attention map visualization ───────────────────────────────────────────────

@torch.no_grad()
def get_attention_map(model, image_tensor, device):
    """
    Extract channel attention weights and produce a spatial heatmap.
    Returns (H, W) float32 in [0, 1].
    """
    model.eval()
    img = image_tensor.to(device)

    # forward pass — get features before attention
    feat = model.features(img)
    feat = model.relu(feat)

    if model.attention is None:
        return None

    _, attn_weights = model.attention(feat)   # (1, 1024)
    # top-k channel selection → project back to spatial
    top_k = 64
    top_channels = attn_weights[0].topk(top_k).indices.cpu().numpy()
    feat_np = feat[0].detach().cpu().numpy()   # (1024, H, W)
    spatial_map = feat_np[top_channels].mean(axis=0)   # (H, W)
    spatial_map = (spatial_map - spatial_map.min()) / (spatial_map.max() - spatial_map.min() + 1e-8)
    return spatial_map.astype(np.float32)


# ── batch saliency generation ─────────────────────────────────────────────────

def generate_batch_saliency(model, dataset, device, cfg, n_samples=10):
    """
    For n_samples images: run inference → get top predicted label →
    generate Grad-CAM → save overlay.
    """
    out_dir = Path(cfg["output_dir"]) / "saliency_maps"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  Generating saliency maps for {n_samples} samples...")

    for idx in range(n_samples):
        image_tensor, label_tensor = dataset[idx]
        image_tensor = image_tensor.unsqueeze(0)   # (1, 3, H, W)

        # inference
        with torch.no_grad():
            logits, _ = model(image_tensor.to(device))
            probs = torch.sigmoid(logits)[0].cpu().numpy()

        top_label_idx = int(probs.argmax())
        top_label     = PATHOLOGY_COLS[top_label_idx]
        confidence    = probs[top_label_idx]

        # grad-cam
        heatmap = generate_gradcam(model, image_tensor, top_label_idx, device)
        image_np = denormalize(image_tensor)
        overlay  = overlay_heatmap(image_np, heatmap)

        # save
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        fig.suptitle(f"Sample {idx} | Prediction: {top_label} ({confidence:.1%})",
                     fontsize=11)

        axes[0].imshow(image_np, cmap="gray" if image_np.mean(axis=2).std() < 30 else None)
        axes[0].set_title("Original X-ray")
        axes[0].axis("off")

        axes[1].imshow(heatmap, cmap="jet")
        axes[1].set_title("Grad-CAM Heatmap")
        axes[1].axis("off")

        axes[2].imshow(overlay)
        axes[2].set_title("Overlay")
        axes[2].axis("off")

        plt.tight_layout()
        save_path = out_dir / f"sample_{idx:03d}_{top_label.replace(' ', '_')}.png"
        plt.savefig(save_path, dpi=120)
        plt.close()
        print(f"  [Saved] {save_path.name}")


# ── healthy vs diseased comparison grid ──────────────────────────────────────

def generate_comparison_grid(model, dataset, device, cfg):
    """
    For each competition label: find one positive and one negative sample,
    generate Grad-CAM for both, save a comparison grid.
    Portfolio key asset.
    """
    out_dir = Path(cfg["output_dir"]) / "saliency_maps"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  Generating healthy vs diseased comparison grid...")

    df = dataset.df
    fig, axes = plt.subplots(
        len(COMPETITION_LABELS), 4,
        figsize=(16, 4 * len(COMPETITION_LABELS))
    )
    fig.suptitle("Healthy vs Diseased — Grad-CAM Comparison", fontsize=14, y=1.01)

    col_titles = ["Healthy (Original)", "Healthy (Grad-CAM)",
                  "Diseased (Original)", "Diseased (Grad-CAM)"]
    for ax, title in zip(axes[0], col_titles):
        ax.set_title(title, fontsize=10, fontweight="bold")

    for row_idx, label in enumerate(COMPETITION_LABELS):
        label_col_idx = PATHOLOGY_COLS.index(label)

        # find one healthy (label=0) and one diseased (label=1)
        healthy_rows  = df[df[label] == 0.0].index.tolist()
        diseased_rows = df[df[label] == 1.0].index.tolist()

        if not healthy_rows or not diseased_rows:
            for ax in axes[row_idx]:
                ax.axis("off")
            continue

        healthy_idx  = healthy_rows[0]
        diseased_idx = diseased_rows[0]

        for col_offset, sample_idx in enumerate([healthy_idx, diseased_idx]):
            image_tensor, _ = dataset[sample_idx]
            image_tensor = image_tensor.unsqueeze(0)
            heatmap  = generate_gradcam(model, image_tensor, label_col_idx, device)
            image_np = denormalize(image_tensor)
            overlay  = overlay_heatmap(image_np, heatmap)

            ax_orig = axes[row_idx][col_offset * 2]
            ax_cam  = axes[row_idx][col_offset * 2 + 1]

            gray = image_np.mean(axis=2)
            ax_orig.imshow(gray, cmap="gray")
            ax_orig.set_ylabel(label, fontsize=9, rotation=90, labelpad=4)
            ax_orig.axis("off")

            ax_cam.imshow(overlay)
            ax_cam.axis("off")

    plt.tight_layout()
    save_path = out_dir / "comparison_grid.png"
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  [Saved] Comparison grid → {save_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def run_explainability(config_path: str = "configs/config.yaml"):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device = get_device()
    print(f"\n{'='*55}")
    print(f"  Explainability — Grad-CAM & Attention Maps")
    print(f"{'='*55}")
    print(f"  Device: {device}")

    model = load_model(cfg, device)

    # dataset (use train set for more variety)
    train_ds = CheXpertDataset(
        csv_path=f"{cfg['data_root']}/train.csv",
        data_root=cfg["data_root"],
        split="val",                    # val transforms (no augmentation)
        image_size=cfg["image_size"],
        uncertainty_policy=cfg["uncertainty_policy"],
        frontal_only=cfg["frontal_only"],
    )

    # batch saliency maps
    generate_batch_saliency(model, train_ds, device, cfg, n_samples=10)

    # healthy vs diseased comparison grid
    generate_comparison_grid(model, train_ds, device, cfg)

    print(f"\n  Explainability complete ✅")
    print(f"  Outputs saved to outputs/saliency_maps/")


if __name__ == "__main__":
    run_explainability()