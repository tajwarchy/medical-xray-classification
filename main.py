import sys
from pathlib import Path
import argparse
import json
import time

import torch
import numpy as np
import yaml
import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

from src.model import CheXpertModel
from src.dataset import PATHOLOGY_COLS, COMPETITION_LABELS
from src.explainability import (
    generate_gradcam, overlay_heatmap, denormalize, get_device
)


# ── preprocessing ─────────────────────────────────────────────────────────────

def preprocess_image(image_path: str, image_size: int):
    """Load and preprocess a single X-ray image."""
    img = Image.open(image_path).convert("RGB")
    img_np = np.array(img)

    transform = A.Compose([
        A.Resize(image_size, image_size),
        A.CenterCrop(image_size, image_size),
        A.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    tensor = transform(image=img_np)["image"].unsqueeze(0)  # (1,3,H,W)
    return tensor, img_np


# ── inference ─────────────────────────────────────────────────────────────────

def run_single_inference(model, image_tensor, device, thresholds):
    """
    Run inference on a single image.
    Returns list of dicts with label, prob, positive flag.
    """
    model.eval()
    with torch.no_grad():
        logits, _ = model(image_tensor.to(device))
        probs = torch.sigmoid(logits)[0].cpu().numpy()

    results = []
    for i, label in enumerate(PATHOLOGY_COLS):
        prob    = float(probs[i])
        thresh  = thresholds.get(label, 0.5)
        results.append({
            "label"     : label,
            "prob"      : prob,
            "threshold" : thresh,
            "positive"  : prob >= thresh,
            "competition": label in COMPETITION_LABELS,
        })
    return results, probs


# ── report generation ─────────────────────────────────────────────────────────

def format_report(results, image_path: str, inference_time_ms: float,
                  saliency_path: str = None):
    """Generate a formatted plain-text clinical report."""
    lines = []
    lines.append("=" * 60)
    lines.append("  CheXpert Inference Report")
    lines.append("=" * 60)
    lines.append(f"  Image          : {Path(image_path).name}")
    lines.append(f"  Inference time : {inference_time_ms:.1f}ms")
    lines.append("")
    lines.append("  ── Competition Labels ──────────────────────────────")
    for r in results:
        if not r["competition"]:
            continue
        status = "[FINDING]" if r["positive"] else "[NORMAL] "
        bar    = _prob_bar(r["prob"])
        lines.append(f"  {status} {r['label']:<28} {r['prob']:>6.1%}  {bar}")
    lines.append("")
    lines.append("  ── All Labels ──────────────────────────────────────")
    for r in results:
        status = "[+]" if r["positive"] else "[ ]"
        lines.append(f"  {status} {r['label']:<30} {r['prob']:>6.1%}  "
                     f"(thresh={r['threshold']:.3f})")
    lines.append("")
    if saliency_path:
        lines.append(f"  Saliency map   : {saliency_path}")
    lines.append("=" * 60)
    return "\n".join(lines)


def _prob_bar(prob, width=20):
    filled = int(prob * width)
    return f"[{'█' * filled}{'░' * (width - filled)}]"


# ── visualization ─────────────────────────────────────────────────────────────

def save_inference_visualization(image_tensor, image_np, probs,
                                  results, model, device,
                                  save_path: Path, image_path: str):
    """
    Save a 4-panel visualization:
    1. Original X-ray
    2. Top prediction Grad-CAM overlay
    3. Competition label probability bars
    4. All label probability bars
    """
    top_idx   = int(probs.argmax())
    top_label = PATHOLOGY_COLS[top_idx]

    heatmap = generate_gradcam(model, image_tensor, top_idx, device)
    overlay = overlay_heatmap(image_np, heatmap)

    fig = plt.figure(figsize=(16, 9))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

    # panel 1 — original
    ax1 = fig.add_subplot(gs[0, 0])
    gray = np.array(Image.open(image_path).convert("L").resize(
        (image_np.shape[1], image_np.shape[0])))
    ax1.imshow(gray, cmap="gray")
    ax1.set_title("Original X-ray", fontsize=10)
    ax1.axis("off")

    # panel 2 — grad-cam overlay
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(overlay)
    ax2.set_title(f"Grad-CAM: {top_label}\n({probs[top_idx]:.1%} confidence)",
                  fontsize=10)
    ax2.axis("off")

    # panel 3 — heatmap only
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(heatmap, cmap="jet")
    ax3.set_title("Activation Heatmap", fontsize=10)
    ax3.axis("off")

    # panel 4 — competition label bars
    ax4 = fig.add_subplot(gs[1, 0:2])
    comp_results = [r for r in results if r["competition"]]
    comp_labels  = [r["label"] for r in comp_results]
    comp_probs   = [r["prob"]  for r in comp_results]
    comp_thresh  = [r["threshold"] for r in comp_results]
    colors = ["#e74c3c" if r["positive"] else "#95a5a6" for r in comp_results]
    bars = ax4.barh(comp_labels, comp_probs, color=colors, edgecolor="white")
    for thresh, y in zip(comp_thresh, range(len(comp_labels))):
        ax4.axvline(x=thresh, color="black", linestyle="--",
                    alpha=0.4, linewidth=0.8)
    ax4.set_xlim(0, 1)
    ax4.set_title("Competition Label Probabilities", fontsize=10)
    ax4.set_xlabel("Probability")
    for bar, prob in zip(bars, comp_probs):
        ax4.text(min(prob + 0.02, 0.95), bar.get_y() + bar.get_height() / 2,
                 f"{prob:.1%}", va="center", fontsize=8)
    ax4.grid(True, alpha=0.2, axis="x")

    # panel 5 — all labels
    ax5 = fig.add_subplot(gs[1, 2])
    all_labels = [r["label"] for r in results]
    all_probs  = [r["prob"]  for r in results]
    colors_all = ["#e74c3c" if r["positive"] else "#bdc3c7" for r in results]
    ax5.barh(all_labels, all_probs, color=colors_all, edgecolor="white", height=0.6)
    ax5.set_xlim(0, 1)
    ax5.set_title("All Label Probabilities", fontsize=10)
    ax5.set_xlabel("Probability")
    ax5.tick_params(axis="y", labelsize=7)
    ax5.grid(True, alpha=0.2, axis="x")

    fig.suptitle(f"CheXpert Inference — {Path(image_path).name}",
                 fontsize=13, fontweight="bold")
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  [Saved] Visualization → {save_path}")


# ── batch inference ───────────────────────────────────────────────────────────

def batch_inference(model, image_dir: str, cfg: dict,
                    thresholds: dict, device, out_dir: Path):
    """Run inference on all images in a directory."""
    image_dir = Path(image_dir)
    images    = list(image_dir.glob("*.jpg")) + \
                list(image_dir.glob("*.png")) + \
                list(image_dir.glob("*.jpeg"))

    if not images:
        print(f"  No images found in {image_dir}")
        return

    print(f"\n  Batch inference on {len(images)} images...")
    summary = {label: 0 for label in PATHOLOGY_COLS}

    for img_path in images:
        image_tensor, image_np = preprocess_image(
            str(img_path), cfg["image_size"]
        )
        results, probs = run_single_inference(
            model, image_tensor, device, thresholds
        )
        for r in results:
            if r["positive"]:
                summary[r["label"]] += 1

        report = format_report(results, str(img_path), 0)
        report_path = out_dir / f"{img_path.stem}_report.txt"
        report_path.write_text(report)

    print(f"\n  Batch Summary — Positive counts:")
    for label, count in summary.items():
        if count > 0:
            print(f"    {label:<30} {count}/{len(images)}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="CheXpert Inference")
    parser.add_argument("--image",     type=str, help="Path to single X-ray image")
    parser.add_argument("--batch_dir", type=str, help="Directory of X-ray images")
    parser.add_argument("--config",    type=str, default="configs/config.yaml")
    parser.add_argument("--no_viz",    action="store_true",
                        help="Skip saving visualization")
    args = parser.parse_args()

    if not args.image and not args.batch_dir:
        parser.error("Provide --image or --batch_dir")

    # load config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = get_device()

    # load thresholds
    thresh_path = Path(cfg["output_dir"]) / "thresholds.json"
    with open(thresh_path) as f:
        thresholds = json.load(f)

    # load model
    model = CheXpertModel(
        num_classes=cfg["num_classes"],
        pretrained=False,
        use_attention=cfg["use_attention"],
    ).to(device)
    ckpt = torch.load(
        Path(cfg["checkpoint_dir"]) / "best_model.pth",
        map_location=device
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"\n  Model loaded — AUC {ckpt['best_auc']:.4f}")

    out_dir = Path(cfg["output_dir"]) / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── single image mode ─────────────────────────────────────────────────────
    if args.image:
        print(f"\n  Running inference on: {args.image}")

        t0 = time.time()
        image_tensor, image_np = preprocess_image(args.image, cfg["image_size"])
        results, probs = run_single_inference(
            model, image_tensor, device, thresholds
        )
        elapsed_ms = (time.time() - t0) * 1000

        # saliency map
        saliency_path = None
        if not args.no_viz:
            stem = Path(args.image).stem
            viz_path = Path(cfg["output_dir"]) / "reports" / f"{stem}_viz.png"
            save_inference_visualization(
                image_tensor, image_np, probs,
                results, model, device, viz_path, args.image
            )
            saliency_path = str(viz_path)

        # print + save report
        report = format_report(results, args.image, elapsed_ms, saliency_path)
        print("\n" + report)

        report_path = out_dir / f"{Path(args.image).stem}_report.txt"
        report_path.write_text(report)
        print(f"\n  [Saved] Report → {report_path}")

    # ── batch mode ────────────────────────────────────────────────────────────
    if args.batch_dir:
        batch_inference(model, args.batch_dir, cfg, thresholds, device, out_dir)


if __name__ == "__main__":
    main()