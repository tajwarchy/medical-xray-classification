import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import yaml
import time
import json
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from src.dataset import CheXpertDataset, PATHOLOGY_COLS, COMPETITION_LABELS
from src.model import CheXpertModel
from src.utils import compute_pos_weights


# ── helpers ───────────────────────────────────────────────────────────────────

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def compute_auc(labels_np: np.ndarray, logits_np: np.ndarray):
    """
    Compute per-label AUC and mean AUC over competition labels.
    Returns (per_label_auc dict, mean_competition_auc float).
    Skips labels with only one class present in labels.
    """
    probs = 1 / (1 + np.exp(-logits_np))   # sigmoid
    per_auc = {}
    for i, col in enumerate(PATHOLOGY_COLS):
        y = labels_np[:, i]
        if len(np.unique(y)) < 2:
            per_auc[col] = float("nan")
            continue
        try:
            per_auc[col] = roc_auc_score(y, probs[:, i])
        except Exception:
            per_auc[col] = float("nan")

    comp_aucs = [per_auc[c] for c in COMPETITION_LABELS
                 if not np.isnan(per_auc[c])]
    mean_auc = float(np.mean(comp_aucs)) if comp_aucs else float("nan")
    return per_auc, mean_auc


def save_checkpoint(model, optimizer, epoch, best_auc, path):
    torch.save({
        "epoch"     : epoch,
        "best_auc"  : best_auc,
        "model_state": model.state_dict(),
        "optim_state": optimizer.state_dict(),
    }, path)


def load_checkpoint(model, optimizer, path, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optim_state"])
    return ckpt["epoch"], ckpt["best_auc"]


# ── evaluation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_labels, all_logits = [], []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits, _ = model(images)
        loss = criterion(logits, labels)
        total_loss += loss.item()
        all_labels.append(labels.cpu().numpy())
        all_logits.append(logits.cpu().numpy())

    all_labels = np.concatenate(all_labels, axis=0)
    all_logits = np.concatenate(all_logits, axis=0)
    per_auc, mean_auc = compute_auc(all_labels, all_logits)
    avg_loss = total_loss / len(loader)
    return avg_loss, per_auc, mean_auc


# ── training phase ────────────────────────────────────────────────────────────

def run_phase(
    phase_name: str,
    model, optimizer, scheduler, criterion,
    train_loader, val_loader,
    epochs: int, device,
    checkpoint_dir: Path,
    history: dict,
    best_auc: float,
    early_stop_patience: int = 5,
):
    no_improve = 0
    print(f"\n{'='*60}")
    print(f"  {phase_name}")
    print(f"{'='*60}")

    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        train_loss = 0.0
        n_batches = len(train_loader)

        for batch_idx, (images, labels) in enumerate(train_loader, 1):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits, _ = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            if batch_idx % 50 == 0 or batch_idx == n_batches:
                elapsed = time.time() - t0
                batches_per_sec = batch_idx / elapsed
                eta_sec = (n_batches - batch_idx) / batches_per_sec
                print(f"  [{phase_name}] Epoch {epoch}/{epochs} "
                    f"| Batch {batch_idx}/{n_batches} "
                    f"| Loss {train_loss/batch_idx:.4f} "
                    f"| {batches_per_sec:.1f} b/s "
                    f"| ETA {eta_sec/60:.1f}min")

        avg_train_loss = train_loss / n_batches
        val_loss, per_auc, mean_auc = evaluate(
            model, val_loader, criterion, device
        )
        scheduler.step()
        epoch_time = time.time() - t0

        # log
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(val_loss)
        history["mean_auc"].append(mean_auc)

        print(f"\n  ── Epoch {epoch}/{epochs} Summary ──")
        print(f"  Train loss : {avg_train_loss:.4f}")
        print(f"  Val loss   : {val_loss:.4f}")
        print(f"  Mean AUC (competition labels) : {mean_auc:.4f}")
        print(f"  Competition label AUCs:")
        for lbl in COMPETITION_LABELS:
            auc_val = per_auc.get(lbl, float("nan"))
            marker = " ◄" if not np.isnan(auc_val) and auc_val >= 0.85 else ""
            print(f"    {lbl:<25} {auc_val:.4f}{marker}")
        print(f"  Epoch time : {epoch_time/60:.1f}min")

        # checkpoint
        save_checkpoint(
            model, optimizer, epoch, mean_auc,
            checkpoint_dir / "last_model.pth"
        )
        if mean_auc > best_auc:
            best_auc = mean_auc
            no_improve = 0
            save_checkpoint(
                model, optimizer, epoch, best_auc,
                checkpoint_dir / "best_model.pth"
            )
            print(f"  ✅ New best AUC: {best_auc:.4f} — checkpoint saved")
        else:
            no_improve += 1
            print(f"  No improvement ({no_improve}/{early_stop_patience})")
            if no_improve >= early_stop_patience:
                print(f"  Early stopping triggered.")
                break

    return best_auc


# ── main ──────────────────────────────────────────────────────────────────────

def train(config_path: str = "configs/config.yaml"):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device = get_device()
    print(f"\n{'='*60}")
    print(f"  CheXpert Training — Device: {device}")
    print(f"{'='*60}")

    # directories
    ckpt_dir = Path(cfg["checkpoint_dir"])
    out_dir  = Path(cfg["output_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    # datasets
    data_root = cfg["data_root"]
    train_ds = CheXpertDataset(
        csv_path=f"{data_root}/train.csv",
        data_root=data_root,
        split="train",
        image_size=cfg["image_size"],
        uncertainty_policy=cfg["uncertainty_policy"],
        frontal_only=cfg["frontal_only"],
    )
    val_ds = CheXpertDataset(
        csv_path=f"{data_root}/valid.csv",
        data_root=data_root,
        split="val",
        image_size=cfg["image_size"],
        uncertainty_policy=cfg["uncertainty_policy"],
        frontal_only=cfg["frontal_only"],
    )
    
    if cfg.get("train_fraction", 1.0) < 1.0:
        n = int(len(train_ds) * cfg["train_fraction"])
        indices = torch.randperm(len(train_ds))[:n].tolist()
        from torch.utils.data import Subset
        train_ds = Subset(train_ds, indices)
        print(f"  Subsampled train set: {n:,} samples ({cfg['train_fraction']*100:.0f}%)")

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=False,
    )

    # model
    model = CheXpertModel(
        num_classes=cfg["num_classes"],
        pretrained=cfg["pretrained"],
        use_attention=cfg["use_attention"],
    ).to(device)

    # loss
    pos_weights = compute_pos_weights(
        csv_path=f"{data_root}/train.csv",
        uncertainty_policy=cfg["uncertainty_policy"],
    ).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)

    # history
    history = {"train_loss": [], "val_loss": [], "mean_auc": []}
    best_auc = 0.0

    resume_phase_b = True   # set to False to train from scratch

    if not resume_phase_b:
        # ── Phase A: frozen backbone ──────────────────────────────────────
        model.freeze_backbone()
        optimizer_a = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg["lr_frozen"],
            weight_decay=cfg["weight_decay"],
        )
        scheduler_a = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer_a, T_max=cfg["epochs_frozen"]
        )
        best_auc = run_phase(
            phase_name="Phase A — Frozen Backbone",
            model=model,
            optimizer=optimizer_a,
            scheduler=scheduler_a,
            criterion=criterion,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=cfg["epochs_frozen"],
            device=device,
            checkpoint_dir=ckpt_dir,
            history=history,
            best_auc=best_auc,
        )

    # ── load best checkpoint before Phase B ──────────────────────────────
    ckpt_path = ckpt_dir / "best_model.pth"
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        best_auc = ckpt["best_auc"]
        print(f"\n  Resumed from checkpoint — best AUC so far: {best_auc:.4f}")
    else:
        print("\n  No checkpoint found, starting Phase B from scratch.")

    # ── Phase B: full fine-tuning ─────────────────────────────────────────
    model.unfreeze_backbone()
    optimizer_b = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["lr_finetune"],
        weight_decay=cfg["weight_decay"],
    )
    scheduler_b = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_b, T_max=cfg["epochs_finetune"]
    )
    best_auc = run_phase(
        phase_name="Phase B — Full Fine-tuning",
        model=model,
        optimizer=optimizer_b,
        scheduler=scheduler_b,
        criterion=criterion,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=7,              # 10 - 3 already done = 7 remaining
        device=device,
        checkpoint_dir=ckpt_dir,
        history=history,
        best_auc=best_auc,
    )

    # ── save history & plot ───────────────────────────────────────────────────
    history_path = out_dir / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\n  [Saved] Training history → {history_path}")

    _plot_training_curves(history, out_dir)
    print(f"\n  Training complete. Best competition AUC: {best_auc:.4f}")


def _plot_training_curves(history: dict, out_dir: Path):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, history["train_loss"], label="Train Loss", marker="o", markersize=3)
    axes[0].plot(epochs, history["val_loss"],   label="Val Loss",   marker="o", markersize=3)
    axes[0].set_title("Loss Curves")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history["mean_auc"], label="Mean AUC (competition)",
                 marker="o", markersize=3, color="green")
    axes[1].axhline(y=0.85, color="red", linestyle="--", alpha=0.7, label="Target 0.85")
    axes[1].set_title("Validation AUC")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("AUC")
    axes[1].set_ylim(0, 1)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # mark phase boundary
    n_frozen = None
    try:
        import yaml
        with open("configs/config.yaml") as f:
            cfg = yaml.safe_load(f)
        n_frozen = cfg.get("epochs_frozen", None)
    except Exception:
        pass
    if n_frozen:
        for ax in axes:
            ax.axvline(x=n_frozen + 0.5, color="orange",
                       linestyle="--", alpha=0.6, label="Phase A→B")

    plt.tight_layout()
    save_path = out_dir / "training_curves.png"
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  [Saved] Training curves → {save_path}")


if __name__ == "__main__":
    train()