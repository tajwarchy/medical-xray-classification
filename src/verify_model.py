import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import time
import yaml
from src.model import CheXpertModel


def verify(config_path: str = "configs/config.yaml"):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\n{'='*55}")
    print(f"  Model Verification")
    print(f"{'='*55}")
    print(f"  Device : {device}")

    # --- build model ---
    model = CheXpertModel(
        num_classes=cfg["num_classes"],
        pretrained=cfg["pretrained"],
        use_attention=cfg["use_attention"],
    ).to(device)

    total  = sum(p.numel() for p in model.parameters())
    print(f"  Total params : {total:,}")

    # --- test frozen phase ---
    print(f"\n  [Phase A] Frozen backbone:")
    model.freeze_backbone()

    # --- dummy forward pass ---
    B = 4
    dummy = torch.randn(B, 3, cfg["image_size"], cfg["image_size"]).to(device)

    print(f"\n  Running forward pass (batch={B}, size={cfg['image_size']})...")
    t0 = time.time()
    with torch.no_grad():
        logits, attn = model(dummy)
    elapsed = time.time() - t0

    print(f"  Forward pass time  : {elapsed*1000:.1f}ms")
    print(f"  Logits shape       : {logits.shape}")       # (4, 14)
    print(f"  Logits range       : {logits.min():.3f} / {logits.max():.3f}")
    if attn is not None:
        print(f"  Attention shape    : {attn.shape}")     # (4, 1024)
        print(f"  Attention range    : {attn.min():.3f} / {attn.max():.3f}")

    # --- test unfrozen phase ---
    print(f"\n  [Phase B] Unfrozen backbone:")
    model.unfreeze_backbone()

    # --- test with loss ---
    print(f"\n  Testing loss computation...")
    from src.utils import compute_pos_weights
    pos_weights = compute_pos_weights(
        csv_path=f"{cfg['data_root']}/train.csv",
        uncertainty_policy=cfg["uncertainty_policy"]
    ).to(device)

    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weights)
    dummy_labels = torch.randint(0, 2, (B, cfg["num_classes"])).float().to(device)

    model.train()
    logits, _ = model(dummy)
    loss = criterion(logits, dummy_labels)
    print(f"  Loss value         : {loss.item():.4f}")
    print(f"  Loss is finite     : {torch.isfinite(loss).item()}")

    # --- inference speed benchmark ---
    print(f"\n  Benchmarking inference speed (10 batches, batch=8)...")
    model.eval()
    dummy_bench = torch.randn(8, 3, cfg["image_size"], cfg["image_size"]).to(device)
    times = []
    with torch.no_grad():
        for _ in range(10):
            t0 = time.time()
            model(dummy_bench)
            times.append(time.time() - t0)
    avg_ms = sum(times) / len(times) * 1000
    print(f"  Avg inference time : {avg_ms:.1f}ms/batch  "
          f"({8/(avg_ms/1000):.1f} images/sec)")

    print(f"\n  Model verification complete ✅")


if __name__ == "__main__":
    verify()