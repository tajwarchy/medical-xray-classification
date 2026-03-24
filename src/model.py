import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import DenseNet121_Weights


class ChannelAttention(nn.Module):
    """
    Lightweight channel attention block (SE-style).
    Learns to re-weight feature channels — helps focus on
    diagnostically relevant features and enables attention visualization.
    """
    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        mid = max(in_channels // reduction, 8)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, in_channels, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.shape
        avg = self.fc(self.avg_pool(x).view(b, c))
        mx  = self.fc(self.max_pool(x).view(b, c))
        attn = self.sigmoid(avg + mx).view(b, c, 1, 1)
        return x * attn, attn.squeeze(-1).squeeze(-1)  # (B,C,H,W), (B,C)


class CheXpertModel(nn.Module):
    """
    DenseNet-121 backbone with optional channel attention and
    a multi-label classification head (14 pathologies).

    forward() returns:
        logits      : (B, 14)  — raw scores, no sigmoid
        attn_weights: (B, C)   — channel attention weights (None if use_attention=False)
    """
    def __init__(
        self,
        num_classes: int = 14,
        pretrained: bool = True,
        use_attention: bool = True,
    ):
        super().__init__()
        self.use_attention = use_attention

        # --- backbone ---
        weights = DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.densenet121(weights=weights)

        # DenseNet-121 feature extractor (everything except classifier)
        self.features = backbone.features   # output: (B, 1024, H/32, W/32)

        # --- optional channel attention ---
        if use_attention:
            self.attention = ChannelAttention(in_channels=1024, reduction=16)
        else:
            self.attention = None

        # --- classifier head ---
        self.relu      = nn.ReLU(inplace=True)
        self.gap       = nn.AdaptiveAvgPool2d(1)   # global average pool
        self.dropout   = nn.Dropout(p=0.5)
        self.classifier = nn.Linear(1024, num_classes)

        # --- init classifier head ---
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, x):
        # extract features
        feat = self.features(x)          # (B, 1024, H, W)
        feat = self.relu(feat)

        attn_weights = None
        if self.use_attention and self.attention is not None:
            feat, attn_weights = self.attention(feat)   # (B,1024,H,W), (B,1024)

        # global average pool + flatten
        pooled = self.gap(feat)          # (B, 1024, 1, 1)
        pooled = pooled.view(pooled.size(0), -1)   # (B, 1024)
        pooled = self.dropout(pooled)

        logits = self.classifier(pooled) # (B, 14)
        return logits, attn_weights

    def freeze_backbone(self):
        """Freeze all layers except attention + classifier (Phase A training)."""
        for param in self.features.parameters():
            param.requires_grad = False
        print("  Backbone frozen. Training: attention + classifier only.")
        self._print_trainable()

    def unfreeze_backbone(self):
        """Unfreeze all layers (Phase B training)."""
        for param in self.features.parameters():
            param.requires_grad = True
        print("  Backbone unfrozen. Training: all layers.")
        self._print_trainable()

    def _print_trainable(self):
        total  = sum(p.numel() for p in self.parameters())
        active = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  Trainable params: {active:,} / {total:,} "
              f"({100*active/total:.1f}%)")