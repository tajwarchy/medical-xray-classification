import torch
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


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


def get_transforms(split: str, image_size: int = 320):
    """
    Returns Albumentations transform pipeline.
    split: 'train' or 'val'
    """
    mean = [0.485, 0.456, 0.406]  # ImageNet stats
    std  = [0.229, 0.224, 0.225]

    if split == "train":
        return A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=10, p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.5
            ),
            A.GaussNoise(p=0.2),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(image_size, image_size),
            A.CenterCrop(image_size, image_size),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])


def apply_uncertainty_policy(val: float, policy: str) -> float:
    """
    Resolve uncertainty label (-1) to a training signal.
    policy:
      'zeros' — treat uncertain as negative (0)
      'ones'  — treat uncertain as positive (1)  [CheXpert paper default]
      'ignore'— mask out uncertain (returns -1, handled in loss)
    """
    if val == -1.0:
        if policy == "zeros":
            return 0.0
        elif policy == "ones":
            return 1.0
        else:  # ignore
            return -1.0
    return val


class CheXpertDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        data_root: str,
        split: str = "train",
        image_size: int = 320,
        uncertainty_policy: str = "ones",
        frontal_only: bool = True,
    ):
        """
        Args:
            csv_path          : path to train.csv or valid.csv
            data_root         : root folder containing the CheXpert data
            split             : 'train' or 'val'
            image_size        : resize target (square)
            uncertainty_policy: 'zeros' | 'ones' | 'ignore'
            frontal_only      : if True, keep only Frontal view images
        """
        self.data_root         = Path(data_root)
        self.split             = split
        self.uncertainty_policy = uncertainty_policy
        self.transform         = get_transforms(split, image_size)

        df = pd.read_csv(csv_path)

        # filter frontal only
        if frontal_only and "Frontal/Lateral" in df.columns:
            df = df[df["Frontal/Lateral"] == "Frontal"].reset_index(drop=True)
            print(f"  [{split}] Frontal-only filter: {len(df):,} samples retained")

        self.df = df
        self.labels = PATHOLOGY_COLS

        print(f"  [{split}] Dataset ready: {len(self.df):,} samples | "
              f"policy={uncertainty_policy} | image_size={image_size}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # --- load image ---
        # Path column in CSV: e.g. "CheXpert-v1.0-small/train/patient00001/..."
        img_path = self.data_root / Path(row["Path"]).relative_to("CheXpert-v1.0-small")
        img = Image.open(img_path).convert("RGB")
        img_np = np.array(img)

        # --- apply transforms ---
        augmented = self.transform(image=img_np)
        image_tensor = augmented["image"]           # (3, H, W) float32

        # --- build label vector ---
        label_vec = []
        for col in self.labels:
            val = row.get(col, np.nan)
            if pd.isna(val):
                val = 0.0                           # NaN → negative
            else:
                val = apply_uncertainty_policy(float(val), self.uncertainty_policy)
            label_vec.append(val)

        label_tensor = torch.tensor(label_vec, dtype=torch.float32)

        return image_tensor, label_tensor

    @property
    def num_classes(self):
        return len(self.labels)