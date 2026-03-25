# Project 5.3A — Medical X-ray Classification (CheXpert)

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange)
![AUC](https://img.shields.io/badge/Mean%20AUC-0.8730-brightgreen)
![License](https://img.shields.io/badge/License-MIT-green)

A production-ready chest X-ray multi-label classifier built on DenseNet-121
with channel attention, Grad-CAM explainability, and clinical metric evaluation.
Trained on the CheXpert dataset (Stanford) across 14 pathology labels.

---

## Results

| Label            | AUC   | Sensitivity | Specificity |
|------------------|-------|-------------|-------------|
| Pleural Effusion | 0.911 | 0.875       | 0.804       |
| Edema            | 0.901 | 0.833       | 0.875       |
| Consolidation    | 0.878 | 0.969       | 0.629       |
| Cardiomegaly     | 0.852 | 0.864       | 0.728       |
| Atelectasis      | 0.823 | 0.853       | 0.654       |
| **Mean (competition)** | **0.873** | | |

Target: ≥ 0.85 AUC ✅

---

## Key Features

- **Multi-label classification** — 14 pathology labels simultaneously
- **Channel attention** — SE-style attention block for feature re-weighting
- **Grad-CAM explainability** — highlights regions driving each prediction
- **Clinical metrics** — AUC, sensitivity, specificity, F1 at optimal threshold
- **Uncertainty label handling** — U-Ones policy (CheXpert paper default)
- **Class imbalance** — per-label weighted BCEWithLogitsLoss
- **Two-phase training** — frozen backbone → full fine-tuning

---

## Tech Stack

- **Framework:** PyTorch
- **Model:** DenseNet-121 (ImageNet pretrained)
- **Augmentation:** Albumentations
- **Explainability:** pytorch-grad-cam
- **Metrics:** scikit-learn
- **Dataset:** CheXpert-v1.0 (Stanford AIMI)

---

## Project Structure
```
medical-xray-classification/
├── src/
│   ├── dataset.py          # CheXpertDataset, transforms, uncertainty policy
│   ├── model.py            # DenseNet-121 + ChannelAttention
│   ├── train.py            # Two-phase training pipeline
│   ├── evaluate.py         # Clinical metrics, AUROC curves
│   ├── explainability.py   # Grad-CAM, attention maps, comparison grid
│   └── utils.py            # EDA, class weight computation
├── configs/
│   └── config.yaml         # All hyperparameters
├── outputs/
│   ├── auroc_curves.png
│   ├── metrics_bar.png
│   ├── confusion_matrices.png
│   ├── training_curves.png
│   ├── saliency_maps/
│   │   └── comparison_grid.png
│   └── reports/
│       └── evaluation_report.txt
├── main.py                 # Inference CLI
└── requirements.txt
```

---

## Setup

### 1. Clone & create environment
```bash
git clone https://github.com/tajwarchy/medical-xray-classification.git
cd medical-xray-classification
conda create -n medical-imaging python=3.10 -y
conda activate medical-imaging
pip install -r requirements.txt
```

### 2. Download CheXpert dataset
Register at [Kaggle](https://www.kaggle.com/datasets/ashery/chexpert) and download
`archive.zip`, then:
```bash
mkdir -p data/chexpert
mv ~/Downloads/archive.zip data/chexpert/
unzip data/chexpert/archive.zip -d data/chexpert/
rm data/chexpert/archive.zip
```

### 3. Verify dataset
```bash
python src/utils.py --inspect --data_root data/chexpert
```

---

## Training
```bash
# Full training (Phase A + Phase B)
python src/train.py

# Resume Phase B from checkpoint
# Set resume_phase_b = True in src/train.py
python src/train.py
```

Training config is in `configs/config.yaml`:
```yaml
image_size: 160
batch_size: 64
epochs_frozen: 3
epochs_finetune: 10
uncertainty_policy: "ones"
frontal_only: true
train_fraction: 0.2
```

---

## Evaluation
```bash
python src/evaluate.py
```

Outputs:
- `outputs/auroc_curves.png` — AUROC curves for all 14 labels
- `outputs/metrics_bar.png` — AUC, sensitivity, specificity per label
- `outputs/confusion_matrices.png` — confusion matrices at optimal threshold
- `outputs/reports/evaluation_report.txt` — full clinical report

---

## Inference
```bash
# Single image
python main.py --image path/to/xray.jpg

# Batch inference
python main.py --batch_dir path/to/folder/
```

Example output:
```
[FINDING] Pleural Effusion    87.3%  [█████████████████░░░]
[NORMAL]  Cardiomegaly        10.9%  [██░░░░░░░░░░░░░░░░░░]
[NORMAL]  Edema                6.1%  [█░░░░░░░░░░░░░░░░░░░]
```

---

## Explainability
```bash
python src/explainability.py
```

Generates:
- Individual Grad-CAM overlays for 10 training samples
- `outputs/saliency_maps/comparison_grid.png` — healthy vs diseased comparison

---

## Limitations

- Trained on frontal views only — lateral X-rays will produce noisier predictions
- Validation set is small (202 samples) — metrics may vary slightly
- Trained on 20% of CheXpert for compute reasons — full dataset would improve AUC
- Not intended for clinical use

---

## License
MIT