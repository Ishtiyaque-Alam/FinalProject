# USCNet for NSCLC-Radiomics Histology Classification

Transformer-based multimodal fusion model with segmentation guidance, adapted from USCNet for NSCLC-Radiomics histology classification (large cell, squamous cell carcinoma, adenocarcinoma, NOS).

## Architecture

- **Visual Transformation**: 3D Patch Embedding + ViT Encoder (pretrained ViT-B/16 weights)
- **Textual Transformation**: EHR clinical features projected to transformer embedding space
- **ViT-UNetSeg**: Multi-scale CNN decoder for tumor segmentation (auxiliary task)
- **MSAF Fusion**: CT-EHR Attention (CEA) + Segmentation-Multimodal Attention (SMA) via cross-attention
- **Classification Head**: Fused multimodal features → 4-class histology prediction

## Setup

```bash
pip install -r requirements.txt
```

## Data

1. Place `NSCLC-Radiomics-Lung1.clinical-version3-Oct-2019.csv` (clinical/EHR data) in the project root.
2. Place `phase1_metadata (1).csv` (CT and segmentation paths) in the project root.
3. Ensure the NSCLC-Radiomics DICOM data is available at the Kaggle path:
   `/kaggle/input/datasets/umutkrdrms/nsclc-radiomics/NSCLC-Radiomics/`

## Training

### Single GPU

```bash
python train.py --batch_size 4 --epochs 50 --lr 1e-4
```

### Multi-GPU (DDP)

```bash
torchrun --nproc_per_node=4 train.py --batch_size 4 --epochs 50 --lr 1e-4
```

### All Arguments

| Argument | Default | Description |
|---|---|---|
| `--batch_size` | 4 | Batch size per GPU |
| `--epochs` | 50 | Total training epochs |
| `--lr` | 1e-4 | Initial learning rate |
| `--weight_decay` | 1e-5 | Weight decay |
| `--drop_rate` | 0.1 | Dropout rate |
| `--freeze_epochs` | 5 | Epochs to keep backbone frozen |
| `--volume_depth` | 64 | CT volume depth |
| `--volume_height` | 128 | CT volume height |
| `--volume_width` | 128 | CT volume width |
| `--val_split` | 0.2 | Validation split ratio |
| `--num_workers` | 4 | DataLoader workers |
| `--checkpoint_dir` | checkpoints | Save directory |
| `--resume` | None | Path to checkpoint to resume |
| `--amp` | True | Mixed precision training |

## Training Strategy

- **Freeze/Unfreeze**: ViT backbone weights are frozen for the first 5 epochs (configurable via `--freeze_epochs`), then unfrozen for full fine-tuning with a reduced learning rate (0.01x).
- **Loss**: Combined Dice (segmentation) + CrossEntropy + Focal (classification) with dynamic weight adjustment.
- **Scheduler**: CosineAnnealingWarmRestarts.
- **Checkpointing**: Best model saved based on validation macro F1 score.

## Project Structure

```
UGCNet/
├── model.py        # USCNet architecture (VTT, ViT-UNetSeg, MSAF, ClassificationHead)
├── dataset.py      # NSCLC-Radiomics dataset loader (DICOM + EHR)
├── train.py        # Training script with DDP, argparse, freeze/unfreeze
├── utils.py        # Losses (Dice, Focal, combined), metrics, dynamic weight adjuster
├── requirements.txt
└── README.md
```
