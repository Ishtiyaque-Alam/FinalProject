"""
Training script for USCNet on NSCLC-Radiomics dataset.

Supports:
  - Distributed Data Parallel (DDP) training
  - Argparse for batch_size, epochs, learning_rate
  - Stratified train/val split (preserves class proportions)
  - Class-weighted CE + Focal loss to handle imbalance
  - Linear LR warm-up for first 3 epochs, then Cosine Annealing
  - Freeze ViT backbone for first N epochs, then unfreeze (ClinicalBERT stays frozen)
  - Best model checkpoint saving

Usage:
  Single GPU:
    python train.py --batch_size 4 --epochs 50 --lr 1e-4

  Multi-GPU DDP:
    torchrun --nproc_per_node=NUM_GPUS train.py --batch_size 4 --epochs 50 --lr 1e-4
"""

import os
import argparse
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, Subset
from sklearn.model_selection import StratifiedShuffleSplit

try:
    from torch.amp import autocast as _autocast, GradScaler
    def autocast(enabled=True):
        return _autocast("cuda", enabled=enabled)
except ImportError:
    from torch.cuda.amp import GradScaler, autocast

from model import USCNet, build_model
from dataset import NSCLCDataset
from utils import USCNetLoss, DynamicWeightAdjuster, compute_metrics, compute_dice_score, compute_class_weights


# ---------------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------------

def collate_fn(batch):
    """Custom collate that keeps ehr_text as a list of strings."""
    ct         = torch.stack([b["ct"] for b in batch])
    seg_gt     = torch.stack([b["seg_gt"] for b in batch])
    labels     = torch.stack([b["label"] for b in batch])
    ehr_text   = [b["ehr_text"] for b in batch]
    patient_ids = [b["patient_id"] for b in batch]
    return {
        "ct":         ct,
        "seg_gt":     seg_gt,
        "ehr_text":   ehr_text,
        "label":      labels,
        "patient_id": patient_ids,
    }


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="USCNet Training for NSCLC-Radiomics")

    # Data paths
    parser.add_argument(
        "--metadata_csv", type=str,
        default="phase1_metadata (1).csv",
        help="Path to phase1_metadata CSV",
    )
    parser.add_argument(
        "--clinical_csv", type=str,
        default="NSCLC-Radiomics-Lung1.clinical-version3-Oct-2019.csv",
        help="Path to clinical data CSV",
    )

    # Training hyperparameters
    parser.add_argument("--batch_size",    type=int,   default=4,    help="Batch size per GPU")
    parser.add_argument("--epochs",        type=int,   default=50,   help="Number of training epochs")
    parser.add_argument("--lr",            type=float, default=1e-4, help="Initial learning rate")
    parser.add_argument("--weight_decay",  type=float, default=1e-5, help="Weight decay")
    parser.add_argument("--drop_rate",     type=float, default=0.1,  help="Dropout rate")
    parser.add_argument("--label_smoothing", type=float, default=0.1, help="Label smoothing epsilon")
    parser.add_argument("--warmup_epochs", type=int,   default=3,    help="Linear LR warm-up epochs")

    # Freeze settings
    parser.add_argument("--freeze_epochs", type=int, default=5, help="Epochs to freeze backbone")

    # Volume dimensions
    parser.add_argument("--volume_depth",  type=int, default=64,  help="Volume depth")
    parser.add_argument("--volume_height", type=int, default=128, help="Volume height")
    parser.add_argument("--volume_width",  type=int, default=128, help="Volume width")

    # Dataset
    parser.add_argument("--gtv_margin",  type=int,   default=10,   help="GTV crop margin in voxels (0 = disabled)")

    # Training settings
    parser.add_argument("--num_workers",     type=int,   default=4,    help="DataLoader workers")
    parser.add_argument("--val_split",       type=float, default=0.2,  help="Validation split ratio")
    parser.add_argument("--seed",            type=int,   default=42,   help="Random seed")
    parser.add_argument("--checkpoint_dir",  type=str,   default="checkpoints", help="Checkpoint directory")
    parser.add_argument("--log_interval",    type=int,   default=10,   help="Log every N steps")
    parser.add_argument("--amp",             action="store_true", default=True, help="Use mixed precision")
    parser.add_argument("--resume",          type=str,   default=None, help="Path to resume checkpoint")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# DDP helpers
# ---------------------------------------------------------------------------

def setup_ddp():
    """Initialize DDP if launched via torchrun."""
    if "RANK" in os.environ:
        rank       = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(local_rank)
        return rank, local_rank, world_size
    return 0, 0, 1


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank):
    return rank == 0


def log(msg, rank=0):
    if is_main_process(rank):
        print(msg, flush=True)


# ---------------------------------------------------------------------------
# Stratified split
# ---------------------------------------------------------------------------

def stratified_split(dataset: NSCLCDataset, val_ratio: float, seed: int):
    """
    Return (train_indices, val_indices) preserving class proportions.
    Falls back to random split if stratification fails (e.g., too few samples).
    """
    labels = dataset.get_labels()
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
    try:
        train_idx, val_idx = next(splitter.split(np.zeros(len(labels)), labels))
        return train_idx.tolist(), val_idx.tolist()
    except ValueError:
        # Not enough samples per class — fall back
        n = len(labels)
        val_n = int(n * val_ratio)
        perm = np.random.permutation(n)
        return perm[val_n:].tolist(), perm[:val_n].tolist()


# ---------------------------------------------------------------------------
# LR warm-up scheduler wrapper
# ---------------------------------------------------------------------------

class LinearWarmupCosineScheduler:
    """
    Linear warm-up for `warmup_epochs` then hand off to a CosineAnnealingWarmRestarts.
    """

    def __init__(self, optimizer, warmup_epochs: int, base_lr: float,
                 cosine_scheduler):
        self.optimizer       = optimizer
        self.warmup_epochs   = warmup_epochs
        self.base_lr         = base_lr
        self.cosine_scheduler = cosine_scheduler
        self._epoch          = 0

    def step(self):
        self._epoch += 1
        if self._epoch <= self.warmup_epochs:
            scale = self._epoch / max(self.warmup_epochs, 1)
            for pg in self.optimizer.param_groups:
                pg["lr"] = pg.get("_base_lr", self.base_lr) * scale
        else:
            self.cosine_scheduler.step()

    def state_dict(self):
        return {
            "_epoch": self._epoch,
            "cosine": self.cosine_scheduler.state_dict(),
        }

    def load_state_dict(self, sd):
        self._epoch = sd["_epoch"]
        self.cosine_scheduler.load_state_dict(sd["cosine"])


def _tag_base_lrs(optimizer):
    """Store the configured LR as _base_lr for warm-up scaling."""
    for pg in optimizer.param_groups:
        pg["_base_lr"] = pg["lr"]


# ---------------------------------------------------------------------------
# Train / Validate
# ---------------------------------------------------------------------------

def train_one_epoch(
    model, dataloader, criterion, optimizer, scaler, device, epoch, args,
    rank=0, weight_adjuster=None,
):
    model.train()
    total_loss = total_dice = total_ce = total_focal = 0.0
    all_preds, all_labels = [], []
    num_batches = 0

    for step, batch in enumerate(dataloader):
        ct      = batch["ct"].to(device, non_blocking=True)
        seg_gt  = batch["seg_gt"].to(device, non_blocking=True)
        ehr_text = batch["ehr_text"]
        labels  = batch["label"].to(device, non_blocking=True)

        optimizer.zero_grad()

        with autocast(enabled=args.amp):
            seg_pred, cls_pred = model(ct, ehr_text)
            losses = criterion(seg_pred, seg_gt, cls_pred, labels)

        scaler.scale(losses["total"]).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss  += losses["total"].item()
        total_dice  += losses["dice"].item()
        total_ce    += losses["ce"].item()
        total_focal += losses["focal"].item()

        preds = cls_pred.argmax(dim=1).detach().cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.detach().cpu().numpy().tolist())
        num_batches += 1

        if weight_adjuster is not None and step % 5 == 0:
            criterion_raw = criterion.module if hasattr(criterion, "module") else criterion
            weight_adjuster.step(criterion_raw, losses)

        if step % args.log_interval == 0:
            log(
                f"  [Epoch {epoch}] Step {step}/{len(dataloader)} | "
                f"Loss: {losses['total'].item():.4f} | "
                f"Dice: {losses['dice'].item():.4f} | "
                f"CE: {losses['ce'].item():.4f}",
                rank,
            )

    avg_loss = total_loss / max(num_batches, 1)
    metrics  = compute_metrics(all_preds, all_labels, num_classes=4)

    return {
        "loss":       avg_loss,
        "dice_loss":  total_dice  / max(num_batches, 1),
        "ce_loss":    total_ce    / max(num_batches, 1),
        "focal_loss": total_focal / max(num_batches, 1),
        **metrics,
    }


@torch.no_grad()
def validate(model, dataloader, criterion, device, args):
    model.eval()
    total_loss = total_dice_score = 0.0
    all_preds, all_labels = [], []
    num_batches = 0

    for batch in dataloader:
        ct      = batch["ct"].to(device, non_blocking=True)
        seg_gt  = batch["seg_gt"].to(device, non_blocking=True)
        ehr_text = batch["ehr_text"]
        labels  = batch["label"].to(device, non_blocking=True)

        with autocast(enabled=args.amp):
            seg_pred, cls_pred = model(ct, ehr_text)
            losses = criterion(seg_pred, seg_gt, cls_pred, labels)

        total_loss       += losses["total"].item()
        total_dice_score += compute_dice_score(seg_pred, seg_gt).item()

        preds = cls_pred.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.cpu().numpy().tolist())
        num_batches += 1

    avg_loss = total_loss / max(num_batches, 1)
    avg_dice = total_dice_score / max(num_batches, 1)
    metrics  = compute_metrics(all_preds, all_labels, num_classes=4)

    return {
        "loss":       avg_loss,
        "dice_score": avg_dice,
        **metrics,
    }


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

def save_checkpoint(model, optimizer, scheduler, scaler, epoch, metrics, path):
    state = {
        "epoch":                epoch,
        "model_state_dict":     (
            model.module.state_dict() if hasattr(model, "module") else model.state_dict()
        ),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "scaler_state_dict":    scaler.state_dict(),
        "metrics":              metrics,
    }
    torch.save(state, path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _build_optimizer(raw_model, args):
    return torch.optim.AdamW(
        [
            {"params": raw_model.vit_encoder.parameters(),              "lr": args.lr * 0.1,  "_base_lr": args.lr * 0.1},
            {"params": raw_model.clinical_bert.projection.parameters(), "lr": args.lr,         "_base_lr": args.lr},
            {"params": raw_model.clinical_bert.norm.parameters(),       "lr": args.lr,         "_base_lr": args.lr},
            {"params": raw_model.seg_decoder.parameters(),              "lr": args.lr,         "_base_lr": args.lr},
            {"params": raw_model.msaf.parameters(),                     "lr": args.lr,         "_base_lr": args.lr},
            {"params": raw_model.classifier.parameters(),               "lr": args.lr,         "_base_lr": args.lr},
        ],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )


def _build_scheduler(optimizer, args):
    cosine = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-7,
    )
    scheduler = LinearWarmupCosineScheduler(
        optimizer,
        warmup_epochs=args.warmup_epochs,
        base_lr=args.lr,
        cosine_scheduler=cosine,
    )
    return scheduler


def main():
    args = parse_args()
    rank, local_rank, world_size = setup_ddp()
    is_distributed = world_size > 1

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    log(f"Using device: {device}, World size: {world_size}", rank)

    # -----------------------------------------------------------------------
    # Dataset — full dataset constructed once; split into stratified subsets
    # -----------------------------------------------------------------------
    volume_size = (args.volume_depth, args.volume_height, args.volume_width)

    full_dataset = NSCLCDataset(
        metadata_csv=args.metadata_csv,
        clinical_csv=args.clinical_csv,
        volume_size=volume_size,
        is_train=True,
        gtv_margin=args.gtv_margin,
    )

    train_idx, val_idx = stratified_split(full_dataset, args.val_split, args.seed)

    # Val subset shares the same dataset but we want no augmentation for val:
    # Construct a separate val dataset object pointing to same CSVs, is_train=False
    val_dataset_full = NSCLCDataset(
        metadata_csv=args.metadata_csv,
        clinical_csv=args.clinical_csv,
        volume_size=volume_size,
        is_train=False,
        gtv_margin=args.gtv_margin,
    )

    train_dataset = Subset(full_dataset, train_idx)
    val_dataset   = Subset(val_dataset_full, val_idx)

    log(
        f"Dataset: {len(full_dataset)} total | "
        f"{len(train_dataset)} train | {len(val_dataset)} val  (stratified)",
        rank,
    )

    # Class weights from the training split only
    train_labels = [full_dataset.get_labels()[i] for i in train_idx]
    class_weights = compute_class_weights(train_labels, num_classes=4, device=device)
    log(f"Class weights: {class_weights.cpu().tolist()}", rank)

    if is_distributed:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        val_sampler   = DistributedSampler(val_dataset, shuffle=False)
    else:
        train_sampler = None
        val_sampler   = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    # -----------------------------------------------------------------------
    # Model
    # -----------------------------------------------------------------------
    model = build_model(args)
    model = model.to(device)

    # Freeze backbone for first N epochs (ViT blocks + ClinicalBERT)
    model.freeze_backbone()
    log(f"Backbone FROZEN for first {args.freeze_epochs} epochs", rank)

    if is_distributed:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    raw_model = model.module if hasattr(model, "module") else model

    # -----------------------------------------------------------------------
    # Loss, Optimizer, Scheduler
    # -----------------------------------------------------------------------
    criterion = USCNetLoss(
        num_classes=4,
        class_weights=class_weights,
        label_smoothing=args.label_smoothing,
    ).to(device)
    weight_adjuster = DynamicWeightAdjuster(temperature=2.0)

    optimizer = _build_optimizer(raw_model, args)
    _tag_base_lrs(optimizer)
    scheduler = _build_scheduler(optimizer, args)
    scaler    = GradScaler(enabled=args.amp)

    # Resume from checkpoint
    start_epoch = 0
    best_val_f1 = 0.0
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        raw_model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if ckpt.get("scheduler_state_dict"):
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_val_f1 = ckpt["metrics"].get("f1", 0.0)
        log(f"Resumed from epoch {start_epoch}, best F1: {best_val_f1:.4f}", rank)

    # -----------------------------------------------------------------------
    # Checkpoint directory
    # -----------------------------------------------------------------------
    if is_main_process(rank):
        os.makedirs(args.checkpoint_dir, exist_ok=True)

    # -----------------------------------------------------------------------
    # Training Loop
    # -----------------------------------------------------------------------
    log("=" * 70, rank)
    log("Starting USCNet Training for NSCLC-Radiomics Histology Classification", rank)
    log("=" * 70, rank)

    history    = []
    val_metrics = {}

    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()

        # Freeze / Unfreeze ViT logic
        if epoch == args.freeze_epochs:
            raw_model.unfreeze_backbone()
            log(f"[Epoch {epoch}] ViT UNFROZEN — fine-tuning begins (BERT stays frozen)", rank)

            # Re-build optimizer with lower ViT LR
            optimizer = torch.optim.AdamW(
                [
                    {"params": raw_model.vit_encoder.parameters(),              "lr": args.lr * 0.01, "_base_lr": args.lr * 0.01},
                    {"params": raw_model.clinical_bert.projection.parameters(), "lr": args.lr,         "_base_lr": args.lr},
                    {"params": raw_model.clinical_bert.norm.parameters(),       "lr": args.lr,         "_base_lr": args.lr},
                    {"params": raw_model.seg_decoder.parameters(),              "lr": args.lr,         "_base_lr": args.lr},
                    {"params": raw_model.msaf.parameters(),                     "lr": args.lr,         "_base_lr": args.lr},
                    {"params": raw_model.classifier.parameters(),               "lr": args.lr,         "_base_lr": args.lr},
                ],
                lr=args.lr,
                weight_decay=args.weight_decay,
            )
            cosine = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=10, T_mult=2, eta_min=1e-7,
            )
            scheduler = LinearWarmupCosineScheduler(
                optimizer,
                warmup_epochs=0,   # no extra warm-up after unfreeze
                base_lr=args.lr,
                cosine_scheduler=cosine,
            )

        if is_distributed:
            train_sampler.set_epoch(epoch)

        log(f"\n[Epoch {epoch + 1}/{args.epochs}] LR: {optimizer.param_groups[0]['lr']:.2e}", rank)

        # Train
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler,
            device, epoch + 1, args, rank, weight_adjuster,
        )
        log(
            f"  Train | Loss: {train_metrics['loss']:.4f} | "
            f"Acc: {train_metrics['accuracy']:.4f} | "
            f"F1: {train_metrics['f1']:.4f}",
            rank,
        )

        # Validate
        val_metrics = validate(model, val_loader, criterion, device, args)
        log(
            f"  Val   | Loss: {val_metrics['loss']:.4f} | "
            f"Acc: {val_metrics['accuracy']:.4f} | "
            f"F1: {val_metrics['f1']:.4f} | "
            f"Dice: {val_metrics['dice_score']:.4f}",
            rank,
        )

        scheduler.step()

        epoch_time = time.time() - epoch_start
        log(f"  Time: {epoch_time:.1f}s", rank)

        # Save best checkpoint
        if is_main_process(rank) and val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            ckpt_path = os.path.join(args.checkpoint_dir, "best_model.pt")
            save_checkpoint(model, optimizer, scheduler, scaler, epoch, val_metrics, ckpt_path)
            log(f"  >>> New best model saved (F1: {best_val_f1:.4f}) -> {ckpt_path}", rank)

        # Periodic checkpoint every 10 epochs
        if is_main_process(rank) and (epoch + 1) % 10 == 0:
            ckpt_path = os.path.join(args.checkpoint_dir, f"checkpoint_epoch_{epoch + 1}.pt")
            save_checkpoint(model, optimizer, scheduler, scaler, epoch, val_metrics, ckpt_path)

        # Log history
        if is_main_process(rank):
            entry = {
                "epoch":      epoch + 1,
                "train_loss": train_metrics["loss"],
                "train_acc":  train_metrics["accuracy"],
                "train_f1":   train_metrics["f1"],
                "val_loss":   val_metrics["loss"],
                "val_acc":    val_metrics["accuracy"],
                "val_f1":     val_metrics["f1"],
                "val_dice":   val_metrics["dice_score"],
                "lr":         optimizer.param_groups[0]["lr"],
                "epoch_time": epoch_time,
            }
            history.append(entry)

    # Save training history
    if is_main_process(rank):
        history_path = os.path.join(args.checkpoint_dir, "training_history.json")
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)
        log(f"\nTraining complete. History saved to {history_path}", rank)
        log(f"Best validation F1: {best_val_f1:.4f}", rank)

        if history:
            log("\nFinal Classification Report (Validation):", rank)
            log(val_metrics.get("report", ""), rank)

    cleanup_ddp()


if __name__ == "__main__":
    main()
