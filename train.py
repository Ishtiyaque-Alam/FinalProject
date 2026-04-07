"""
Training script for USCNet on NSCLC-Radiomics dataset.

Supports:
  - Distributed Data Parallel (DDP) training
  - Argparse for batch_size, epochs, learning_rate
  - Freeze backbone for first 5 epochs, then unfreeze
  - Best model checkpoint saving
  - Mixed precision training

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
from torch.utils.data import DataLoader, DistributedSampler, random_split
try:
    from torch.amp import autocast as _autocast, GradScaler
    def autocast(enabled=True):
        return _autocast("cuda", enabled=enabled)
except ImportError:
    from torch.cuda.amp import GradScaler, autocast

from model import USCNet, build_model
from dataset import NSCLCDataset
from utils import USCNetLoss, DynamicWeightAdjuster, compute_metrics, compute_dice_score


def collate_fn(batch):
    """Custom collate that keeps ehr_text as a list of strings."""
    ct = torch.stack([b["ct"] for b in batch])
    seg_gt = torch.stack([b["seg_gt"] for b in batch])
    labels = torch.stack([b["label"] for b in batch])
    ehr_text = [b["ehr_text"] for b in batch]
    patient_ids = [b["patient_id"] for b in batch]
    return {
        "ct": ct,
        "seg_gt": seg_gt,
        "ehr_text": ehr_text,
        "label": labels,
        "patient_id": patient_ids,
    }


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
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size per GPU")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Initial learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay")
    parser.add_argument("--drop_rate", type=float, default=0.1, help="Dropout rate")

    # Freeze settings
    parser.add_argument("--freeze_epochs", type=int, default=5, help="Epochs to freeze backbone")

    # Volume dimensions
    parser.add_argument("--volume_depth", type=int, default=64, help="Volume depth")
    parser.add_argument("--volume_height", type=int, default=128, help="Volume height")
    parser.add_argument("--volume_width", type=int, default=128, help="Volume width")

    # Training settings
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--val_split", type=float, default=0.2, help="Validation split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Checkpoint directory")
    parser.add_argument("--log_interval", type=int, default=10, help="Log every N steps")
    parser.add_argument("--amp", action="store_true", default=True, help="Use mixed precision")
    parser.add_argument("--resume", type=str, default=None, help="Path to resume checkpoint")

    return parser.parse_args()


def setup_ddp():
    """Initialize DDP if launched via torchrun."""
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
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


def train_one_epoch(
    model, dataloader, criterion, optimizer, scaler, device, epoch, args,
    rank=0, weight_adjuster=None,
):
    model.train()
    total_loss = 0.0
    total_dice = 0.0
    total_ce = 0.0
    total_focal = 0.0
    all_preds = []
    all_labels = []
    num_batches = 0

    for step, batch in enumerate(dataloader):
        ct = batch["ct"].to(device, non_blocking=True)
        seg_gt = batch["seg_gt"].to(device, non_blocking=True)
        ehr_text = batch["ehr_text"]  # list of strings for ClinicalBERT
        labels = batch["label"].to(device, non_blocking=True)

        optimizer.zero_grad()

        with autocast(enabled=args.amp):
            seg_pred, cls_pred = model(ct, ehr_text)
            losses = criterion(seg_pred, seg_gt, cls_pred, labels)

        scaler.scale(losses["total"]).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += losses["total"].item()
        total_dice += losses["dice"].item()
        total_ce += losses["ce"].item()
        total_focal += losses["focal"].item()

        preds = cls_pred.argmax(dim=1).detach().cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.detach().cpu().numpy().tolist())
        num_batches += 1

        if weight_adjuster is not None and step % 5 == 0:
            weight_adjuster.step(
                criterion.module if hasattr(criterion, "module") else criterion,
                losses,
            )

        if step % args.log_interval == 0:
            log(
                f"  [Epoch {epoch}] Step {step}/{len(dataloader)} | "
                f"Loss: {losses['total'].item():.4f} | "
                f"Dice: {losses['dice'].item():.4f} | "
                f"CE: {losses['ce'].item():.4f}",
                rank,
            )

    avg_loss = total_loss / max(num_batches, 1)
    metrics = compute_metrics(all_preds, all_labels, num_classes=4)

    return {
        "loss": avg_loss,
        "dice_loss": total_dice / max(num_batches, 1),
        "ce_loss": total_ce / max(num_batches, 1),
        "focal_loss": total_focal / max(num_batches, 1),
        **metrics,
    }


@torch.no_grad()
def validate(model, dataloader, criterion, device, args):
    model.eval()
    total_loss = 0.0
    total_dice_score = 0.0
    all_preds = []
    all_labels = []
    num_batches = 0

    for batch in dataloader:
        ct = batch["ct"].to(device, non_blocking=True)
        seg_gt = batch["seg_gt"].to(device, non_blocking=True)
        ehr_text = batch["ehr_text"]  # list of strings for ClinicalBERT
        labels = batch["label"].to(device, non_blocking=True)

        with autocast(enabled=args.amp):
            seg_pred, cls_pred = model(ct, ehr_text)
            losses = criterion(seg_pred, seg_gt, cls_pred, labels)

        total_loss += losses["total"].item()
        total_dice_score += compute_dice_score(seg_pred, seg_gt).item()

        preds = cls_pred.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.cpu().numpy().tolist())
        num_batches += 1

    avg_loss = total_loss / max(num_batches, 1)
    avg_dice = total_dice_score / max(num_batches, 1)
    metrics = compute_metrics(all_preds, all_labels, num_classes=4)

    return {
        "loss": avg_loss,
        "dice_score": avg_dice,
        **metrics,
    }


def save_checkpoint(model, optimizer, scheduler, scaler, epoch, metrics, path):
    state = {
        "epoch": epoch,
        "model_state_dict": (
            model.module.state_dict() if hasattr(model, "module") else model.state_dict()
        ),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "scaler_state_dict": scaler.state_dict(),
        "metrics": metrics,
    }
    torch.save(state, path)


def main():
    args = parse_args()
    rank, local_rank, world_size = setup_ddp()
    is_distributed = world_size > 1

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    log(f"Using device: {device}, World size: {world_size}", rank)

    # -----------------------------------------------------------------------
    # Dataset & DataLoaders
    # -----------------------------------------------------------------------
    volume_size = (args.volume_depth, args.volume_height, args.volume_width)
    full_dataset = NSCLCDataset(
        metadata_csv=args.metadata_csv,
        clinical_csv=args.clinical_csv,
        volume_size=volume_size,
        is_train=True,
    )

    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )

    log(f"Dataset: {len(full_dataset)} total, {train_size} train, {val_size} val", rank)

    if is_distributed:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None

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

    # Freeze backbone for first N epochs
    model.freeze_backbone()
    log(f"Backbone FROZEN for first {args.freeze_epochs} epochs", rank)

    if is_distributed:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    raw_model = model.module if hasattr(model, "module") else model

    # -----------------------------------------------------------------------
    # Loss, Optimizer, Scheduler
    # -----------------------------------------------------------------------
    criterion = USCNetLoss(num_classes=4).to(device)
    weight_adjuster = DynamicWeightAdjuster(temperature=2.0)

    optimizer = torch.optim.AdamW(
        [
            {"params": raw_model.vit_encoder.parameters(), "lr": args.lr * 0.1},
            {"params": raw_model.clinical_bert.projection.parameters(), "lr": args.lr},
            {"params": raw_model.clinical_bert.norm.parameters(), "lr": args.lr},
            {"params": raw_model.seg_decoder.parameters(), "lr": args.lr},
            {"params": raw_model.msaf.parameters(), "lr": args.lr},
            {"params": raw_model.classifier.parameters(), "lr": args.lr},
        ],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-7,
    )
    scaler = GradScaler(enabled=args.amp)

    # Resume from checkpoint
    start_epoch = 0
    best_val_f1 = 0.0
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        raw_model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if ckpt["scheduler_state_dict"]:
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

    history = []

    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()

        # Freeze / Unfreeze logic
        if epoch == args.freeze_epochs:
            raw_model.unfreeze_backbone()
            log(f"[Epoch {epoch}] ViT UNFROZEN - fine-tuning begins (BERT stays frozen)", rank)

            # Reset optimizer — ViT gets a lower LR, BERT projection stays at full LR
            optimizer = torch.optim.AdamW(
                [
                    {"params": raw_model.vit_encoder.parameters(), "lr": args.lr * 0.01},
                    {"params": raw_model.clinical_bert.projection.parameters(), "lr": args.lr},
                    {"params": raw_model.clinical_bert.norm.parameters(), "lr": args.lr},
                    {"params": raw_model.seg_decoder.parameters(), "lr": args.lr},
                    {"params": raw_model.msaf.parameters(), "lr": args.lr},
                    {"params": raw_model.classifier.parameters(), "lr": args.lr},
                ],
                lr=args.lr,
                weight_decay=args.weight_decay,
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=10, T_mult=2, eta_min=1e-7,
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
            save_checkpoint(
                model, optimizer, scheduler, scaler, epoch, val_metrics, ckpt_path,
            )
            log(f"  >>> New best model saved (F1: {best_val_f1:.4f}) -> {ckpt_path}", rank)

        # Save periodic checkpoint
        if is_main_process(rank) and (epoch + 1) % 10 == 0:
            ckpt_path = os.path.join(args.checkpoint_dir, f"checkpoint_epoch_{epoch + 1}.pt")
            save_checkpoint(
                model, optimizer, scheduler, scaler, epoch, val_metrics, ckpt_path,
            )

        # Log history
        if is_main_process(rank):
            entry = {
                "epoch": epoch + 1,
                "train_loss": train_metrics["loss"],
                "train_acc": train_metrics["accuracy"],
                "train_f1": train_metrics["f1"],
                "val_loss": val_metrics["loss"],
                "val_acc": val_metrics["accuracy"],
                "val_f1": val_metrics["f1"],
                "val_dice": val_metrics["dice_score"],
                "lr": optimizer.param_groups[0]["lr"],
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
