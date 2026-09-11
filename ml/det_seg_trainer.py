"""
Detection/segmentation trainer for Phase 7.
Focused on mAP/mIoU metrics and loss computation specific to dense prediction.
"""
from pathlib import Path
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassJaccardIndex
from torchmetrics.detection import MeanAveragePrecision
from tqdm.auto import tqdm

from .config import TrainerConfig
from .trainer import BaseTrainer


class DetectionTrainer(BaseTrainer):
    """Training loop for SSD detection models on VOC."""

    LOGGER_PREFIX = "det_trainer"

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        cfg: TrainerConfig,
        device: torch.device,
        save_dir: Path | str,
        run_name: str,
        num_classes: int = 21,
        wandb_run=None,
        epoch_callback: Optional[Callable[[int, nn.Module], None]] = None,
        metrics_callback: Optional[Callable[[dict], None]] = None,
        log_file: Optional[Path] = None,
    ):
        super().__init__(
            model, train_loader, val_loader, cfg, device, save_dir, run_name,
            num_classes=num_classes, wandb_run=wandb_run, epoch_callback=epoch_callback,
            metrics_callback=metrics_callback, log_file=log_file,
        )
        # mAP metric (CPU-based, accumulates over batches)
        self.mAP_metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")

    def _build_criterion(self) -> None:
        return None

    def _init_history_schema(self) -> dict:
        return {
            "train_loss_bbox": [], "train_loss_cls": [], "train_loss_total": [],
            "val_mAP": [], "val_mAP50": [], "epoch_time_s": [], "peak_gpu_mem_mb": [],
            "lr": [], "images_per_sec": [], "avg_batch_time_s": [],
            "cpu_percent": [], "ram_used_mb": [],
        }

    def _primary_metric_key(self) -> str:
        return "val_mAP"

    def _log_epoch(self, epoch, m) -> None:
        self.logger.info(
            f"Epoch {epoch + 1}/{self.cfg.epochs} | "
            f"Loss: {m['train_loss_total']:.3f} (bbox:{m['train_loss_bbox']:.3f} cls:{m['train_loss_cls']:.3f}) | "
            f"mAP: {m['val_mAP']:.4f} (@.50: {m['val_mAP50']:.4f}) | "
            f"LR: {m['lr']:.2e} | "
            f"Time: {m['epoch_time_s']:.1f}s"
        )

    def _shape_result(self, history, best_epoch, epochs_run, total_training_time_s, wandb_run_id) -> dict:
        history["total_time_s"] = total_training_time_s
        history["best_epoch"] = best_epoch
        history["best_val_mAP"] = history["val_mAP"][best_epoch] if history["val_mAP"] else -1.0
        self.logger.info(f"Training complete. Best mAP: {history['best_val_mAP']:.4f} at epoch {best_epoch + 1}")
        return history

    def _train_one_epoch(self, model: nn.Module, optimizer: torch.optim.Optimizer, scaler, criterion) -> dict:
        """Train one epoch, return {train_loss_bbox, train_loss_cls, train_loss_total}."""
        model.train()
        cfg = self.cfg
        total_loss_bbox = 0.0
        total_loss_cls = 0.0
        n_batches = 0

        for images, targets in tqdm(self.train_loader, desc="Train", leave=False):
            images = [img.to(self.device) for img in images]
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

            optimizer.zero_grad()

            if cfg.use_amp and scaler:
                with torch.amp.autocast("cuda"):
                    loss_dict = model(images, targets)
                    loss = loss_dict["bbox_regression"] + loss_dict["classification"]
                scaler.scale(loss).backward()
                if cfg.grad_clip_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                # Forward pass (training mode, SSD computes loss)
                loss_dict = model(images, targets)
                loss = loss_dict["bbox_regression"] + loss_dict["classification"]
                loss.backward()
                if cfg.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
                optimizer.step()

            total_loss_bbox += loss_dict["bbox_regression"].item()
            total_loss_cls += loss_dict["classification"].item()
            n_batches += 1

        loss_bbox = total_loss_bbox / max(n_batches, 1)
        loss_cls = total_loss_cls / max(n_batches, 1)
        return {"train_loss_bbox": loss_bbox, "train_loss_cls": loss_cls, "train_loss_total": loss_bbox + loss_cls}

    def _validate(self, model: nn.Module, criterion) -> dict:
        """Validate, return {val_mAP, val_mAP50}."""
        model.eval()
        self.mAP_metric.reset()

        with torch.no_grad():
            for images, targets in tqdm(self.val_loader, desc="Val", leave=False):
                images = [img.to(self.device) for img in images]

                # Forward pass (eval mode, SSD returns detections)
                predictions = model(images)

                # Convert boxes to CPU for metric computation
                preds_cpu = []
                for pred in predictions:
                    preds_cpu.append({
                        "boxes": pred["boxes"].cpu(),
                        "scores": pred["scores"].cpu(),
                        "labels": pred["labels"].cpu(),
                    })

                targets_cpu = [{k: v.cpu() for k, v in t.items()} for t in targets]

                # Update metric
                self.mAP_metric.update(preds_cpu, targets_cpu)

        # Compute metric
        metrics = self.mAP_metric.compute()
        mAP = metrics["map"].item()  # mAP averaged over all IoU thresholds
        mAP50 = metrics["map_50"].item()  # mAP@.5

        return {"val_mAP": mAP, "val_mAP50": mAP50}


class SegmentationTrainer(BaseTrainer):
    """Training loop for DeepLab segmentation models on VOC."""

    LOGGER_PREFIX = "seg_trainer"

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        cfg: TrainerConfig,
        device: torch.device,
        save_dir: Path | str,
        run_name: str,
        num_classes: int = 21,
        wandb_run=None,
        epoch_callback: Optional[Callable[[int, nn.Module], None]] = None,
        metrics_callback: Optional[Callable[[dict], None]] = None,
        log_file: Optional[Path] = None,
    ):
        super().__init__(
            model, train_loader, val_loader, cfg, device, save_dir, run_name,
            num_classes=num_classes, wandb_run=wandb_run, epoch_callback=epoch_callback,
            metrics_callback=metrics_callback, log_file=log_file,
        )
        # mIoU metric. VOC masks use 255 as a "void"/boundary label — excluded via ignore_index.
        self.iou_metric = MulticlassJaccardIndex(
            num_classes=self.num_classes, ignore_index=255, average="macro"
        )

    def _build_criterion(self) -> None:
        return None

    def _init_history_schema(self) -> dict:
        return {
            "train_loss": [], "val_loss": [], "val_mIoU": [],
            "epoch_time_s": [], "peak_gpu_mem_mb": [],
            "lr": [], "images_per_sec": [], "avg_batch_time_s": [],
            "cpu_percent": [], "ram_used_mb": [],
        }

    def _primary_metric_key(self) -> str:
        return "val_mIoU"

    def _log_epoch(self, epoch, m) -> None:
        self.logger.info(
            f"Epoch {epoch + 1}/{self.cfg.epochs} | "
            f"Loss: {m['train_loss']:.3f} (val: {m['val_loss']:.3f}) | "
            f"mIoU: {m['val_mIoU']:.4f} | "
            f"LR: {m['lr']:.2e} | "
            f"Time: {m['epoch_time_s']:.1f}s"
        )

    def _shape_result(self, history, best_epoch, epochs_run, total_training_time_s, wandb_run_id) -> dict:
        history["total_time_s"] = total_training_time_s
        history["best_epoch"] = best_epoch
        history["best_val_mIoU"] = history["val_mIoU"][best_epoch] if history["val_mIoU"] else -1.0
        self.logger.info(f"Training complete. Best mIoU: {history['best_val_mIoU']:.4f} at epoch {best_epoch + 1}")
        return history

    def _train_one_epoch(self, model: nn.Module, optimizer: torch.optim.Optimizer, scaler, criterion) -> dict:
        """Train one epoch, return {train_loss}."""
        model.train()
        cfg = self.cfg
        total_loss = 0.0
        n_batches = 0

        for images, masks in tqdm(self.train_loader, desc="Train", leave=False):
            images = images.to(self.device)
            masks = masks.to(self.device)

            optimizer.zero_grad()

            if cfg.use_amp and scaler:
                with torch.amp.autocast("cuda"):
                    logits = model(images)
                    loss = F.cross_entropy(logits, masks, ignore_index=255, label_smoothing=cfg.label_smoothing)
                scaler.scale(loss).backward()
                if cfg.grad_clip_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(images)
                loss = F.cross_entropy(logits, masks, ignore_index=255, label_smoothing=cfg.label_smoothing)
                loss.backward()
                if cfg.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
                optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return {"train_loss": total_loss / max(n_batches, 1)}

    def _validate(self, model: nn.Module, criterion) -> dict:
        """Validate, return {val_loss, val_mIoU}."""
        model.eval()
        self.iou_metric.reset()
        total_loss = 0.0
        n_batches = 0

        with torch.no_grad():
            for images, masks in tqdm(self.val_loader, desc="Val", leave=False):
                images = images.to(self.device)
                masks = masks.to(self.device)

                logits = model(images)
                loss = F.cross_entropy(logits, masks, ignore_index=255, label_smoothing=self.cfg.label_smoothing)
                total_loss += loss.item()
                n_batches += 1

                preds = logits.argmax(dim=1)
                self.iou_metric.update(preds.cpu(), masks.cpu())

        val_loss = total_loss / max(n_batches, 1)
        mIoU = self.iou_metric.compute().item()
        return {"val_loss": val_loss, "val_mIoU": mIoU}


class TinyLoader:
    """Wrap a short list of batches in a DataLoader-like object (exposes batch_size)."""

    def __init__(self, batches, batch_size: int = 2):
        self.batches = batches
        self.batch_size = batch_size

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)


def demo():
    """Smoke check: 10-image overfit test."""
    from .det_seg_data import DetSegDataConfig, create_voc_detection_loaders
    from .det_seg_models import build_ssd_detector
    import os

    print("Loading tiny dataset (10 images)...")
    cfg = DetSegDataConfig(
        img_size=256,
        voc_root=os.path.expanduser("~/.cache/torchvision/datasets"),
        batch_size=2,
        num_workers=0,
    )
    _, _, train_loader, _ = create_voc_detection_loaders(cfg)

    # Take only first 5 batches = ~10 images
    small_loader = [batch for i, batch in enumerate(train_loader) if i < 5]

    print("Building SSD detector...")
    model = build_ssd_detector("alexnet_bottleneck", num_classes=21, image_size=256)

    print("Running 10-epoch overfit test...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg_train = TrainerConfig(epochs=10, lr=1e-3, use_amp=False)
    tiny_loader = TinyLoader(small_loader)

    trainer = DetectionTrainer(
        model, tiny_loader, tiny_loader, cfg_train, device,
        save_dir="/tmp/det_smoke_check", run_name="overfit_test"
    )

    history = trainer.fit()

    # Check that loss decreased and mAP increased
    assert history["train_loss_total"][-1] < history["train_loss_total"][0], "Loss did not decrease!"
    assert history["val_mAP"][-1] > history["val_mAP"][0], "mAP did not increase!"

    print(f"\n✓✓✓ OVERFIT TEST PASSED ✓✓✓")
    print(f"  Loss: {history['train_loss_total'][0]:.3f} → {history['train_loss_total'][-1]:.3f}")
    print(f"  mAP:  {history['val_mAP'][0]:.4f} → {history['val_mAP'][-1]:.4f}")


def demo_segmentation():
    """Smoke check: 10-image overfit test for segmentation."""
    from .det_seg_data import DetSegDataConfig, create_voc_segmentation_loaders
    from .det_seg_models import build_deeplabv3_segmenter
    import os

    print("Loading tiny dataset (10 images)...")
    cfg = DetSegDataConfig(
        img_size=256,
        voc_root=os.path.expanduser("~/.cache/torchvision/datasets"),
        batch_size=2,
        num_workers=0,
    )
    _, _, train_loader, _ = create_voc_segmentation_loaders(cfg)

    # Take only first 5 batches = ~10 images
    small_loader = [batch for i, batch in enumerate(train_loader) if i < 5]

    print("Building DeepLabV3 segmenter...")
    model = build_deeplabv3_segmenter("alexnet_bottleneck", num_classes=21, image_size=256)

    print("Running 10-epoch overfit test...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg_train = TrainerConfig(epochs=10, lr=1e-3, use_amp=False)
    tiny_loader = TinyLoader(small_loader)

    trainer = SegmentationTrainer(
        model, tiny_loader, tiny_loader, cfg_train, device,
        save_dir="/tmp/seg_smoke_check", run_name="overfit_test"
    )

    history = trainer.fit()

    # Check that loss decreased and mIoU increased
    assert history["train_loss"][-1] < history["train_loss"][0], "Loss did not decrease!"
    assert history["val_mIoU"][-1] > history["val_mIoU"][0], "mIoU did not increase!"

    print(f"\n✓✓✓ OVERFIT TEST PASSED ✓✓✓")
    print(f"  Loss: {history['train_loss'][0]:.3f} → {history['train_loss'][-1]:.3f}")
    print(f"  mIoU: {history['val_mIoU'][0]:.4f} → {history['val_mIoU'][-1]:.4f}")


if __name__ == "__main__":
    demo()
    demo_segmentation()
