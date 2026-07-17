"""
Detection/segmentation trainer for Phase 7.
Focused on mAP/mIoU metrics and loss computation specific to dense prediction.
"""
import json
import logging
import time
from pathlib import Path
from typing import Callable, Optional

import psutil
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchmetrics.detection import MeanAveragePrecision
from tqdm.auto import tqdm

from .checkpoint import save_checkpoint, load_resume_state
from .config import TrainerConfig
from .profiling import GpuSampler


class DetectionTrainer:
    """Training loop for SSD detection models on VOC."""

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
        log_file: Optional[Path] = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.device = device
        self.save_dir = Path(save_dir)
        self.run_name = run_name
        self.num_classes = num_classes
        self.wandb_run = wandb_run
        self.epoch_callback = epoch_callback
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # mAP metric (CPU-based, accumulates over batches)
        self.mAP_metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")

        self.logger = logging.getLogger(f"det_trainer.{run_name}")
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            self.logger.addHandler(logging.StreamHandler())
            if log_file is not None:
                fh = logging.FileHandler(log_file)
                fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
                self.logger.addHandler(fh)

    def fit(self, resume_from: Optional[Path] = None) -> dict:
        """Run train/val loop, checkpoint best-mAP, return history dict."""
        cfg = self.cfg
        model = self.model.to(self.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

        start_epoch = 0
        best_val_mAP = 0.0
        best_epoch = 0
        patience_counter = 0
        elapsed_time_s = 0.0
        history = {
            "train_loss_bbox": [], "train_loss_cls": [], "train_loss_total": [],
            "val_mAP": [], "val_mAP50": [], "epoch_time_s": [], "peak_gpu_mem_mb": [],
            "lr": [], "images_per_sec": [], "avg_batch_time_s": [],
            "cpu_percent": [], "ram_used_mb": [],
        }
        psutil.cpu_percent(interval=None)

        # Load state if resuming
        if resume_from is not None and Path(resume_from).exists():
            state = load_resume_state(resume_from, model, optimizer, scheduler, None, device=str(self.device))
            start_epoch = state["epoch"] + 1
            best_val_mAP = state.get("best_val_mAP", 0.0)
            best_epoch = start_epoch - 1
            patience_counter = state.get("patience_counter", 0)
            elapsed_time_s = state.get("elapsed_time_s", 0.0)
            for k, v in state["history"].items():
                if k in history:
                    history[k] = v

        best_path = self.save_dir / f"{self.run_name}_best.pth"
        resume_path = self.save_dir / f"{self.run_name}_resume.pth"
        train_start = time.time()

        for epoch in range(start_epoch, cfg.epochs):
            epoch_start = time.time()

            if self.epoch_callback is not None:
                self.epoch_callback(epoch, model)

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(self.device)

            with GpuSampler() as gpu_sampler:
                train_loss_bbox, train_loss_cls = self._train_one_epoch(model, optimizer)
                val_mAP, val_mAP50 = self._validate(model)

            gpu_metrics = gpu_sampler.summary()
            scheduler.step()
            lr = optimizer.param_groups[0]["lr"]

            epoch_time = time.time() - epoch_start
            peak_mem = (
                torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)
                if torch.cuda.is_available() else 0.0
            )
            n_batches = len(self.train_loader)
            images_per_sec = (n_batches * self.train_loader.batch_size) / epoch_time if epoch_time > 0 else None
            avg_batch_time_s = epoch_time / n_batches if n_batches else None
            cpu_percent = psutil.cpu_percent(interval=None)
            ram_used_mb = psutil.virtual_memory().used / (1024 ** 2)

            train_loss_total = train_loss_bbox + train_loss_cls
            history["train_loss_bbox"].append(train_loss_bbox)
            history["train_loss_cls"].append(train_loss_cls)
            history["train_loss_total"].append(train_loss_total)
            history["val_mAP"].append(val_mAP)
            history["val_mAP50"].append(val_mAP50)
            history["epoch_time_s"].append(epoch_time)
            history["peak_gpu_mem_mb"].append(peak_mem)
            history["lr"].append(lr)
            history["images_per_sec"].append(images_per_sec)
            history["avg_batch_time_s"].append(avg_batch_time_s)
            history["cpu_percent"].append(cpu_percent)
            history["ram_used_mb"].append(ram_used_mb)

            epoch_metrics = {
                "train_loss_bbox": train_loss_bbox, "train_loss_cls": train_loss_cls,
                "train_loss_total": train_loss_total, "val_mAP": val_mAP, "val_mAP50": val_mAP50,
                "lr": lr, "epoch_time_s": epoch_time, "peak_gpu_mem_mb": peak_mem,
                "images_per_sec": images_per_sec, "avg_batch_time_s": avg_batch_time_s,
                "cpu_percent": cpu_percent, "ram_used_mb": ram_used_mb,
                **gpu_metrics,
            }

            if self.wandb_run is not None:
                self.wandb_run.log(epoch_metrics, step=epoch + 1)

            self.logger.info(
                f"Epoch {epoch + 1}/{cfg.epochs} | "
                f"Loss: {train_loss_total:.3f} (bbox:{train_loss_bbox:.3f} cls:{train_loss_cls:.3f}) | "
                f"mAP: {val_mAP:.4f} (@.50: {val_mAP50:.4f}) | "
                f"LR: {lr:.2e} | "
                f"Time: {epoch_time:.1f}s"
            )

            # Early stopping on mAP
            if val_mAP > best_val_mAP:
                best_val_mAP = val_mAP
                best_epoch = epoch
                patience_counter = 0
                torch.save(model.state_dict(), best_path)
                self.logger.info(f"  ✓ Best mAP so far! Saved to {best_path}")
            else:
                patience_counter += 1

            # Save resume checkpoint
            save_checkpoint(
                resume_path, model, optimizer, scheduler, epoch,
                {"best_val_mAP": best_val_mAP, "patience_counter": patience_counter, "history": history,
                 "elapsed_time_s": elapsed_time_s + (time.time() - train_start)},
            )

            if cfg.early_stopping_patience is not None and patience_counter >= cfg.early_stopping_patience:
                self.logger.info(f"Early stopping triggered after {cfg.early_stopping_patience} epochs without improvement")
                break

        elapsed_time_s += time.time() - train_start
        history["total_time_s"] = elapsed_time_s
        history["best_epoch"] = best_epoch
        history["best_val_mAP"] = best_val_mAP

        self.logger.info(f"Training complete. Best mAP: {best_val_mAP:.4f} at epoch {best_epoch + 1}")
        return history

    def _train_one_epoch(self, model: nn.Module, optimizer: torch.optim.Optimizer) -> tuple:
        """Train one epoch, return (loss_bbox, loss_cls)."""
        model.train()
        total_loss_bbox = 0.0
        total_loss_cls = 0.0
        n_batches = 0

        for images, targets in tqdm(self.train_loader, desc="Train", leave=False):
            images = [img.to(self.device) for img in images]
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

            optimizer.zero_grad()

            # Forward pass (training mode, SSD computes loss)
            loss_dict = model(images, targets)

            loss = loss_dict["bbox_regression"] + loss_dict["classification"]
            loss.backward()

            # Gradient clipping if configured
            if self.cfg.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.cfg.grad_clip_norm)

            optimizer.step()

            total_loss_bbox += loss_dict["bbox_regression"].item()
            total_loss_cls += loss_dict["classification"].item()
            n_batches += 1

        return total_loss_bbox / max(n_batches, 1), total_loss_cls / max(n_batches, 1)

    def _validate(self, model: nn.Module) -> tuple:
        """Validate, return (mAP, mAP@.5)."""
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

        return mAP, mAP50


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
    trainer = DetectionTrainer(
        model, small_loader, small_loader, cfg_train, device,
        save_dir="/tmp/det_smoke_check", run_name="overfit_test"
    )

    # Override loader for this tiny test
    trainer.train_loader = small_loader
    trainer.val_loader = small_loader

    history = trainer.fit()

    # Check that loss decreased and mAP increased
    assert history["train_loss_total"][-1] < history["train_loss_total"][0], "Loss did not decrease!"
    assert history["val_mAP"][-1] > history["val_mAP"][0], "mAP did not increase!"

    print(f"\n✓✓✓ OVERFIT TEST PASSED ✓✓✓")
    print(f"  Loss: {history['train_loss_total'][0]:.3f} → {history['train_loss_total'][-1]:.3f}")
    print(f"  mAP:  {history['val_mAP'][0]:.4f} → {history['val_mAP'][-1]:.4f}")


if __name__ == "__main__":
    demo()
