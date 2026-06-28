import time
from pathlib import Path
from typing import Callable, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .checkpoint import save_checkpoint, load_checkpoint
from .config import TrainerConfig


class Trainer:
    """
    Single training loop for FP32 and QAT runs.

    For QAT, pass use_amp=False in cfg and supply epoch_callback via
    make_qat_callback(freeze_bn_epoch, disable_observer_epoch).

    Skip/resume logic lives in the notebook loop — instantiate one Trainer
    per model and call fit(resume_from=...) when resuming.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        cfg: TrainerConfig,
        device: torch.device,
        save_dir: Path | str,
        run_name: str,
        num_classes: int = 200,
        wandb_run=None,
        epoch_callback: Optional[Callable[[int, nn.Module], None]] = None,
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

    def fit(self, resume_from: Optional[Path] = None) -> dict:
        """Run train/val loop, checkpoint best, return history dict."""
        cfg = self.cfg
        model = self.model.to(self.device)
        criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
        scaler = torch.amp.GradScaler("cuda") if cfg.use_amp else None

        start_epoch = 0
        if resume_from is not None and Path(resume_from).exists():
            start_epoch, _ = load_checkpoint(resume_from, model, optimizer, scheduler, device=str(self.device))
            start_epoch += 1

        best_val_acc = 0.0
        best_epoch = 0
        patience_counter = 0
        history: dict[str, list] = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
        best_path = self.save_dir / f"{self.run_name}_best.pth"

        for epoch in range(start_epoch, cfg.epochs):
            epoch_start = time.time()

            if self.epoch_callback is not None:
                self.epoch_callback(epoch, model)

            train_loss, train_acc = self._train_one_epoch(model, optimizer, scaler, criterion)
            val_loss, val_acc = self._validate(model, criterion)
            scheduler.step()

            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            if self.wandb_run is not None:
                self.wandb_run.log(
                    {"train_loss": train_loss, "train_acc": train_acc,
                     "val_loss": val_loss, "val_acc": val_acc},
                    step=epoch + 1,
                )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
                patience_counter = 0
                save_checkpoint(best_path, model, optimizer, scheduler, epoch, {"val_acc": val_acc})
                if self.wandb_run is not None:
                    self.wandb_run.log({"best_val_acc": best_val_acc})
            else:
                patience_counter += 1

            elapsed = time.time() - epoch_start
            print(f"Epoch {epoch+1:3d}/{cfg.epochs} | "
                  f"train_loss={train_loss:.4f} train_acc={train_acc:.2f}% | "
                  f"val_loss={val_loss:.4f} val_acc={val_acc:.2f}% | "
                  f"time={elapsed:.1f}s")

            if cfg.early_stopping_patience and patience_counter >= cfg.early_stopping_patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        return {"best_val_accuracy": best_val_acc, "best_epoch": best_epoch, "history": history}

    @torch.no_grad()
    def evaluate(self, loader: Optional[DataLoader] = None, topk: tuple = (1, 5)) -> dict:
        """Val loop returning {top1, top5, loss} using torchmetrics."""
        from torchmetrics.classification import MulticlassAccuracy
        from torchmetrics import MeanMetric

        loader = loader or self.val_loader
        model = self.model.eval().to(self.device)
        criterion = nn.CrossEntropyLoss()

        accs = {
            k: MulticlassAccuracy(num_classes=self.num_classes, top_k=k).to(self.device)
            for k in topk
        }
        loss_m = MeanMetric().to(self.device)

        for data, target in loader:
            data, target = data.to(self.device), target.to(self.device)
            out = model(data)
            loss_m.update(criterion(out, target))
            for acc in accs.values():
                acc.update(out, target)

        return {
            "loss": loss_m.compute().item(),
            **{f"top{k}": accs[k].compute().item() * 100 for k in topk},
        }

    def _train_one_epoch(self, model, optimizer, scaler, criterion) -> tuple[float, float]:
        model.train()
        cfg = self.cfg
        total_loss = correct = total = 0

        for data, target in (bar := tqdm(self.train_loader, desc="Training")):
            data, target = data.to(self.device), target.to(self.device)
            optimizer.zero_grad()

            if cfg.use_amp and scaler:
                with torch.amp.autocast("cuda"):
                    out = model(data)
                    loss = criterion(out, target)
                scaler.scale(loss).backward()
                if cfg.grad_clip_norm:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                out = model(data)
                loss = criterion(out, target)
                loss.backward()
                if cfg.grad_clip_norm:
                    nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
                optimizer.step()

            total_loss += loss.item() * target.size(0)
            correct += out.argmax(1).eq(target).sum().item()
            total += target.size(0)
            bar.set_postfix(loss=f"{total_loss/total:.4f}", acc=f"{100*correct/total:.2f}%")

        return total_loss / total, 100 * correct / total

    @torch.no_grad()
    def _validate(self, model, criterion) -> tuple[float, float]:
        model.eval()
        total_loss = correct = total = 0

        for data, target in (bar := tqdm(self.val_loader, desc="Validation")):
            data, target = data.to(self.device), target.to(self.device)
            out = model(data)
            total_loss += criterion(out, target).item() * target.size(0)
            correct += out.argmax(1).eq(target).sum().item()
            total += target.size(0)
            bar.set_postfix(loss=f"{total_loss/total:.4f}", acc=f"{100*correct/total:.2f}%")

        return total_loss / total, 100 * correct / total
