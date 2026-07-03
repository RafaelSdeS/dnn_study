import json
import logging
import time
from pathlib import Path
from typing import Callable, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .checkpoint import save_checkpoint, load_resume_state
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

        self.logger = logging.getLogger(f"trainer.{run_name}")
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            self.logger.addHandler(logging.StreamHandler())
            if log_file is not None:
                fh = logging.FileHandler(log_file)
                fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
                self.logger.addHandler(fh)

    def fit(self, resume_from: Optional[Path] = None) -> dict:
        """Run train/val loop, checkpoint best, return history dict."""
        cfg = self.cfg
        model = self.model.to(self.device)
        criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
        scaler = torch.amp.GradScaler("cuda") if cfg.use_amp else None

        start_epoch = 0
        best_val_acc = 0.0
        best_val_top5 = 0.0
        best_val_loss = float("inf")
        best_epoch = 0
        patience_counter = 0
        wandb_run_id = self.wandb_run.id if self.wandb_run else None
        history: dict[str, list] = {
            "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "val_top5": [],
            "epoch_time_s": [], "peak_gpu_mem_mb": [], "lr": [],
        }

        # Load full training state if resuming
        if resume_from is not None and Path(resume_from).exists():
            state = load_resume_state(resume_from, model, optimizer, scheduler, scaler, device=str(self.device))
            start_epoch = state["epoch"] + 1
            best_val_acc = state["best_val_acc"]
            best_val_top5 = state["best_val_top5"]
            best_epoch = start_epoch - 1
            patience_counter = state["patience_counter"]
            for k, v in state["history"].items():
                if k in history:
                    history[k] = v
            wandb_run_id = state["wandb_run_id"]

        best_path = self.save_dir / f"{self.run_name}_best.pth"
        resume_path = self.save_dir / f"{self.run_name}_resume.pth"
        meta_path = self.save_dir / f"{self.run_name}_meta.json"
        train_start = time.time()

        epoch = start_epoch - 1  # ponytail: keeps epoch bound if resume already reached cfg.epochs
        for epoch in range(start_epoch, cfg.epochs):
            epoch_start = time.time()

            if self.epoch_callback is not None:
                self.epoch_callback(epoch, model)

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(self.device)

            train_loss, train_acc, avg_grad_norm = self._train_one_epoch(model, optimizer, scaler, criterion)
            val_loss, val_acc, val_top5 = self._validate(model, criterion)
            scheduler.step()
            lr = optimizer.param_groups[0]["lr"]

            epoch_time = time.time() - epoch_start
            peak_mem = (
                torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)
                if torch.cuda.is_available() else 0.0
            )

            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            history["val_top5"].append(val_top5)
            history["epoch_time_s"].append(epoch_time)
            history["peak_gpu_mem_mb"].append(peak_mem)
            history["lr"].append(lr)
            if avg_grad_norm is not None:
                history.setdefault("grad_norm", []).append(avg_grad_norm)

            if self.wandb_run is not None:
                log_dict = {
                    "train_loss": train_loss, "train_acc": train_acc,
                    "val_loss": val_loss, "val_acc": val_acc, "val_top5": val_top5,
                    "lr": lr, "epoch_time_s": epoch_time, "peak_gpu_mem_mb": peak_mem,
                }
                if avg_grad_norm is not None:
                    log_dict["grad_norm"] = avg_grad_norm
                self.wandb_run.log(log_dict, step=epoch + 1)

            # Save resume checkpoint every epoch (full training state for recovery)
            save_checkpoint(
                resume_path, model, optimizer, scheduler, epoch, {"val_acc": val_acc},
                scaler=scaler,
                best_val_acc=best_val_acc,
                best_val_top5=best_val_top5,
                history=history,
                wandb_run_id=wandb_run_id,
                patience_counter=patience_counter,
            )
            # Write metadata sidecar for quick access
            meta_path.write_text(json.dumps({"epoch": epoch, "best_val_acc": best_val_acc, "wandb_run_id": wandb_run_id}))

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_val_top5 = val_top5
                best_val_loss = val_loss
                best_epoch = epoch
                patience_counter = 0
                save_checkpoint(best_path, model, optimizer, scheduler, epoch, {"val_acc": val_acc})
                if self.wandb_run is not None:
                    self.wandb_run.log({"best_val_acc": best_val_acc})
                    self.wandb_run.save(str(best_path))
            else:
                patience_counter += 1

            self.logger.info(
                "Epoch %3d/%d | train_loss=%.4f train_acc=%.2f%% | "
                "val_loss=%.4f val_acc=%.2f%% val_top5=%.2f%% | "
                "lr=%.2e peak_mem=%.0fMB time=%.1fs",
                epoch + 1, cfg.epochs, train_loss, train_acc,
                val_loss, val_acc, val_top5, lr, peak_mem, epoch_time,
            )

            if cfg.early_stopping_patience and patience_counter >= cfg.early_stopping_patience:
                self.logger.info("Early stopping at epoch %d", epoch + 1)
                break

        final_val_top1 = history["val_acc"][-1] if history["val_acc"] else 0.0
        final_val_top5 = history["val_top5"][-1] if history["val_top5"] else 0.0
        total_training_time_s = time.time() - train_start
        total_time_str = time.strftime("%H:%M:%S", time.gmtime(total_training_time_s))

        self.logger.info(
            "\n================= Run Summary =================\n"
            "Model          : %s\n"
            "Epochs         : %d\n"
            "Best Val Top-1 : %.2f%%\n"
            "Best Val Top-5 : %.2f%%\n"
            "Final Val Top-1: %.2f%%\n"
            "Final Val Top-5: %.2f%%\n"
            "Best Val Loss  : %.4f\n"
            "Total Time     : %s\n"
            "===============================================",
            self.run_name, epoch + 1, best_val_acc, best_val_top5,
            final_val_top1, final_val_top5, best_val_loss, total_time_str,
        )

        return {
            "best_val_top1": best_val_acc,
            "best_val_top5": best_val_top5,
            "best_val_loss": best_val_loss,
            "final_val_top1": final_val_top1,
            "final_val_top5": final_val_top5,
            "best_epoch": best_epoch,
            "total_training_time_s": total_training_time_s,
            "total_training_time": total_time_str,
            "history": history,
            "wandb_run_id": wandb_run_id,
        }

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

    @torch.no_grad()
    def benchmark(self, loader: Optional[DataLoader] = None, warmup: int = 100) -> dict:
        """Time inference over val_loader; returns latency_ms_per_image and throughput_img_per_s."""
        loader = loader or self.val_loader
        model = self.model.eval().to(self.device)
        n_warmup = 0

        # warmup
        for data, _ in loader:
            data = data.to(self.device)
            model(data)
            n_warmup += data.size(0)
            if n_warmup >= warmup:
                break

        total_images = 0
        t0 = time.perf_counter()
        for data, _ in loader:
            data = data.to(self.device)
            model(data)
            total_images += data.size(0)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        latency_ms = elapsed / total_images * 1000
        throughput = total_images / elapsed
        return {"latency_ms_per_image": latency_ms, "throughput_img_per_s": throughput, "device": str(self.device)}

    def _train_one_epoch(self, model, optimizer, scaler, criterion) -> tuple[float, float, float | None]:
        model.train()
        cfg = self.cfg
        total_loss = correct = total = 0
        total_norm = 0.0

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
                    total_norm += nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm).item()
                scaler.step(optimizer)
                scaler.update()
            else:
                out = model(data)
                loss = criterion(out, target)
                loss.backward()
                if cfg.grad_clip_norm:
                    total_norm += nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm).item()
                optimizer.step()

            total_loss += loss.item() * target.size(0)
            correct += out.argmax(1).eq(target).sum().item()
            total += target.size(0)
            bar.set_postfix(loss=f"{total_loss/total:.4f}", acc=f"{100*correct/total:.2f}%")

        avg_norm = total_norm / len(self.train_loader) if cfg.grad_clip_norm else None
        return total_loss / total, 100 * correct / total, avg_norm

    @torch.no_grad()
    def _validate(self, model, criterion) -> tuple[float, float, float]:
        model.eval()
        total_loss = correct1 = correct5 = total = 0

        for data, target in (bar := tqdm(self.val_loader, desc="Validation")):
            data, target = data.to(self.device), target.to(self.device)
            out = model(data)
            total_loss += criterion(out, target).item() * target.size(0)
            correct1 += out.argmax(1).eq(target).sum().item()
            correct5 += out.topk(5, dim=1).indices.eq(target.unsqueeze(1)).any(dim=1).sum().item()
            total += target.size(0)
            bar.set_postfix(loss=f"{total_loss/total:.4f}", top1=f"{100*correct1/total:.2f}%", top5=f"{100*correct5/total:.2f}%")

        return total_loss / total, 100 * correct1 / total, 100 * correct5 / total
