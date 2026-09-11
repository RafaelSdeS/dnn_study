import json
import logging
import time
from pathlib import Path
from typing import Callable, Optional

import psutil
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .checkpoint import save_checkpoint, load_resume_state
from .config import TrainerConfig
from .profiling import GpuSampler


class BaseTrainer:
    """
    Shared fit() loop for classification/detection/segmentation trainers: optimizer/scheduler/
    scaler construction, resume-state restore, per-epoch bookkeeping (GpuSampler, wandb,
    metrics_callback, early stopping, stop_requested), envelope checkpointing (both the
    per-epoch resume file and the best-metric file), and best-checkpoint reload before
    returning. Concrete subclasses implement the hooks below; each keeps its own fit()
    return shape via _shape_result() -- Trainer's is a tested, widely-consumed contract
    (ml/reporting.py::make_run_summary, several notebooks), so unifying the loop mechanism
    does not mean unifying every subclass's return keys.
    """

    LOGGER_PREFIX = "trainer"

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
        metrics_callback: Optional[Callable[[dict], None]] = None,
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
        self.metrics_callback = metrics_callback
        self.stop_requested = False
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.logger = logging.getLogger(f"{self.LOGGER_PREFIX}.{run_name}")
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            self.logger.addHandler(logging.StreamHandler())
            if log_file is not None:
                fh = logging.FileHandler(log_file)
                fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
                self.logger.addHandler(fh)

    def request_stop(self) -> None:
        self.stop_requested = True

    # ── hooks concrete subclasses implement ─────────────────────────────
    def _build_criterion(self) -> Optional[nn.Module]:
        raise NotImplementedError

    def _init_history_schema(self) -> dict:
        raise NotImplementedError

    def _train_one_epoch(self, model, optimizer, scaler, criterion) -> dict:
        raise NotImplementedError

    def _validate(self, model, criterion) -> dict:
        raise NotImplementedError

    def _primary_metric_key(self) -> str:
        raise NotImplementedError

    def _log_epoch(self, epoch: int, epoch_metrics: dict) -> None:
        raise NotImplementedError

    def _shape_result(self, history: dict, best_epoch: int, epochs_run: int,
                       total_training_time_s: float, wandb_run_id) -> dict:
        raise NotImplementedError

    def fit(self, resume_from: Optional[Path] = None) -> dict:
        """Run train/val loop, checkpoint best, return this subclass's result dict."""
        cfg = self.cfg
        model = self.model.to(self.device)
        criterion = self._build_criterion()
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        if cfg.warmup_epochs > 0:
            warmup = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.1, total_iters=cfg.warmup_epochs
            )
            cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=cfg.epochs - cfg.warmup_epochs
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer, schedulers=[warmup, cosine], milestones=[cfg.warmup_epochs]
            )
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
        scaler = torch.amp.GradScaler("cuda") if cfg.use_amp else None

        primary_key = self._primary_metric_key()
        start_epoch = 0
        best_primary = -1.0
        best_epoch = 0
        patience_counter = 0
        wandb_run_id = self.wandb_run.id if self.wandb_run else None
        elapsed_time_s = 0.0
        history: dict[str, list] = self._init_history_schema()
        psutil.cpu_percent(interval=None)  # prime the delta-since-last-call counter

        # Load full training state if resuming
        if resume_from is not None and Path(resume_from).exists():
            state = load_resume_state(
                resume_from, model, optimizer, scheduler, scaler,
                device=str(self.device), reset_scheduler=cfg.reset_scheduler_on_resume,
            )
            start_epoch = state["epoch"] + 1
            best_primary = state["best_val_acc"]
            patience_counter = state["patience_counter"]
            elapsed_time_s = state["elapsed_time_s"]
            for k, v in state["history"].items():
                if k in history:
                    history[k] = v
            wandb_run_id = state["wandb_run_id"]
            # best_epoch isn't in the checkpoint -- recover it from the restored history so
            # the resumed run's summary reports the whole run, not just the segment after
            # the resume.
            if history[primary_key]:
                best_epoch = max(range(len(history[primary_key])), key=history[primary_key].__getitem__)
            else:
                best_epoch = start_epoch - 1
            if cfg.reset_scheduler_on_resume:
                # scheduler was just constructed fresh above (correct T_max for the new
                # cfg.epochs) but never stepped -- fast-forward it to start_epoch instead
                # of loading the checkpoint's stale, wrong-T_max scheduler state.
                for _ in range(start_epoch):
                    scheduler.step()

        best_path = self.save_dir / f"{self.run_name}_best.pth"
        resume_path = self.save_dir / f"{self.run_name}_resume.pth"
        meta_path = self.save_dir / f"{self.run_name}_meta.json"
        train_start = time.monotonic()

        epoch = start_epoch - 1  # ponytail: keeps epoch bound if resume already reached cfg.epochs
        for epoch in range(start_epoch, cfg.epochs):
            epoch_start = time.monotonic()

            if self.epoch_callback is not None:
                self.epoch_callback(epoch, model)

            if self.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(self.device)

            with GpuSampler() as gpu_sampler:
                train_metrics = self._train_one_epoch(model, optimizer, scaler, criterion)
                val_metrics = self._validate(model, criterion)
            gpu_metrics = gpu_sampler.summary()
            scheduler.step()
            lr = optimizer.param_groups[0]["lr"]

            epoch_time = time.monotonic() - epoch_start
            peak_mem = (
                torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)
                if self.device.type == "cuda" else 0.0
            )
            n_batches = len(self.train_loader)
            images_per_sec = (n_batches * self.train_loader.batch_size) / epoch_time if epoch_time > 0 else None
            avg_batch_time_s = epoch_time / n_batches if n_batches else None
            cpu_percent = psutil.cpu_percent(interval=None)
            ram_used_mb = psutil.virtual_memory().used / (1024 ** 2)

            epoch_metrics = {
                **train_metrics, **val_metrics,
                "lr": lr, "epoch_time_s": epoch_time, "peak_gpu_mem_mb": peak_mem,
                "images_per_sec": images_per_sec, "avg_batch_time_s": avg_batch_time_s,
                "cpu_percent": cpu_percent, "ram_used_mb": ram_used_mb,
                **gpu_metrics,
            }
            for key, value in epoch_metrics.items():
                history.setdefault(key, []).append(value)

            if self.wandb_run is not None:
                self.wandb_run.log(epoch_metrics, step=epoch + 1)

            if self.metrics_callback is not None:
                self.metrics_callback({"epoch": epoch + 1, **epoch_metrics})

            # Save resume checkpoint every epoch (full training state for recovery)
            save_checkpoint(
                resume_path, model, optimizer, scheduler, epoch, val_metrics,
                scaler=scaler,
                best_val_acc=best_primary,
                history=history,
                wandb_run_id=wandb_run_id,
                patience_counter=patience_counter,
                elapsed_time_s=elapsed_time_s + (time.monotonic() - train_start),
            )
            # Write metadata sidecar for quick access
            meta_path.write_text(json.dumps({
                "epoch": epoch, f"best_{primary_key}": best_primary, "wandb_run_id": wandb_run_id,
            }))

            current_primary = epoch_metrics[primary_key]
            if current_primary > best_primary:
                best_primary = current_primary
                best_epoch = epoch
                patience_counter = 0
                save_checkpoint(best_path, model, optimizer, scheduler, epoch, val_metrics)
                self.logger.info("  ✓ Best %s so far! Saved to %s", primary_key, best_path)
                if self.wandb_run is not None:
                    self.wandb_run.log({f"best_{primary_key}": best_primary})
                    self.wandb_run.save(str(best_path))
            else:
                patience_counter += 1

            self._log_epoch(epoch, epoch_metrics)

            if cfg.early_stopping_patience and patience_counter >= cfg.early_stopping_patience:
                self.logger.info("Early stopping at epoch %d", epoch + 1)
                break

            if self.stop_requested:
                self.logger.info("Stop requested after epoch %d", epoch + 1)
                break

        # Restore the best checkpoint before returning. Without this, self.model keeps the
        # LAST epoch's weights, so a caller evaluating right after fit() (or building a
        # results summary from self.model) measures a different model than the one saved as
        # _best.pth -- the same checkpoint the QAT stage then picks up. See CLAUDE.md's
        # 2026-08-29 Trainer.fit() bug note for the classification-side history of this.
        if best_path.exists():
            best_state = torch.load(best_path, map_location=str(self.device), weights_only=False)
            model.load_state_dict(best_state.get("model_state_dict", best_state))
            self.logger.info("Restored best checkpoint (epoch %d) into the model", best_epoch + 1)

        total_training_time_s = elapsed_time_s + (time.monotonic() - train_start)
        return self._shape_result(history, best_epoch, epoch + 1, total_training_time_s, wandb_run_id)


class Trainer(BaseTrainer):
    """
    Single training loop for FP32 and QAT runs.

    For QAT, pass use_amp=False in cfg and supply epoch_callback via
    make_qat_callback(freeze_bn_epoch, disable_observer_epoch).

    Skip/resume logic lives in the notebook loop — instantiate one Trainer
    per model and call fit(resume_from=...) when resuming.
    """

    def _build_criterion(self) -> nn.Module:
        return nn.CrossEntropyLoss(label_smoothing=self.cfg.label_smoothing)

    def _init_history_schema(self) -> dict:
        return {
            "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "val_top5": [],
            "epoch_time_s": [], "peak_gpu_mem_mb": [], "lr": [],
            "images_per_sec": [], "avg_batch_time_s": [], "cpu_percent": [], "ram_used_mb": [],
            "gpu_power_avg_w": [], "gpu_utilization_pct": [], "gpu_temp_avg_c": [],
            "gpu_memory_used_avg_mb": [], "gpu_energy_wh": [],
        }

    def _primary_metric_key(self) -> str:
        return "val_acc"

    def _log_epoch(self, epoch, m) -> None:
        self.logger.info(
            "Epoch %3d/%d | train_loss=%.4f train_acc=%.2f%% | "
            "val_loss=%.4f val_acc=%.2f%% val_top5=%.2f%% | "
            "lr=%.2e peak_mem=%.0fMB time=%.1fs",
            epoch + 1, self.cfg.epochs, m["train_loss"], m["train_acc"],
            m["val_loss"], m["val_acc"], m["val_top5"], m["lr"], m["peak_gpu_mem_mb"], m["epoch_time_s"],
        )

    def _shape_result(self, history, best_epoch, epochs_run, total_training_time_s, wandb_run_id) -> dict:
        # best_val_loss is the loss AT the best-accuracy epoch (not the minimum loss).
        best_val_acc = history["val_acc"][best_epoch] if history["val_acc"] else -1.0
        best_val_top5 = history["val_top5"][best_epoch] if history["val_top5"] else 0.0
        best_val_loss = history["val_loss"][best_epoch] if history["val_loss"] else float("inf")
        final_val_top1 = history["val_acc"][-1] if history["val_acc"] else 0.0
        final_val_top5 = history["val_top5"][-1] if history["val_top5"] else 0.0
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
            self.run_name, epochs_run, best_val_acc, best_val_top5,
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
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        latency_ms = elapsed / total_images * 1000
        throughput = total_images / elapsed
        return {"latency_ms_per_image": latency_ms, "throughput_img_per_s": throughput, "device": str(self.device)}

    def _train_one_epoch(self, model, optimizer, scaler, criterion) -> dict:
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

        result = {"train_loss": total_loss / total, "train_acc": 100 * correct / total}
        if cfg.grad_clip_norm:
            result["grad_norm"] = total_norm / len(self.train_loader)
        return result

    @torch.no_grad()
    def _validate(self, model, criterion) -> dict:
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

        return {"val_loss": total_loss / total, "val_acc": 100 * correct1 / total, "val_top5": 100 * correct5 / total}
