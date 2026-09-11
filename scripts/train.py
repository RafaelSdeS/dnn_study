from __future__ import annotations

import argparse
import json
import logging
import signal
import tempfile
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import torch
from torch.utils.tensorboard import SummaryWriter

import ml.model_registrations  # noqa: F401 — populates MODEL_REGISTRY
from configs.loader import load_config
from ml import (
    MODEL_REGISTRY,
    DataConfig,
    TrainerConfig,
    QATConfig,
    Trainer,
    auto_resume_path,
    build_qat,
    build_runtime_paths,
    capture_provenance,
    compress_checkpoint,
    compute_flops,
    convert_to_int8,
    create_imagenet_loaders,
    create_results_summary,
    disk_mb,
    gzip_mb,
    ensure_dataset_path,
    expand_path,
    load_profile,
    make_model_runs,
    make_qat_callback,
    make_run_summary,
    load_best_model,
    save_resolved_config,
    set_global_seed,
)
from ml.winograd_bridge import bridge_provenance, dump_wino_calibration, load_qat_wino_model


def _resolve_model_names(model_names: list[str] | str | None) -> list[str]:
    if not model_names or model_names == "all":
        return list(MODEL_REGISTRY.keys())
    if isinstance(model_names, str):  # `models: alexnet_fire` in YAML, not a one-item list
        model_names = [model_names]
    # Fail loudly: a typo in an experiment's models: list used to silently drop that
    # model, which on PCAD only shows up hours later as a missing summary.
    unknown = [name for name in model_names if name not in MODEL_REGISTRY]
    if unknown:
        raise ValueError(f"Unknown model(s) {unknown}; registered: {sorted(MODEL_REGISTRY)}")
    return list(model_names)


def _append_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    if not rows:
        return
    try:
        import pandas as pd
    except Exception:
        output_path.write_text(json.dumps(rows, indent=2, default=str))
        return
    df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def _maybe_init_wandb(enabled: bool, run_name: str, config: dict[str, Any], mode: str = "offline", group: str | None = None, resume_id: str | None = None):
    if not enabled:
        return None
    try:
        import wandb
    except Exception:
        return None

    return wandb.init(
        project=config.get("wandb_project", "alexnet_kernel_research"),
        name=run_name,
        group=group,
        mode=mode,
        id=resume_id,
        resume="allow" if resume_id else None,
        config=config,
    )


def _build_data_config(base_cfg: dict[str, Any], experiment_cfg: dict[str, Any]) -> DataConfig:
    data_cfg = DataConfig(**base_cfg)
    overrides = experiment_cfg.get("data", {}) or {}
    for key, value in overrides.items():
        setattr(data_cfg, key, value)
    return data_cfg


def _build_trainer_config(base_cfg: dict[str, Any], experiment_cfg: dict[str, Any]) -> TrainerConfig:
    trainer_cfg = TrainerConfig(**base_cfg)
    overrides = experiment_cfg.get("training", {}) or {}
    return replace(trainer_cfg, **overrides)


def _build_qat_config(base_cfg: dict[str, Any], experiment_cfg: dict[str, Any]) -> QATConfig:
    qat_cfg = QATConfig(**base_cfg)
    overrides = experiment_cfg.get("qat", {}) or {}
    return replace(qat_cfg, **overrides)


def _build_qat_wino_config(base_cfg: dict[str, Any], experiment_cfg: dict[str, Any]) -> QATConfig:
    qat_wino_cfg = QATConfig(**base_cfg)
    overrides = experiment_cfg.get("qat_wino", {}) or {}
    return replace(qat_wino_cfg, **overrides)


def _apply_smoke_override(experiment_cfg: dict[str, Any]) -> dict[str, Any]:
    """Cap fp32/QAT/qat_wino epochs to 1 so a full pipeline run (data/model/checkpoint/
    QAT-convert) finishes in minutes locally, to catch bugs before a PCAD submission."""
    experiment_cfg["training"] = {**experiment_cfg.get("training", {}), "epochs": 1, "warmup_epochs": 0}
    experiment_cfg["qat"] = {**experiment_cfg.get("qat", {}), "epochs": 1}
    experiment_cfg["qat_wino"] = {**experiment_cfg.get("qat_wino", {}), "epochs": 1}
    return experiment_cfg


def _stop_requested(trainer: Trainer, stage: str, model_name: str, writer, wandb_run) -> bool:
    """True if a stop signal cut this stage's fit() short -- SIGUSR1 is Slurm's pre-timeout
    warning (train.sbatch), SIGTERM/SIGINT the rest. Going on to the next stage would build
    it on the truncated model and keep the process alive past the wall clock, so
    train.sbatch's requeue would never run. The caller ends the whole run instead (models
    already finished keep their summaries); the requeued job resumes this stage from its
    _resume.pth."""
    if not trainer.stop_requested:
        return False
    logging.getLogger(f"pcad_runner.{model_name}").warning(
        "Stop requested during %s of %s -- exiting before later stages and the summary.", stage, model_name)
    if writer is not None:
        writer.close()
    if wandb_run is not None:
        wandb_run.finish()
    return True


def run_experiment(experiment_cfg: dict[str, Any], runtime_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    runtime_root = expand_path(runtime_cfg.get("root"), default="outputs/local") or Path("outputs/local")
    runtime_paths = build_runtime_paths(runtime_root)

    data_cfg = _build_data_config(load_config("data.yaml"), experiment_cfg)
    trainer_cfg = _build_trainer_config(load_config("training.yaml"), experiment_cfg)
    qat_cfg = _build_qat_config(load_config("qat.yaml"), experiment_cfg)
    qat_wino_cfg = _build_qat_wino_config(load_config("qat_wino.yaml"), experiment_cfg)
    uniform_hparams = bool(experiment_cfg.get("uniform_hparams", False))
    experiment_name = experiment_cfg.get("name") or "experiment"
    stage_list = experiment_cfg.get("stages", ["fp32", "qat", "int8"])
    selected_models = _resolve_model_names(experiment_cfg.get("models"))  # before the dataset: fail fast

    seed = int(experiment_cfg.get("seed", data_cfg.seed))
    data_cfg.seed = seed
    set_global_seed(seed)

    dataset_path = ensure_dataset_path(runtime_cfg)
    data_cfg.dataset_path = str(dataset_path)

    train_ds, val_ds, train_loader, val_loader = create_imagenet_loaders(data_cfg, persistent_workers=runtime_cfg.get("persistent_workers", False))

    results_rows: list[dict[str, Any]] = []
    torch.backends.quantized.engine = runtime_cfg.get("quantized_engine", "fbgemm")
    device = torch.device(runtime_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    provenance = capture_provenance()
    # *_fpga models and the qat_wino stage run Winograd-FPGA code (ml/winograd_bridge.py),
    # so this repo's git hash alone doesn't pin what trained them.
    if "qat_wino" in stage_list or any(name.endswith("_fpga") for name in selected_models):
        provenance["winograd_fpga"] = bridge_provenance()

    for model_name in selected_models:
        spec = MODEL_REGISTRY[model_name]
        model_run_name = f"{experiment_name}_{model_name}"
        run_root, checkpoints_dir, logs_dir, tb_dir, results_dir = make_model_runs(runtime_paths.root, experiment_name, model_name)

        # uniform_hparams (Fase 2 do plano): o registry tem lr/weight_decay
        # por-modelo (register_model(lr=...), tunado para o melhor resultado de
        # cada rede) -- o protocolo de budget unico exige o MESMO valor para
        # todas, senao "mesmo lr entre redes" seria furado em silencio aqui.
        model_cfg = trainer_cfg if uniform_hparams else replace(
            trainer_cfg,
            lr=spec.get("lr", trainer_cfg.lr),
            weight_decay=spec.get("weight_decay", trainer_cfg.weight_decay),
        )
        resolved_config = {
            "experiment": experiment_cfg,
            "runtime": runtime_cfg,
            "data": asdict(data_cfg),
            "training": asdict(model_cfg),
            "qat": asdict(qat_cfg),
            "qat_wino": asdict(qat_wino_cfg),
            "selected_model": model_name,
            "stage_list": stage_list,
            "provenance": provenance,
        }
        save_resolved_config(run_root, resolved_config)

        log_file = logs_dir / f"{model_name}.log"
        writer = SummaryWriter(log_dir=str(tb_dir / model_name)) if runtime_cfg.get("tensorboard", True) else None
        epoch_metrics: list[dict[str, Any]] = []

        def metrics_callback(metrics: dict[str, Any]) -> None:
            epoch_metrics.append(metrics)
            epoch = metrics.get("epoch")
            for key, value in metrics.items():
                if key == "epoch" or value is None or writer is None:
                    continue
                if isinstance(value, (int, float)):
                    writer.add_scalar(key, value, epoch)

        logger = logging.getLogger(f"pcad_runner.{model_name}")
        logger.setLevel(logging.INFO)
        logger.handlers.clear()
        stream = logging.StreamHandler()
        stream.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(stream)

        trainer = None

        def _request_stop(_signum, _frame):
            # 1st signal: stop after this epoch (resume checkpoint stays valid). A 2nd one, or one
            # before any trainer exists, aborts now -- like a plain Ctrl+C.
            if trainer is None or trainer.stop_requested:
                raise KeyboardInterrupt
            trainer.request_stop()

        signal.signal(signal.SIGTERM, _request_stop)
        signal.signal(signal.SIGINT, _request_stop)
        signal.signal(signal.SIGUSR1, _request_stop)  # Slurm pre-timeout warning (see train.sbatch)

        fp32_fit = {}
        fp32_eval = {}
        int8_eval = None
        int8_benchmark = None
        fp32_benchmark = {"latency_ms_per_image": None, "throughput_img_per_s": None}
        best_model_path = checkpoints_dir / f"{model_name}_best.pth"
        int8_path = checkpoints_dir / f"qat_{model_name}.pth"
        wandb_run = _maybe_init_wandb(
            runtime_cfg.get("wandb", False),
            model_run_name,
            resolved_config,
            mode=runtime_cfg.get("wandb_mode", "offline"),
            group=experiment_name,
        )

        if "fp32" in stage_list:
            resume_from = auto_resume_path(checkpoints_dir, model_name)
            if best_model_path.exists() and resume_from is None:
                logger.info("Skipping FP32 stage for %s; best checkpoint exists.", model_name)
                cached_model = load_best_model(model_name, spec["ctor"], checkpoints_dir, device)
                trainer = Trainer(
                    cached_model,
                    train_loader,
                    val_loader,
                    model_cfg,
                    device,
                    checkpoints_dir,
                    model_name,
                    num_classes=data_cfg.num_classes,
                    wandb_run=wandb_run,
                    metrics_callback=metrics_callback,
                    log_file=log_file,
                )
                fp32_eval = trainer.evaluate(topk=(1, 5))
                fp32_benchmark = trainer.benchmark(warmup=int(runtime_cfg.get("benchmark_warmup", 100)))
            else:
                trainer = Trainer(
                    spec["ctor"]().to(device),
                    train_loader,
                    val_loader,
                    model_cfg,
                    device,
                    checkpoints_dir,
                    model_name,
                    num_classes=data_cfg.num_classes,
                    wandb_run=wandb_run,
                    metrics_callback=metrics_callback,
                    log_file=log_file,
                )
                fp32_fit = trainer.fit(resume_from=resume_from)
                if _stop_requested(trainer, "fp32", model_name, writer, wandb_run):
                    break
                fp32_eval = trainer.evaluate(topk=(1, 5))
                fp32_benchmark = trainer.benchmark(warmup=int(runtime_cfg.get("benchmark_warmup", 100)))

        qat_model = None
        qat_fit = {}
        if "qat" in stage_list:
            qat_model = build_qat(model_name, save_dir=checkpoints_dir, device=device)
            qat_cfg_run = replace(model_cfg, epochs=qat_cfg.epochs, lr=qat_cfg.lr, weight_decay=qat_cfg.weight_decay, use_amp=False)
            resume_from = auto_resume_path(checkpoints_dir, f"qat_{model_name}")
            qat_best_path = checkpoints_dir / f"qat_{model_name}_best.pth"
            if qat_best_path.exists() and resume_from is None:
                logger.info("Skipping QAT stage for %s; best checkpoint exists.", model_name)
                # build_qat() started from the FP32 weights; without this the int8 stage would
                # convert an untrained, uncalibrated QAT model
                qat_best = torch.load(qat_best_path, map_location=str(device), weights_only=False)
                qat_model.load_state_dict(qat_best.get("model_state_dict", qat_best))
            else:
                trainer = Trainer(
                    qat_model,
                    train_loader,
                    val_loader,
                    qat_cfg_run,
                    device,
                    checkpoints_dir,
                    f"qat_{model_name}",
                    num_classes=data_cfg.num_classes,
                    wandb_run=wandb_run,
                    epoch_callback=make_qat_callback(qat_cfg.freeze_bn_epoch, qat_cfg.disable_observer_epoch),
                    metrics_callback=metrics_callback,
                    log_file=logs_dir / f"qat_{model_name}.log",
                )
                qat_fit = trainer.fit(resume_from=resume_from)
                if _stop_requested(trainer, "qat", model_name, writer, wandb_run):
                    break
                trainer.logger.info("QAT training complete for %s", model_name)

        if "qat_wino" in stage_list:
            # Fase 2 do plano de avaliacao Winograd (accelerator-numeric QAT,
            # not fbgemm's per-channel one): fine-tune with the FPGA's own
            # per-tensor pow2 requant + F(4,3) transform in the forward, via
            # scripts/avaliacao_redes/qat_wino.py in the sibling Winograd-FPGA
            # repo (via ml/winograd_bridge.py). No int8-convert step here -- unlike
            # the fbgemm path, WinoQuantConv2d.forward() already simulates the
            # quantized pipeline in eval mode, so evaluating this checkpoint directly
            # *is* the accelerator-numeric accuracy (top1_int8_accel proxy).
            qat_wino_model = load_qat_wino_model(model_name, spec, checkpoints_dir, device)
            qat_wino_cfg_run = replace(
                model_cfg, epochs=qat_wino_cfg.epochs, lr=qat_wino_cfg.lr,
                weight_decay=qat_wino_cfg.weight_decay, use_amp=False,
            )
            wino_best_path = checkpoints_dir / f"qat_wino_{model_name}_best.pth"
            resume_from = auto_resume_path(checkpoints_dir, f"qat_wino_{model_name}")
            qat_wino_fit: dict[str, Any] = {}
            trainer = Trainer(
                qat_wino_model,
                train_loader,
                val_loader,
                qat_wino_cfg_run,
                device,
                checkpoints_dir,
                f"qat_wino_{model_name}",
                num_classes=data_cfg.num_classes,
                wandb_run=wandb_run,
                metrics_callback=metrics_callback,
                log_file=logs_dir / f"qat_wino_{model_name}.log",
            )
            if wino_best_path.exists() and resume_from is None:
                logger.info("Skipping qat_wino fit for %s; best checkpoint exists.", model_name)
                wino_ckpt = torch.load(wino_best_path, map_location=str(device), weights_only=False)
                qat_wino_model.load_state_dict(wino_ckpt.get("model_state_dict", wino_ckpt))
            else:
                qat_wino_fit = trainer.fit(resume_from=resume_from)
                if _stop_requested(trainer, "qat_wino", model_name, writer, wandb_run):
                    break
                trainer.logger.info("qat_wino training complete for %s", model_name)
            qat_wino_eval = trainer.evaluate(topk=(1, 5))
            wino_summary = {
                "name": model_name,
                "stage": "qat_wino",
                "fit": qat_wino_fit,
                "eval": qat_wino_eval,
                "calibration": dump_wino_calibration(qat_wino_model),
                "checkpoint": str(wino_best_path),
            }
            create_results_summary(wino_summary, resolved_config, results_dir / f"{model_name}_qat_wino_summary.json")

        if "int8" in stage_list:
            int8_model = None
            if qat_model is not None:
                int8_model = convert_to_int8(qat_model)
                torch.save(int8_model, int8_path)
            elif int8_path.exists():
                int8_model = torch.load(int8_path, map_location="cpu")

            if int8_model is not None:
                int8_model = int8_model.to("cpu")
                int8_trainer = Trainer(
                    int8_model,
                    train_loader,
                    val_loader,
                    replace(model_cfg, use_amp=False),
                    torch.device("cpu"),
                    checkpoints_dir,
                    f"qat_{model_name}",
                    num_classes=data_cfg.num_classes,
                    wandb_run=wandb_run,
                    log_file=logs_dir / f"qat_{model_name}_int8.log",
                )
                int8_eval = int8_trainer.evaluate(topk=(1, 5))
                int8_benchmark = int8_trainer.benchmark(warmup=int(runtime_cfg.get("benchmark_warmup", 100)))

        fp32_model = load_best_model(model_name, spec["ctor"], checkpoints_dir, device) if best_model_path.exists() else spec["ctor"]().to(device)
        flops_results = compute_flops(fp32_model)
        fp32_size_mb = disk_mb(best_model_path)
        int8_size_mb = disk_mb(int8_path) if int8_path.exists() else None
        compress_checkpoint(best_model_path)
        if int8_path.exists():
            compress_checkpoint(int8_path)
        fp32_gzip_mb = gzip_mb(best_model_path)
        int8_gzip_mb = gzip_mb(int8_path) if int8_path.exists() else None

        summary = make_run_summary(
            name=model_name,
            mode="/".join(stage_list),
            fit_results=fp32_fit or qat_fit,
            fp32_eval=fp32_eval or {"top1": None, "top5": None, "loss": None},
            params_m=sum(p.numel() for p in fp32_model.parameters()) / 1e6,
            fp32_size_mb=fp32_size_mb or 0.0,
            int8_size_mb=int8_size_mb or 0.0,
            fp32_benchmark=fp32_benchmark,
            flops_results=flops_results,
            int8_eval=int8_eval,
            int8_benchmark=int8_benchmark,
            fp32_gzip_mb=fp32_gzip_mb,
            int8_gzip_mb=int8_gzip_mb,
        )
        create_results_summary(summary, resolved_config, results_dir / f"{model_name}_summary.json")
        results_rows.append(summary)
        if writer is not None:
            writer.flush()
            writer.close()
        if wandb_run is not None:
            wandb_run.finish()

    _append_csv(results_rows, runtime_paths.aggregates / f"{experiment_name}_comparison.csv")
    return results_rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a reproducible training experiment locally or on PCAD.")
    parser.add_argument("--experiment", default="default", help="Experiment config file name or path")
    parser.add_argument("--runtime", default="local", help="Runtime profile name or path")
    parser.add_argument("--device", default=None, help="Override device, e.g. cuda or cpu")
    parser.add_argument("--model", default=None, help="Restrict the experiment to a single registered model name")
    parser.add_argument("--dry-run", action="store_true", help="Load configs and print the resolved run plan")
    parser.add_argument("--smoke", action="store_true", help="Cap epochs to 1 for a fast local pipeline check before a PCAD submission")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    experiment_cfg = load_profile(args.experiment, "experiments")
    runtime_cfg = load_profile(args.runtime, "runtime")
    if args.device:
        runtime_cfg["device"] = args.device
    if args.model:
        experiment_cfg["models"] = [args.model]
    if args.smoke:
        experiment_cfg = _apply_smoke_override(experiment_cfg)
        runtime_cfg["wandb"] = False  # smoke runs discard all output; an offline wandb run would escape the temp dir
    if args.dry_run:
        print(json.dumps({"experiment": experiment_cfg, "runtime": runtime_cfg}, indent=2, default=str))
        return 0
    if args.smoke:
        # Everything run_experiment() writes (checkpoints/logs/tensorboard/resolved_config.json/
        # aggregates CSV) lands under this temp root and is discarded on exit -- a smoke run
        # must never touch or overwrite a real run's output. On failure, the exception's
        # traceback still prints to stderr before cleanup, which is the "error message" a smoke
        # run should surface -- no separate error log needed.
        with tempfile.TemporaryDirectory(prefix="smoke_") as tmp_root:
            runtime_cfg["root"] = tmp_root
            run_experiment(experiment_cfg, runtime_cfg)
    else:
        run_experiment(experiment_cfg, runtime_cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
