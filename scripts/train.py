from __future__ import annotations

import argparse
import json
import logging
import signal
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import torch
import yaml
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
    compute_flops,
    convert_to_int8,
    create_imagenet_loaders,
    create_results_summary,
    disk_mb,
    gzip_mb,
    expand_path,
    make_qat_callback,
    make_run_summary,
    load_best_model,
    resolve_dataset_train_path,
    set_global_seed,
)


def _load_yaml(path_or_name: str) -> dict[str, Any]:
    candidate = Path(path_or_name)
    if candidate.exists():
        with candidate.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}
    return load_config(path_or_name)


def _load_profile(name_or_path: str, subdir: str) -> dict[str, Any]:
    candidate = Path(name_or_path)
    if candidate.exists():
        return _load_yaml(name_or_path)
    return load_config(f"{subdir}/{name_or_path}.yaml")


def _resolve_model_names(model_names: list[str] | str | None) -> list[str]:
    if not model_names or model_names == "all":
        return list(MODEL_REGISTRY.keys())
    return [name for name in model_names if name in MODEL_REGISTRY]


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


def _load_runtime_config(runtime_name: str) -> dict[str, Any]:
    return _load_profile(runtime_name, "runtime")


def _load_experiment_config(experiment_name: str) -> dict[str, Any]:
    return _load_profile(experiment_name, "experiments")


def _ensure_dataset_path(runtime_cfg: dict[str, Any]) -> Path:
    dataset_root = expand_path(runtime_cfg.get("dataset_root"))
    dataset_train = resolve_dataset_train_path(dataset_root) if dataset_root else None
    if dataset_train is not None:
        return dataset_train

    if not runtime_cfg.get("use_kagglehub_fallback", True):
        raise FileNotFoundError("dataset_root is unset and KaggleHub fallback is disabled")

    import kagglehub

    dataset_path = kagglehub.dataset_download(runtime_cfg.get("kaggle_dataset", "akash2sharma/tiny-imagenet"))
    return Path(dataset_path) / "tiny-imagenet-200" / "train"


def _make_model_runs(root: Path, experiment_name: str, model_name: str) -> tuple[Path, Path, Path, Path, Path]:
    run_root = root / experiment_name / model_name
    checkpoints = run_root / "checkpoints"
    logs = run_root / "logs"
    tensorboard = run_root / "tensorboard"
    results = run_root / "results"
    for path in (checkpoints, logs, tensorboard, results):
        path.mkdir(parents=True, exist_ok=True)
    return run_root, checkpoints, logs, tensorboard, results


def _save_resolved_config(run_root: Path, config: dict[str, Any]) -> None:
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "resolved_config.json").write_text(json.dumps(config, indent=2, default=str))


def run_experiment(experiment_cfg: dict[str, Any], runtime_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    runtime_root = expand_path(runtime_cfg.get("root"), default="outputs/local") or Path("outputs/local")
    runtime_paths = build_runtime_paths(runtime_root)

    data_cfg = _build_data_config(load_config("data.yaml"), experiment_cfg)
    trainer_cfg = _build_trainer_config(load_config("training.yaml"), experiment_cfg)
    qat_cfg = _build_qat_config(load_config("qat.yaml"), experiment_cfg)

    seed = int(experiment_cfg.get("seed", data_cfg.seed))
    data_cfg.seed = seed
    set_global_seed(seed)

    dataset_path = _ensure_dataset_path(runtime_cfg)
    data_cfg.dataset_path = str(dataset_path)

    train_ds, val_ds, train_loader, val_loader = create_imagenet_loaders(data_cfg, persistent_workers=runtime_cfg.get("persistent_workers", False))

    experiment_name = experiment_cfg.get("name") or "experiment"
    stage_list = experiment_cfg.get("stages", ["fp32", "qat", "int8"])
    selected_models = _resolve_model_names(experiment_cfg.get("models"))
    if not selected_models:
        raise ValueError("No valid model names were selected")

    results_rows: list[dict[str, Any]] = []
    torch.backends.quantized.engine = runtime_cfg.get("quantized_engine", "fbgemm")
    device = torch.device(runtime_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

    for model_name in selected_models:
        spec = MODEL_REGISTRY[model_name]
        model_run_name = f"{experiment_name}_{model_name}"
        run_root, checkpoints_dir, logs_dir, tb_dir, results_dir = _make_model_runs(runtime_paths.root, experiment_name, model_name)

        model_cfg = replace(trainer_cfg, lr=spec.get("lr", trainer_cfg.lr))
        resolved_config = {
            "experiment": experiment_cfg,
            "runtime": runtime_cfg,
            "data": asdict(data_cfg),
            "training": asdict(model_cfg),
            "qat": asdict(qat_cfg),
            "selected_model": model_name,
            "stage_list": stage_list,
        }
        _save_resolved_config(run_root, resolved_config)

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
            if trainer is not None:
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
                fp32_eval = trainer.evaluate(topk=(1, 5))
                fp32_benchmark = trainer.benchmark(warmup=int(runtime_cfg.get("benchmark_warmup", 100)))

        qat_model = None
        qat_fit = {}
        if "qat" in stage_list:
            qat_model = build_qat(model_name, save_dir=checkpoints_dir, device=device)
            qat_cfg_run = replace(model_cfg, epochs=qat_cfg.epochs, lr=qat_cfg.lr, weight_decay=qat_cfg.weight_decay, use_amp=False)
            resume_from = auto_resume_path(checkpoints_dir, f"qat_{model_name}")
            if (checkpoints_dir / f"qat_{model_name}_best.pth").exists() and resume_from is None:
                logger.info("Skipping QAT stage for %s; best checkpoint exists.", model_name)
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
                trainer.logger.info("QAT training complete for %s", model_name)

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

    _append_csv(results_rows, runtime_paths.results / f"{experiment_name}_comparison.csv")
    return results_rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a reproducible training experiment locally or on PCAD.")
    parser.add_argument("--experiment", default="default", help="Experiment config file name or path")
    parser.add_argument("--runtime", default="local", help="Runtime profile name or path")
    parser.add_argument("--device", default=None, help="Override device, e.g. cuda or cpu")
    parser.add_argument("--model", default=None, help="Restrict the experiment to a single registered model name")
    parser.add_argument("--dry-run", action="store_true", help="Load configs and print the resolved run plan")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    experiment_cfg = _load_experiment_config(args.experiment)
    runtime_cfg = _load_runtime_config(args.runtime)
    if args.device:
        runtime_cfg["device"] = args.device
    if args.model:
        experiment_cfg["models"] = [args.model]
    if args.dry_run:
        print(json.dumps({"experiment": experiment_cfg, "runtime": runtime_cfg}, indent=2, default=str))
        return 0
    run_experiment(experiment_cfg, runtime_cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
