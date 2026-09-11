from __future__ import annotations

import argparse
import importlib
import json
import logging
import os
import signal
import sys
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
    capture_provenance,
    compress_checkpoint,
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


def _build_qat_wino_config(base_cfg: dict[str, Any], experiment_cfg: dict[str, Any]) -> QATConfig:
    qat_wino_cfg = QATConfig(**base_cfg)
    overrides = experiment_cfg.get("qat_wino", {}) or {}
    return replace(qat_wino_cfg, **overrides)


def _apply_smoke_override(experiment_cfg: dict[str, Any]) -> dict[str, Any]:
    """Cap fp32/QAT epochs to 1 so a full pipeline run (data/model/checkpoint/
    QAT-convert) finishes in minutes locally, to catch bugs before a PCAD submission."""
    experiment_cfg["training"] = {**experiment_cfg.get("training", {}), "epochs": 1, "warmup_epochs": 0}
    experiment_cfg["qat"] = {**experiment_cfg.get("qat", {}), "epochs": 1}
    return experiment_cfg


def _import_qat_wino():
    """Bridge to the Winograd-FPGA sibling repo's accelerator-numeric QAT
    (scripts/avaliacao_redes/qat_wino.py). Not vendored here on purpose --
    that repo's README explains why the two projects don't cross-import model
    *definitions* (this repo would silently change FPGA-side numbers), but
    qat_wino.convert() is a generic nn.Conv2d->WinoQuantConv2d swap, not a
    model definition, so importing the module itself is fine.
    """
    default_root = Path.home() / "Documents" / "Winograd-FPGA" / "scripts" / "avaliacao_redes"
    root = Path(os.environ.get("WINOGRAD_FPGA_ROOT", default_root)).expanduser()
    if not (root / "qat_wino.py").exists():
        raise FileNotFoundError(
            f"qat_wino.py not found at {root} -- set WINOGRAD_FPGA_ROOT to the "
            "Winograd-FPGA repo's scripts/avaliacao_redes directory"
        )
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return importlib.import_module("qat_wino")


def _load_qat_wino_model(model_name: str, spec: dict[str, Any], checkpoints_dir: Path, device):
    """FP32 best checkpoint -> qat_wino.convert() in place.

    A fresh instance built this way every time (rather than caching) so its
    state_dict keys always match a qat_wino checkpoint saved from a model
    built the same way -- convert() adds buffers (act_absmax, post_shift,
    sat_frac, BT/AT/G) that a plain ctor() instance doesn't have.
    """
    qat_wino = _import_qat_wino()
    model = load_best_model(model_name, spec["ctor"], checkpoints_dir, device, eval_mode=False)
    trocadas = qat_wino.convert(model)
    if not trocadas:
        raise RuntimeError(f"{model_name}: qat_wino.convert() found no eligible 3x3 conv to replace")
    # Safety net for the class of bug this stage almost shipped with: convert()
    # silently SKIPS any 3x3 conv with stride/groups/dilation != 1 (see its own
    # "nao elegivel: converta antes (SS4)" comment) rather than erroring. A model
    # registered without first applying the SS4 stride-2->stride-1+maxpool
    # conversion (models/wino_adapted.py) would train qat_wino fine with that
    # one layer silently left un-Winograd'd -- no exception, just a wrong number.
    ineligible = [n for n, m in model.named_modules()
                  if isinstance(m, torch.nn.Conv2d) and m.kernel_size[0] == 3
                  and (m.stride[0] != 1 or m.groups != 1 or m.dilation[0] != 1)]
    if ineligible:
        logging.getLogger(f"pcad_runner.{model_name}").warning(
            "%s: %d 3x3 conv(s) left un-Winograd'd by qat_wino.convert() "
            "(stride/groups/dilation not eligible): %s -- register a "
            "models/wino_adapted.py wrapper that converts stride-2 first",
            model_name, len(ineligible), ineligible)
    logging.getLogger(f"pcad_runner.{model_name}").info(
        "qat_wino converted %d conv layer(s): %s", len(trocadas), trocadas)
    return model.to(device)


def _dump_wino_calibration(model) -> dict[str, Any]:
    """Per-layer act_absmax/post_shift/sat_frac -- the §5.1 calibration fields
    the plan calls irrecoverable without a re-eval. Duck-typed (not isinstance)
    because qat_wino.make_wino_conv() builds its conv class inside a closure.
    """
    calib: dict[str, Any] = {}
    for name, m in model.named_modules():
        if hasattr(m, "act_absmax") and hasattr(m, "post_shift") and hasattr(m, "sat_frac"):
            calib[name or "root"] = {
                "act_absmax": float(m.act_absmax.item()),
                "post_shift": int(round(float(m.post_shift.item()))),
                "sat_frac": float(m.sat_frac.item()),
            }
    return calib


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
    qat_wino_cfg = _build_qat_wino_config(load_config("qat_wino.yaml"), experiment_cfg)
    uniform_hparams = bool(experiment_cfg.get("uniform_hparams", False))

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
    provenance = capture_provenance()

    for model_name in selected_models:
        spec = MODEL_REGISTRY[model_name]
        model_run_name = f"{experiment_name}_{model_name}"
        run_root, checkpoints_dir, logs_dir, tb_dir, results_dir = _make_model_runs(runtime_paths.root, experiment_name, model_name)

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

        if "qat_wino" in stage_list:
            # Fase 2 do plano de avaliacao Winograd (accelerator-numeric QAT,
            # not fbgemm's per-channel one): fine-tune with the FPGA's own
            # per-tensor pow2 requant + F(4,3) transform in the forward, via
            # scripts/avaliacao_redes/qat_wino.py in the sibling Winograd-FPGA
            # repo. No int8-convert step here -- unlike the fbgemm path,
            # WinoQuantConv2d.forward() already simulates the quantized
            # pipeline in eval mode, so evaluating this checkpoint directly
            # *is* the accelerator-numeric accuracy (top1_int8_accel proxy).
            qat_wino_model = _load_qat_wino_model(model_name, spec, checkpoints_dir, device)
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
                trainer.logger.info("qat_wino training complete for %s", model_name)
            qat_wino_eval = trainer.evaluate(topk=(1, 5))
            wino_summary = {
                "name": model_name,
                "stage": "qat_wino",
                "fit": qat_wino_fit,
                "eval": qat_wino_eval,
                "calibration": _dump_wino_calibration(qat_wino_model),
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
    experiment_cfg = _load_experiment_config(args.experiment)
    runtime_cfg = _load_runtime_config(args.runtime)
    if args.device:
        runtime_cfg["device"] = args.device
    if args.model:
        experiment_cfg["models"] = [args.model]
    if args.smoke:
        experiment_cfg = _apply_smoke_override(experiment_cfg)
    if args.dry_run:
        print(json.dumps({"experiment": experiment_cfg, "runtime": runtime_cfg}, indent=2, default=str))
        return 0
    run_experiment(experiment_cfg, runtime_cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
