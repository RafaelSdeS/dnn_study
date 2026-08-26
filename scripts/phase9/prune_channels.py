"""Structured (channel) pruning CLI — Phase 9 Task 2.

    python -m scripts.phase9.prune_channels --model alexnet_bottleneck --ratio 0.4 --runtime local --dry-run
    python -m scripts.phase9.prune_channels --model alexnet_bottleneck --ratio 0.4 --runtime local --evaluate
    python -m scripts.phase9.prune_channels --model alexnet_bottleneck --ratio 0.4 --runtime local --finetune-epochs 200

Mechanics pass (ideas/PHASE9_PLAN.md Task 2): prunes each `_AlexBottleneck` block's
internal squeeze width, then checks the result still forward-passes and (with
--evaluate) still runs through Trainer.evaluate() without shape errors — no
fine-tuning, accuracy after pruning is expected to be poor.

--finetune-epochs closes the gap the plan leaves open ("a fine-tuning loop to
recover pruned accuracy... remains future work"): fine-tunes the pruned model
(FP32), then runs it through the project's standard QAT->INT8 flow, producing a
*_summary.json directly comparable to every other model's.
"""

from __future__ import annotations

import argparse
import signal
from dataclasses import replace
from pathlib import Path

import torch

import ml.model_registrations  # noqa: F401 — populates MODEL_REGISTRY
from ml import (
    MODEL_REGISTRY,
    DataConfig,
    QATConfig,
    TrainerConfig,
    Trainer,
    auto_resume_path,
    build_qat_from_model,
    build_runtime_paths,
    compress_checkpoint,
    compute_flops,
    convert_to_int8,
    create_imagenet_loaders,
    create_results_summary,
    disk_mb,
    gzip_mb,
    load_best_model,
    make_qat_callback,
    make_run_summary,
    set_global_seed,
)
from ml.pruning import bottleneck_prune_plan, prune_model_channels
from configs.loader import load_config
from scripts.train import _ensure_dataset_path, _load_profile, _make_model_runs, _save_resolved_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Structured channel pruning for _AlexBottleneck-based models.")
    parser.add_argument("--model", required=True, help="Registered model name (e.g. alexnet_bottleneck)")
    parser.add_argument("--ratio", type=float, default=0.4, help="Fraction of each bottleneck's mid_ch to remove")
    parser.add_argument("--runtime", default="local", help="Runtime profile name (resolves the default checkpoint path)")
    parser.add_argument("--experiment", default="default", help="Experiment name (resolves the default checkpoint path)")
    parser.add_argument("--checkpoint", default=None, help="Explicit checkpoint path override (skips runtime/experiment resolution)")
    parser.add_argument("--dry-run", action="store_true", help="Print the channel-removal plan only; no checkpoint, no model build")
    parser.add_argument("--evaluate", action="store_true", help="Also run Trainer.evaluate() on the pruned, unfine-tuned model (needs the dataset)")
    parser.add_argument("--finetune-epochs", type=int, default=0,
                         help="If > 0, fine-tune the pruned model (FP32) for up to this many epochs, "
                              "then run it through QAT->INT8. 0 (default) skips fine-tuning entirely.")
    parser.add_argument("--finetune-lr", type=float, default=1e-4,
                         help="Fine-tune learning rate (between full training's 3e-4 and QAT's 1e-5)")
    parser.add_argument("--finetune-patience", type=int, default=20, help="Early-stopping patience for the fine-tune stage")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.model not in MODEL_REGISTRY:
        raise SystemExit(f"Unknown model {args.model!r}; registered: {list(MODEL_REGISTRY)}")
    spec = MODEL_REGISTRY[args.model]

    plan = bottleneck_prune_plan(spec["ctor"](), args.ratio)
    if not plan:
        raise SystemExit(f"{args.model} has no _AlexBottleneck blocks; nothing to prune.")

    print(f"Pruning plan for {args.model} (ratio={args.ratio}):")
    for name, before, after in plan:
        print(f"  {name:30s} mid_ch {before:4d} -> {after:4d}")

    if args.dry_run:
        return 0

    set_global_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
        model = spec["ctor"]()
        state = torch.load(checkpoint_path, map_location=str(device), weights_only=False)
        model.load_state_dict(state.get("model_state_dict", state))
        model = model.to(device)
    else:
        runtime_cfg = _load_profile(args.runtime, "runtime")
        runtime_paths = build_runtime_paths(runtime_cfg.get("root", "outputs/local"))
        _, checkpoints_dir, _, _, _ = _make_model_runs(runtime_paths.root, args.experiment, args.model)
        checkpoint_path = checkpoints_dir / f"{args.model}_best.pth"
        if not checkpoint_path.exists():
            if args.finetune_epochs > 0:
                raise SystemExit(f"No checkpoint at {checkpoint_path} -- can't fine-tune pruned random-init weights.")
            print(f"No checkpoint at {checkpoint_path}; pruning random-init weights (shape check only).")
            model = spec["ctor"]().to(device)
        else:
            model = load_best_model(args.model, spec["ctor"], checkpoints_dir, device)

    before_params = sum(p.numel() for p in model.parameters())
    prune_model_channels(model, args.ratio)
    after_params = sum(p.numel() for p in model.parameters())
    print(f"params: {before_params:,} -> {after_params:,}  ({after_params / before_params:.1%})")

    model.eval()
    x = torch.randn(1, 3, 64, 64, device=device)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (1, 200), f"unexpected output shape: {y.shape}"
    assert all(m.groups == 1 for m in model.modules() if isinstance(m, torch.nn.Conv2d)), \
        "pruning produced a non-dense Conv2d — Winograd-eligibility broken"
    print("Forward pass OK, every remaining Conv2d still dense (groups=1).")

    if args.evaluate:
        runtime_cfg = _load_profile(args.runtime, "runtime")
        runtime_paths = build_runtime_paths(runtime_cfg.get("root", "outputs/local"))
        data_cfg = DataConfig(**load_config("data.yaml"))
        dataset_path = _ensure_dataset_path(runtime_cfg)
        data_cfg.dataset_path = str(dataset_path)
        _, _, _, val_loader = create_imagenet_loaders(data_cfg)

        trainer_cfg = TrainerConfig(**load_config("training.yaml"))
        trainer = Trainer(
            model, val_loader, val_loader, trainer_cfg, device,
            runtime_paths.root / "phase9_pruning", f"{args.model}_pruned",
            num_classes=data_cfg.num_classes,
        )
        metrics = trainer.evaluate(topk=(1, 5))
        print(f"Pruned (no fine-tune) | top1={metrics['top1']:.2f}% | top5={metrics['top5']:.2f}% | loss={metrics['loss']:.4f}")

    if args.finetune_epochs > 0:
        runtime_cfg = _load_profile(args.runtime, "runtime")
        data_cfg = DataConfig(**load_config("data.yaml"))
        dataset_path = _ensure_dataset_path(runtime_cfg)
        data_cfg.dataset_path = str(dataset_path)
        _finetune_and_quantize(args, spec, model, data_cfg, runtime_cfg, device)

    return 0


def _finetune_and_quantize(
    args: argparse.Namespace,
    spec: dict,
    pruned_model: torch.nn.Module,
    data_cfg: DataConfig,
    runtime_cfg: dict,
    device: torch.device,
) -> None:
    """Fine-tune a pruned model (FP32), then run it through the project's standard QAT->INT8
    flow — closes the gap ideas/PHASE9_PLAN.md leaves open (H2's fine-tuning loop, future work).
    Mirrors scripts/train.py's per-model FP32->QAT->INT8 body, but starting from an
    already-pruned model instead of a fresh MODEL_REGISTRY ctor.
    """
    run_name = f"{args.model}_pruned_r{args.ratio}"
    runtime_paths = build_runtime_paths(runtime_cfg.get("root", "outputs/local"))
    run_root, checkpoints_dir, logs_dir, _tb_dir, results_dir = _make_model_runs(
        runtime_paths.root, "phase9_pruning_finetune", run_name
    )
    _save_resolved_config(run_root, {
        "model": args.model, "ratio": args.ratio, "finetune_epochs": args.finetune_epochs,
        "finetune_lr": args.finetune_lr, "finetune_patience": args.finetune_patience,
        "runtime": runtime_cfg,
    })

    _train_ds, _val_ds, train_loader, val_loader = create_imagenet_loaders(
        data_cfg, persistent_workers=runtime_cfg.get("persistent_workers", False)
    )
    torch.backends.quantized.engine = runtime_cfg.get("quantized_engine", "fbgemm")

    # SIGUSR1 (Slurm's pre-timeout warning, see train.sbatch) -> request_stop() -> Trainer writes
    # its resume checkpoint and exits cleanly, same pattern scripts/train.py uses.
    active_trainer: dict[str, Trainer | None] = {"trainer": None}

    def _request_stop(_signum, _frame):
        if active_trainer["trainer"] is not None:
            active_trainer["trainer"].request_stop()

    signal.signal(signal.SIGTERM, _request_stop)
    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGUSR1, _request_stop)

    finetune_cfg = replace(
        TrainerConfig(**load_config("training.yaml")),
        epochs=args.finetune_epochs, lr=args.finetune_lr, early_stopping_patience=args.finetune_patience,
    )
    trainer = Trainer(
        pruned_model, train_loader, val_loader, finetune_cfg, device,
        checkpoints_dir, run_name, num_classes=data_cfg.num_classes,
        log_file=logs_dir / f"{run_name}.log",
    )
    active_trainer["trainer"] = trainer
    fit_results = trainer.fit(resume_from=auto_resume_path(checkpoints_dir, run_name))
    fp32_eval = trainer.evaluate(topk=(1, 5))
    fp32_benchmark = trainer.benchmark(warmup=int(runtime_cfg.get("benchmark_warmup", 100)))
    print(f"Pruned + fine-tuned (FP32) | top1={fp32_eval['top1']:.2f}% | top5={fp32_eval['top5']:.2f}%")

    # QAT needs the *best* fine-tuned checkpoint (not necessarily the last epoch) loaded into a
    # freshly-pruned model of the same shape -- build_qat()'s usual ctor()-fresh path would give
    # back the unpruned architecture, so the prune step is repeated on a clean instance instead.
    best_model = spec["ctor"]()
    prune_model_channels(best_model, args.ratio)
    best_path = checkpoints_dir / f"{run_name}_best.pth"
    state = torch.load(best_path, map_location=str(device), weights_only=False)
    best_model.load_state_dict(state["model_state_dict"])
    best_model = best_model.to(device)

    qat_cfg = QATConfig(**load_config("qat.yaml"))
    qat_model = build_qat_from_model(best_model, args.model, device)
    qat_trainer_cfg = replace(finetune_cfg, epochs=qat_cfg.epochs, lr=qat_cfg.lr, weight_decay=qat_cfg.weight_decay, use_amp=False)
    qat_trainer = Trainer(
        qat_model, train_loader, val_loader, qat_trainer_cfg, device,
        checkpoints_dir, f"qat_{run_name}", num_classes=data_cfg.num_classes,
        epoch_callback=make_qat_callback(qat_cfg.freeze_bn_epoch, qat_cfg.disable_observer_epoch),
        log_file=logs_dir / f"qat_{run_name}.log",
    )
    active_trainer["trainer"] = qat_trainer
    qat_trainer.fit(resume_from=auto_resume_path(checkpoints_dir, f"qat_{run_name}"))

    int8_model = convert_to_int8(qat_model)
    int8_path = checkpoints_dir / f"qat_{run_name}.pth"
    torch.save(int8_model, int8_path)
    int8_trainer = Trainer(
        int8_model.to("cpu"), val_loader, val_loader, replace(qat_trainer_cfg, use_amp=False),
        torch.device("cpu"), checkpoints_dir, f"qat_{run_name}", num_classes=data_cfg.num_classes,
    )
    int8_eval = int8_trainer.evaluate(topk=(1, 5))
    int8_benchmark = int8_trainer.benchmark(warmup=int(runtime_cfg.get("benchmark_warmup", 100)))
    print(f"Pruned + fine-tuned (INT8) | top1={int8_eval['top1']:.2f}% | top5={int8_eval['top5']:.2f}%")

    flops_results = compute_flops(best_model)
    fp32_size_mb = disk_mb(best_path)
    int8_size_mb = disk_mb(int8_path)
    compress_checkpoint(best_path)
    compress_checkpoint(int8_path)
    fp32_gzip_mb = gzip_mb(best_path)
    int8_gzip_mb = gzip_mb(int8_path)

    summary = make_run_summary(
        name=run_name, mode="prune_finetune_qat_int8", fit_results=fit_results,
        fp32_eval=fp32_eval, params_m=sum(p.numel() for p in best_model.parameters()) / 1e6,
        fp32_size_mb=fp32_size_mb or 0.0, int8_size_mb=int8_size_mb or 0.0,
        fp32_benchmark=fp32_benchmark, flops_results=flops_results,
        int8_eval=int8_eval, int8_benchmark=int8_benchmark,
        fp32_gzip_mb=fp32_gzip_mb, int8_gzip_mb=int8_gzip_mb,
    )
    summary["prune_ratio"] = args.ratio
    summary_path = results_dir / f"{run_name}_summary.json"
    create_results_summary(summary, {"model": args.model, "ratio": args.ratio}, summary_path)
    print(f"Saved {summary_path}")


if __name__ == "__main__":
    raise SystemExit(main())
