#!/usr/bin/env python
"""
CLI for Phase 7 detection/segmentation experiments.
Mirrors scripts/train.py structure but for VOC detection/segmentation.

Usage:
    python scripts/train_det_seg.py detection --model alexnet_bottleneck --dry-run
    python scripts/train_det_seg.py detection --model alexnet_bottleneck --runtime pcad
"""
import argparse
import json
import signal
import sys
import tempfile
from pathlib import Path
from dataclasses import asdict, replace

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import yaml

# Default 'file_descriptor' sharing strategy exhausts /dev/shm when many workers pass large
# (512px) tensors between processes — this is what caused job 805529's mid-epoch segfault.
# 'file_system' avoids the shared-memory limit entirely, letting num_workers scale safely.
torch.multiprocessing.set_sharing_strategy("file_system")

from ml import (
    DetSegDataConfig, TrainerConfig, DetectionTrainer, SegmentationTrainer,
    create_voc_detection_loaders, create_voc_segmentation_loaders,
    build_ssd_detector, build_deeplabv3_segmenter,
)
from ml.det_seg_models import (
    build_qat_ssd_detector, convert_ssd_to_int8, compute_anchor_recall,
    build_qat_deeplabv3_segmenter, convert_deeplabv3_to_int8,
)
from ml.quantization import make_qat_callback
from ml.reporting import compute_detection_summary, compute_segmentation_summary
from ml.runtime import capture_provenance, expand_path, load_runtime_root, set_global_seed
from configs.loader import load_config


# Everything detection and segmentation don't share. The stage pipeline itself (fp32 ->
# qat -> int8, run_id naming, checkpoint hand-off between stages) is identical, and one
# copy of it means a fix lands once instead of twice.
TASKS = {
    "detection": dict(
        prefix="ssd", config="detection.yaml", label="SSD detector",
        loaders=create_voc_detection_loaders, build=build_ssd_detector,
        build_qat=build_qat_ssd_detector, to_int8=convert_ssd_to_int8,
        trainer=DetectionTrainer, summary=compute_detection_summary,
        int8_metrics=(("val_mAP", "mAP@[.5:.95]"), ("val_mAP50", "mAP@.5")),
    ),
    "segmentation": dict(
        prefix="seg", config="segmentation.yaml", label="DeepLabV3 segmenter",
        loaders=create_voc_segmentation_loaders, build=build_deeplabv3_segmenter,
        build_qat=build_qat_deeplabv3_segmenter, to_int8=convert_deeplabv3_to_int8,
        trainer=SegmentationTrainer, summary=compute_segmentation_summary,
        int8_metrics=(("val_loss", "val loss"), ("val_mIoU", "mIoU")),
    ),
}


def run_id_for(args, stage: str) -> str:
    """<ssd|seg>_<model>_<stage>[_pretrained][_<experiment>] -- also how the qat/int8 stages
    find the previous stage's checkpoint. _pretrained keeps a pretrained-init sweep from
    clobbering the from-scratch one's checkpoints/logs."""
    run_id = f"{TASKS[args.task]['prefix']}_{args.model}_{stage}"
    if args.pretrained_ckpt:
        run_id += "_pretrained"
    if args.experiment:
        run_id += f"_{args.experiment}"
    return run_id


def run(args):
    """Run one detection/segmentation stage (FP32, QAT, or INT8)."""
    task = TASKS[args.task]
    print(f"\n{'='*60}")
    print(f"{args.task.upper()} EXPERIMENT: {args.model} [{args.stage.upper()}]")
    print(f"{'='*60}\n")

    # Load configs
    base_cfg = load_config(task["config"])
    data_cfg = DetSegDataConfig(**base_cfg.get("data", {}))
    trainer_cfg = TrainerConfig(**base_cfg.get("trainer", {}))

    # Override from experiment config if provided
    if args.experiment:
        exp_cfg = load_config(f"experiments/{args.experiment}.yaml")
        if args.model in exp_cfg:  # per-model-keyed format (e.g. phase_7_detection.yaml)
            exp_cfg = exp_cfg[args.model]
        data_cfg = replace(data_cfg, **exp_cfg.get("data", {}))
        trainer_cfg = replace(trainer_cfg, **exp_cfg.get("trainer", {}))

    set_global_seed(data_cfg.seed)

    trainer = None

    def _request_stop(_signum, _frame):
        if trainer is not None:
            trainer.request_stop()

    signal.signal(signal.SIGTERM, _request_stop)
    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGUSR1, _request_stop)  # Slurm pre-timeout warning (see train.sbatch)

    # Adjust trainer config for QAT (shorter epochs, lower lr, no AMP).
    # Disabling AMP roughly doubles activation memory at the same batch size, which
    # OOM'd on 24GB cards at img_size 512 (e.g. job 811098) — halve batch size to
    # compensate.
    if args.stage == "qat":
        trainer_cfg = replace(trainer_cfg, epochs=100, lr=1e-5, use_amp=False)
        data_cfg = replace(data_cfg, batch_size=max(1, data_cfg.batch_size // 2))

    if args.smoke:
        trainer_cfg = replace(trainer_cfg, epochs=1)

    run_id = run_id_for(args, args.stage)
    run_dir = Path(args.save_dir) / run_id

    # ponytail: dry-run stays read-only — writing config.yaml first would clobber the
    # provenance record of an existing run that shares this run_id.
    if args.dry_run:
        print(f"\n[DRY-RUN] Would run {args.stage.upper()} {args.task} in {run_dir}. Exiting.")
        return

    run_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config_out = run_dir / "config.yaml"
    with open(config_out, "w") as f:
        yaml.dump({"data": asdict(data_cfg), "trainer": asdict(trainer_cfg), "stage": args.stage}, f)
    print(f"Config saved to {config_out}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _load_data():
        print(f"\nLoading VOC {args.task} data...")
        data_cfg.voc_root = expand_path(data_cfg.voc_root)
        train_ds, val_ds, train_loader, val_loader = task["loaders"](data_cfg)
        print(f"  Train: {len(train_ds)} | Val: {len(val_ds)}")
        return train_loader, val_loader

    def _previous_stage_ckpt(stage: str) -> Path:
        prev_id = run_id_for(args, stage)
        ckpt = Path(args.save_dir) / prev_id / f"{prev_id}_best.pth"
        if not ckpt.exists():
            print(f"ERROR: {stage.upper()} checkpoint not found at {ckpt}")
            print(f"Make sure you run {stage.upper()} training first: "
                  f"python {__file__} {args.task} --model {args.model} --stage {stage}")
            sys.exit(1)
        return ckpt

    def _load_into(model, ckpt: Path):
        ckpt_state = torch.load(ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt_state.get("model_state_dict", ckpt_state))
        model.to(device)
        print(f"  ✓ Loaded {ckpt}")

    # Data is loaded at a different point in each stage on purpose (fp32 before the model
    # is built, qat/int8 after) -- kept as-is so model init and data RNG draws stay in the
    # same order as every run already on disk.
    if args.stage in ("fp32", "qat"):
        if args.stage == "fp32":
            train_loader, val_loader = _load_data()
            print(f"\nBuilding {task['label']} ({args.model})...")
            if args.pretrained_ckpt:
                print(f"  Initializing backbone from: {args.pretrained_ckpt}")
            model = task["build"](
                args.model, num_classes=21, image_size=data_cfg.img_size,
                pretrained_ckpt=args.pretrained_ckpt,
            )
            print(f"  Model ready. Parameter count: {sum(p.numel() for p in model.parameters()):,}")

            # Anchor-recall pre-flight gate: mAP is capped regardless of training quality
            # if ground-truth boxes aren't covered by any default box. max_samples=1000
            # (compute_anchor_recall's default) keeps this to a few seconds even on the
            # full VOC07-test val set — the earlier "too slow" complaint was actually an
            # unbounded max_samples check, since fixed.
            if args.task == "detection" and not args.skip_anchor_check:
                recall = compute_anchor_recall(model, val_loader, iou_threshold=0.5)
                print(f"  Anchor recall @IoU 0.5: {recall:.3f}")
                if recall < 0.95:
                    print(f"ABORT: anchor recall {recall:.3f} < 0.95 — fix anchor config first "
                          f"(or pass --skip-anchor-check to override).")
                    sys.exit(1)
            epoch_callback = None
            print(f"\nStarting FP32 training...")
        else:
            print(f"\nLoading FP32 checkpoint...")
            ckpt = _previous_stage_ckpt("fp32")
            model = task["build"](args.model, num_classes=21, image_size=data_cfg.img_size)
            _load_into(model, ckpt)
            print(f"Preparing model for QAT...")
            model = task["build_qat"](model, device)
            print(f"  ✓ Model prepared with fused Conv-BN and fake-quant observers")
            train_loader, val_loader = _load_data()
            # QAT training with epoch callback for observer scheduling
            epoch_callback = make_qat_callback(freeze_bn_epoch=3, disable_observer_epoch=8)
            print(f"\nStarting QAT fine-tuning...")

        trainer = task["trainer"](
            model, train_loader, val_loader, trainer_cfg, device,
            save_dir=run_dir, run_name=run_id,
            num_classes=21,
            epoch_callback=epoch_callback,
            log_file=run_dir / f"{run_id}.log"
        )
        history = trainer.fit(resume_from=run_dir / f"{run_id}_resume.pth")
        history["summary"] = task["summary"](
            model, data_cfg.img_size, val_loader, device,
            checkpoint_path=run_dir / f"{run_id}_best.pth",
        )

    elif args.stage == "int8":
        print(f"\nLoading QAT checkpoint...")
        ckpt = _previous_stage_ckpt("qat")
        model_qat = task["build"](args.model, num_classes=21, image_size=data_cfg.img_size)
        # True architecture param count, from the untouched FP32 skeleton -- quantized modules
        # pack weights as torch.qint8 buffers, not nn.Parameter, so counting on the converted
        # INT8 model itself (below) silently undercounts.
        true_params_m = sum(p.numel() for p in model_qat.parameters()) / 1e6
        model_qat = task["build_qat"](model_qat, device)
        _load_into(model_qat, ckpt)

        print(f"Converting to INT8...")
        model_int8 = task["to_int8"](model_qat)
        print(f"  ✓ INT8 conversion complete (backbone on CPU)")

        train_loader, val_loader = _load_data()

        # INT8 convert/inference is CPU-only (project convention) — model_int8 already lives on
        # CPU post-conversion, so the eval trainer must target CPU too, not `device` (which is
        # CUDA when available and would send images to the wrong device).
        print(f"\nEvaluating INT8 model...")
        trainer = task["trainer"](
            model_int8, train_loader, val_loader, trainer_cfg, torch.device("cpu"),
            save_dir=run_dir, run_name=run_id,
            num_classes=21,
            log_file=run_dir / f"{run_id}.log"
        )

        # Run validation only (no training)
        val_metrics = trainer._validate(model_int8, criterion=None)
        history = {key: [val_metrics[key]] for key, _ in task["int8_metrics"]}
        history["note"] = "INT8 evaluation only (no training)"
        for key, label in task["int8_metrics"]:
            print(f"  INT8 {label}: {val_metrics[key]:.4f}")

        # Save the converted checkpoint and its real size -- fp32/qat get one from fit(); int8
        # has no fit() call, so past runs' metrics.json has accuracy but no summary (see
        # scripts/phase7/backfill_int8_size*.py for backfilling those).
        int8_ckpt_path = run_dir / f"{run_id}_best.pth"
        torch.save(model_int8.state_dict(), int8_ckpt_path)
        history["summary"] = task["summary"](
            model_int8, data_cfg.img_size, val_loader, torch.device("cpu"),
            checkpoint_path=int8_ckpt_path,
        )
        history["summary"]["params_m"] = true_params_m

    # Save final results
    results_path = run_dir / "metrics.json"
    with open(results_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Save git hash for reproducibility
    (run_dir / "git_hash.txt").write_text(capture_provenance()["git_hash"] + "\n")


def main():
    parser = argparse.ArgumentParser(description="Phase 7 detection/segmentation training")
    parser.add_argument("task", choices=list(TASKS), help="Task to run")
    parser.add_argument("--model", default="alexnet_bottleneck", help="Model architecture")
    parser.add_argument("--stage", choices=["fp32", "qat", "int8"], default="fp32", help="Training stage")
    parser.add_argument("--experiment", help="Experiment config name (optional)")
    parser.add_argument("--runtime", choices=["local", "pcad"], default="local", help="Where to run")
    parser.add_argument("--save-dir", default=None, help="Output directory (default: <runtime root>/phase_7_detection_segmentation)")
    parser.add_argument("--dry-run", action="store_true", help="Don't train, just show config")
    parser.add_argument("--smoke", action="store_true", help="Cap epochs to 1 for a fast local pipeline check")
    parser.add_argument("--skip-anchor-check", action="store_true", help="Skip the anchor-recall pre-flight gate")
    parser.add_argument(
        "--pretrained-ckpt", type=Path, default=None,
        help="Tiny-ImageNet classification checkpoint to init the SSD backbone from (fp32 stage only)",
    )

    args = parser.parse_args()
    if args.save_dir is None:
        args.save_dir = str(load_runtime_root(args.runtime) / "phase_7_detection_segmentation")

    if args.runtime == "pcad":
        print("\n[CLUSTER MODE] Would submit to PCAD. Use: sbatch scripts/slurm/det_seg.sbatch")
        return

    if args.smoke:
        # Checkpoints/logs/metrics.json/git_hash.txt all land under args.save_dir -- redirect it
        # to a temp dir so a smoke run never overwrites a real run's output; deleted on exit
        # either way, so a failure's traceback (printed to stderr before cleanup) is the only
        # trace it leaves, which is the point.
        with tempfile.TemporaryDirectory(prefix="smoke_") as tmp_dir:
            args.save_dir = tmp_dir
            run(args)
        return

    run(args)


if __name__ == "__main__":
    main()
