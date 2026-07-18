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
import logging
import os
import sys
from pathlib import Path
from dataclasses import asdict, replace

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import yaml

from ml import (
    DetSegDataConfig, TrainerConfig, DetectionTrainer,
    create_voc_detection_loaders, build_ssd_detector,
)
from ml.det_seg_models import build_qat_ssd_detector, convert_ssd_to_int8
from ml.quantization import make_qat_callback
from ml.runtime import expand_path


def load_yaml(path: str) -> dict:
    """Load YAML config file."""
    with open(expand_path(path)) as f:
        return yaml.safe_load(f)


def run_detection(args):
    """Run detection training (FP32, QAT, or INT8 stage)."""
    print(f"\n{'='*60}")
    print(f"DETECTION EXPERIMENT: {args.model} [{args.stage.upper()}]")
    print(f"{'='*60}\n")

    # Load configs
    data_cfg = DetSegDataConfig(**load_yaml("configs/detection.yaml").get("data", {}))
    trainer_cfg = TrainerConfig(**load_yaml("configs/detection.yaml").get("trainer", {}))

    # Override from experiment config if provided
    if args.experiment:
        exp_cfg = load_yaml(f"configs/experiments/{args.experiment}.yaml")
        data_cfg = replace(data_cfg, **exp_cfg.get("data", {}))
        trainer_cfg = replace(trainer_cfg, **exp_cfg.get("trainer", {}))

    # Adjust trainer config for QAT (shorter epochs, lower lr, no AMP)
    if args.stage == "qat":
        trainer_cfg = replace(trainer_cfg, epochs=15, lr=1e-5, use_amp=False)

    # Setup paths
    stage_suffix = {"fp32": "fp32", "qat": "qat", "int8": "int8"}[args.stage]
    run_id = f"ssd_{args.model}_{stage_suffix}"
    run_dir = Path(args.save_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config_out = run_dir / "config.yaml"
    with open(config_out, "w") as f:
        yaml.dump({"data": asdict(data_cfg), "trainer": asdict(trainer_cfg), "stage": args.stage}, f)
    print(f"Config saved to {config_out}")

    if args.dry_run:
        print(f"\n[DRY-RUN] Would run {args.stage.upper()} detection. Exiting.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.stage == "fp32":
        # ========== FP32 Training ==========
        # Load data
        print(f"\nLoading VOC detection data...")
        data_cfg.voc_root = expand_path(data_cfg.voc_root)
        train_ds, val_ds, train_loader, val_loader = create_voc_detection_loaders(data_cfg)
        print(f"  Train: {len(train_ds)} | Val: {len(val_ds)}")

        # Build model
        print(f"\nBuilding SSD detector ({args.model})...")
        model = build_ssd_detector(args.model, num_classes=21, image_size=256)
        print(f"  Model ready. Parameter count: {sum(p.numel() for p in model.parameters()):,}")

        # Train
        print(f"\nStarting FP32 training...")
        trainer = DetectionTrainer(
            model, train_loader, val_loader, trainer_cfg, device,
            save_dir=run_dir, run_name=run_id,
            num_classes=21,
            log_file=run_dir / f"{run_id}.log"
        )

        history = trainer.fit(resume_from=run_dir / f"{run_id}_resume.pth")

    elif args.stage == "qat":
        # ========== QAT Fine-tuning ==========
        # Load FP32 checkpoint
        print(f"\nLoading FP32 checkpoint...")
        fp32_run_id = f"ssd_{args.model}_fp32"
        fp32_ckpt = Path(args.save_dir) / fp32_run_id / f"{fp32_run_id}_best.pth"
        if not fp32_ckpt.exists():
            print(f"ERROR: FP32 checkpoint not found at {fp32_ckpt}")
            print(f"Make sure you run FP32 training first: python {__file__} detection --model {args.model} --stage fp32")
            return

        model = build_ssd_detector(args.model, num_classes=21, image_size=256)
        ckpt_state = torch.load(fp32_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt_state)
        model.to(device)
        print(f"  ✓ Loaded FP32 checkpoint: {fp32_ckpt}")

        # Prepare for QAT
        print(f"Preparing model for QAT...")
        model_qat = build_qat_ssd_detector(model, device)
        print(f"  ✓ Model prepared with fused Conv-BN and fake-quant observers")

        # Load data
        print(f"\nLoading VOC detection data...")
        data_cfg.voc_root = expand_path(data_cfg.voc_root)
        train_ds, val_ds, train_loader, val_loader = create_voc_detection_loaders(data_cfg)
        print(f"  Train: {len(train_ds)} | Val: {len(val_ds)}")

        # QAT training with epoch callback for observer scheduling
        print(f"\nStarting QAT fine-tuning...")
        epoch_callback = make_qat_callback(freeze_bn_epoch=3, disable_observer_epoch=8)
        trainer = DetectionTrainer(
            model_qat, train_loader, val_loader, trainer_cfg, device,
            save_dir=run_dir, run_name=run_id,
            num_classes=21,
            epoch_callback=epoch_callback,
            log_file=run_dir / f"{run_id}.log"
        )

        history = trainer.fit(resume_from=run_dir / f"{run_id}_resume.pth")

    elif args.stage == "int8":
        # ========== INT8 Conversion & Evaluation ==========
        # Load QAT checkpoint
        print(f"\nLoading QAT checkpoint...")
        qat_run_id = f"ssd_{args.model}_qat"
        qat_ckpt = Path(args.save_dir) / qat_run_id / f"{qat_run_id}_best.pth"
        if not qat_ckpt.exists():
            print(f"ERROR: QAT checkpoint not found at {qat_ckpt}")
            print(f"Make sure you run QAT training first: python {__file__} detection --model {args.model} --stage qat")
            return

        model_qat = build_ssd_detector(args.model, num_classes=21, image_size=256)
        model_qat = build_qat_ssd_detector(model_qat, device)
        ckpt_state = torch.load(qat_ckpt, map_location=device, weights_only=False)
        model_qat.load_state_dict(ckpt_state)
        model_qat.to(device)
        print(f"  ✓ Loaded QAT checkpoint: {qat_ckpt}")

        # Convert to INT8
        print(f"Converting to INT8...")
        model_int8 = convert_ssd_to_int8(model_qat)
        print(f"  ✓ INT8 conversion complete (backbone on CPU)")

        # Load data
        print(f"\nLoading VOC detection data...")
        data_cfg.voc_root = expand_path(data_cfg.voc_root)
        train_ds, val_ds, train_loader, val_loader = create_voc_detection_loaders(data_cfg)
        print(f"  Train: {len(train_ds)} | Val: {len(val_ds)}")

        # Evaluate INT8 model
        print(f"\nEvaluating INT8 model...")
        trainer = DetectionTrainer(
            model_int8, train_loader, val_loader, trainer_cfg, device,
            save_dir=run_dir, run_name=run_id,
            num_classes=21,
            log_file=run_dir / f"{run_id}.log"
        )

        # Run validation only (no training)
        val_mAP, val_mAP50 = trainer._validate(model_int8)
        history = {
            "val_mAP": [val_mAP],
            "val_mAP50": [val_mAP50],
            "note": "INT8 evaluation only (no training)"
        }
        print(f"  INT8 mAP@[.5:.95]: {val_mAP:.4f}")
        print(f"  INT8 mAP@.5: {val_mAP50:.4f}")

    # Save final results
    results_path = run_dir / "metrics.json"
    with open(results_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Save git hash for reproducibility
    os.system(f"git rev-parse HEAD > {run_dir / 'git_hash.txt'}")


def run_segmentation(args):
    """Run segmentation training (placeholder)."""
    print(f"\n{'='*60}")
    print(f"SEGMENTATION EXPERIMENT: {args.model} (PLACEHOLDER)")
    print(f"{'='*60}\n")
    print("[PLACEHOLDER] Segmentation training not yet implemented")


def main():
    parser = argparse.ArgumentParser(description="Phase 7 detection/segmentation training")
    parser.add_argument("task", choices=["detection", "segmentation"], help="Task to run")
    parser.add_argument("--model", default="alexnet_bottleneck", help="Model architecture")
    parser.add_argument("--stage", choices=["fp32", "qat", "int8"], default="fp32", help="Training stage")
    parser.add_argument("--experiment", help="Experiment config name (optional)")
    parser.add_argument("--runtime", choices=["local", "pcad"], default="local", help="Where to run")
    parser.add_argument("--save-dir", default="runs", help="Output directory")
    parser.add_argument("--dry-run", action="store_true", help="Don't train, just show config")

    args = parser.parse_args()

    if args.runtime == "pcad":
        print("\n[CLUSTER MODE] Would submit to PCAD. Use: sbatch scripts/slurm/det_seg.sbatch")
        return

    if args.task == "detection":
        run_detection(args)
    else:
        run_segmentation(args)


if __name__ == "__main__":
    main()
