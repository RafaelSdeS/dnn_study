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
from ml.runtime import expand_path


def load_yaml(path: str) -> dict:
    """Load YAML config file."""
    with open(expand_path(path)) as f:
        return yaml.safe_load(f)


def run_detection(args):
    """Run detection training."""
    print(f"\n{'='*60}")
    print(f"DETECTION EXPERIMENT: {args.model}")
    print(f"{'='*60}\n")

    # Load configs
    data_cfg = DetSegDataConfig(**load_yaml("configs/detection.yaml").get("data", {}))
    trainer_cfg = TrainerConfig(**load_yaml("configs/detection.yaml").get("trainer", {}))

    # Override from experiment config if provided
    if args.experiment:
        exp_cfg = load_yaml(f"configs/experiments/{args.experiment}.yaml")
        data_cfg = replace(data_cfg, **exp_cfg.get("data", {}))
        trainer_cfg = replace(trainer_cfg, **exp_cfg.get("trainer", {}))

    # Setup paths
    run_id = f"ssd_{args.model}_fp32"
    run_dir = Path(args.save_dir) / "phase7" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config_out = run_dir / "config.yaml"
    with open(config_out, "w") as f:
        yaml.dump({"data": asdict(data_cfg), "trainer": asdict(trainer_cfg)}, f)
    print(f"Config saved to {config_out}")

    if args.dry_run:
        print("\n[DRY-RUN] Would run detection training. Exiting.")
        return

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
    print(f"\nStarting training...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = DetectionTrainer(
        model, train_loader, val_loader, trainer_cfg, device,
        save_dir=run_dir, run_name=run_id,
        num_classes=21,
        log_file=run_dir / f"{run_id}.log"
    )

    history = trainer.fit(resume_from=run_dir / f"{run_id}_resume.pth")

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
