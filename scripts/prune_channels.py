"""Structured (channel) pruning CLI — Phase 9 Task 2.

    python -m scripts.prune_channels --model alexnet_bottleneck --ratio 0.4 --runtime local --dry-run
    python -m scripts.prune_channels --model alexnet_bottleneck --ratio 0.4 --runtime local --evaluate

Mechanics-only measurement pass (ideas/PHASE9_PLAN.md Task 2): prunes each
`_AlexBottleneck` block's internal squeeze width, then checks the result still
forward-passes and (with --evaluate) still runs through Trainer.evaluate() without
shape errors. No fine-tuning here — accuracy after pruning is expected to be poor.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

import ml.model_registrations  # noqa: F401 — populates MODEL_REGISTRY
from ml import (
    MODEL_REGISTRY,
    DataConfig,
    TrainerConfig,
    Trainer,
    build_runtime_paths,
    create_imagenet_loaders,
    load_best_model,
    set_global_seed,
)
from ml.pruning import bottleneck_prune_plan, prune_model_channels
from configs.loader import load_config
from scripts.train import _ensure_dataset_path, _load_profile, _make_model_runs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Structured channel pruning for _AlexBottleneck-based models.")
    parser.add_argument("--model", required=True, help="Registered model name (e.g. alexnet_bottleneck)")
    parser.add_argument("--ratio", type=float, default=0.4, help="Fraction of each bottleneck's mid_ch to remove")
    parser.add_argument("--runtime", default="local", help="Runtime profile name (resolves the default checkpoint path)")
    parser.add_argument("--experiment", default="default", help="Experiment name (resolves the default checkpoint path)")
    parser.add_argument("--checkpoint", default=None, help="Explicit checkpoint path override (skips runtime/experiment resolution)")
    parser.add_argument("--dry-run", action="store_true", help="Print the channel-removal plan only; no checkpoint, no model build")
    parser.add_argument("--evaluate", action="store_true", help="Also run Trainer.evaluate() on the pruned model (needs the dataset)")
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
        data_cfg = DataConfig(**load_config("data.yaml"))
        dataset_path = _ensure_dataset_path(runtime_cfg)
        data_cfg.dataset_path = str(dataset_path)
        _, _, _, val_loader = create_imagenet_loaders(data_cfg)

        trainer_cfg = TrainerConfig(**load_config("training.yaml"))
        trainer = Trainer(
            model, val_loader, val_loader, trainer_cfg, device,
            Path("outputs") / "phase9_pruning", f"{args.model}_pruned",
            num_classes=data_cfg.num_classes,
        )
        metrics = trainer.evaluate(topk=(1, 5))
        print(f"Pruned (no fine-tune) | top1={metrics['top1']:.2f}% | top5={metrics['top5']:.2f}% | loss={metrics['loss']:.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
