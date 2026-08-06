"""Phase 6 Extension — Winograd INT8 Quantization Error Analysis.

Measures the error contribution of Winograd F(2×2,3×3) transforms when applied
to INT8-quantized convolutions, isolating whether the transform's additive mixing
amplifies quantization error beyond direct INT8 convolution.

Usage:
    python -m scripts.winograd_quant_error --experiment phase6 --runtime local [--dry-run]
    python -m scripts.winograd_quant_error --experiment phase6 --runtime pcad [--dry-run]

Outputs JSON to: outputs/{runtime}/phase6/winograd_quant_error.json
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import torch

from ml.quantization import load_best_model, build_qat_from_model, convert_to_int8
from ml.quantization_advanced import (
    winograd_eligible_layers,
    compute_winograd_quant_error,
)
from ml.registry import MODEL_REGISTRY
from ml.runtime import set_global_seed, build_runtime_paths
from ml.config import DataConfig
from ml.data import create_imagenet_loaders
import ml.model_registrations  # noqa: F401 — populates MODEL_REGISTRY


def _load_runtime_config(runtime_name: str) -> dict:
    from configs.loader import load_config
    return load_config(f"runtime/{runtime_name}.yaml")


def _load_experiment_config(experiment_name: str) -> dict:
    from configs.loader import load_config
    return load_config(f"experiments/{experiment_name}.yaml")


def _ensure_dataset_path(runtime_cfg: dict) -> Path:
    from ml.runtime import expand_path
    dataset_root = expand_path(runtime_cfg.get("dataset_root"))
    if dataset_root and dataset_root.exists():
        return dataset_root
    import kagglehub
    return Path(kagglehub.dataset_download("akash2sharma/tiny-imagenet"))


def main():
    parser = argparse.ArgumentParser(
        description="Analyze Winograd INT8 quantization error for phase 6 models."
    )
    parser.add_argument("--experiment", default="phase6", help="Experiment config file name")
    parser.add_argument("--runtime", default="local", help="Runtime profile (local or pcad)")
    parser.add_argument("--device", default=None, help="Override device (cuda/cpu)")
    parser.add_argument("--dry-run", action="store_true", help="Show config and exit")

    args = parser.parse_args()

    runtime_cfg = _load_runtime_config(args.runtime)
    if args.device:
        runtime_cfg["device"] = args.device

    # Load configs
    try:
        exp_cfg = _load_experiment_config(args.experiment)
    except FileNotFoundError:
        exp_cfg = {"models": list(MODEL_REGISTRY.keys())}

    if args.dry_run:
        print(json.dumps({"experiment": exp_cfg, "runtime": runtime_cfg}, indent=2, default=str))
        return

    # Setup
    set_global_seed(42)
    device = torch.device(runtime_cfg["device"])
    torch.backends.quantized.engine = runtime_cfg.get("quantized_engine", "fbgemm")

    runtime_root = Path(runtime_cfg.get("root", "outputs/local"))
    output_dir = runtime_root / "phase6"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "winograd_quant_error.json"
    log_file = output_dir / "winograd_quant_error.log"

    logger = logging.getLogger("winograd_analysis")
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(log_file)
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(handler)
    logger.addHandler(logging.StreamHandler(sys.stdout))

    logger.info(f"Starting Winograd INT8 quantization error analysis")
    logger.info(f"Runtime: {args.runtime}, Device: {device}")
    logger.info(f"Output: {output_file}")

    # Load data
    data_cfg = DataConfig()
    _, val_loader = create_imagenet_loaders(data_cfg)

    # Results
    results = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "experiment": args.experiment,
            "runtime": args.runtime,
            "device": str(device),
        },
        "models": {}
    }

    # Process each model
    model_names = exp_cfg.get("models", list(MODEL_REGISTRY.keys()))
    for model_name in model_names:
        if model_name not in MODEL_REGISTRY:
            logger.warning(f"Model {model_name} not in registry, skipping")
            continue

        logger.info(f"Analyzing {model_name}...")
        spec = MODEL_REGISTRY[model_name]

        try:
            # Load best checkpoint (or uncalibrated QAT for INT8)
            save_dir = runtime_root / "phase3" / model_name if model_name.startswith("alexnet") else runtime_root / "phase5" / model_name
            save_dir = Path(str(save_dir).replace("phase3", "outputs/local").replace("phase5", "outputs/local"))  # hack: find checkpoints
            if not save_dir.exists():
                save_dir = runtime_root / "outputs" / model_name  # fallback
            if not save_dir.exists():
                logger.warning(f"  Checkpoint dir not found for {model_name}, trying to build uncalibrated QAT")
                model = spec["ctor"](num_classes=200).to(device)
            else:
                model = load_best_model(model_name, spec["ctor"], save_dir, device, eval_mode=True)

            # Check Winograd eligibility
            eligible = winograd_eligible_layers(model)
            logger.info(f"  {len(eligible)} Winograd-eligible layer(s)")

            if not eligible:
                logger.info(f"  No eligible layers, skipping")
                results["models"][model_name] = {"eligible_layers": 0, "error": "no_eligible"}
                continue

            # Compute error
            winograd_err = compute_winograd_quant_error(model, val_loader, device, n_samples=512)
            results["models"][model_name] = {
                "eligible_layers": len(eligible),
                **winograd_err,
            }
            logger.info(f"  Mean ratio: {winograd_err.get('mean_ratio', 'N/A'):.4f}")

        except Exception as e:
            logger.error(f"  Error processing {model_name}: {e}")
            results["models"][model_name] = {"error": str(e)}

    # Save results
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {output_file}")


if __name__ == "__main__":
    main()
