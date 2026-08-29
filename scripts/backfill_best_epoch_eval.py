"""Backfill FP32/INT8 accuracy for models hit by the last-epoch-not-best-epoch bug.

ml/trainer.py's Trainer.fit() used to return without reloading `{run_name}_best.pth` into
self.model, so scripts/train.py's post-fit evaluate() call measured the LAST epoch's
weights, not the best one -- while build_qat()/load_best_model() (the QAT stage's own
starting point) correctly loaded the best checkpoint. FP32 and INT8 ended up measured on
different weights, which read as spurious "INT8 accuracy gains" for any run with a long
post-peak early-stopping tail. Fixed in ml/trainer.py (fit() now restores best_path before
returning); see tests/test_trainer_smoke.py::test_fit_restores_best_checkpoint_not_last_epoch.

This backfills the models trained BEFORE that fix, using only checkpoints already on disk
-- no retraining. For each model:
  - Reload `{model}_best.pth[.gz]` (the correct FP32 weights were always saved, only the
    post-fit evaluate() read the wrong ones) and re-evaluate.
  - Where the full-precision QAT best checkpoint also survived (`qat_{model}_best.pth`),
    rebuild INT8 from it too -- same "load best, then convert" flow
    scripts/train_det_seg.py's int8 stage already used explicitly for detection/segmentation.

Every other model in the project is NOT affected and is left untouched: the Phase 1-4
notebooks and notebooks/phase_8_efficient_vit/vit_qat_phase8.ipynb all call
load_best_model() before evaluating (never relied on Trainer.fit()'s return-value model),
and scripts/train_det_seg.py's int8 stage always reloads the QAT *_best.pth explicitly.

INT8 caveat (all 6 models, for two different reasons): swin_pico_w2/w4/w8,
swin_pico_poolmixer, and hybrid_bottleneck_swin only kept the already-converted INT8 model
from their QAT run's LAST epoch (`qat_{model}.pth.gz`) -- the full-precision QAT best
checkpoint was never saved for them, so there's nothing to rebuild from without re-running
the QAT stage itself (~80-100 epochs each), not just re-evaluating.
alexnet_fire_bypass DID keep `qat_alexnet_fire_bypass_best.pth`, but that checkpoint was
trained 2026-07-23 -- BEFORE commit 565fef4 (2026-08-15) changed models/compensation.py's
_FireModule to route its concat through FloatFunctional instead of a bare torch.cat. The
checkpoint's state_dict has no `*.cat.activation_post_process.*` keys at all (that submodule
didn't exist yet), so it can't be loaded into today's build_qat_from_model() output --
loading with strict=False would silently leave that observer at its uncalibrated default
(it was never trained), producing a meaningless INT8 number instead of a merely-imprecise
one. This script detects the mismatch and falls back to the previously-measured INT8 value
for alexnet_fire_bypass too, printing why. Net effect: only FP32 accuracy is corrected here
for all 6 models; every INT8 number is left as previously measured (see report/ic_report.tex
for how the two caveats differ in what they say about that number's remaining meaning).

Needs the Tiny ImageNet-200 val split, downloaded via kagglehub on first run (same fallback
scripts/train.py uses) -- reconstructs the identical deterministic 90/10 split (seed 42) as
the original training runs, since every affected experiment config uses configs/data.yaml's
defaults (img_size/train_val_split/seed) unchanged.

Usage:
    python -m scripts.backfill_best_epoch_eval
    python -m scripts.backfill_best_epoch_eval --dry-run
"""
from __future__ import annotations

import argparse
import gzip
import json
import shutil
import tempfile
from dataclasses import replace
from pathlib import Path

import torch

import ml.model_registrations  # noqa: F401 -- populates MODEL_REGISTRY as an import side effect
from ml.config import DataConfig, TrainerConfig
from ml.data import create_imagenet_loaders
from ml.quantization import build_qat_from_model, convert_to_int8
from ml.registry import MODEL_REGISTRY
from ml.trainer import Trainer

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRATCH_DIR = REPO_ROOT / "outputs" / "local" / "_backfill_scratch"

# (model_name, run_dir relative to REPO_ROOT)
MODELS = [
    ("swin_pico_w2", "outputs/pcad/phase8/swin_pico_w2"),
    ("swin_pico_w4", "outputs/pcad/phase8/swin_pico_w4"),
    ("swin_pico_w8", "outputs/pcad/phase8/swin_pico_w8"),
    ("swin_pico_poolmixer", "outputs/pcad/phase8/swin_pico_poolmixer"),
    ("hybrid_bottleneck_swin", "outputs/pcad/phase8/hybrid_bottleneck_swin"),
    ("alexnet_fire_bypass",
     "outputs/pcad/phase_9_bypass_ablation/fire_bypass_large_scale/alexnet_fire_bypass"),
]


def _load_state_dict(path: Path, device: torch.device) -> dict:
    """torch.load a checkpoint that may be gzip-compressed (*.pth.gz)."""
    if path.suffix == ".gz":
        with tempfile.NamedTemporaryFile(suffix=".pth") as tmp:
            with gzip.open(path, "rb") as src:
                shutil.copyfileobj(src, tmp)
            tmp.flush()
            ckpt = torch.load(tmp.name, map_location=str(device), weights_only=False)
    else:
        ckpt = torch.load(path, map_location=str(device), weights_only=False)
    return ckpt.get("model_state_dict", ckpt)


def _find_checkpoint(run_dir: Path, stem: str) -> Path | None:
    plain = run_dir / "checkpoints" / f"{stem}.pth"
    gz = run_dir / "checkpoints" / f"{stem}.pth.gz"
    if plain.exists():
        return plain
    if gz.exists():
        return gz
    return None


def _ensure_dataset_path() -> Path:
    """Same fallback scripts/train.py's _ensure_dataset_path uses for the `local` runtime
    (configs/runtime/local.yaml has dataset_root: null, use_kagglehub_fallback: true)."""
    import kagglehub
    dataset_path = kagglehub.dataset_download("akash2sharma/tiny-imagenet")
    return Path(dataset_path) / "tiny-imagenet-200" / "train"


def evaluate_fp32(name: str, run_dir: Path, val_loader, device: torch.device) -> dict:
    ckpt_path = _find_checkpoint(run_dir, f"{name}_best")
    if ckpt_path is None:
        raise FileNotFoundError(f"no {name}_best checkpoint under {run_dir}")
    model = MODEL_REGISTRY[name]["ctor"]()
    model.load_state_dict(_load_state_dict(ckpt_path, device))
    model = model.to(device).eval()

    trainer = Trainer(
        model, val_loader, val_loader, TrainerConfig(), device,
        SCRATCH_DIR, f"{name}_fp32_reeval", num_classes=200,
    )
    result = trainer.evaluate(topk=(1, 5))
    print(f"  [{name}] FP32 (best epoch, re-evaluated): top1={result['top1']:.2f}% "
          f"top5={result['top5']:.2f}% loss={result['loss']:.4f}")
    return result


def evaluate_int8_from_qat_best(name: str, run_dir: Path, val_loader, device: torch.device) -> dict | None:
    ckpt_path = _find_checkpoint(run_dir, f"qat_{name}_best")
    if ckpt_path is None:
        print(f"  [{name}] no qat_{name}_best checkpoint on disk -- INT8 left as previously "
              f"measured (final QAT epoch, not best; see module docstring CAVEAT)")
        return None

    qat_model = build_qat_from_model(MODEL_REGISTRY[name]["ctor"](), name, device)
    state = _load_state_dict(ckpt_path, device)
    try:
        qat_model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        first_line = str(exc).strip().splitlines()[0]
        print(f"  [{name}] qat_{name}_best checkpoint doesn't match today's model code "
              f"(architecture changed since this ran -- {first_line}); INT8 left as "
              f"previously measured, not rebuilt")
        return None
    int8_model = convert_to_int8(qat_model).to("cpu")

    trainer = Trainer(
        int8_model, val_loader, val_loader, replace(TrainerConfig(), use_amp=False),
        torch.device("cpu"), SCRATCH_DIR, f"{name}_int8_reeval", num_classes=200,
    )
    result = trainer.evaluate(topk=(1, 5))
    print(f"  [{name}] INT8 (best QAT epoch, rebuilt+re-evaluated): top1={result['top1']:.2f}% "
          f"top5={result['top5']:.2f}% loss={result['loss']:.4f}")
    return result


def _update_summary(summary_path: Path, fp32_eval: dict, int8_eval: dict | None) -> None:
    summary = json.loads(summary_path.read_text())
    summary["fp32_top1"] = fp32_eval["top1"]
    summary["fp32_top5"] = fp32_eval["top5"]
    summary["fp32_loss"] = fp32_eval["loss"]
    summary["fp32_top1_top5_gap"] = fp32_eval["top5"] - fp32_eval["top1"]
    if int8_eval is not None:
        summary["int8_top1"] = int8_eval["top1"]
        summary["int8_top5"] = int8_eval["top5"]
        summary["int8_loss"] = int8_eval["loss"]
        summary["int8_top1_top5_gap"] = int8_eval["top5"] - int8_eval["top1"]
    if summary.get("int8_top1") is not None:
        summary["quantization_drop_top1"] = summary["fp32_top1"] - summary["int8_top1"]
    summary_path.write_text(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                         help="Evaluate and print, but don't write the summary JSONs.")
    args = parser.parse_args()

    torch.backends.quantized.engine = "fbgemm"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Resolving Tiny ImageNet-200 (kagglehub fallback, cached after first run)...")
    dataset_path = _ensure_dataset_path()
    data_cfg = DataConfig(dataset_path=str(dataset_path))
    _, val_ds, _, val_loader = create_imagenet_loaders(data_cfg)
    print(f"Val split: {len(val_ds)} images (seed={data_cfg.seed}, "
          f"train_val_split={data_cfg.train_val_split})")

    for name, run_dir_rel in MODELS:
        run_dir = REPO_ROOT / run_dir_rel
        summary_path = run_dir / "results" / f"{name}_summary.json"
        print(f"\n=== {name} ===")
        if not summary_path.exists():
            print(f"  SKIP: no summary at {summary_path}")
            continue

        fp32_eval = evaluate_fp32(name, run_dir, val_loader, device)
        int8_eval = evaluate_int8_from_qat_best(name, run_dir, val_loader, device)

        if args.dry_run:
            print(f"  [{name}] --dry-run: not writing {summary_path}")
        else:
            _update_summary(summary_path, fp32_eval, int8_eval)
            print(f"  [{name}] wrote corrected fields to {summary_path}")


if __name__ == "__main__":
    main()
