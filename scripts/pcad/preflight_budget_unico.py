"""Pre-flight check for configs/experiments/budget_unico.yaml before burning a
PCAD SLURM allocation on it.

⚠️ Passing this on a laptop only proves the CODE is right. It does NOT prove
PCAD is ready -- run it there too, inside the same conda env train.sbatch
activates (`conda activate alexnet_rafael`), before `scripts.cluster
submit-sweep`. The one thing this repo's own environment.yml/requirements.txt
never had to account for is scripts/train.py's new "qat_wino" stage, which
needs the SIBLING Winograd-FPGA repo's scripts/avaliacao_redes/{qat_wino.py,
models_wino.py} + scripts/{gen_f43_system_vectors.py,validate_transforms_f43.py}
+ scripts/avaliacao_redes/{net_manifest.py,eligibility_wino.py} (6 pure
numpy/torch files, no Vivado/RTL) reachable at $WINOGRAD_FPGA_ROOT/.. or at
~/Documents/Winograd-FPGA (the default). If PCAD's home directory doesn't
have that tree, every *_fpga model's FP32 stage AND every model's qat_wino
stage fail at construction time -- this check catches that here, not after
a job has sat in the SLURM queue for hours.

Usage:
    python -m scripts.pcad.preflight_budget_unico
    python -m scripts.pcad.preflight_budget_unico --experiment budget_unico --runtime pcad
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_RESULTS: list[tuple[str, bool, str]] = []


def _check(label: str, fn) -> None:
    try:
        detail = fn()
        print(f"  OK   {label}" + (f" -- {detail}" if detail else ""))
        _RESULTS.append((label, True, str(detail or "")))
    except Exception as e:
        print(f"  FAIL {label}: {e}")
        _RESULTS.append((label, False, str(e)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default="budget_unico")
    ap.add_argument("--runtime", default="pcad")
    args = ap.parse_args()

    scripts_dir = str(Path(__file__).resolve().parents[1])
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)

    print("[1] core Python dependencies (environment.yml / requirements.txt)")
    for mod in ("torch", "torchvision", "yaml", "pandas", "tensorboard",
                "wandb", "kagglehub", "torchmetrics"):
        _check(mod, lambda mod=mod: __import__(mod))

    print("\n[2] CUDA")
    def _cuda():
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("torch.cuda.is_available() is False -- would train on CPU")
        return torch.cuda.get_device_name(0)
    _check("torch.cuda.is_available()", _cuda)

    print("\n[3] Winograd-FPGA bridge (qat_wino stage -- see WINOGRAD_FPGA_ROOT above)")
    def _bridge():
        from train import _import_qat_wino
        return _import_qat_wino().__file__
    _check("qat_wino.py importable", _bridge)

    print(f"\n[4] experiment config: {args.experiment!r} / runtime: {args.runtime!r}")
    exp_cfg, rt_cfg = {}, {}
    def _cfg():
        nonlocal exp_cfg, rt_cfg
        from configs.loader import load_config
        exp_cfg = load_config(f"experiments/{args.experiment}.yaml")
        rt_cfg = load_config(f"runtime/{args.runtime}.yaml")
        return f"{len(exp_cfg.get('models', []))} models, stages={exp_cfg.get('stages')}"
    _check("load yaml", _cfg)

    print(f"\n[5] every model in {args.experiment!r} constructs (this is what would fail on PCAD if "
          f"Winograd-FPGA isn't synced there -- 8 of these models import it at construction time)")
    if exp_cfg.get("models"):
        import ml.model_registrations  # noqa: F401 -- populates MODEL_REGISTRY
        from ml import MODEL_REGISTRY
        for name in exp_cfg["models"]:
            def _build(name=name):
                if name not in MODEL_REGISTRY:
                    raise KeyError(f"{name!r} not in MODEL_REGISTRY")
                m = MODEL_REGISTRY[name]["ctor"]()
                n = sum(p.numel() for p in m.parameters())
                return f"{n / 1e6:.2f}M params"
            _check(name, _build)

    print("\n[6] dataset resolution")
    if rt_cfg:
        def _dataset():
            from ml import expand_path, resolve_dataset_train_path
            root = expand_path(rt_cfg.get("dataset_root"))
            train_path = resolve_dataset_train_path(root) if root else None
            if train_path is not None:
                return str(train_path)
            if rt_cfg.get("use_kagglehub_fallback", True):
                return f"dataset_root={root} not found -- kagglehub fallback ENABLED, will download on first run"
            raise FileNotFoundError(f"dataset_root={root} not found and kagglehub fallback disabled")
        _check("dataset_root / kagglehub fallback", _dataset)

    n_fail = sum(1 for _, ok, _ in _RESULTS if not ok)
    print(f"\n{'ALL CHECKS PASSED' if n_fail == 0 else f'{n_fail} CHECK(S) FAILED'} "
          f"({len(_RESULTS)} total) -- {'ready to submit' if n_fail == 0 else 'fix before scripts.cluster submit-sweep'}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
