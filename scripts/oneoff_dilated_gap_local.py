"""
One-off local smoke run for alexnet_dilated_gap.

Lighter budget than the PCAD large_scale protocol (configs/experiments/
alexnet_dilated_gap.yaml: 1000 epochs/patience 20 FP32, 100 epochs QAT) --
100 epochs/patience 5 FP32, 20 epochs QAT here -- to get a quick local
accuracy/sanity number on the RTX 4060 without committing it to the full
budget. Not a source of truth for the report: the PCAD config (unchanged)
is what produces the number comparable to the other large_scale models.

Usage:
    python -m scripts.oneoff_dilated_gap_local
"""
from scripts.train import run_experiment, _load_runtime_config

EXPERIMENT_CFG = {
    "name": "alexnet_dilated_gap_local_oneoff",
    "models": ["alexnet_dilated_gap"],
    "seed": 42,
    "stages": ["fp32", "qat", "int8"],
    "data": {
        "batch_size": 64,
        "num_workers": 4,
        "pin_memory": True,
        "train_val_split": 0.9,
    },
    "training": {
        "epochs": 100,
        "lr": 3.0e-4,
        "weight_decay": 5.0e-4,
        "label_smoothing": 0.1,
        "grad_clip_norm": None,
        "use_amp": True,
        "early_stopping_patience": 5,
    },
    "qat": {
        "epochs": 20,
        "lr": 1.0e-5,
        "weight_decay": 5.0e-4,
        "freeze_bn_epoch": 3,
        "disable_observer_epoch": 5,
    },
}

if __name__ == "__main__":
    run_experiment(EXPERIMENT_CFG, _load_runtime_config("local"))
