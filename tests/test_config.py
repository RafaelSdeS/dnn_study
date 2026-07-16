"""Every YAML under configs/{experiments,runtime,slurm} must load into its dataclass cleanly.

Catches config drift (typo'd keys, missing fields) before it reaches a Slurm job.
"""
from pathlib import Path

import yaml

from configs.loader import load_config
from scripts.train import _build_data_config, _build_qat_config, _build_trainer_config

CONFIGS_DIR = Path(__file__).resolve().parents[1] / "configs"


def _experiment_names():
    return sorted(p.stem for p in (CONFIGS_DIR / "experiments").glob("*.yaml"))


def _runtime_names():
    return sorted(p.stem for p in (CONFIGS_DIR / "runtime").glob("*.yaml"))


def _slurm_names():
    return sorted(p.stem for p in (CONFIGS_DIR / "slurm").glob("*.yaml"))


def test_every_experiment_config_resolves_into_dataclasses():
    base_data = load_config("data.yaml")
    base_training = load_config("training.yaml")
    base_qat = load_config("qat.yaml")

    for name in _experiment_names():
        experiment_cfg = load_config(f"experiments/{name}.yaml")
        data_cfg = _build_data_config(base_data, experiment_cfg)
        trainer_cfg = _build_trainer_config(base_training, experiment_cfg)
        qat_cfg = _build_qat_config(base_qat, experiment_cfg)

        assert data_cfg.batch_size > 0, name
        assert trainer_cfg.epochs > 0, name
        assert qat_cfg.epochs > 0, name
        assert experiment_cfg.get("models"), f"{name} has no models"
        assert experiment_cfg.get("stages"), f"{name} has no stages"


def test_large_scale_experiment_has_the_expected_budgets():
    experiment_cfg = load_config("experiments/large_scale.yaml")
    assert experiment_cfg["training"]["epochs"] == 1000
    assert experiment_cfg["training"]["early_stopping_patience"] == 20
    assert experiment_cfg["qat"]["epochs"] == 100
    assert len(experiment_cfg["models"]) == 12


def test_every_runtime_config_is_a_dict_with_a_root():
    for name in _runtime_names():
        runtime_cfg = yaml.safe_load((CONFIGS_DIR / "runtime" / f"{name}.yaml").read_text())
        assert isinstance(runtime_cfg, dict), name
        assert runtime_cfg.get("root"), name


def test_every_slurm_config_has_a_partition_and_gres():
    for name in _slurm_names():
        slurm_cfg = yaml.safe_load((CONFIGS_DIR / "slurm" / f"{name}.yaml").read_text())
        assert isinstance(slurm_cfg, dict), name
        assert slurm_cfg.get("partition"), name
        assert slurm_cfg.get("gres"), name


def test_tupi_4090_slurm_config_has_requeue_and_signal():
    slurm_cfg = yaml.safe_load((CONFIGS_DIR / "slurm" / "tupi_4090.yaml").read_text())
    assert slurm_cfg["requeue"] is True
    assert slurm_cfg["signal"] == "B:USR1@300"
    assert slurm_cfg["time"] == "24:00:00"
