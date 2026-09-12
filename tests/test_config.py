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
        if "models" not in experiment_cfg:
            continue  # Phase 7 configs (scripts/train_det_seg.py) use a per-model schema, not this one
        data_cfg = _build_data_config(base_data, experiment_cfg)
        trainer_cfg = _build_trainer_config(base_training, experiment_cfg)
        qat_cfg = _build_qat_config(base_qat, experiment_cfg)

        assert data_cfg.batch_size > 0, name
        assert trainer_cfg.epochs > 0, name
        assert qat_cfg.epochs > 0, name
        assert experiment_cfg.get("models"), f"{name} has no models"
        assert experiment_cfg.get("stages"), f"{name} has no stages"


def test_experiment_name_matches_filename_stem():
    """scripts/train.py:251 uses experiment_cfg["name"] (not the --experiment filename) to build
    outputs/<runtime>/<name>/ -- a config whose declared name drifts from its own filename silently
    forks the run tree (e.g. name: phase8 inside phase_8_efficient_vit.yaml wrote to outputs/*/phase8/).
    """
    stems = set(_experiment_names())
    for name in stems:
        experiment_cfg = load_config(f"experiments/{name}.yaml")
        if "models" not in experiment_cfg:
            continue  # Phase 7 configs use a per-model schema with no top-level `name`
        declared = experiment_cfg.get("name")
        # None falls back to "experiment" in scripts/train.py; a match to this file's own
        # stem or another real experiment's stem (a resume config sharing its parent's
        # output dir, e.g. large_scale_fire_residual_resume.yaml -> name: large_scale) is fine.
        assert declared is None or declared == name or declared in stems, (
            f"{name}.yaml declares name={declared!r}, which is neither its own filename "
            f"nor an existing experiment -- this becomes outputs/<runtime>/{declared}/, "
            f"silently diverging from the config file name"
        )


def test_large_scale_experiment_has_the_expected_budgets():
    experiment_cfg = load_config("experiments/large_scale.yaml")
    assert experiment_cfg["training"]["epochs"] == 1000
    assert experiment_cfg["training"]["early_stopping_patience"] == 50
    assert experiment_cfg["qat"]["epochs"] == 100
    assert len(experiment_cfg["models"]) == 12


def test_extends_merges_parent_and_child_fields():
    """configs/loader.py's `extends:` resolution: parent loads first, child's dict-valued
    keys merge field-by-field on top (not a wholesale block replace) and win on conflicts.
    """
    child = load_config("experiments/phase_8_efficient_vit.yaml")
    parent = load_config("experiments/_protocols/phase_8_vit.yaml")

    # fields only in the parent protocol survive into the child
    assert child["training"]["lr"] == parent["training"]["lr"]
    assert child["qat"]["epochs"] == parent["qat"]["epochs"]
    # a field the child adds on top of the parent block is present, not overwritten away
    assert child["training"]["reset_scheduler_on_resume"] is True
    # "extends" itself never leaks into the resolved config
    assert "extends" not in child


def test_protocol_fragments_are_not_treated_as_experiments():
    """configs/experiments/_protocols/*.yaml are merge fragments, not runnable experiments --
    _experiment_names()'s non-recursive glob must not pick them up.
    """
    assert not any(name.startswith("_protocols") for name in _experiment_names())


def test_every_classification_experiment_extends_a_protocol_fragment():
    """The extends:/_protocols mechanism is what keeps every train.py-style run on a declared,
    shared protocol instead of a hand-copied one. Read the raw YAML -- load_config pops `extends`.
    """
    for name in _experiment_names():
        raw = yaml.safe_load((CONFIGS_DIR / "experiments" / f"{name}.yaml").read_text())
        if "models" not in raw:
            continue  # Phase 7 configs (scripts/train_det_seg.py) use a per-model schema, not this one
        parent = raw.get("extends")
        assert parent, f"{name}.yaml has no `extends:` -- inline protocol, see configs/experiments/_protocols/"
        parent_path = (CONFIGS_DIR / "experiments" / parent).with_suffix(".yaml")
        assert parent_path.is_file(), f"{name}.yaml extends {parent!r}, which does not exist"
        assert parent_path.parent.name == "_protocols", f"{name}.yaml extends {parent!r}, not a _protocols/ fragment"


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
    assert slurm_cfg["signal"] == "B:USR1@900"
    assert slurm_cfg["open_mode"] == "append"
    assert slurm_cfg["time"] == "24:00:00"
