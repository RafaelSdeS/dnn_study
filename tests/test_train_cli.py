"""scripts/train.py CLI: --model filtering and model-name resolution."""
import pytest

import ml.model_registrations  # noqa: F401 — populates MODEL_REGISTRY
from scripts.train import _apply_smoke_override, _resolve_model_names, build_parser


def test_model_flag_is_parsed():
    args = build_parser().parse_args(["--experiment", "large_scale", "--model", "alexnet_fire"])
    assert args.model == "alexnet_fire"


def test_model_flag_defaults_to_none():
    args = build_parser().parse_args(["--experiment", "large_scale"])
    assert args.model is None


def test_resolve_model_names_rejects_unknown_models():
    assert _resolve_model_names(["alexnet_fire"]) == ["alexnet_fire"]
    with pytest.raises(ValueError, match="not_a_real_model"):
        _resolve_model_names(["alexnet_fire", "not_a_real_model"])


def test_resolve_model_names_accepts_a_bare_string():
    assert _resolve_model_names("alexnet_fire") == ["alexnet_fire"]


def test_resolve_model_names_all_returns_full_registry():
    from ml.registry import MODEL_REGISTRY
    assert set(_resolve_model_names("all")) == set(MODEL_REGISTRY.keys())


def test_model_cli_override_replaces_experiment_models_list():
    experiment_cfg = {"models": ["alexnet_tv", "vgg_style"]}
    # mirrors the override scripts/train.py's main() applies when --model is passed
    experiment_cfg["models"] = ["alexnet_fire"]
    assert _resolve_model_names(experiment_cfg["models"]) == ["alexnet_fire"]


def test_smoke_flag_is_parsed():
    assert build_parser().parse_args(["--experiment", "large_scale"]).smoke is False
    assert build_parser().parse_args(["--experiment", "large_scale", "--smoke"]).smoke is True


def test_smoke_override_caps_epochs_and_keeps_other_overrides():
    experiment_cfg = {"training": {"epochs": 1000, "lr": 5e-4}, "qat": {"epochs": 100},
                      "qat_wino": {"epochs": 15, "lr": 5e-5}}
    result = _apply_smoke_override(experiment_cfg)
    assert result["training"]["epochs"] == 1
    assert result["training"]["warmup_epochs"] == 0
    assert result["training"]["lr"] == 5e-4
    assert result["qat"]["epochs"] == 1
    assert result["qat_wino"] == {"epochs": 1, "lr": 5e-5}


def test_smoke_override_handles_missing_blocks():
    assert _apply_smoke_override({}) == {
        "training": {"epochs": 1, "warmup_epochs": 0}, "qat": {"epochs": 1}, "qat_wino": {"epochs": 1},
    }
