"""scripts/train.py CLI: --model filtering and model-name resolution."""
import ml.model_registrations  # noqa: F401 — populates MODEL_REGISTRY
from scripts.train import _resolve_model_names, build_parser


def test_model_flag_is_parsed():
    args = build_parser().parse_args(["--experiment", "large_scale", "--model", "alexnet_fire"])
    assert args.model == "alexnet_fire"


def test_model_flag_defaults_to_none():
    args = build_parser().parse_args(["--experiment", "large_scale"])
    assert args.model is None


def test_resolve_model_names_filters_to_registered_models():
    assert _resolve_model_names(["alexnet_fire"]) == ["alexnet_fire"]
    assert _resolve_model_names(["not_a_real_model"]) == []
    assert _resolve_model_names(["alexnet_fire", "not_a_real_model"]) == ["alexnet_fire"]


def test_resolve_model_names_all_returns_full_registry():
    from ml.registry import MODEL_REGISTRY
    assert set(_resolve_model_names("all")) == set(MODEL_REGISTRY.keys())


def test_model_cli_override_replaces_experiment_models_list():
    experiment_cfg = {"models": ["alexnet_tv", "vgg_style"]}
    # mirrors the override scripts/train.py's main() applies when --model is passed
    experiment_cfg["models"] = ["alexnet_fire"]
    assert _resolve_model_names(experiment_cfg["models"]) == ["alexnet_fire"]
