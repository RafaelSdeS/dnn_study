"""Regression test: ml.model_registrations must populate MODEL_REGISTRY with the full sweep set.

scripts/train.py used to omit `import ml.model_registrations`, leaving MODEL_REGISTRY empty and every
`python -m scripts.train` invocation failing with "No valid model names were selected". This guards
against that regressing silently again.
"""
import ml.model_registrations  # noqa: F401 — populates MODEL_REGISTRY as a side effect
from ml.registry import MODEL_REGISTRY

EXPECTED_MODELS = {
    "alexnet_tv", "vgg_style", "mobilenetv2", "resnet18tv",
    "alexnet_bottleneck", "alexnet_depthwisesep", "alexnet_fire", "alexnet_smallkernel",
    "alexnet_final_bottleneck_residual", "alexnet_final_fire_residual",
    "alexnet_final_bottleneck_fire", "alexnet_final_depthwise_fire",
}


def test_model_registrations_populates_the_full_sweep_set():
    assert EXPECTED_MODELS <= MODEL_REGISTRY.keys()


def test_every_registration_has_a_constructor_and_fuse_map():
    for name, spec in MODEL_REGISTRY.items():
        assert callable(spec["ctor"]), f"{name} has no callable ctor"
        assert isinstance(spec["fuse_map"], list), f"{name} has a non-list fuse_map"
