"""Regression test: ml.model_registrations must populate MODEL_REGISTRY with the full sweep set.

scripts/train.py used to omit `import ml.model_registrations`, leaving MODEL_REGISTRY empty and every
`python -m scripts.train` invocation failing with "No valid model names were selected". This guards
against that regressing silently again.
"""
import torch

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


def test_every_experiment_only_names_registered_models():
    """scripts/train.py now raises on an unknown name -- catch the typo here, not on PCAD."""
    from pathlib import Path
    from configs.loader import load_config

    for path in sorted((Path(__file__).parents[1] / "configs" / "experiments").glob("*.yaml")):
        models = load_config(f"experiments/{path.name}").get("models")
        if isinstance(models, list):  # phase_7_*.yaml key per-model instead
            assert set(models) <= MODEL_REGISTRY.keys(), f"{path.name}: {set(models) - MODEL_REGISTRY.keys()}"


def test_every_registration_has_a_constructor_and_fuse_map():
    for name, spec in MODEL_REGISTRY.items():
        assert callable(spec["ctor"]), f"{name} has no callable ctor"
        assert isinstance(spec["fuse_map"], list), f"{name} has a non-list fuse_map"


CONTROL_MODELS = ["alexnet_adapted_orig_fc", "alexnet_adapted_orig_gap", "alexnet_adapted_2x2_fc",
                  "alexnet_adapted_2x2_gap", "alexnet_3x3_gap_bn"]
# Phase 11 geometry factorial: final map before the classifier's AdaptiveAvgPool at 64x64 input
# (alexnet_adapted_orig_fc_pt is checked separately -- it needs the ImageNet weights).
GEO_MAPS = {
    "alexnet_geo_s4_p3_fc": 1, "alexnet_geo_s4_p3_gap": 1, "alexnet_geo_s4_p3_fc_k3": 1,
    "alexnet_geo_s2_p3_fc": 3, "alexnet_geo_s4_p2_fc": 3,
    "alexnet_geo_s2_pk3n2_fc": 7, "alexnet_geo_s2_pk2n3_fc": 4, "alexnet_geo_s2_p2_drop_fc": 8,
}


def test_geometry_factorial_models_hit_their_documented_maps_and_dropout_counts():
    x = torch.randn(1, 3, 64, 64)
    for name, size in GEO_MAPS.items():
        model = MODEL_REGISTRY[name]["ctor"]().eval()
        assert tuple(model.features[:-1](x).shape) == (1, 256, size, size), name
        assert sum(isinstance(m, torch.nn.Dropout) for m in model.modules()) == (2 if name.endswith("_drop_fc") else 0), name


def test_geometry_factorial_original_corner_is_torchvision_alexnet_geometry():
    """alexnet_geo_s4_p3_fc + Dropout must BE AlexNetTV(pretrained=False)'s layout (conv/pool
    kernel-stride-padding, Linear shapes, parameter count) -- else the 'walk back to torchvision'
    ends somewhere else and the Dropout/stride/pooling read-outs compare the wrong thing."""
    from functools import partial
    from ml.model_registrations import _S4P3
    from models import AlexNetAdapted, AlexNetTV

    def layout(m):
        feats = [(type(l).__name__, getattr(l, "kernel_size", None), getattr(l, "stride", None), getattr(l, "padding", None))
                 for l in m.features if isinstance(l, (torch.nn.Conv2d, torch.nn.MaxPool2d))]
        linears = [tuple(l.weight.shape) for l in m.classifier if isinstance(l, torch.nn.Linear)]
        return feats, linears, sum(p.numel() for p in m.parameters())

    assert layout(partial(AlexNetAdapted, dropout=0.5, **_S4P3)()) == layout(AlexNetTV(pretrained=False))


def test_pretrained_adapted_model_loads_the_imagenet_convs_and_first_two_linears():
    import pytest
    from torchvision.models import alexnet

    try:
        tv = alexnet(weights="IMAGENET1K_V1")
    except Exception as e:  # no cached weights and no network
        pytest.skip(f"ImageNet weights unavailable: {e}")
    model = MODEL_REGISTRY["alexnet_adapted_orig_fc_pt"]["ctor"]().eval()
    for i in (0, 3, 6, 8, 10):
        assert torch.equal(model.features[i].weight, tv.features[i].weight), i
    mine = [m for m in model.classifier if isinstance(m, torch.nn.Linear)]
    theirs = [m for m in tv.classifier if isinstance(m, torch.nn.Linear)]
    assert torch.equal(mine[0].weight, theirs[0].weight) and torch.equal(mine[1].weight, theirs[1].weight)
    assert tuple(mine[2].weight.shape) == (200, 4096)  # fresh head, not the 1000-class one
    assert tuple(model.features[:-1](torch.randn(1, 3, 64, 64)).shape) == (1, 256, 8, 8)


def test_geometry_control_models_survive_the_qat_to_int8_path():
    """The controls run FP32 -> QAT -> INT8 (Phase 11 protocol), so the hand-written fuse maps must
    fuse every conv and the 2x2 ZeroPad2d must survive a real INT8 convert (quantized input)."""
    from ml.quantization import build_qat_from_model, convert_to_int8

    for name in CONTROL_MODELS + list(GEO_MAPS):
        spec, model = MODEL_REGISTRY[name], MODEL_REGISTRY[name]["ctor"]()
        root = getattr(model, spec["fuse_root_attr"]) if spec.get("fuse_root_attr") else model
        assert len(spec["fuse_map"]) == sum(isinstance(m, torch.nn.Conv2d) for m in model.features), name
        for group in spec["fuse_map"]:  # every group must be Conv -> (BN ->) ReLU, in the padded layout too
            kinds = [type(root.get_submodule(i)) for i in group]
            assert kinds[0] is torch.nn.Conv2d and kinds[-1] is torch.nn.ReLU, f"{name}: {group} -> {kinds}"
        qat_model = build_qat_from_model(model, name, torch.device("cpu"))
        out = convert_to_int8(qat_model.eval())(torch.randn(2, 3, 64, 64))
        assert out.shape == (2, 200), name


def test_alexnet_adapted_is_3x3_family_layer_for_layer_at_3x3_and_keeps_geometry_for_other_kernels():
    """The geometry/BN controls are only controls if AlexNetAdapted(3x3) IS AlexNet3x3FC/GAP and the
    11-5-3-3-3 default keeps the same 8x8 map -- else the kernel sweep silently changes geometry."""
    from models import AlexNet3x3FC, AlexNet3x3GAP, AlexNetAdapted

    for ref, head in [(AlexNet3x3FC, "fc"), (AlexNet3x3GAP, "gap")]:
        shapes = lambda m: {k: tuple(v.shape) for k, v in m.state_dict().items()}
        assert shapes(AlexNetAdapted(kernels=(3,) * 5, head=head)) == shapes(ref()), head

    x = torch.randn(1, 3, 64, 64)
    for name in CONTROL_MODELS:
        model = MODEL_REGISTRY[name]["ctor"]().eval()
        assert tuple(model.features[:-1](x).shape) == (1, 256, 8, 8), name
        assert model(x).shape == (1, 200), name
    assert not any(isinstance(m, torch.nn.BatchNorm2d) for m in MODEL_REGISTRY["alexnet_adapted_orig_fc"]["ctor"]().modules())
    assert any(isinstance(m, torch.nn.BatchNorm2d) for m in MODEL_REGISTRY["alexnet_3x3_gap_bn"]["ctor"]().modules())


def test_phase_11_kernel_swap_only_changes_the_kernel():
    """AlexNetTV/VGG16(kernel_size=...): every variant of a family must reach the same feature
    map size at 64x64 -- proves the kernel swap left channels/pool structure alone -- while
    actually using a different kernel size (else the swap silently no-opped)."""
    x = torch.randn(1, 3, 64, 64)

    alexnet_variants = ["alexnet_tv_scratch", "alexnet_tv_3x3", "alexnet_tv_2x2"]
    kernel_sets = set()
    for name in alexnet_variants:
        model = MODEL_REGISTRY[name]["ctor"]()
        assert tuple(model.features(x).shape) == (1, 256, 1, 1), name
        kernel_sets.add(frozenset(m.kernel_size for m in model.features if isinstance(m, torch.nn.Conv2d)))
    assert len(kernel_sets) == len(alexnet_variants), "some AlexNet variants share identical kernels"

    vgg_variants = ["vgg16", "vgg16_2x2"]
    kernel_sets = set()
    for name in vgg_variants:
        model = MODEL_REGISTRY[name]["ctor"]()
        assert tuple(model.features(x).shape) == (1, 512, 2, 2), name
        kernel_sets.add(frozenset(m.kernel_size for m in model.features if isinstance(m, torch.nn.Conv2d)))
    assert len(kernel_sets) == len(vgg_variants), "vgg16 and vgg16_2x2 share identical kernels"
