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


def _layout(m):
    """Every module that shapes the computation, in order -- two nets with equal layouts are the same net."""
    out = []
    for x in m.modules():
        if isinstance(x, torch.nn.Conv2d):
            out.append(("conv", x.in_channels, x.out_channels, x.kernel_size, x.stride, x.padding))
        elif isinstance(x, (torch.nn.MaxPool2d, torch.nn.AdaptiveAvgPool2d, torch.nn.ZeroPad2d)):
            out.append((type(x).__name__, getattr(x, "kernel_size", None), getattr(x, "stride", None),
                        getattr(x, "output_size", None), getattr(x, "padding", None)))
        elif isinstance(x, torch.nn.Linear):
            out.append(("linear", x.in_features, x.out_features))
        elif isinstance(x, (torch.nn.Dropout, torch.nn.BatchNorm2d)):
            out.append((type(x).__name__, getattr(x, "p", None)))
    return out


def test_factorial_grid_is_complete_and_every_gap_cell_runs():
    """208 cells = 4 kernels x 2 strides x 2 pool kernels x 2 pool counts x (FC x BN x Dropout + GAP x BN)
    + 16 pretrained; each GAP cell (cheap to build) runs and has one Conv-(BN-)ReLU fuse group per conv."""
    fx = [n for n in MODEL_REGISTRY if n.startswith("alexnet_fx_")]
    assert len(fx) == 208 and sum(n.endswith("_pt") for n in fx) == 16
    x = torch.randn(1, 3, 64, 64)
    for name in (n for n in fx if "_gap" in n):
        spec, model = MODEL_REGISTRY[name], MODEL_REGISTRY[name]["ctor"]().eval()
        assert model(x).shape == (1, 200), name
        for group in spec["fuse_map"]:
            kinds = [type(model.features.get_submodule(i)) for i in group]
            assert kinds[0] is torch.nn.Conv2d and kinds[-1] is torch.nn.ReLU, f"{name}: {group} -> {kinds}"
        assert len(spec["fuse_map"]) == 5, name


def test_factorial_cells_already_trained_elsewhere_are_the_same_net():
    """FX_EXISTING reuses 19 runs as factorial cells -- only valid if each is layer for layer that cell
    (the *_pt cells share their non-pt twin's layout; the ImageNet load is tested above)."""
    from ml.model_registrations import FX_EXISTING
    from models import AlexNetTV

    for cell, run in FX_EXISTING.items():
        if cell.endswith("_pt"):
            continue
        model = run.split("/")[1]
        ref = AlexNetTV(pretrained=False, kernel_size=3) if model == "alexnet_tv_3x3" else MODEL_REGISTRY[model]["ctor"]()
        assert _layout(MODEL_REGISTRY[cell]["ctor"]()) == _layout(ref), (cell, run)


def test_factorial_cells_survive_the_qat_to_int8_path():
    """One GAP cell per kernel x BN (the fuse map depends on nothing else) through the real QAT->INT8 path."""
    from ml.quantization import build_qat_from_model, convert_to_int8

    for kernel in ("orig", "k3", "k2", "mix"):
        for bn in ("", "_bn"):
            name = f"alexnet_fx_{kernel}_s4_pk3n2_gap{bn}"  # a stride-4 2x2 stem and a crossed pool: the newest paths
            qat_model = build_qat_from_model(MODEL_REGISTRY[name]["ctor"](), name, torch.device("cpu"))
            assert convert_to_int8(qat_model.eval())(torch.randn(2, 3, 64, 64)).shape == (2, 200), name


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


VGG_FX_MAPS = {(1, 5): 2, (1, 4): 4, (2, 5): 1, (2, 4): 2}  # (stem stride, pool count) -> final map at 64x64


def test_vgg_factorial_grid_is_complete_and_only_stride_and_pool_count_set_the_map():
    """168 cells = 6 kernel patterns x 2 strides x 2 pool kernels x 2 pool counts x 3 heads + 24 pretrained (3x3).
    The kernel and pool-kernel factors are only clean if they never change the map size -- checked on every cell
    (meta device: shapes only, no weights), plus each fuse group being Conv-BN-ReLU in the padded layout."""
    from ml.model_registrations import VGG_FX_KERNELS

    fx = [n for n in MODEL_REGISTRY if n.startswith("vgg_fx_")]
    assert len(fx) == 168 and sum(n.endswith("_pt") for n in fx) == 24
    for name in (n for n in fx if not n.endswith("_pt")):
        spec = MODEL_REGISTRY[name]
        with torch.device("meta"):
            model = spec["ctor"]()
            out = model.features(torch.zeros(1, 3, 64, 64))
        kw = spec["ctor"].keywords
        assert tuple(out.shape) == (1, 512, *(2 * [VGG_FX_MAPS[kw["stem_stride"], kw["pool_count"]]])), name
        assert tuple(m.kernel_size[0] for m in model.features if isinstance(m, torch.nn.Conv2d)) == kw["kernels"], name
        assert [type(model.features.get_submodule(i)) for g in spec["fuse_map"] for i in g] == \
            [torch.nn.Conv2d, torch.nn.BatchNorm2d, torch.nn.ReLU] * 13, name
    assert sum(k == 3 for k in VGG_FX_KERNELS["alt32"]) == 7 and sum(k == 3 for k in VGG_FX_KERNELS["early2"]) == 6


def test_vgg_factorial_reused_cell_is_vgg16_layer_for_layer():
    """vgg16's run stands in for vgg_fx_k3_s1_pk2n5_fc_d -- only valid if it is the same net, same state_dict."""
    from ml.model_registrations import VGG_FX_EXISTING
    from models import VGG16

    for cell, run in VGG_FX_EXISTING.items():
        with torch.device("meta"):
            mine, ref = MODEL_REGISTRY[cell]["ctor"](), MODEL_REGISTRY[run.split("/")[1]]["ctor"]()
        assert _layout(mine) == _layout(ref), cell
        assert {k: v.shape for k, v in mine.state_dict().items()} == {k: v.shape for k, v in ref.state_dict().items()}, cell
    with torch.device("meta"):  # the no-Dropout FC level keeps the same indices (Dropout(0.0)), so the same fuse map
        assert MODEL_REGISTRY["vgg_fx_k3_s1_pk2n5_fc"]["ctor"]().state_dict().keys() == VGG16().state_dict().keys()


def test_vgg_factorial_cells_survive_the_qat_to_int8_path():
    """The fuse map changes with kernel pattern x stem stride (ZeroPad2d positions): a 2x2 stride-1 stem, a strided
    2x2 stem and an alternating pattern, with the overlapping pool, through the real QAT->INT8 path (GAP head --
    the FC head's classifier fusion is vgg16's, tested in test_quantization)."""
    from ml.quantization import build_qat_from_model, convert_to_int8

    for name in ("vgg_fx_k2_s1_pk3n5_gap", "vgg_fx_k2_s2_pk3n4_gap", "vgg_fx_alt23_s1_pk3n4_gap"):
        qat_model = build_qat_from_model(MODEL_REGISTRY[name]["ctor"](), name, torch.device("cpu"))
        assert convert_to_int8(qat_model.eval())(torch.randn(2, 3, 64, 64)).shape == (2, 200), name


def test_pretrained_vgg_cell_loads_vgg16_bn_and_matches_it_in_eval():
    """vgg16_bn's convs carry a bias, ours don't (BN follows) -- the load folds it into running_mean, which must
    leave the features' eval output unchanged; with the FC head the first two Linears load too."""
    import pytest
    from torchvision.models import vgg16_bn

    try:
        tv = vgg16_bn(weights="IMAGENET1K_V1").eval()
    except Exception as e:  # no cached weights and no network
        pytest.skip(f"ImageNet weights unavailable: {e}")
    model = MODEL_REGISTRY["vgg_fx_k3_s1_pk2n5_fc_d_pt"]["ctor"]().eval()
    x = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        assert torch.allclose(model.features(x), tv.features(x), atol=1e-4)
    assert torch.equal(model.classifier[0].weight, tv.classifier[0].weight)
    assert torch.equal(model.classifier[3].weight, tv.classifier[3].weight)
    assert tuple(model.classifier[6].weight.shape) == (200, 4096)
