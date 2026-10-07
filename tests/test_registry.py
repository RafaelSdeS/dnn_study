"""Regression test: ml.model_registrations must populate MODEL_REGISTRY with the full sweep set.

scripts/train.py used to omit `import ml.model_registrations`, leaving MODEL_REGISTRY empty and every
`python -m scripts.train` invocation failing with "No valid model names were selected". This guards
against that regressing silently again.
"""
import torch

import ml.model_registrations  # noqa: F401 — populates MODEL_REGISTRY as a side effect
from ml.registry import MODEL_REGISTRY
from models.baselines import SymmetricPad2d

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
    fuse every conv and the 2x2 SymmetricPad2d must survive a real INT8 convert (quantized input)."""
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
        elif isinstance(x, (torch.nn.MaxPool2d, torch.nn.AdaptiveAvgPool2d, torch.nn.ZeroPad2d, SymmetricPad2d)):
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
    from ml.model_registrations import CELL_FACTORS, FX_KERNELS

    fx = [n for n, f in CELL_FACTORS.items() if f["family"] == "alexnet" and f["kernels"] in FX_KERNELS]
    assert len(fx) == 208 and sum(CELL_FACTORS[n]["pretrained"] for n in fx) == 16
    x = torch.randn(1, 3, 64, 64)
    for name in (n for n in fx if CELL_FACTORS[n]["head"] == "gap"):
        spec, model = MODEL_REGISTRY[name], MODEL_REGISTRY[name]["ctor"]().eval()
        assert model(x).shape == (1, 200), name
        for group in spec["fuse_map"]:
            kinds = [type(model.features.get_submodule(i)) for i in group]
            assert kinds[0] is torch.nn.Conv2d and kinds[-1] is torch.nn.ReLU, f"{name}: {group} -> {kinds}"
        assert len(spec["fuse_map"]) == 5, name


def test_every_cell_name_says_what_the_net_is():
    """The descriptive names are only useful if true: on every registered cell (meta device, shapes only), the kernels,
    conv1 stride, max-pools, last-map side, head, Dropout and BN read off the built net match its name."""
    from ml.model_registrations import CELL_FACTORS, FX_KERNELS, VGG_FX_KERNELS, _cell_name

    for name, f in CELL_FACTORS.items():
        kernels = {"alexnet": FX_KERNELS, "vgg16": VGG_FX_KERNELS}[f["family"]]
        assert _cell_name(**f) == name, name
        with torch.device("meta"):
            m = MODEL_REGISTRY[name]["ctor"]()
            feats = [x for x in m.features if not isinstance(x, torch.nn.AdaptiveAvgPool2d)]
            side = torch.nn.Sequential(*feats)(torch.zeros(1, 3, 64, 64)).shape[-1]
        convs = [x for x in m.features if isinstance(x, torch.nn.Conv2d)]
        pools = [x for x in m.features if isinstance(x, torch.nn.MaxPool2d)]
        if f["kernels"] in kernels:
            assert tuple(c.kernel_size[0] for c in convs) == kernels[f["kernels"]], name
        elif f["kernels"].endswith("stacked"):  # k3x3stacked / k2x2stacked: every conv that size, two per stage
            assert {c.kernel_size[0] for c in convs} == {int(f["kernels"][1])} and len(convs) == 10, name
        assert convs[0].stride[0] == f["stride"] and side == f["map_side"], name
        assert len(pools) == f["pool_count"] and {p.kernel_size for p in pools} == {f["pool_kernel"]}, name
        assert any(isinstance(x, torch.nn.BatchNorm2d) for x in m.modules()) == f["bn"], name
        assert (sum(isinstance(x, torch.nn.Linear) for x in m.classifier.modules()) == 1) == (f["head"] == "gap"), name
        assert any(isinstance(x, torch.nn.Dropout) and x.p > 0 for x in m.modules()) == f["dropout"], name


def test_named_reference_cells_are_the_reference_nets():
    """The cells the report names after a known net must be that net layer for layer: AlexNet trained from scratch
    (torchvision's layout), VGG16 + BN, and the Phase 2 AlexNet3x3 FC/GAP the 64px layout started from."""
    from models import AlexNetTV, VGG16

    twins = {"alexnet_k11-5-3_stride4_3pool3x3_map1_fcdrop_nobn": lambda: AlexNetTV(pretrained=False),
             "alexnet_k3x3_stride4_3pool3x3_map1_fcdrop_nobn": lambda: AlexNetTV(pretrained=False, kernel_size=3),
             "alexnet_k3x3_stride2_2pool2x2_map8_fc_nobn": MODEL_REGISTRY["alexnet_3x3_fc"]["ctor"],
             "alexnet_k3x3_stride2_2pool2x2_map8_gap_nobn": MODEL_REGISTRY["alexnet_3x3_gap"]["ctor"]}
    for cell, ref in twins.items():
        assert _layout(MODEL_REGISTRY[cell]["ctor"]()) == _layout(ref()), cell
    with torch.device("meta"):
        mine, ref = MODEL_REGISTRY["vgg16_k3x3_stride1_5pool2x2_map2_fcdrop_bn"]["ctor"](), VGG16()
    assert _layout(mine) == _layout(ref)
    assert {k: v.shape for k, v in mine.state_dict().items()} == {k: v.shape for k, v in ref.state_dict().items()}


def test_factorial_cells_survive_the_qat_to_int8_path():
    """One GAP cell per kernel x BN (the fuse map depends on nothing else) through the real QAT->INT8 path."""
    from ml.quantization import build_qat_from_model, convert_to_int8

    for kernel, side in (("k11-5-3", 3), ("k3x3", 3), ("k2x2", 3), ("kalt3-2", 3)):
        for bn in ("nobn", "bn"):
            name = f"alexnet_{kernel}_stride4_2pool3x3_map{side}_gap_{bn}"  # a stride-4 2x2 stem + crossed pool: the newest paths
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
    from ml.model_registrations import CELL_FACTORS, VGG_FX_KERNELS

    fx = [n for n, f in CELL_FACTORS.items() if f["family"] == "vgg16"]
    assert len(fx) == 168 and sum(CELL_FACTORS[n]["pretrained"] for n in fx) == 24
    for name in (n for n in fx if not CELL_FACTORS[n]["pretrained"]):
        spec = MODEL_REGISTRY[name]
        with torch.device("meta"):
            model = spec["ctor"]()
            out = model.features(torch.zeros(1, 3, 64, 64))
        kw = spec["ctor"].keywords
        assert tuple(out.shape) == (1, 512, *(2 * [VGG_FX_MAPS[kw["stem_stride"], kw["pool_count"]]])), name
        assert tuple(m.kernel_size[0] for m in model.features if isinstance(m, torch.nn.Conv2d)) == kw["kernels"], name
        assert [type(model.features.get_submodule(i)) for g in spec["fuse_map"] for i in g] == \
            [torch.nn.Conv2d, torch.nn.BatchNorm2d, torch.nn.ReLU] * 13, name
    assert sum(k == 3 for k in VGG_FX_KERNELS["kalt3-2"]) == 7 and sum(k == 3 for k in VGG_FX_KERNELS["k2x2then3x3"]) == 6


def test_vgg_no_dropout_fc_cell_keeps_vgg16s_state_dict_layout():
    """The no-Dropout FC level keeps the same indices (Dropout(0.0)), so the same fuse map as VGG16 itself."""
    from models import VGG16

    with torch.device("meta"):
        assert MODEL_REGISTRY["vgg16_k3x3_stride1_5pool2x2_map2_fc_bn"]["ctor"]().state_dict().keys() == VGG16().state_dict().keys()


def test_vgg_factorial_cells_survive_the_qat_to_int8_path():
    """The fuse map changes with kernel pattern x stem stride (ZeroPad2d positions): a 2x2 stride-1 stem, a strided
    2x2 stem and an alternating pattern, with the overlapping pool, through the real QAT->INT8 path (GAP head --
    the FC head's classifier fusion is vgg16's, tested in test_quantization)."""
    from ml.quantization import build_qat_from_model, convert_to_int8

    for name in ("vgg16_k2x2_stride1_5pool3x3_map2_gap_bn", "vgg16_k2x2_stride2_4pool3x3_map2_gap_bn",
                 "vgg16_kalt2-3_stride1_4pool3x3_map4_gap_bn"):
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
    model = MODEL_REGISTRY["vgg16_k3x3_stride1_5pool2x2_map2_fcdrop_bn_pretrained"]["ctor"]().eval()
    x = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        assert torch.allclose(model.features(x), tv.features(x), atol=1e-4)
    assert torch.equal(model.classifier[0].weight, tv.classifier[0].weight)
    assert torch.equal(model.classifier[3].weight, tv.classifier[3].weight)
    assert tuple(model.classifier[6].weight.shape) == (200, 4096)


def test_symmetric_pad_has_no_net_shift():
    """C2sp (Wu et al. 2019): the four channel groups pad four different corners, so averaged over channels the padded
    map is symmetric under a 180-degree turn -- one-sided padding is not -- and the 2x2 conv after it keeps the size."""
    y = SymmetricPad2d()(torch.ones(1, 8, 5, 5))
    assert y.shape == (1, 8, 6, 6)
    assert torch.equal(y.mean(1), y.mean(1).flip(-1, -2))
    assert SymmetricPad2d()(torch.ones(1, 3, 5, 5)).shape == (1, 3, 6, 6)  # VGG's RGB stem: 3 groups, 3 corners


def test_replicated_fc_weights_fold_away_exactly():
    """ml.reporting.replicated_fc_weights counts the FC-head weights that only multiply AdaptiveAvgPool copies (the
    figures' effective cost, 2026-10-07): folding the pool into fc1 gives the same output with exactly that many fewer
    weights. A 4x4 map resampled to 6x6 -- overlapping windows, the non-trivial case."""
    from ml.reporting import replicated_fc_weights

    torch.manual_seed(0)
    m = MODEL_REGISTRY["alexnet_k3x3_stride4_2pool2x2_map4_fcdrop_nobn"]["ctor"]().eval()
    pool, fc1 = m.features[-1], next(layer for layer in m.classifier if isinstance(layer, torch.nn.Linear))
    with torch.no_grad():
        feats = m.features[:-1](torch.randn(2, 3, 64, 64))
        c, h, w = feats.shape[1:]
        pooling = pool(torch.eye(h * w).view(h * w, 1, h, w)).flatten(1)  # map position -> pooled cells
        folded = torch.einsum("ocp,ip->oci", fc1.weight.view(fc1.out_features, c, -1), pooling).reshape(fc1.out_features, -1)
        assert torch.allclose(fc1(pool(feats).flatten(1)), feats.flatten(1) @ folded.T + fc1.bias, atol=1e-4)
    assert fc1.weight.numel() - folded.numel() == replicated_fc_weights(m) > 0
