"""QAT must fake-quantize both weights and activations, including through residual adds.

Regression test for the ResNet18TV bug: its BasicBlock used a raw `out += identity` instead of
FloatFunctional, so the residual add wasn't instrumented and converting to INT8 either produced wrong
results or crashed. Fixed by switching to torchvision's quantizable resnet18. This test builds QAT +
converts to INT8 for every residual-bearing model in the sweep and asserts the forward pass still works.
"""
import copy

import pytest
import torch
import torch.nn as nn

import torch.ao.nn.intrinsic.qat as nniqat
import torch.ao.quantization as tq

import ml.model_registrations  # noqa: F401 -- populates MODEL_REGISTRY
from ml.quantization import build_qat_from_model, convert_to_int8, find_fuse_groups, make_qat_callback, prepare_qat_model
from ml.registry import MODEL_REGISTRY
from models.baselines import ResNet18TV
from models.final_architecture import AlexNetFinalBottleneckResidual, AlexNetFinalFireResidual

# One model per family the report trains (phase_11_families.yaml + the Phase 11 comparisons), plus the factorial's
# cell types: FC/GAP heads, BN, Dropout, 2x2 + SymmetricPad2d, stride 4, torchvision's AlexNet head. No FC-head VGG
# (~130M params: too much RAM for a laptop test); vgg16's Linear-ReLU pairs go through the same Sequential pass.
REPORT_MODELS = [
    "alexnet_3x3_fc", "alexnet_3x3_gap", "alexnet_tv_scratch", "alexnet_tv_mixed_alt_gap", "alexnet_mixed_fc_bn",
    "alexnet_stacked", "alexnet_stacked_gap_nobn", "alexnet_smallkernel_fc", "alexnet_adapted_2x2_fc",
    "alexnet_k2x2_stride4_3pool3x3_map1_fcdrop_bn", "alexnet_k11-5-3_stride2_2pool3x3_map7_gap_nobn", "alexnet_3x3_gap_bn", "alexnet_bottleneck",
    "alexnet_fire", "alexnet_factorized", "alexnet_groupconv", "alexnet_depthwisesep", "alexnet_residual",
    "alexnet_dilated_fc", "alexnet_dilated_gap", "alexnet_small_kernel_with_bn", "tinyhybridnet", "tinymobilenetv2",
    "alexnet_final_bottleneck_fire", "alexnet_final_fire_residual", "alexnet_final_bottleneck_residual",
    "alexnet_final_depthwise_fire", "alexnet_fire_bypass", "vgg_style", "mobilenetv2", "resnet18tv",
    "vgg16_kalt3-2_stride2_4pool2x2_map2_gap_bn", "vgg16_k2x2_stride1_5pool2x2_map2_gap_bn",  # the latter: C2sp on RGB
    "alexnet_k2x2stacked_stride2_2pool2x2_map8_gap_bn", "mobilenetv2_scratch", "resnet18tv_scratch",
]


def _qat(name):
    return build_qat_from_model(MODEL_REGISTRY[name]["ctor"](), name, torch.device("cpu"))


RESIDUAL_MODELS = [
    ("alexnet_final_bottleneck_residual", AlexNetFinalBottleneckResidual),
    ("alexnet_final_fire_residual", AlexNetFinalFireResidual),
    ("resnet18tv", lambda: ResNet18TV(pretrained=False)),
]


def test_residual_models_quantize_to_int8_without_crashing():
    for name, ctor in RESIDUAL_MODELS:
        model = ctor()
        fuse_map = find_fuse_groups(model)
        qat_model = prepare_qat_model(model, fuse_map)
        qat_model.eval()

        int8_model = convert_to_int8(qat_model)
        out = int8_model(torch.randn(2, 3, 64, 64))

        assert out.shape == (2, 200), f"{name} produced the wrong output shape after INT8 convert"


def test_prepare_qat_attaches_weight_and_activation_fake_quant():
    model = AlexNetFinalBottleneckResidual()
    fuse_map = find_fuse_groups(model)
    qat_model = prepare_qat_model(model, fuse_map)

    has_weight_fake_quant = any(hasattr(m, "weight_fake_quant") for m in qat_model.modules())
    has_activation_fake_quant = any(
        hasattr(m, "activation_post_process")
        and type(m.activation_post_process).__name__ != "Identity"
        for m in qat_model.modules()
    )

    assert has_weight_fake_quant, "no weight fake-quantizer found after prepare_qat"
    assert has_activation_fake_quant, "no activation fake-quantizer found after prepare_qat"


def test_int8_definition_is_8bit_activations_and_symmetric_per_channel_weights():
    """The literature's INT8 (Jacob et al. 2018; Wu et al. 2020 Sec. 6): activations 0..255 -- not fbgemm's default
    reduce_range, which made them 7-bit until 2026-10-03 -- and weights per-channel symmetric in [-127, 127]."""
    qat = _qat("alexnet_3x3_fc")
    weights = [m.weight_fake_quant for m in qat.modules() if hasattr(m, "weight_fake_quant")]
    acts = [m for m in qat.modules() if isinstance(m, tq.FakeQuantizeBase) and all(m is not w for w in weights)]
    assert weights and acts
    for fq in weights:
        obs = fq.activation_post_process
        assert (obs.quant_min, obs.quant_max, obs.qscheme) == (-127, 127, torch.per_channel_symmetric)
    for fq in acts:
        obs = fq.activation_post_process
        assert (obs.quant_min, obs.quant_max, obs.reduce_range) == (0, 255, False), type(obs)


@pytest.mark.parametrize("name", REPORT_MODELS)
def test_every_relu_is_fused_into_the_layer_that_feeds_it(name):
    """Conv/Linear/add + ReLU must be fused (Jacob et al. 2018; LiteRT fused activations): an unfused ReLU leaves its
    producer's observer on the pre-ReLU range, half of it spent on negatives. Until 2026-10-03 every FC head except
    vgg16's ran its Linear-ReLU pairs unfused -- an INT8 handicap only FC heads had."""
    qat, ran = _qat(name).eval(), []
    for n, m in qat.named_modules():
        if type(m) is nn.ReLU:
            m.register_forward_hook(lambda mod, i, o, n=n: ran.append(n))
    with torch.no_grad():
        qat(torch.randn(2, 3, 64, 64))
    assert not ran, f"{name}: standalone ReLU(s) {ran}"


@pytest.mark.parametrize("name", ["alexnet_3x3_gap", "alexnet_3x3_fc", "alexnet_adapted_2x2_gap",
                                  "alexnet_final_fire_residual"])
def test_int8_kernels_reproduce_the_fake_quant_model(name):
    """The converted model must compute what QAT simulated (Jacob et al. 2018); a gap means the kernels round,
    saturate or requantize differently -- e.g. fbgemm's int16 saturation on full-range activations without VNNI
    (2026-10-03, this laptop's AVX-VNNI-only i7: alexnet_3x3_gap 50.8 dB on onednn, 42.7 on fbgemm 8-bit, 48.4 on the
    old fbgemm 7-bit). The model is put in its deployed state first -- BN running stats settled on data, as a trained
    checkpoint has them, observers calibrated on what eval computes. Random weights still amplify 1-LSB rounding
    differences with depth, the same on every backend (resnet18 ~3 dB per block), so the real check is
    scripts/train.py's agreement_qat_int8 on trained models; this one catches a broken kernel or definition."""
    torch.manual_seed(0)
    fp = MODEL_REGISTRY[name]["ctor"]().train()
    with torch.no_grad():
        for m in fp.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.momentum = None  # cumulative average
        for _ in range(5):
            fp(torch.randn(16, 3, 64, 64))
    qat = build_qat_from_model(fp, name, torch.device("cpu"))
    qat.apply(nniqat.freeze_bn_stats)
    qat.eval()
    with torch.no_grad():
        for _ in range(3):
            qat(torch.randn(16, 3, 64, 64))
    qat.apply(tq.disable_observer)
    x = torch.randn(16, 3, 64, 64)
    with torch.no_grad():
        fq, int8 = qat(x), convert_to_int8(copy.deepcopy(qat))(x)
    sqnr_db = 10 * torch.log10(fq.pow(2).sum() / (fq - int8).pow(2).sum())
    assert sqnr_db > 30, f"{name}: INT8 vs fake-quant SQNR {sqnr_db:.1f} dB"


def test_fuse_root_models_are_fused_in_the_qat_copy_not_the_input():
    """Regression test for the 2026-09-30 QAT fusion bug: prepare_qat_model deep-copied the model but fused
    the caller's fuse_root (a submodule of the *original*), so all 43 fuse_root_attr models trained QAT
    unfused -- observer before the ReLU, BN left unfolded -- and the input model was mutated."""
    for name, fused_type in [("alexnet_3x3_gap", nniqat.ConvReLU2d), ("vgg16_2x2", nniqat.ConvBnReLU2d)]:
        model = MODEL_REGISTRY[name]["ctor"]()
        qat_model = build_qat_from_model(model, name, torch.device("cpu"))
        assert isinstance(qat_model.features[0], fused_type), (name, type(qat_model.features[0]))
        assert type(model.features[0]) is nn.Conv2d, f"{name}: the input model was fused in place"


def test_qat_callback_still_applies_on_the_first_epoch_after_a_resume_past_it():
    qat = prepare_qat_model(nn.Sequential(nn.Conv2d(3, 4, 3), nn.BatchNorm2d(4), nn.ReLU(inplace=False)),
                            [["0", "1", "2"]])
    make_qat_callback(freeze_bn_epoch=3, disable_observer_epoch=5)(7, qat)  # resumed at epoch 7
    assert qat[0].freeze_bn  # a module attribute, not state_dict -- lost on resume unless re-applied
    fake_quants = [m for m in qat.modules() if isinstance(m, torch.ao.quantization.FakeQuantizeBase)]
    assert fake_quants and all(int(m.observer_enabled[0]) == 0 for m in fake_quants)


def test_qat_callback_with_no_observer_freeze_keeps_ranges_adapting():
    """A registry qat_disable_observer_epoch=None (vgg16's pre-fusion-fix override, docs/logs/PHASE11_LOG.md):
    BN must still freeze, but observers must stay enabled no matter how late the epoch."""
    qat = prepare_qat_model(nn.Sequential(nn.Conv2d(3, 4, 3), nn.BatchNorm2d(4), nn.ReLU(inplace=False)),
                            [["0", "1", "2"]])
    make_qat_callback(freeze_bn_epoch=3, disable_observer_epoch=None)(99, qat)
    assert qat[0].freeze_bn
    fake_quants = [m for m in qat.modules() if isinstance(m, torch.ao.quantization.FakeQuantizeBase)]
    assert fake_quants and all(int(m.observer_enabled[0]) == 1 for m in fake_quants)


def test_int8_avg_pool_requantizes_instead_of_inheriting_its_input_scale():
    """Regression test for the quantized-GAP gap (docs/logs/PHASE11_LOG.md): eager INT8 avg pooling keeps its
    input's scale, and QAT had no observer after the pool, so fake-quant and INT8 disagreed on every GAP model.
    One outlier sets the per-tensor input scale to ~1; channel c's mean is 4c/64 < 0.5 of a step, so the old
    INT8 GAP rounded channels 1-7 to 0 while the fake-quant model kept them."""
    class Gap(nn.Module):
        def __init__(self):
            super().__init__()
            self.quant, self.pool, self.dequant = torch.ao.quantization.QuantStub(), nn.AdaptiveAvgPool2d(1), \
                torch.ao.quantization.DeQuantStub()

        def forward(self, x):
            return self.dequant(self.pool(self.quant(x))).flatten(1)

    x = torch.zeros(1, 8, 8, 8)
    x[0, 0, 0, 0] = 127.0
    for c in range(1, 8):
        x[0, c].view(-1)[:4 * c] = 1.0
    qat = prepare_qat_model(Gap(), fuse_pairs=[]).eval()
    qat(x)  # calibrate
    qat.apply(torch.ao.quantization.disable_observer)
    with torch.no_grad():
        fq, int8 = qat(x), convert_to_int8(qat)(x)
    assert torch.allclose(int8, fq, atol=1e-3), (fq, int8)
    assert torch.allclose(int8[0, 1:], torch.arange(1, 8) * 4 / 64, atol=0.02), int8  # the means survive


def test_logits_layer_has_int8_weights_and_an_fp32_output_so_int8_logits_do_not_tie():
    """2026-10-02 audit (docs/logs/PHASE11_LOG.md, "Float logits layer"): the logits Linear's output went through an
    8-bit fake-quant, leaving 8-31 distinct values per image and 14-31% top-1 ties in QAT/INT8. Wu et al. 2020: INT8
    input and weights, the int32 accumulator rescaled to FP32 since no quantized layer reads it. resnet18tv because
    its input QuantStub is registered after fc: the logits layer must be found in execution order."""
    from ml.quantization import _FloatLogits
    x = torch.randn(4, 3, 64, 64)
    for name, model in [("alexnet_3x3_gap", MODEL_REGISTRY["alexnet_3x3_gap"]["ctor"]()),
                        ("resnet18tv", ResNet18TV(pretrained=False))]:
        qat = build_qat_from_model(model, name, torch.device("cpu")).eval()
        qat(x)  # calibrate
        qat.apply(tq.disable_observer)
        int8 = convert_to_int8(qat)
        heads = [m for m in int8.modules() if isinstance(m, _FloatLogits)]
        assert len(heads) == 1 and heads[0].linear.weight is None and heads[0].qweight.dtype == torch.int8, name
        assert any(isinstance(m, torch.ao.nn.quantized.Quantize) for m in int8.modules()), f"{name}: input not quantized"
        with torch.no_grad():
            out = int8(x)
            qat_head = next(m for m in qat.modules() if isinstance(m, _FloatLogits))
            fake_quant_w = qat_head.weight_fake_quant(qat_head.linear.weight)
        assert not out.is_quantized and all(len(torch.unique(row)) == 200 for row in out), name
        # the stored int8 codes are exactly the weights QAT trained against
        assert torch.allclose(heads[0].qweight.float() * heads[0].weight_fake_quant.scale[:, None], fake_quant_w), name


def test_validation_does_not_calibrate_qat_observers_on_val_data():
    """FakeQuantize ignores eval(): Trainer's per-epoch validation used to update activation ranges from the val
    split. frozen_observers must leave the ranges untouched and restore each module's own flag afterwards."""
    from ml.trainer import frozen_observers
    qat = prepare_qat_model(nn.Sequential(nn.Conv2d(3, 4, 3), nn.ReLU(inplace=False)), [["0", "1"]])
    qat(torch.randn(2, 3, 8, 8))
    fqs = [m for m in qat.modules() if isinstance(m, torch.ao.quantization.FakeQuantizeBase)]
    fqs[0].disable_observer()  # a mixed state, as after make_qat_callback on some modules
    before = [(m.observer_enabled.clone(), m.scale.clone()) for m in fqs]
    with frozen_observers(qat.eval()):
        qat(100 * torch.randn(2, 3, 8, 8))
    for m, (flag, scale) in zip(fqs, before):
        assert torch.equal(m.observer_enabled, flag) and torch.equal(m.scale, scale)
