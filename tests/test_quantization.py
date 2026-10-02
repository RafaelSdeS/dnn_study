"""QAT must fake-quantize both weights and activations, including through residual adds.

Regression test for the ResNet18TV bug: its BasicBlock used a raw `out += identity` instead of
FloatFunctional, so the residual add wasn't instrumented and converting to INT8 either produced wrong
results or crashed. Fixed by switching to torchvision's quantizable resnet18. This test builds QAT +
converts to INT8 for every residual-bearing model in the sweep and asserts the forward pass still works.
"""
import torch
import torch.nn as nn

import torch.ao.nn.intrinsic.qat as nniqat

from ml.model_registrations import CLASSIFIER_FUSE_MAP_VGG16
from ml.quantization import build_qat_from_model, convert_to_int8, find_fuse_groups, make_qat_callback, prepare_qat_model
from ml.registry import MODEL_REGISTRY
from models.baselines import AlexNetTV, ResNet18TV, VGG16
from models.final_architecture import AlexNetFinalBottleneckResidual, AlexNetFinalFireResidual

# CLAUDE.md mandates fbgemm for real training runs (PCAD's x86 GPU nodes), but dev/CI boxes
# (e.g. ARM, or an x86 box without AVX2) may lack it — fall back to whatever's supported so this
# test still exercises the same QAT graph-construction logic everywhere.
torch.backends.quantized.engine = (
    "fbgemm" if "fbgemm" in torch.backends.quantized.supported_engines
    else torch.backends.quantized.supported_engines[0]
)

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


def test_vgg16_classifier_linear_relu_fusion_for_qat():
    """Regression test for the QAT collapse in docs/logs/PHASE11_LOG.md: vgg16's classifier.0
    Linear feeds a heavy-tailed raw output (p999 ~27k, max ~115k) straight into an activation
    observer unless Linear+ReLU are fused first, the same mechanism already used for every
    Conv-BN-ReLU stage."""
    assert MODEL_REGISTRY["vgg16"]["classifier_fuse_map"] == CLASSIFIER_FUSE_MAP_VGG16
    assert MODEL_REGISTRY["vgg16_2x2"]["classifier_fuse_map"] == CLASSIFIER_FUSE_MAP_VGG16

    qat_model = build_qat_from_model(VGG16(kernel_size=3), "vgg16", torch.device("cpu"))
    assert isinstance(qat_model.classifier[0], nniqat.LinearReLU)
    assert isinstance(qat_model.classifier[3], nniqat.LinearReLU)
    assert not isinstance(qat_model.classifier[6], nniqat.LinearReLU)  # logits layer, no ReLU after it

    # models without classifier_fuse_map are unaffected
    alexnet_qat = build_qat_from_model(AlexNetTV(pretrained=False), "alexnet_tv_scratch", torch.device("cpu"))
    assert not isinstance(alexnet_qat.classifier[0], nniqat.LinearReLU)


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


def test_logits_layer_stays_float_so_int8_logits_do_not_tie():
    """2026-10-02 audit (docs/logs/PHASE11_LOG.md, "Float logits layer"): the logits Linear's output went through an
    8-bit fake-quant, leaving 8-31 distinct values per image and 14-31% top-1 ties in QAT/INT8. resnet18tv because
    its input QuantStub is registered after fc: the logits layer must be found in execution order."""
    from ml.quantization import _FloatLogits
    x = torch.randn(4, 3, 64, 64)
    for name, model in [("alexnet_3x3_gap", MODEL_REGISTRY["alexnet_3x3_gap"]["ctor"]()),
                        ("resnet18tv", ResNet18TV(pretrained=False))]:
        qat = build_qat_from_model(model, name, torch.device("cpu")).eval()
        qat(x)  # calibrate
        int8 = convert_to_int8(qat)
        heads = [m for m in int8.modules() if isinstance(m, _FloatLogits)]
        assert len(heads) == 1 and type(heads[0].linear) is nn.Linear, name
        assert any(isinstance(m, torch.ao.nn.quantized.Quantize) for m in int8.modules()), f"{name}: input not quantized"
        out = int8(x)
        assert not out.is_quantized and all(len(torch.unique(row)) == 200 for row in out), name


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
