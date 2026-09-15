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


def test_qat_callback_still_applies_on_the_first_epoch_after_a_resume_past_it():
    qat = prepare_qat_model(nn.Sequential(nn.Conv2d(3, 4, 3), nn.BatchNorm2d(4), nn.ReLU(inplace=False)),
                            [["0", "1", "2"]])
    make_qat_callback(freeze_bn_epoch=3, disable_observer_epoch=5)(7, qat)  # resumed at epoch 7
    assert qat[0].freeze_bn  # a module attribute, not state_dict -- lost on resume unless re-applied
    fake_quants = [m for m in qat.modules() if isinstance(m, torch.ao.quantization.FakeQuantizeBase)]
    assert fake_quants and all(int(m.observer_enabled[0]) == 0 for m in fake_quants)
