"""QAT must fake-quantize both weights and activations, including through residual adds.

Regression test for the ResNet18TV bug: its BasicBlock used a raw `out += identity` instead of
FloatFunctional, so the residual add wasn't instrumented and converting to INT8 either produced wrong
results or crashed. Fixed by switching to torchvision's quantizable resnet18. This test builds QAT +
converts to INT8 for every residual-bearing model in the sweep and asserts the forward pass still works.
"""
import torch

from ml.quantization import convert_to_int8, find_fuse_groups, prepare_qat_model
from models.baselines import ResNet18TV
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
