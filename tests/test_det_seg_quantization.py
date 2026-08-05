"""Regression test for the Phase 7 SSD QAT->INT8 dead-branch bug.

build_qat_ssd_detector's qconfig propagated to backbone_full.classifier/.quant/.dequant
and to .features layers past the deepest tap — all structurally present but never called
by DetSegBackbone.forward. prepare_qat wrapped them with observers that never saw data,
so tq.convert() hard-asserted on their inf min/max (broke INT8 conversion for every model,
e.g. jobs 811687/811101/811132). Fixed by setting .qconfig = None on those dead branches.
"""
import torch

from ml.det_seg_models import build_qat_ssd_detector, build_ssd_detector, convert_ssd_to_int8

torch.backends.quantized.engine = (
    "fbgemm" if "fbgemm" in torch.backends.quantized.supported_engines
    else torch.backends.quantized.supported_engines[0]
)

BACKBONES = ["alexnet_bottleneck", "alexnet_fire", "alexnet_tv"]


def test_ssd_qat_converts_to_int8_without_crashing():
    for arch in BACKBONES:
        device = torch.device("cpu")
        model = build_ssd_detector(arch, num_classes=21, image_size=64)
        qat_model = build_qat_ssd_detector(model, device)
        qat_model.train()

        with torch.no_grad():
            qat_model.backbone(torch.randn(1, 3, 64, 64))  # calibrate observers

        int8_model = convert_ssd_to_int8(qat_model)
        with torch.no_grad():
            out = int8_model.backbone(torch.randn(1, 3, 64, 64))

        assert len(out) == len(model.backbone.out_channels), f"{arch} produced the wrong number of feature levels after INT8 convert"
